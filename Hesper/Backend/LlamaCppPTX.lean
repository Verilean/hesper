import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.Layers.Linear
import Hesper.Backend.CUDA

/-!
# llama.cpp PTX loader (Phase 0 hybrid prototype)

Loads externally-compiled PTX from llama.cpp (mmvq.cu, quantize.cu) and
exposes typed launch helpers.  Enabled via env flag `HESPER_USE_LLAMACPP_PTX=1`.

Requires both PTXes present at:
  `/tmp/llamacpp_ptx/mmvq.ptx`
  `/tmp/llamacpp_ptx/quantize.ptx`

Target kernels (sm_89):
  * Q4_K decode matmul — `_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEv...`
  * Q6_K decode matmul — `_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEv...`
  * Q8_1 quantize      — `_Z13quantize_q8_1PKfPvlllllj5uint3`

See docs/llama-fusion-analysis/09-hybrid-prototype.md §0.3-0.4 for ABI.
-/

namespace Hesper.LlamaCppPTX

open Hesper.CUDA

/-- Mangled symbols. -/
def q4kMatmulSymbol : String :=
  "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"

def q6kMatmulSymbol : String :=
  "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"

def q8_1QuantizeSymbol : String :=
  "_Z13quantize_q8_1PKfPvlllllj5uint3"

/-- Q4_K mat-mat (mmq) kernel — prefill path. mmq_x=64, need_check=false.
    Source: `mmq.cuh`'s `mul_mat_q<GGML_TYPE_Q4_K, 64, false>`.
    See `docs/llama-fusion-analysis/31-mmq-q4k-ptx-extraction.md` for ABI. -/
def mmqQ4KMatmulSymbol_x64 : String :=
  "_Z9mul_mat_qIL9ggml_type12ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii"

/-- Cached compiled kernel handles. -/
structure Kernels where
  q4kMatmul : CUfunction
  q6kMatmul : CUfunction
  q8_1Quantize : CUfunction
  /-- Prefill mmq Q4_K kernel — `none` if `mmq_q4k.ptx` was not present. -/
  mmqQ4K_x64 : Option CUfunction := none

initialize kernelsRef : IO.Ref (Option Kernels) ← IO.mkRef none

/-- Default paths.  Override by caller if needed. -/
def defaultMmvqPath : String := "/tmp/llamacpp_ptx/mmvq.ptx"
def defaultQuantizePath : String := "/tmp/llamacpp_ptx/quantize.ptx"
def defaultMmqQ4KPath : String := "/tmp/llamacpp_ptx/mmq_q4k.ptx"

/-- Load PTX modules and resolve target kernels.  Idempotent.
    `mmq_q4k.ptx` is optional — if absent (or driver rejects the PTX
    version), `mmqQ4K_x64` is left `none` so existing decode-only
    callers keep working. See
    `docs/llama-fusion-analysis/31-mmq-q4k-ptx-extraction.md` for the
    extraction recipe and the open PTX-8.7-vs-driver issue. -/
def loadKernels (mmvqPath : String := defaultMmvqPath)
    (quantizePath : String := defaultQuantizePath)
    (mmqQ4KPath : String := defaultMmqQ4KPath) : IO Kernels := do
  match ← kernelsRef.get with
  | some k => return k
  | none =>
    let mmvqSrc ← IO.FS.readFile mmvqPath
    let qSrc ← IO.FS.readFile quantizePath
    let mmvqMod ← cuModuleLoadData mmvqSrc
    let qMod ← cuModuleLoadData qSrc
    let q4k ← cuModuleGetFunction mmvqMod q4kMatmulSymbol
    let q6k ← cuModuleGetFunction mmvqMod q6kMatmulSymbol
    let q8q ← cuModuleGetFunction qMod q8_1QuantizeSymbol
    let mmqQ4K ← (do
      try
        let present ← System.FilePath.pathExists mmqQ4KPath
        if !present then return none
        let mmqSrc ← IO.FS.readFile mmqQ4KPath
        let mmqMod ← cuModuleLoadData mmqSrc
        let f ← cuModuleGetFunction mmqMod mmqQ4KMatmulSymbol_x64
        return some f
      catch e =>
        IO.eprintln s!"[LlamaCppPTX] mmq Q4_K load failed: {e.toString}"
        return none)
    let k : Kernels :=
      { q4kMatmul := q4k, q6kMatmul := q6k, q8_1Quantize := q8q,
        mmqQ4K_x64 := mmqQ4K }
    kernelsRef.set (some k)
    return k

/-- Check env flag `HESPER_USE_LLAMACPP_PTX` (truthy = "1"). -/
def isEnabled : IO Bool := do
  match ← IO.getEnv "HESPER_USE_LLAMACPP_PTX" with
  | some "1" => return true
  | _ => return false

/-! ## Argument packing

CUDA's `cuLaunchKernel` takes `void**` where each entry points at an arg's
value.  For mixed-type args (u64, u32, uint3, 32-byte struct) we pack every
value into a single `ByteArray` and pass per-arg byte offsets. -/

/-- Append a `UInt64` little-endian to the buffer.  Returns new size. -/
private def pushU64 (buf : ByteArray) (v : UInt64) : ByteArray := Id.run do
  let mut b := buf
  for i in [0:8] do
    b := b.push ((v >>> (i.toUInt64 * 8)).toUInt8)
  return b

private def pushU32 (buf : ByteArray) (v : UInt32) : ByteArray := Id.run do
  let mut b := buf
  for i in [0:4] do
    b := b.push ((v >>> (i.toUInt32 * 8)).toUInt8)
  return b

/-- Pad buffer up to `align`-byte boundary (align must be power of two). -/
private def alignTo (buf : ByteArray) (align : Nat) : ByteArray := Id.run do
  let mut b := buf
  while b.size % align != 0 do
    b := b.push 0
  return b

/-- Launch a kernel, routing through the CUDA-Graph capture stream when
    active.  This is the critical fix for the hybrid-PTX garbage: when
    `HESPER_CUDA_GRAPHS=1` is on and decode is mid-capture, kernel launches
    on the default stream are NOT captured into the graph.  Such kernels
    still execute, but they run OUTSIDE the graph's serialized ordering —
    meaning hesper's (captured, replay-ordered) kernels race with override's
    (default-stream, always-live) kernels on shared buffers (Q8_1 scratch,
    per-layer input/output buffers, KV cache).  The race corrupts logits
    with cumulative error that manifests as garbage tokens.

    Fix: use `cuLaunchKernelRawOnStream` with the capture stream when set,
    so override launches become graph nodes just like hesper's kernels. -/
private def launchOnCaptureStream (func : CUfunction)
    (gx gy gz bx byDim bz : UInt32) (smem : UInt32)
    (bytes : ByteArray) (offsets : Array USize) : IO Unit := do
  match ← Hesper.cudaCaptureStream.get with
  | some s => cuLaunchKernelRawOnStream func gx gy gz bx byDim bz smem bytes offsets s
  | none   => cuLaunchKernelRaw func gx gy gz bx byDim bz smem bytes offsets

/-! ## Q8_1 quantize launch

Signature (9 params):
  quantize_q8_1(const float *x, void *vy,
                int64_t ne00, s01, s02, s03, ne0,
                uint32_t ne1, uint3 ne2_fastdiv)

For single-row Q8_1 quantize (decode path):
  ne00 = ne0 = inDim
  s01 = s02 = s03 = 0
  ne1 = 1
  ne2 = (0, 0, 1)  (fastdiv values for divisor=1)

Launch: block=(256,1,1), grid=(ceil(inDim/256), 1, 1). -/
def launchQuantizeQ8_1 (k : Kernels) (xBuf : CUdeviceptr) (yBuf : CUdeviceptr)
    (inDim : Nat) : IO Unit := do
  if inDim % 32 != 0 then
    throw (IO.userError s!"launchQuantizeQ8_1: inDim {inDim} not a multiple of QK8_1=32")
  let mut bytes : ByteArray := ByteArray.empty
  let mut offsets : Array USize := #[]
  -- param 0: const float *x
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes xBuf.toUInt64
  -- param 1: void *vy
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes yBuf.toUInt64
  -- params 2-6: int64_t ne00, s01, s02, s03, ne0
  for v in [inDim, 0, 0, 0, inDim] do
    offsets := offsets.push bytes.size.toUSize
    bytes := pushU64 bytes v.toUInt64
  -- param 7: uint32_t ne1 = 1
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  -- param 8: uint3 ne2_fastdiv = init_fastdiv_values(1) = (mp=1, L=0, d=1)
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  bytes := pushU32 bytes 0
  bytes := pushU32 bytes 1
  let grid := (inDim + 255) / 256
  launchOnCaptureStream k.q8_1Quantize
    grid.toUInt32 1 1
    256 1 1
    0
    bytes offsets

/-! ## Q4_K / Q6_K matmul launch

Signature (19 params) for `ncols_dst=1, has_fusion=false, small_k=false`:
  mul_mat_vec_q(const void *vx, const void *vy, const int32_t *ids,
                ggml_cuda_mm_fusion_args_device fusion,   // 32-byte all-zero
                float *dst,
                uint32_t ncols_x,                         // K-dim = inDim
                uint3 nchannels_y,                        // (0,0,1)
                uint32_t stride_row_x, stride_col_y, stride_col_dst,
                uint3 channel_ratio,                      // (0,0,1)
                uint32_t stride_channel_x, stride_channel_y, stride_channel_dst,
                uint3 sample_ratio,                       // (0,0,1)
                uint32_t stride_sample_x, stride_sample_y, stride_sample_dst,
                uint32_t ids_stride)

Block: (warp_size=32, nwarps=4, 1)
Grid : (outDim / rows_per_cuda_block, 1, 1).  On sm_89 for typical decode
       (ncols_dst=1, small_k=false), `rows_per_cuda_block = 1`, so grid.x = outDim.
       (`small_k` only triggers when blocks_per_row_x < nwarps * 2 = 8, i.e. inDim<2048,
       which does not occur in Gemma 4 E4B.)

Strides (verified against mmvq.cu host code, `ggml_cuda_op_mul_mat_vec_q`):
  stride_row_x   = ne00 / ggml_blck_size(type) = inDim / 256  (Q4_K/Q6_K block count/row)
                   Kernel uses it as `kbx + i*stride_row_x` in block-index space.
  stride_col_y   = src1_padded_row_size / QK8_1 = inDim / 32  (Q8_1 blocks per row)
  stride_col_dst = outDim (rows written per column)
-/
private def packMatmulArgs (weightBuf : CUdeviceptr) (q8Buf : CUdeviceptr)
    (dstBuf : CUdeviceptr) (inDim outDim : Nat)
    (blocksPerRowX : Nat) : ByteArray × Array USize := Id.run do
  let mut bytes : ByteArray := ByteArray.empty
  let mut offsets : Array USize := #[]
  -- 0: const void *vx (weights)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes weightBuf.toUInt64
  -- 1: const void *vy (Q8_1 input)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes q8Buf.toUInt64
  -- 2: const int32_t *ids (NULL)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- 3: ggml_cuda_mm_fusion_args_device (32 bytes, 8-byte aligned, all zero)
  bytes := alignTo bytes 8
  offsets := offsets.push bytes.size.toUSize
  for _ in [0:32] do
    bytes := bytes.push 0
  -- 4: float *dst
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes dstBuf.toUInt64
  -- 5: uint32_t ncols_x = inDim
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes inDim.toUInt32
  -- 6: uint3 nchannels_y — llama.cpp's `init_fastdiv_values(d)` packs
  -- (mp, L, divisor) in (x, y, z).  For d=1: L=0, mp = ((1<<32)*0/1)+1 = 1,
  -- so the correct fastdiv triple is (1, 0, 1), NOT (0, 0, 1).  Passing
  -- (0, 0, 1) made `fastdiv(n, triple) = (n + __umulhi(n, 0)) >> 0 = n`
  -- which WRAPS for channel_dst > 0 in any non-trivial launch config,
  -- but since we always launch with channel_dst=0, fastdiv returned 0
  -- and channel_y was computed as `fastmodulo(0, (0,0,1)) = 0 - 0*1 = 0`,
  -- which is fine; however ANY kernel path that later multiplies by
  -- `nchannels_y.z` (the divisor field) saw value 1 correctly, but
  -- intermediate `fastdiv` temporaries in PTX can still diverge when
  -- mp=0.  Using the real (1, 0, 1) is safer and matches llama.cpp.
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1  -- mp for divisor=1
  bytes := pushU32 bytes 0  -- L
  bytes := pushU32 bytes 1  -- divisor
  -- 7-9: uint32_t stride_row_x, stride_col_y, stride_col_dst
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes blocksPerRowX.toUInt32
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes (inDim / 32).toUInt32
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes outDim.toUInt32
  -- 10: uint3 channel_ratio — same fastdiv(1) triple (1, 0, 1).
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  bytes := pushU32 bytes 0
  bytes := pushU32 bytes 1
  -- 11-13: stride_channel_{x,y,dst} = 0
  for _ in [0:3] do
    offsets := offsets.push bytes.size.toUSize
    bytes := pushU32 bytes 0
  -- 14: uint3 sample_ratio — same fastdiv(1) triple (1, 0, 1).
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  bytes := pushU32 bytes 0
  bytes := pushU32 bytes 1
  -- 15-17: stride_sample_{x,y,dst} = 0
  for _ in [0:3] do
    offsets := offsets.push bytes.size.toUSize
    bytes := pushU32 bytes 0
  -- 18: ids_stride = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  return (bytes, offsets)

/-- Launch llama.cpp's Q4_K matmul.
    `weightBuf` — Q4_K blocks (144 B each)
    `q8Buf`     — Q8_1 blocks (36 B each) produced by `launchQuantizeQ8_1`
    `dstBuf`    — f32 output
    Strides assume a single-row vec×mat (standard decode path). -/
def launchMulMatVecQ4K (k : Kernels) (weightBuf q8Buf dstBuf : CUdeviceptr)
    (inDim outDim : Nat) : IO Unit := do
  if inDim % 256 != 0 then
    throw (IO.userError s!"Q4_K requires inDim % 256 == 0, got {inDim}")
  -- stride_row_x is in Q4_K block units (host passes ne00 / blck_size(Q4_K)=256).
  let blocksPerRowX := inDim / 256
  let (bytes, offsets) := packMatmulArgs weightBuf q8Buf dstBuf inDim outDim blocksPerRowX
  -- On sm_89 for ncols_dst=1 with inDim ≥ 2048: rows_per_cuda_block = 1.
  launchOnCaptureStream k.q4kMatmul
    outDim.toUInt32 1 1
    32 4 1
    0
    bytes offsets

/-- Launch llama.cpp's Q6_K matmul (same host signature as Q4_K, different kernel).
    `weightBuf` — Q6_K blocks (210 B each, 256 weights/block).
    Same stride semantics as Q4_K. -/
def launchMulMatVecQ6K (k : Kernels) (weightBuf q8Buf dstBuf : CUdeviceptr)
    (inDim outDim : Nat) : IO Unit := do
  if inDim % 256 != 0 then
    throw (IO.userError s!"Q6_K requires inDim % 256 == 0, got {inDim}")
  let blocksPerRowX := inDim / 256
  let (bytes, offsets) := packMatmulArgs weightBuf q8Buf dstBuf inDim outDim blocksPerRowX
  launchOnCaptureStream k.q6kMatmul
    outDim.toUInt32 1 1
    32 4 1
    0
    bytes offsets

/-! ## Installation as `Hesper.Layers.Linear` override

Per-layer Q8_1 scratch buffer is sized `(inDim/32) * 36 B`.  To keep the
override signature clean (no backend context in signature), we cache one
scratch buffer per (inDim) globally.  A single decode call runs in one
thread so no contention.  Buffer is lazily allocated on first use.
-/

initialize scratchQ8Ref : IO.Ref (Array (Nat × CUdeviceptr)) ← IO.mkRef #[]

private def getQ8Scratch (inDim : Nat) : IO CUdeviceptr := do
  let arr ← scratchQ8Ref.get
  match arr.find? (·.1 == inDim) with
  | some (_, p) => return p
  | none =>
    let bytes : USize := ((inDim / 32) * 36).toUSize
    let p ← cuMalloc bytes
    scratchQ8Ref.modify (·.push (inDim, p))
    return p

/-- Debug knobs.  Default: Q4_K on; Q6_K falls through to hesper kernel
    (its 210-byte block layout hasn't been cross-checked against hesper yet). -/
initialize enableQ4K : IO.Ref Bool ← IO.mkRef true
initialize enableQ6K : IO.Ref Bool ← IO.mkRef false

/-- Debug counter: how many Q4_K (+Q6_K) calls have reached the override
    since it was installed.  Reset to 0 at install time; monotonically
    incremented on every call.  Used by the `HESPER_LLAMACPP_CALL_FROM` /
    `HESPER_LLAMACPP_CALL_UNTIL` gating below to bisect which specific
    override invocation (not layer — layer has Q/K/V/gate/up/down × 42 =
    252 matmul calls) is responsible for the garbage output.  Single-
    threaded decode so no locking needed. -/
initialize q4kCallCounter : IO.Ref Nat ← IO.mkRef 0

/-- Inclusive [from, until) range of override calls on which to fire the
    llama.cpp kernel.  Calls outside this range return `false` and fall
    back to hesper's kernel.  Defaults (from=0, until=None) = all calls.
    Set via `HESPER_LLAMACPP_CALL_FROM=<n>` / `HESPER_LLAMACPP_CALL_UNTIL=<m>`. -/
initialize callFrom : IO.Ref Nat ← IO.mkRef 0
initialize callUntil : IO.Ref (Option Nat) ← IO.mkRef none

/-- Install `llamaCppDp4aOverride` so `forwardDP4A` dispatches to the llama.cpp
    PTX path.  Call once after CUDA init if `HESPER_USE_LLAMACPP_PTX=1`.
    Per-quant toggles: `HESPER_LLAMACPP_Q4K` (default on), `HESPER_LLAMACPP_Q6K`
    (default off).  Call-range toggles: `HESPER_LLAMACPP_CALL_FROM`,
    `HESPER_LLAMACPP_CALL_UNTIL` for bisection. -/
def installOverride : IO Unit := do
  let k ← loadKernels
  let q4on ← enableQ4K.get
  let q6on ← enableQ6K.get
  -- Match env overrides: HESPER_LLAMACPP_Q4K=0 / HESPER_LLAMACPP_Q6K=1
  let q4on := match ← IO.getEnv "HESPER_LLAMACPP_Q4K" with
              | some "0" => false | _ => q4on
  let q6on := match ← IO.getEnv "HESPER_LLAMACPP_Q6K" with
              | some "1" => true | _ => q6on
  enableQ4K.set q4on
  enableQ6K.set q6on
  -- Call-range gating for bisection.
  let fromN := (← IO.getEnv "HESPER_LLAMACPP_CALL_FROM").bind String.toNat? |>.getD 0
  let untilOpt := (← IO.getEnv "HESPER_LLAMACPP_CALL_UNTIL").bind String.toNat?
  callFrom.set fromN
  callUntil.set untilOpt
  q4kCallCounter.set 0
  IO.println s!"[hesper-llamacpp] Q4_K override = {q4on}, Q6_K override = {q6on}, call range = [{fromN}, {untilOpt})"
  Hesper.Layers.Linear.llamaCppDp4aOverride.set (some fun inPtr wPtr outPtr inDim outDim tag => do
    let useLlama := match tag with
      | 0 => q4on  -- Q4_K
      | 1 => q6on  -- Q6_K (lm_head)
      | _ => false
    if !useLlama then
      return false  -- fall through to hesper's kernel
    -- Call counter + range gating.
    let n ← q4kCallCounter.get
    q4kCallCounter.set (n + 1)
    let fromN ← callFrom.get
    let untilOpt ← callUntil.get
    let inRange := n ≥ fromN &&
      (match untilOpt with | some u => n < u | none => true)
    if !inRange then
      return false  -- fall back to hesper for this call
    let q8Ptr ← getQ8Scratch inDim
    launchQuantizeQ8_1 k inPtr q8Ptr inDim
    match tag with
    | 0 => launchMulMatVecQ4K k wPtr q8Ptr outPtr inDim outDim
    | 1 => launchMulMatVecQ6K k wPtr q8Ptr outPtr inDim outDim
    | _ => throw (IO.userError s!"llamacpp override: unknown quant tag {tag}")
    return true)

/-- Auto-install if env flag set.  Returns `true` if the override is now live. -/
def autoInstall : IO Bool := do
  if ← isEnabled then
    installOverride
    IO.println "[hesper] HESPER_USE_LLAMACPP_PTX=1 — llama.cpp PTX override installed."
    return true
  else
    return false

/-! ## flash_attn_ext_vec<D=256, ncols=1, K=f16, V=f16, false>

Single-token decode path with f16 K/V cache.  Used by Gemma 4 (head_dim=256).

Mangled symbol (from cuobjdump on `fattn-vec-instance-f16-f16.cu.o`):
  `_Z18flash_attn_ext_vecILi256ELi1EL9ggml_type1ELS0_1ELb0EE...`

Launch (from `launch_fattn` in `fattn-common.cuh` line 1048):
  block_dim = (32, 4, 1) = 128 threads
  blocks_num = (ntiles_x, parallel_blocks, ntiles_z_gqa * K->ne[2] * Q->ne[3])
             = (1, 1, numHeads)  for ncols=1, single sequence, parallel_blocks=1
  smem = 0 (kernel uses internal __shared__ arrays)

Kernel signature: 38 params (8 ptrs, then float×4 + u32 + float + i32 + uint3 +
22 i32/i64 strides).  Mangling tail `iiiiiiiiiiiliiliiiiil` decodes:
  i32 ne00, uint3 ne01, i32 ne02, i32 ne03,
       i32 nb01, i32 nb02, i32 nb03,
  i32 ne10, i32 ne11, i32 ne12, i32 ne13,
       i32 nb11, i32 nb12, i64 nb13,
       i32 nb21, i32 nb22, i64 nb23,
       i32 ne31, i32 ne32, i32 ne33,
       i32 nb31, i32 nb32, i64 nb33

For decode at single-block tile (cacheLen ≤ 256), parallel_blocks=1 → no
combine kernel needed and dst_meta is unused.  We pass null for mask, sinks,
KV_max, dst_meta. -/
def fattnVecF16F16D256Symbol : String :=
  "_Z18flash_attn_ext_vecILi256ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_S2_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil"

def defaultFattnVecF16F16Path : String := "/tmp/llamacpp_ptx/fattn_vec_f16f16.ptx"

initialize fattnKernelRef : IO.Ref (Option CUfunction) ← IO.mkRef none

def loadFattnVecF16F16Kernel
    (path : String := defaultFattnVecF16F16Path) : IO CUfunction := do
  match ← fattnKernelRef.get with
  | some k => return k
  | none =>
    let src ← IO.FS.readFile path
    let m ← cuModuleLoadData src
    let f ← cuModuleGetFunction m fattnVecF16F16D256Symbol
    fattnKernelRef.set (some f)
    return f

/-- Pack args + launch `flash_attn_ext_vec<D=256, ncols=1, K=f16, V=f16, false>`.

  qBuf      f32 [numHeads × headDim]                Q for single token
  kBuf      f16 [numKVHeads × maxSeqLen × headDim]  K cache (f16!)
  vBuf      f16 [numKVHeads × maxSeqLen × headDim]  V cache (f16!)
  outBuf    f32 [numHeads × headDim]                attention output
  numHeads, numKVHeads, headDim (=256), cacheLen, maxSeqLen, scale

Strides (bytes):
  nb01 = numHeads * headDim * 4  (Q is one row of all heads, ne01=1 so unused)
  nb02 = headDim * 4
  nb03 = nb02 * numHeads
  nb11 = headDim * 2  (sizeof half)
  nb12 = maxSeqLen * nb11
  nb13 = numKVHeads * nb12
  nb21 = nb11
  nb22 = nb12
  nb23 = nb13
-/
def launchFlashAttnVecF16F16D256
    (k : CUfunction)
    (qBuf : CUdeviceptr) (kBuf : CUdeviceptr) (vBuf : CUdeviceptr)
    (outBuf : CUdeviceptr)
    (numHeads numKVHeads headDim cacheLen maxSeqLen : Nat)
    (scale : Float) : IO Unit := do
  if headDim != 256 then
    throw (IO.userError s!"launchFlashAttnVecF16F16D256: headDim must be 256, got {headDim}")
  if cacheLen > 256 then
    throw (IO.userError s!"launchFlashAttnVecF16F16D256: cacheLen {cacheLen} > 256 (nbatch_fa). Single-block path only; need combine for longer.")
  let mut bytes : ByteArray := ByteArray.empty
  let mut offsets : Array USize := #[]
  -- 0: const char *Q
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes qBuf.toUInt64
  -- 1: const char *K
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes kBuf.toUInt64
  -- 2: const char *V
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes vBuf.toUInt64
  -- 3: const char *mask = NULL
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- 4: const char *sinks = NULL
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- 5: const int *KV_max = NULL
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- 6: float *dst
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes outBuf.toUInt64
  -- 7: float2 *dst_meta = NULL  (parallel_blocks=1 path, gridDim.y=1)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- 8: float scale
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes scale.toFloat32.toBits
  -- 9: float max_bias = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 10: float m0 = 1
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes (1.0 : Float).toFloat32.toBits
  -- 11: float m1 = 1
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes (1.0 : Float).toFloat32.toBits
  -- 12: uint32_t n_head_log2 = 1 (max_bias=0 → 2^0)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  -- 13: float logit_softcap = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 14: int32_t ne00 = headDim
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes headDim.toUInt32
  -- 15: uint3 ne01 = init_fastdiv_values(Q->ne[1]=1) = (mp=1, L=0, d=1)
  bytes := alignTo bytes 4
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1  -- mp
  bytes := pushU32 bytes 0  -- L
  bytes := pushU32 bytes 1  -- divisor
  -- 16: int32_t ne02 = numHeads
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes numHeads.toUInt32
  -- 17: int32_t ne03 = 1
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  let nb01 := numHeads * headDim * 4  -- Q row stride (unused since ne01=1)
  let nb02 := headDim * 4              -- Q per-head stride
  let nb03 := numHeads * nb02
  -- 18: i32 nb01
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb01.toUInt32
  -- 19: i32 nb02
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb02.toUInt32
  -- 20: i32 nb03
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb03.toUInt32
  -- 21: i32 ne10 = headDim
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes headDim.toUInt32
  -- 22: i32 ne11 = cacheLen (K positions)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes cacheLen.toUInt32
  -- 23: i32 ne12 = numKVHeads
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes numKVHeads.toUInt32
  -- 24: i32 ne13 = 1
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 1
  let nb11 := headDim * 2          -- f16 row stride
  let nb12 := maxSeqLen * nb11
  let nb13 := numKVHeads * nb12
  -- 25: i32 nb11
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb11.toUInt32
  -- 26: i32 nb12
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb12.toUInt32
  -- 27: i64 nb13
  bytes := alignTo bytes 8
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes nb13.toUInt64
  -- 28: i32 nb21 = nb11
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb11.toUInt32
  -- 29: i32 nb22 = nb12
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes nb12.toUInt32
  -- 30: i64 nb23 = nb13
  bytes := alignTo bytes 8
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes nb13.toUInt64
  -- 31: i32 ne31 = 0 (no mask)
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 32: i32 ne32 = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 33: i32 ne33 = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 34: i32 nb31 = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 35: i32 nb32 = 0
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU32 bytes 0
  -- 36: i64 nb33 = 0
  bytes := alignTo bytes 8
  offsets := offsets.push bytes.size.toUSize
  bytes := pushU64 bytes 0
  -- Launch: blocks=(1, 1, numHeads), block_dim=(32, 4, 1), smem=0
  launchOnCaptureStream k
    1 1 numHeads.toUInt32
    32 4 1
    0
    bytes offsets

end Hesper.LlamaCppPTX
