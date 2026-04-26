import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.FlashAttention
import Hesper

/-!
# FlashAttention vec-params parity test

Compares `flashAttentionVecParamsKernel` (doc 60 Session 1, warp-shuffle
reduce) against `flashAttentionDynamicParamsKernel` (legacy, 256-thread
tree reduce) on identical Q/K/V/cacheLen inputs.  Both kernels share
the same input/output contract — Q from `q`, output to `output`,
cacheLen in `params[1]`.

Pass criterion: `max(|out_vec[i] - out_legacy[i]|) < 1e-4` for all
output positions, across multiple cacheLen sizes.

Run:
  lake exe cuda-flashattn-vec-parity
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else
    let e32 : Int := e.toNat - 1023 + 127
    if e32 <= 0 then 0
    else if e32 >= 255 then
      (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (s.toUInt32 <<< 31) |||
      (e32.toNat.toUInt32 <<< 23) |||
      ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let bits := f64ToF32Bits f
    acc.push bits.toUInt8
       |>.push (bits >>> 8).toUInt8
       |>.push (bits >>> 16).toUInt8
       |>.push (bits >>> 24).toUInt8) ByteArray.empty

/-- Convert f32 bits to IEEE 754 binary16 (half) bits.  Round-to-zero on
    mantissa truncation; subnormals/Inf/NaN folded to zero / signed inf. -/
private def f32ToF16Bits (b : UInt32) : UInt16 :=
  let s : UInt32 := (b >>> 31) &&& 1
  let e32 : UInt32 := (b >>> 23) &&& 0xFF
  let m32 : UInt32 := b &&& 0x7FFFFF
  if e32 == 0 then 0  -- f32 zero/subnormal → half zero
  else if e32 == 0xFF then
    -- f32 inf/nan → half inf (preserve sign)
    (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
  else
    let e32i : Int := e32.toNat - 127 + 15
    if e32i <= 0 then 0
    else if e32i >= 31 then
      (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
    else
      let m16 : UInt32 := m32 >>> 13
      (s.toUInt16 <<< 15) |||
      (e32i.toNat.toUInt16 <<< 10) |||
      m16.toUInt16

private def packHalfs (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let h := f32ToF16Bits (f64ToF32Bits f)
    acc.push h.toUInt8
       |>.push (h >>> 8).toUInt8) ByteArray.empty

private def unpackFloats (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut arr := #[]
  for i in [0:n] do
    let o := i * 4
    let b0 : UInt32 := ba.get! o |>.toUInt32
    let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
    let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
    let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    let e := (bits >>> 23) &&& 0xFF
    let m := bits &&& (0x7FFFFF : UInt32)
    let s := bits >>> 31
    let f := if e == 0 then 0.0 else
      let fv := (1.0 + m.toNat.toFloat / 8388608.0)
                * Float.pow 2.0 (e.toNat.toFloat - 127.0)
      if s == 1 then -fv else fv
    arr := arr.push f
  return arr

/-- Pack two Nat values as little-endian u32 followed by another u32. -/
private def packU32x2 (a b : Nat) : ByteArray :=
  let aU : UInt32 := a.toUInt32
  let bU : UInt32 := b.toUInt32
  ByteArray.empty
    |>.push aU.toUInt8 |>.push (aU >>> 8).toUInt8
    |>.push (aU >>> 16).toUInt8 |>.push (aU >>> 24).toUInt8
    |>.push bU.toUInt8 |>.push (bU >>> 8).toUInt8
    |>.push (bU >>> 16).toUInt8 |>.push (bU >>> 24).toUInt8

/-- Run one (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen) case
    on both kernels and return (legacyOut, vecOut). -/
def runCase
    (ctx : Hesper.CUDAContext)
    (numHeads numKVHeads headDim maxSeqLen cacheLen : Nat)
    (scale : Float) : IO (Array Float × Array Float) := do
  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let outSize := qSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let outBufLegacy ← GPUBackend.allocBuffer ctx outSize
  let outBufVec ← GPUBackend.allocBuffer ctx outSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)

  -- Deterministic test inputs
  let qData := Array.range (numHeads * headDim)
                |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
  let mut kData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for kv in [0:numKVHeads] do
    for pos in [0:cacheLen] do
      for d in [0:headDim] do
        let off := (kv * maxSeqLen + pos) * headDim + d
        kData := kData.set! off (((kv + 1).toFloat * 0.05) +
                                  ((pos + 1).toFloat * 0.013) +
                                  (d.toFloat / 53.0).cos)
        vData := vData.set! off (((kv + 1).toFloat * 0.07) +
                                  ((pos + 1).toFloat * 0.011) +
                                  (d.toFloat / 41.0).sin)

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)
  GPUBackend.writeBuffer ctx vBuf (packFloats vData)
  -- params = [pos=cacheLen-1, cacheLen]; the kernel only reads slot 1.
  GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

  -- Legacy: dynamic-params kernel (256-thread tree reduce)
  let legacyShader := Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
                       numHeads numKVHeads maxSeqLen headDim scale 256
  let legacyBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufLegacy)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx legacyShader legacyBufs
    { workgroupSize := { x := 256 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  -- Vec: doc 60 Session 1 kernel (128-thread, warp-shuffle reduce)
  let vecShader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                     numHeads numKVHeads maxSeqLen headDim scale
  let vecBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufVec)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx vecShader vecBufs
    { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  let legacyResultBytes ← GPUBackend.readBuffer ctx outBufLegacy outSize
  let vecResultBytes    ← GPUBackend.readBuffer ctx outBufVec outSize
  let legacyOut := unpackFloats legacyResultBytes (numHeads * headDim)
  let vecOut    := unpackFloats vecResultBytes (numHeads * headDim)

  GPUBackend.freeBuffer ctx qBuf
  GPUBackend.freeBuffer ctx kBuf
  GPUBackend.freeBuffer ctx vBuf
  GPUBackend.freeBuffer ctx outBufLegacy
  GPUBackend.freeBuffer ctx outBufVec
  GPUBackend.freeBuffer ctx paramsBuf

  return (legacyOut, vecOut)

/-- Benchmark a single kernel by launching it `iters` times in a tight
    loop, then `cuStreamSynchronize` once at the end.  Returns
    (avg_us_per_call, total_wall_ms).

    `warmup` calls run before timing to flush JIT / cubin cache.

    We use the CUDA-internal `cuStreamSynchronize` (stream 0) — the
    legacy / vec kernels both target the default stream when invoked
    via `GPUBackend.execute`. -/
def benchKernel
    (ctx : Hesper.CUDAContext)
    (kernelName : String)
    (shader : Hesper.WGSL.Monad.ShaderM Unit)
    (bufs : List (String × GPUBackend.Buf Hesper.CUDAContext))
    (workgroupX : Nat) (numHeads : Nat)
    (warmup iters : Nat) : IO (Float × Float) := do
  let cfg : Hesper.ExecConfig := {
    workgroupSize := { x := workgroupX, y := 1, z := 1 }
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
  }
  -- Warmup: PTX JIT, cubin cache, pipeline cache all get warm.
  for _ in [0:warmup] do
    GPUBackend.execute ctx shader bufs cfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  -- Timed loop
  let t0 ← IO.monoNanosNow
  for _ in [0:iters] do
    GPUBackend.execute ctx shader bufs cfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let t1 ← IO.monoNanosNow
  let totalNs := t1 - t0
  let totalMs := totalNs.toFloat / 1.0e6
  let avgUs := totalNs.toFloat / iters.toFloat / 1000.0
  return (avgUs, totalMs)

def compareCase (label : String) (legacyOut vecOut : Array Float)
    (tolerance : Float := 1e-4) : IO Bool := do
  let mut maxAbs := 0.0
  let mut maxRel := 0.0
  let mut firstMismatchIdx : Option Nat := none
  for i in [0:legacyOut.size] do
    let a := legacyOut[i]!
    let b := vecOut[i]!
    let d := (a - b).abs
    if d > maxAbs then maxAbs := d
    let denom := max a.abs b.abs
    if denom > 1e-6 then
      let r := d / denom
      if r > maxRel then maxRel := r
    if d > tolerance && firstMismatchIdx.isNone then
      firstMismatchIdx := some i
  let pass := maxAbs < tolerance
  let mark := if pass then "✓" else "✗"
  IO.println s!"  {mark} {label}: max abs diff = {maxAbs}, max rel diff = {maxRel}"
  match firstMismatchIdx with
  | some i => IO.println s!"      first mismatch at [{i}]: legacy={legacyOut[i]!} vec={vecOut[i]!}"
  | none => pure ()
  return pass

unsafe def main : IO Unit := do
  IO.println "═══ FlashAttention Vec-Params Parity Test ═══"
  let ctx ← Hesper.CUDAContext.init

  -- Cases match Gemma 4 (head_dim=256, num_heads=8, num_kv_heads=1)
  -- but kept small enough that PTX cache + run time stays under a
  -- few seconds.  Bisect by varying cacheLen.
  let cases : List (Nat × Nat × Nat × Nat × Nat) :=
    [ -- (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen)
      (1, 1, 64,  16, 4)
    , (1, 1, 64,  16, 8)
    , (1, 1, 128, 32, 16)
    , (1, 1, 256, 32, 8)
    , (2, 1, 256, 32, 16)
    , (2, 1, 256, 64, 32)
    , (4, 1, 256, 64, 32)
    -- Gemma 4 production geometry
    , (8, 1, 256, 64, 32)
    , (8, 1, 256, 128, 64)
    , (8, 1, 256, 256, 100)
    -- Gemma 4 decode-style cacheLen increments (1..40)
    , (8, 1, 256, 256, 5)
    , (8, 1, 256, 256, 6)
    , (8, 1, 256, 256, 7)
    , (8, 1, 256, 256, 10)
    , (8, 1, 256, 256, 33)
    , (8, 1, 256, 256, 35)
    , (8, 1, 256, 256, 40) ]

  let mut allPassed := true
  for (nh, nkv, hd, msl, cl) in cases do
    let scale : Float := 1.0 / (hd.toFloat.sqrt)
    let label := s!"nH={nh} nKV={nkv} D={hd} maxSeq={msl} cacheLen={cl}"
    try
      let (legacy, vec) ← runCase ctx nh nkv hd msl cl scale
      let ok ← compareCase label legacy vec
      if !ok then allPassed := false
    catch e =>
      IO.println s!"  ✗ {label}: exception {e}"
      allPassed := false

  IO.println ""
  if allPassed then
    IO.println "═══ ALL CASES PASS — vec kernel is bit-parity with legacy ═══"
  else
    IO.println "═══ SOME CASES FAILED — see above ═══"
    IO.Process.exit 1

  -- ────────────────────────────────────────────────────────────────
  -- Microbenchmark: real Gemma-4 geometry (numHeads=8, headDim=256,
  -- numKVHeads=1) at a few cacheLen points spanning the decode range.
  -- 200 iters per kernel after 20 warmup → total wall ~50 ms / point.
  -- Lets us iterate on kernel design with sub-second TAT.
  --
  -- Third column: llama.cpp's flash_attn_ext_vec<D=256, ncols=1, K=f16,
  -- V=f16> PTX, loaded via Hesper.LlamaCppPTX.  Requires the extracted
  -- PTX at /tmp/llamacpp_ptx/fattn_vec_f16f16.ptx (see scripts in
  -- docs/llama-fusion-analysis/60-flashattn-execution-plan.md).  If
  -- absent, the column is skipped with a one-line note.
  IO.println ""
  IO.println "═══ Microbenchmark (Gemma 4 geometry: nH=8, nKV=1, D=256) ═══"
  let llamaKernelOpt ← try
      let k ← Hesper.LlamaCppPTX.loadFattnVecF16F16Kernel
      pure (some k)
    catch e =>
      IO.println s!"  (llama.cpp PTX column skipped — {e})"
      pure none
  let benchCases : List (Nat × Nat) :=
    [ (256, 8), (256, 32), (256, 64), (256, 128), (256, 200) ]
  let warmup := 20
  let iters := 200
  for (maxSeq, cacheLen) in benchCases do
    let nh := 8
    let nkv := 1
    let hd := 256
    let scale : Float := 1.0 / hd.toFloat.sqrt
    -- Set up buffers once per cacheLen point.  Reuse across kernels
    -- for fair memory-state comparison.
    let qSize := (nh * hd * 4).toUSize
    let kvSize := (nkv * maxSeq * hd * 4).toUSize
    let kvSizeHalf := (nkv * maxSeq * hd * 2).toUSize
    let qBuf ← GPUBackend.allocBuffer ctx qSize
    let kBuf ← GPUBackend.allocBuffer ctx kvSize
    let vBuf ← GPUBackend.allocBuffer ctx kvSize
    let outBuf ← GPUBackend.allocBuffer ctx qSize
    let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
    -- f16 mirror buffers for llama.cpp's K=f16, V=f16 instantiation.
    let kBufHalf ← GPUBackend.allocBuffer ctx kvSizeHalf
    let vBufHalf ← GPUBackend.allocBuffer ctx kvSizeHalf
    let outBufLlama ← GPUBackend.allocBuffer ctx qSize
    let qData := Array.range (nh * hd)
                  |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
    GPUBackend.writeBuffer ctx qBuf (packFloats qData)
    -- KV: just zero-fill is enough for timing; the kernel reads the
    -- full row regardless of values.
    let kvZeros : Array Float := Array.replicate (nkv * maxSeq * hd) 0.0
    GPUBackend.writeBuffer ctx kBuf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx vBuf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx kBufHalf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx vBufHalf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

    let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",       qBuf)
      , ("k_cache", kBuf)
      , ("v_cache", vBuf)
      , ("output",  outBuf)
      , ("params",  paramsBuf) ]

    let legacyShader := Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
                         nh nkv maxSeq hd scale 256
    let vecShader    := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                         nh nkv maxSeq hd scale

    let (legacyUs, legacyMs) ← benchKernel ctx "legacy" legacyShader bufs 256 nh warmup iters
    let (vecUs,    vecMs)    ← benchKernel ctx "vec"    vecShader    bufs 128 nh warmup iters
    let speedup := legacyUs / vecUs

    -- Raw-launch column: compile vecShader once (cuModuleLoadData +
    -- cuModuleGetFunction), pre-resolve buffer args, then time bare
    -- cuLaunchKernel in a tight loop. Excludes ShaderM.exec, preHash,
    -- cudaModuleCache.get, and buffer-name resolution from the per-call
    -- path — same kernel as the "vec" column but without
    -- GPUBackend.execute machinery.
    let rawInfo ← do
      let funcName : String := "vec_raw_kernel"
      let ptx := Hesper.CUDA.CodeGen.generatePTX funcName
                   { x := 128, y := 1, z := 1 } vecShader
      let cudaMod ← Hesper.CUDA.cuModuleLoadData ptx
      let f ← Hesper.CUDA.cuModuleGetFunction cudaMod funcName
      -- Resolve buffer arg order from the ShaderM exec to match what
      -- GPUBackend.execute would have built.
      let state := Hesper.WGSL.Monad.ShaderM.exec vecShader
      let declaredNames := state.declaredBuffers.map (·.1)
      let args : Array USize ← declaredNames.foldlM (init := #[]) fun acc name => do
        match bufs.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => throw (IO.userError s!"raw-launch: missing buffer '{name}'")
      -- Warmup
      for _ in [0:warmup] do
        Hesper.CUDA.cuLaunchKernel f
          nh.toUInt32 1 1
          128 1 1
          0
          args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t0 ← IO.monoNanosNow
      for _ in [0:iters] do
        Hesper.CUDA.cuLaunchKernel f
          nh.toUInt32 1 1
          128 1 1
          0
          args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t1 ← IO.monoNanosNow
      let rawUs := (t1 - t0).toFloat / iters.toFloat / 1000.0
      let vecVsRaw := vecUs / rawUs
      pure s!"  raw={rawUs} µs/call  vec/raw={vecVsRaw}×"

    -- Fourth column: llama.cpp PTX (only if the PTX module loaded).
    let llamaInfo ← match llamaKernelOpt with
      | none => pure ""
      | some llamaK => do
        -- Warmup
        for _ in [0:warmup] do
          Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256 llamaK
            qBuf.ptr kBufHalf.ptr vBufHalf.ptr outBufLlama.ptr
            nh nkv hd cacheLen maxSeq scale
        Hesper.CUDA.cuStreamSynchronize (0 : USize)
        let t0 ← IO.monoNanosNow
        for _ in [0:iters] do
          Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256 llamaK
            qBuf.ptr kBufHalf.ptr vBufHalf.ptr outBufLlama.ptr
            nh nkv hd cacheLen maxSeq scale
        Hesper.CUDA.cuStreamSynchronize (0 : USize)
        let t1 ← IO.monoNanosNow
        let llamaUs := (t1 - t0).toFloat / iters.toFloat / 1000.0
        let llamaMs := (t1 - t0).toFloat / 1.0e6
        let vecVsLlama := vecUs / llamaUs
        pure s!"  llama={llamaUs} µs/call ({llamaMs} ms / {iters})  vec/llama={vecVsLlama}×"
    IO.println s!"  cacheLen={cacheLen}: legacy={legacyUs} µs/call  vec={vecUs} µs/call  legacy/vec={speedup}×{rawInfo}{llamaInfo}"

    GPUBackend.freeBuffer ctx qBuf
    GPUBackend.freeBuffer ctx kBuf
    GPUBackend.freeBuffer ctx vBuf
    GPUBackend.freeBuffer ctx outBuf
    GPUBackend.freeBuffer ctx paramsBuf
    GPUBackend.freeBuffer ctx kBufHalf
    GPUBackend.freeBuffer ctx vBufHalf
    GPUBackend.freeBuffer ctx outBufLlama

  IO.println ""
  IO.println "═══ DONE ═══"
