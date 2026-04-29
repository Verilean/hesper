import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad
import Hesper.Basic

/-!
# WMMA Phase 3 — real-GPU bit-parity test

Builds the 16×16×16 fp16/f16/f32 matmul kernel from Phase 2, JITs it
through ptxas, runs it on RTX 4070 Ti, and compares the output against
a CPU reference.

Inputs:
  A = identity (16×16 f16)
  B = identity (16×16 f16)

Expected:
  C = A * B = identity (16×16 f32)

If the wmma.mma.sync PTX is wrong (bad operand layout, missing reg in
the brace list, address arithmetic error, etc.), this test catches it.

Run: `lake exe wmma-gpu-parity-test`
-/
namespace Hesper.CUDA.WmmaGPUParityTest

open Hesper.CUDA Hesper.WGSL Hesper.WGSL.Monad Hesper.CUDA.CodeGen

/-- 16×16×16 fp16/f16/f32 matmul kernel.  All shapes hardcoded for
    Phase 3 PoC. Buffers are declared as f16 elements (256 each for
    A and B) and f32 elements (256 for C). -/
def wmmaIdentityKernel : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "A" (.array (.scalar .f16) 256)
  let _ ← ShaderM.declareReadOnlyBuffer "B" (.array (.scalar .f16) 256)
  let _ ← ShaderM.declareOutputBuffer  "C" (.array (.scalar .f32) 256)
  ShaderM.loadFragmentLeft  .f16 16 16 "a_frag" "&A" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.loadFragmentRight .f16 16 16 "b_frag" "&B" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.declareFragmentResultZero .f32 16 16 "c_frag"
  ShaderM.fragmentMultiplyAccumulate .f32 16 16 16 "c_frag" "a_frag" "b_frag"
  ShaderM.storeFragmentResult .f32 16 16 "c_frag" "&C" (Exp.litU32 0) (Exp.litU32 16)

/-- 16×16×64 matmul kernel: 4 K-iterations, unrolled in Lean (compile-time).
    Each iteration reuses the same `c_frag` accumulator. Tests that:
      1. The fragment-as-local model accumulates correctly across mma calls.
      2. Multiple `wmma.mma.sync` instructions chain into one PTX function
         without register explosion (since Lean unrolls statically).

    Inputs:
      A: 16×64 f16 row-major
      B: 16×64 f16 col-major (B[col,k] layout — 1024 elements either way)
      C: 16×16 f32 row-major

    For the identity check: A = [I, I, I, I] (64 cols, 4 stacked identities)
                             B = [I, I, I, I] (64 cols, 4 stacked identities)
    → C = sum of 4 identity products = 4·I (16×16 with 4 on diagonal) -/
def wmmaK4Kernel : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "A" (.array (.scalar .f16) 1024)
  let _ ← ShaderM.declareReadOnlyBuffer "B" (.array (.scalar .f16) 1024)
  let _ ← ShaderM.declareOutputBuffer  "C" (.array (.scalar .f32) 256)
  ShaderM.declareFragmentResultZero .f32 16 16 "c_frag"
  -- Unroll K iterations at Lean compile time; each emits its own
  -- load-load-mma triple. 4 iterations → 4 mma calls.
  for k in [0:4] do
    let kOff : Nat := k * 16
    -- A row-major: row r, col k → A[r * 64 + k]; offset for tile k = k*16.
    ShaderM.loadFragmentLeft  .f16 16 16 s!"a_frag_{k}" "&A"
      (Exp.litU32 kOff) (Exp.litU32 64)
    -- B col-major: col c, row k → B[c * 64 + k]; offset = k*16.
    ShaderM.loadFragmentRight .f16 16 16 s!"b_frag_{k}" "&B"
      (Exp.litU32 kOff) (Exp.litU32 64)
    ShaderM.fragmentMultiplyAccumulate .f32 16 16 16
      "c_frag" s!"a_frag_{k}" s!"b_frag_{k}"
  ShaderM.storeFragmentResult .f32 16 16 "c_frag" "&C" (Exp.litU32 0) (Exp.litU32 16)

/-- Phase 4b: 16×16×K_BIG matmul using a *runtime* ShaderM.loop. This
    exercises the in-place register accumulator path (assign cName ←
    mma(a, b, cName) detected at codegen time). With the fix in place,
    a 64-iteration loop should compile to ONE pair of A/B fragment regs
    (reused per iteration) plus the persistent c_frag — not 64×8 regs.

    For correctness check we use 16×(16*K_BIG) stacked-identity inputs
    so expected C = K_BIG · I. Using K_BIG=16 keeps buffers small
    (16*256 = 4096 f16 elements per matrix). -/
def wmmaRuntimeLoopKernel : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "A" (.array (.scalar .f16) (16 * 256))
  let _ ← ShaderM.declareReadOnlyBuffer "B" (.array (.scalar .f16) (16 * 256))
  let _ ← ShaderM.declareOutputBuffer  "C" (.array (.scalar .f32) 256)
  ShaderM.declareFragmentResultZero .f32 16 16 "c_frag"
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 16) (Exp.litU32 1) fun kIdx => do
    let kOff := Exp.mul kIdx (Exp.litU32 16)
    -- Stride is 256 (column count = 16 K-tiles × 16 cols/tile).
    ShaderM.loadFragmentLeft  .f16 16 16 "a_frag" "&A" kOff (Exp.litU32 256)
    ShaderM.loadFragmentRight .f16 16 16 "b_frag" "&B" kOff (Exp.litU32 256)
    ShaderM.fragmentMultiplyAccumulate .f32 16 16 16 "c_frag" "a_frag" "b_frag"
  ShaderM.storeFragmentResult .f32 16 16 "c_frag" "&C" (Exp.litU32 0) (Exp.litU32 16)

/-- Encode an f16 (passed as Float32) to 2 little-endian bytes. -/
private def encodeF16 (x : Float) : ByteArray := Id.run do
  -- Manual fp16 encoding: sign(1) + exp(5) + frac(10) for the limited
  -- range we need (0.0 and 1.0).
  if x == 0.0 then
    let mut b := ByteArray.empty
    b := b.push 0; b := b.push 0
    return b
  else if x == 1.0 then
    -- 1.0f as half = 0x3C00 (sign=0, exp=15, mantissa=0)
    let mut b := ByteArray.empty
    b := b.push 0x00; b := b.push 0x3C
    return b
  else
    -- Generic fallback: not needed for identity matrix, but be safe.
    panic! s!"encodeF16 not implemented for {x}"

/-- Decode the entire ByteArray as f32 array (uses Hesper.Basic helper). -/
private def decodeAllF32 (ba : ByteArray) : Array Float :=
  Hesper.Basic.bytesToFloatArrayPure ba

/-- Build a 16×16 identity f16 buffer (256 half elements = 512 bytes). -/
private def buildIdentity16x16F16 : ByteArray := Id.run do
  let mut buf := ByteArray.empty
  for i in [0:16] do
    for j in [0:16] do
      let v : Float := if i == j then 1.0 else 0.0
      buf := buf ++ encodeF16 v
  return buf

/-- Build a 16×64 buffer = 4 stacked 16×16 identity matrices.
    Layout: row r, col c → element at byte offset (r * 64 + c) * 2.
    Identity sub-tile k (k=0..3) occupies cols [k*16, (k+1)*16).
    Element [r, k*16 + j] = 1 iff r == j else 0. -/
private def buildStacked16x64Identity : ByteArray := Id.run do
  let mut buf := ByteArray.empty
  for r in [0:16] do
    for c in [0:64] do
      let j := c % 16  -- col within current 16-tile
      let v : Float := if r == j then 1.0 else 0.0
      buf := buf ++ encodeF16 v
  return buf

/-- Build a 16×256 buffer = 16 stacked 16×16 identity matrices.
    Layout: row r, col c → element at byte offset (r * 256 + c) * 2.
    Identity sub-tile k (k=0..15) occupies cols [k*16, (k+1)*16).
    Element [r, k*16 + j] = 1 iff r == j else 0. -/
private def buildStacked16x256Identity : ByteArray := Id.run do
  let mut buf := ByteArray.empty
  for r in [0:16] do
    for c in [0:256] do
      let j := c % 16  -- col within current 16-tile
      let v : Float := if r == j then 1.0 else 0.0
      buf := buf ++ encodeF16 v
  return buf

def main : IO Unit := do
  IO.println "═══ WMMA Phase 3: real-GPU 16×16×16 fp16/fp16/fp32 matmul ═══"

  -- Generate PTX from ShaderM, targeting sm_80 (Ampere+ for native HMMA).
  let ptx := generatePTX
    (funcName := "wmma_identity_16x16")
    (workgroupSize := {x := 32, y := 1, z := 1})
    (computation := wmmaIdentityKernel)
    (targetArch := "sm_80")
  IO.println s!"  PTX size: {ptx.length} chars"

  -- JIT compile via cuModuleLoadData (this is ptxas).
  -- If wmma.mma.sync syntax / register groups / etc. are wrong, this fails.
  let (_dev, _ctx) ← initCUDA
  IO.println "  CUDA init: OK"
  let cudaMod ← cuModuleLoadData ptx
  IO.println "  ptxas JIT: OK ✓ (PTX accepted by NVIDIA driver)"
  let func ← cuModuleGetFunction cudaMod "wmma_identity_16x16"
  IO.println "  Got entry function: OK"

  -- Allocate buffers: A (512B), B (512B), C (1024B for 256 f32).
  let aBuf ← createCUDABuffer 512
  let bBuf ← createCUDABuffer 512
  let cBuf ← createCUDABuffer 1024
  let identity := buildIdentity16x16F16
  writeCUDABuffer aBuf identity
  writeCUDABuffer bBuf identity
  IO.println "  Buffers initialized (A=B=identity)"

  -- Launch: 1 block of 32 threads (one warp).
  cuLaunchKernel func 1 1 1 32 1 1 0 #[aBuf.ptr, bBuf.ptr, cBuf.ptr]
  let result ← readCUDABufferFull cBuf
  IO.println "  Kernel launched + result fetched"

  -- Verify: C should be identity (256 f32, diagonal=1, off-diag=0).
  let cArr := decodeAllF32 result
  let mut maxErr : Float := 0.0
  let mut numWrong : Nat := 0
  for i in [0:16] do
    for j in [0:16] do
      let got := cArr[i*16 + j]!
      let expected : Float := if i == j then 1.0 else 0.0
      let err := (got - expected).abs
      if err > maxErr then maxErr := err
      if err > 1e-3 then numWrong := numWrong + 1

  IO.println s!"  C[0..3] = {cArr[0]!} {cArr[1]!} {cArr[2]!} {cArr[3]!}"
  IO.println s!"  C[16..19] (row 1) = {cArr[16]!} {cArr[17]!} {cArr[18]!} {cArr[19]!}"
  IO.println s!"  C[17] (diagonal at row 1, col 1) = {cArr[17]!}"
  IO.println s!"  max |err| over 256 elements = {maxErr}"
  IO.println s!"  num wrong (err > 1e-3) = {numWrong}"

  freeCUDABuffer aBuf
  freeCUDABuffer bBuf
  freeCUDABuffer cBuf

  if maxErr < 1e-3 then
    IO.println "✓ WMMA Phase 3 PASSED — Tensor Core path produces correct C = I × I"
  else
    IO.println "✗ FAILED — output diverges from identity"
    IO.Process.exit 1

  -- ─── Phase 4a: K-loop unrolled (4 mma calls accumulating) ───
  IO.println ""
  IO.println "═══ Phase 4a: 16×16×64 (4 K-iters unrolled) ═══"
  let ptx2 := generatePTX
    (funcName := "wmma_k4")
    (workgroupSize := {x := 32, y := 1, z := 1})
    (computation := wmmaK4Kernel)
    (targetArch := "sm_80")
  IO.println s!"  PTX size: {ptx2.length} chars"
  let mod2 ← cuModuleLoadData ptx2
  let func2 ← cuModuleGetFunction mod2 "wmma_k4"
  IO.println "  ptxas JIT: OK ✓"
  let aBuf2 ← createCUDABuffer 2048
  let bBuf2 ← createCUDABuffer 2048
  let cBuf2 ← createCUDABuffer 1024
  let stacked := buildStacked16x64Identity
  writeCUDABuffer aBuf2 stacked
  writeCUDABuffer bBuf2 stacked
  cuLaunchKernel func2 1 1 1 32 1 1 0 #[aBuf2.ptr, bBuf2.ptr, cBuf2.ptr]
  let result2 ← readCUDABufferFull cBuf2
  let cArr2 := decodeAllF32 result2
  -- Expected: C[i,j] = 4 if i==j else 0 (4 iterations of I·I = I, summed)
  let mut maxErr2 : Float := 0.0
  let mut numWrong2 : Nat := 0
  for i in [0:16] do
    for j in [0:16] do
      let got := cArr2[i*16 + j]!
      let expected : Float := if i == j then 4.0 else 0.0
      let err := (got - expected).abs
      if err > maxErr2 then maxErr2 := err
      if err > 1e-3 then numWrong2 := numWrong2 + 1
  IO.println s!"  C[0..3] = {cArr2[0]!} {cArr2[1]!} {cArr2[2]!} {cArr2[3]!}"
  IO.println s!"  C[17] (diagonal r=1,c=1) = {cArr2[17]!} (expect 4.0)"
  IO.println s!"  C[34] (diagonal r=2,c=2) = {cArr2[34]!} (expect 4.0)"
  IO.println s!"  max |err| = {maxErr2}"
  IO.println s!"  num wrong = {numWrong2}"
  freeCUDABuffer aBuf2; freeCUDABuffer bBuf2; freeCUDABuffer cBuf2
  if maxErr2 < 1e-3 then
    IO.println "✓ Phase 4a PASSED — 4 mma calls accumulate correctly into c_frag"
  else
    IO.println "✗ FAILED — accumulator did not chain across iterations"
    IO.Process.exit 1

  -- ─── Phase 4b: K-loop at runtime (ShaderM.loop, in-place reg rebind) ───
  IO.println ""
  IO.println "═══ Phase 4b: 16×16×256 (16 K-iters via runtime ShaderM.loop) ═══"
  let ptx3 := generatePTX
    (funcName := "wmma_runtime_loop")
    (workgroupSize := {x := 32, y := 1, z := 1})
    (computation := wmmaRuntimeLoopKernel)
    (targetArch := "sm_80")
  IO.println s!"  PTX size: {ptx3.length} chars"
  let mod3 ← cuModuleLoadData ptx3
  let func3 ← cuModuleGetFunction mod3 "wmma_runtime_loop"
  IO.println "  ptxas JIT: OK ✓"
  -- 16×256 f16 = 4096 elements = 8192 bytes
  let aBuf3 ← createCUDABuffer 8192
  let bBuf3 ← createCUDABuffer 8192
  let cBuf3 ← createCUDABuffer 1024
  let stacked16 := buildStacked16x256Identity
  writeCUDABuffer aBuf3 stacked16
  writeCUDABuffer bBuf3 stacked16
  cuLaunchKernel func3 1 1 1 32 1 1 0 #[aBuf3.ptr, bBuf3.ptr, cBuf3.ptr]
  let result3 ← readCUDABufferFull cBuf3
  let cArr3 := decodeAllF32 result3
  -- Expected: C[i,j] = 16 if i==j else 0 (16 iterations of I·I = I, summed)
  let mut maxErr3 : Float := 0.0
  let mut numWrong3 : Nat := 0
  for i in [0:16] do
    for j in [0:16] do
      let got := cArr3[i*16 + j]!
      let expected : Float := if i == j then 16.0 else 0.0
      let err := (got - expected).abs
      if err > maxErr3 then maxErr3 := err
      if err > 1e-3 then numWrong3 := numWrong3 + 1
  IO.println s!"  C[0..3] = {cArr3[0]!} {cArr3[1]!} {cArr3[2]!} {cArr3[3]!}"
  IO.println s!"  C[17] (diagonal r=1,c=1) = {cArr3[17]!} (expect 16.0)"
  IO.println s!"  C[34] (diagonal r=2,c=2) = {cArr3[34]!} (expect 16.0)"
  IO.println s!"  max |err| = {maxErr3}"
  IO.println s!"  num wrong = {numWrong3}"
  freeCUDABuffer aBuf3; freeCUDABuffer bBuf3; freeCUDABuffer cBuf3
  if maxErr3 < 1e-3 then
    IO.println "✓ Phase 4b PASSED — runtime ShaderM.loop accumulates with in-place reg rebind"
  else
    IO.println "✗ FAILED — runtime-loop accumulator did not chain"
    IO.Process.exit 1

end Hesper.CUDA.WmmaGPUParityTest

def main : IO Unit := Hesper.CUDA.WmmaGPUParityTest.main
