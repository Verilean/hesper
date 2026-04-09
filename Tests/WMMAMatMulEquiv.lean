import Hesper
import Hesper.WGSL.MatMul
import Hesper.WGSL.Execute
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Float16
import Hesper.Basic

/-!
# WMMA MatMul Equivalence Harness

Verifies `matMulTransposeF16WMMAKernel` against the existing non-WMMA
`matMulTransposeF16Kernel` on random inputs.

For each test shape (M, K, N):
  1. Generate deterministic random `A : f32[M, K]` and `B : f16[N, K]`.
  2. Run `matMulTransposeF16Kernel` → golden `C_ref : f32[M, N]`.
  3. Run `matMulTransposeF16WMMAKernel` → `C_wmma : f32[M, N]`.
  4. Require `maxAbs / rms(ref) < 5e-2` and `cosine(wmma, ref) > 0.999`.

The tolerance is loose because the WMMA path casts A to f16 before
accumulating, so it legitimately has ~2^-10 ULP noise per K-step.
-/

open Hesper.WebGPU
open Hesper.WGSL

namespace Tests.WMMAMatMulEquiv

/-- xorshift32 PRNG for deterministic random sequences. -/
structure Rng where
  state : UInt32

def Rng.next (r : Rng) : UInt32 × Rng :=
  let x1 := r.state ^^^ (r.state <<< 13)
  let x2 := x1 ^^^ (x1 >>> 17)
  let x3 := x2 ^^^ (x2 <<< 5)
  (x3, ⟨x3⟩)

def Rng.float (r : Rng) : Float × Rng :=
  let (u, r') := r.next
  let x := (u.toNat.toFloat / 4294967296.0) * 2.0 - 1.0   -- [-1, 1)
  (x, r')

structure Shape where
  M : Nat
  N : Nat
  K : Nat
  seed : UInt32
  name : String

def genFloats (n : Nat) (rng : Rng) : Array Float × Rng := Id.run do
  let mut r := rng
  let mut a : Array Float := Array.empty
  for _ in [:n] do
    let (x, r') := r.float
    a := a.push x
    r := r'
  pure (a, r)

/-- Max absolute error between two arrays. -/
def maxAbsError (a b : Array Float) : Float := Id.run do
  let n := min a.size b.size
  let mut m : Float := 0.0
  for i in [:n] do
    let d := a.getD i 0.0 - b.getD i 0.0
    let ad := if d < 0.0 then -d else d
    if ad > m then m := ad
  pure m

def rms (a : Array Float) : Float := Id.run do
  let mut s : Float := 0.0
  for x in a do s := s + x * x
  pure (Float.sqrt (s / a.size.toFloat))

def cosine (a b : Array Float) : Float := Id.run do
  let n := min a.size b.size
  let mut dot : Float := 0.0
  let mut na : Float := 0.0
  let mut nb : Float := 0.0
  for i in [:n] do
    let ai := a.getD i 0.0
    let bi := b.getD i 0.0
    dot := dot + ai * bi
    na  := na  + ai * ai
    nb  := nb  + bi * bi
  let denom := Float.sqrt na * Float.sqrt nb
  if denom == 0.0 then pure 1.0 else pure (dot / denom)

/-- Upload an `Array Float` to a storage buffer. -/
def uploadF32 (device : Device) (arr : Array Float) : IO Buffer := do
  let buf ← createBuffer device {
    size := (arr.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let bytes ← Hesper.Basic.floatArrayToBytes arr
  writeBuffer device buf 0 bytes
  pure buf

/-- Pack an `Array Float` into a storage buffer of f16 values (stored
    as u32 halves to match the non-WMMA kernel's `b` buffer layout). -/
def uploadF16AsU32 (device : Device) (arr : Array Float) : IO Buffer := do
  let fa : FloatArray := FloatArray.mk arr
  let f16 ← Hesper.Float16.fromFloatArray fa
  let buf ← createBuffer device {
    size := f16.data.size.toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  writeBuffer device buf 0 f16.data
  pure buf

def runShape (device : Device) (shape : Shape)
    (absTol : Float := 5e-2) (cosTol : Float := 0.999) :
    IO (Bool × Float × Float × Float) := do
  let (aData, rng1) := genFloats (shape.M * shape.K) ⟨shape.seed⟩
  let (bData, _)    := genFloats (shape.N * shape.K) rng1
  let cfg : MatMul.Config := { M := shape.M, N := shape.N, K := shape.K }

  -- Upload A (f32) and B (f16 packed as u32).
  let aBuf ← uploadF32 device aData
  let bBuf ← uploadF16AsU32 device bData

  -- Output buffers for the two kernels.
  let mkOut : IO Buffer := createBuffer device {
    size := (shape.M * shape.N * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let cRef ← mkOut
  let cWmma ← mkOut

  -- Run the golden (non-WMMA) kernel.
  let refBufs : List (String × Buffer) :=
    [("a", aBuf), ("b", bBuf), ("c", cRef)]
  Execute.executeShaderNamed device
    (MatMul.matMulTransposeF16Kernel cfg)
    refBufs
    (Execute.ExecutionConfig.dispatch1D (shape.M * shape.N))

  -- Run the WMMA kernel.
  let wmmaConfig : Execute.ExecutionConfig := {
    funcName := "main"
    workgroupSize := { x := 32, y := 1, z := 1 }
    numWorkgroups := (shape.N / 16, shape.M / 16, 1)
    extensions := ["f16", "chromium_experimental_subgroup_matrix"]
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")]
  }
  let wmmaBufs : List (String × Buffer) :=
    [("a", aBuf), ("b", bBuf), ("c", cWmma)]
  Execute.executeShaderNamed device
    (MatMul.matMulTransposeF16WMMAKernel cfg)
    wmmaBufs
    wmmaConfig

  -- Compare.
  let refArr ← Hesper.WebGPU.BufferOps.downloadFloatArray device cRef (shape.M * shape.N)
  let wmmaArr ← Hesper.WebGPU.BufferOps.downloadFloatArray device cWmma (shape.M * shape.N)
  let abs := maxAbsError wmmaArr refArr
  let r := rms refArr
  let denom := if r < 1e-9 then 1e-9 else r
  let absRel := abs / denom
  let cos := cosine wmmaArr refArr
  pure (absRel < absTol && cos > cosTol, abs, absRel, cos)

def shapes : List Shape :=
  [ { M := 16,  K := 16,  N := 16,  seed := 0xDEADBEEF, name := "tiny" }
  , { M := 16,  K := 32,  N := 16,  seed := 0x1337C0DE, name := "K=32" }
  , { M := 16,  K := 64,  N := 32,  seed := 0xBADF00D,  name := "wider-N" }
  , { M := 32,  K := 64,  N := 32,  seed := 0xCAFEBABE, name := "M=32" }
  , { M := 16,  K := 256, N := 16,  seed := 0xFEEDFACE, name := "deep-K" }
  , { M := 64,  K := 256, N := 64,  seed := 0xABCDEF01, name := "64-cube" }
  , { M := 16,  K := 2560,N := 640, seed := 0x11111111, name := "gemma4-lm-ish-small" }
  , { M := 32,  K := 2560,N := 640, seed := 0x22222222, name := "gemma4-lm-ish-mid" }
  , { M := 128, K := 2560,N := 2560,seed := 0x33333333, name := "big-square" }
  ]

def main : IO UInt32 := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  WMMA MatMul Equivalence Harness"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  let hasSm  ← Execute.hasSubgroupMatrixSupport device
  let hasF16 ← Execute.hasShaderF16Support device
  if !hasSm || !hasF16 then
    IO.println "  ⚠ Device lacks SubgroupMatrix or ShaderF16 → cannot run WMMA kernel."
    IO.println "  (This is expected on adapters without the Chromium subgroup matrix feature.)"
    return 0

  let mut pass := 0
  let mut fail := 0
  for s in shapes do
    let (ok, absErr, absRel, cos) ← runShape device s
    let tag := if ok then "✓" else "✗"
    IO.println s!"  {tag} {s.name}  M={s.M} K={s.K} N={s.N}  maxAbs={absErr}  absRel={absRel}  cos={cos}"
    if ok then pass := pass + 1 else fail := fail + 1

  IO.println ""
  IO.println s!"  Total: {pass} passed, {fail} failed"

  if fail == 0 then
    IO.println "  ✅ WMMA matmul matches the non-WMMA reference."
    return 0
  else
    IO.println "  ❌ WMMA matmul diverged from the reference."
    return 1

end Tests.WMMAMatMulEquiv

def main : IO UInt32 := Tests.WMMAMatMulEquiv.main
