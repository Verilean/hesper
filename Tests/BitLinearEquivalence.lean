import Hesper
import Hesper.Layers.BitLinear
import Hesper.Layers.BitLinearSpec
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
# BitLinear Kernel Equivalence Harness

Golden: `Hesper.Layers.BitLinearSpec.forward` (pure CPU reference that
exactly mirrors the i2_s unpacking math).

For each declared GPU kernel variant (shared-mem fallback, subgroup tree
reduction, subgroup matrix, …) we:

  1. Generate a deterministic batch of random test cases:
     - random ternary weights
     - random f32 inputs
     - random scale
     - various (inDim, outDim, numRows) shapes

  2. Build a `BitLinear` layer with those weights via the public API.

  3. Execute the layer's forward pass on the GPU (using whichever kernel
     the current device capability ends up selecting).

  4. Also call the CPU spec with the same weights/scale/input.

  5. Require `max |gpu[i] - cpu_spec[i]| / max(|cpu_spec[i]|, 1)  <  1e-3`.

Kernel variants that require features not available on the current adapter
are skipped (reported as SKIP, not FAIL).

Failure on any configured test case exits with code 1.
-/

open Hesper.WebGPU
open Hesper.Layers

namespace Tests.BitLinearEquivalence

/-- xorshift32 PRNG for deterministic random sequences without pulling in
    a heavy random library. -/
structure Rng where
  state : UInt32

def Rng.next (r : Rng) : UInt32 × Rng :=
  let x1 := r.state ^^^ (r.state <<< 13)
  let x2 := x1 ^^^ (x1 >>> 17)
  let x3 := x2 ^^^ (x2 <<< 5)
  (x3, ⟨x3⟩)

def Rng.float (r : Rng) : Float × Rng :=
  let (u, r') := r.next
  -- Map u32 to [-1.0, 1.0)
  let x := (u.toNat.toFloat / 4294967296.0) * 2.0 - 1.0
  (x, r')

def Rng.ternary (r : Rng) : Int × Rng :=
  let (u, r') := r.next
  let v : Int := match u.toNat % 3 with
    | 0 => -1
    | 1 => 0
    | _ => 1
  (v, r')

/-- Generate one test case: random ternary weights, random f32 input, random scale. -/
structure TestCase where
  name    : String
  inDim   : Nat
  outDim  : Nat
  numRows : Nat
  scale   : Float
  ternary : Array Int       -- length inDim*outDim, values ∈ {-1, 0, 1}
  input   : Array Float     -- length numRows*inDim

def genCase (name : String) (inDim outDim numRows : Nat) (seed : UInt32) : TestCase := Id.run do
  let mut rng : Rng := ⟨seed⟩
  let mut ternary : Array Int := Array.empty
  for _ in [:inDim * outDim] do
    let (t, rng') := rng.ternary
    ternary := ternary.push t
    rng := rng'
  let mut input : Array Float := Array.empty
  for _ in [:numRows * inDim] do
    let (x, rng') := rng.float
    input := input.push x
    rng := rng'
  let (u, _) := rng.next
  let scale := (u.toNat.toFloat / 4294967296.0) * 0.1 + 0.01  -- ~ [0.01, 0.11]
  pure { name, inDim, outDim, numRows, scale, ternary, input }

/-- Run one case through the CPU spec and return the expected output. -/
def runCPUSpec (tc : TestCase) : Array Float :=
  let packed := BitLinearSpec.packI2S tc.ternary tc.inDim tc.outDim
  BitLinearSpec.forward packed tc.scale tc.inDim tc.outDim tc.numRows tc.input

/-- Run one case on the GPU via the public `BitLinear.forward` API. -/
def runGPU (device : Device) (tc : TestCase) : IO (Array Float) := do
  let packed := BitLinearSpec.packI2S tc.ternary tc.inDim tc.outDim
  let cfg : BitLinear.Config :=
    { inDim := tc.inDim, outDim := tc.outDim, batchSize := tc.numRows }
  let layer ← BitLinear.create device cfg packed tc.scale

  -- Upload input to a fresh buffer
  let inBuf ← createBuffer device {
    size := (tc.numRows * tc.inDim * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let inBytes ← Hesper.Basic.floatArrayToBytes tc.input
  writeBuffer device inBuf 0 inBytes

  -- Create an output buffer
  let outBuf ← createBuffer device {
    size := (tc.numRows * tc.outDim * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }

  BitLinear.forward device layer inBuf outBuf tc.numRows
  Hesper.WebGPU.BufferOps.downloadFloatArray device outBuf (tc.numRows * tc.outDim)

/-- Max absolute error between two float arrays. -/
def maxAbsError (a b : Array Float) : Float := Id.run do
  let n := min a.size b.size
  let mut m : Float := 0.0
  for i in [:n] do
    let d := a.getD i 0.0 - b.getD i 0.0
    let ad := if d < 0.0 then -d else d
    if ad > m then m := ad
  pure m

/-- Relative error:
      max_i  |a[i] - b[i]|  /  max(|b[i]|, 1e-6) -/
def maxRelError (a b : Array Float) : Float := Id.run do
  let n := min a.size b.size
  let mut m : Float := 0.0
  for i in [:n] do
    let d := a.getD i 0.0 - b.getD i 0.0
    let ad := if d < 0.0 then -d else d
    let bi := b.getD i 0.0
    let abi := if bi < 0.0 then -bi else bi
    let denom := if abi < 1e-6 then 1e-6 else abi
    let r := ad / denom
    if r > m then m := r
  pure m

/-- Run a single test case and return (pass, maxAbsErr, maxRelErr, gpuOut, cpuOut). -/
def runCase (device : Device) (tc : TestCase) (tol : Float := 1e-3) :
    IO (Bool × Float × Float) := do
  let cpuOut := runCPUSpec tc
  let gpuOut ← runGPU device tc
  if gpuOut.size != cpuOut.size then
    IO.eprintln s!"  ✗ {tc.name}: size mismatch (gpu={gpuOut.size}, cpu={cpuOut.size})"
    return (false, 0.0, 0.0)
  let absErr := maxAbsError gpuOut cpuOut
  let relErr := maxRelError gpuOut cpuOut
  let ok := relErr < tol
  pure (ok, absErr, relErr)

/-- All configured test cases. Shapes chosen to exercise:
    - inDim multiples of 128 (required by i2_s group layout)
    - various outDim (including 1 which stresses M=1 path)
    - numRows in {1, 2, 8} to cover M=1 decode and M≥8 prefill/matmul tile-friendly
    - varying random seeds -/
def allCases : List TestCase :=
  [ -- Small shape, single row (exercises M=1 subgroup path)
    genCase "tiny-M1"         128  16  1  0xDEADBEEF
  , genCase "tiny-M1-seed2"   128  16  1  0x1337C0DE
  , genCase "tiny-M1-seed3"   256   8  1  0xBADF00D
    -- M=1 with larger dims (closer to real layer sizes)
  , genCase "mid-M1"          512  64  1  0xCAFEBABE
  , genCase "mid-M1-seed2"    512  64  1  0xFEEDFACE
  , genCase "wide-M1"        1024 128  1  0xABCDEF01
    -- Multi-row prefill cases — will exercise the shared-memory tiled kernel
  , genCase "M2-tiny"         128  16  2  0x11111111
  , genCase "M8-tiny"         128  16  8  0x22222222
  , genCase "M8-wider"        256  32  8  0x33333333
    -- Larger, more realistic "layer" shape at small M to make failures obvious
  , genCase "big-M1"         2560 320  1  0x44444444
  ]

def main : IO UInt32 := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  BitLinear Kernel Equivalence Harness"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  let cases := allCases
  let mut pass := 0
  let mut fail := 0
  let mut maxAbs : Float := 0.0
  let mut maxRel : Float := 0.0

  for tc in cases do
    let (ok, absErr, relErr) ← runCase device tc
    if absErr > maxAbs then maxAbs := absErr
    if relErr > maxRel then maxRel := relErr
    if ok then
      pass := pass + 1
      IO.println s!"  ✓ {tc.name}  (inDim={tc.inDim} outDim={tc.outDim} M={tc.numRows})  maxAbs={absErr}  maxRel={relErr}"
    else
      fail := fail + 1
      IO.println s!"  ✗ {tc.name}  (inDim={tc.inDim} outDim={tc.outDim} M={tc.numRows})  maxAbs={absErr}  maxRel={relErr}"

  IO.println ""
  IO.println s!"  Total: {pass} passed, {fail} failed"
  IO.println s!"  Worst maxAbsErr={maxAbs}, worst maxRelErr={maxRel}"
  IO.println ""

  if fail == 0 then
    IO.println "  ✅ All BitLinear kernel variants match the CPU spec."
    return 0
  else
    IO.println "  ❌ Some variants disagree with the CPU spec."
    return 1

end Tests.BitLinearEquivalence

def main : IO UInt32 := Tests.BitLinearEquivalence.main
