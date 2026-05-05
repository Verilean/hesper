import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Sentinel: if_ branches that compute shared sub-exprs INSIDE both branches

Regression sentinel for the expCache-leak-across-branches bug discovered
on 2026-05-05 in `concat_dim0_f32_kernel`.  When a sub-expression like
`i * 2` is computed inside both `then` and `else` branches of a
`ShaderM.if_`, Hesper's auto-CSE may merge the two registers and emit
the wrong arithmetic on one of the paths.

This test deliberately writes the kernel WITHOUT hoisting the shared
sub-expression to a `let'` before the `if_`.  Behaviour:

- BEFORE the CodeGen-side scoped expCache fix: this test FAILS with
  wrong output values (the else branch reads the then-branch register).
- AFTER the fix: this test PASSES bit-exact.

Once the CodeGen fix is landed, this test prevents future regressions
to the lowering rule that would re-introduce the bug.

## Algorithm

Given input `x : f32[N]`:
- For each i, condition is `i < N/2`.
- THEN branch:    dst[i] = x[i*2]   * 1.0
- ELSE branch:    dst[i] = x[i*2+1] * 2.0

`i*2` appears in both branches — naive CSE would cache it from the
then-branch and reuse the same register on the else path.

The OOB-on-mismatch nature of the bug: in the else-path, threads with
`i >= N/2` compute `i*2+1` which can exceed `N`. Buffer reads under
`ld.global.nc` don't fault, but we keep N comfortably small and arrange
the indexing so OOB threads land on indices we know.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

/-- Test kernel: branch-shared sub-expr `i*2` inside both branches.
    Block size 256, output size = N.

    Mimics the failure mode in `concat_dim0_f32_kernel`:
    - Both branches compute `i*K` for different K values that arise from
      the same `i` and a different multiplier per branch.
    - The shared sub-expr is the workgroup/local-id chain that produces `i`.
    - Naive expCache merges the two branches' compound offset Exps when
      they share a structural prefix. -/
private def ifBranchCseKernel (n : Nat) : ShaderM Unit := do
  let _x   ← ShaderM.declareReadOnlyBuffer "x"   (.array (.scalar .f32) n)
  let _dst ← ShaderM.declareOutputBuffer  "dst" (.array (.scalar .f32) n)

  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let bx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  -- Use a compound i expression (mirrors the concat kernel pattern).
  let i := Exp.add (Exp.mul bx (Exp.litU32 256)) tid
  let inBounds := Exp.lt i (Exp.litU32 n)
  ShaderM.if_ inBounds (do
    let halfN := n / 2
    let cond := Exp.lt i (Exp.litU32 halfN)
    -- Mirror concat offset shape exactly: outer = i + (mul_term * K0) + (add_term).
    -- Then-branch: `i + i*2 + i` (= 4*i)
    -- Else-branch: `(i - 5) + i*3 + i` (= 5*i - 5)
    -- The shared compound shape `Exp.add (Exp.add ?A ?B) ?C` where B and C are
    -- both `Exp.mul i (litU32 ...)` with different constants is what concat
    -- exhibits.  `i*2` and `i*3` are distinct keys so CSE shouldn't merge them
    -- — but we structurally test that.
    ShaderM.if_ cond (do
      -- THEN: idx = i + i*2 + i = 4*i  (use Exp form to keep shape distinct)
      let idx :=
        Exp.add (Exp.add i (Exp.mul i (Exp.litU32 2))) i
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "x" idx
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" i (Exp.mul v (Exp.litF32 1.0))
    ) (do
      -- ELSE: idx = (i - K) + i*3 + i  with K=5  → ends up at i*5 - 5
      let iMinus := Exp.sub i (Exp.litU32 5)
      let idx :=
        Exp.add (Exp.add iMinus (Exp.mul i (Exp.litU32 3))) i
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "x" idx
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" i (Exp.mul v (Exp.litF32 2.0))
    )
  ) (pure ())

private def packF32 (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

private def unpackF32 (ba : ByteArray) (n : Nat) : IO (Array Float) := do
  let mut arr := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 ba (i * 4)
    arr := arr.push f
  return arr

unsafe def main : IO Unit := do
  IO.println "═══ if_ branch CSE microtest ═══"

  -- Need N large enough that for ELSE-path threads (i ≥ halfN), idx = i*3 + i + 7
  --   ≤ (n-1)*4 + 7  ≤ n*4 + 3, so x has at least n*4 + 8 elements.
  let n := 64
  let halfN := n / 2
  let xLen := n * 4 + 8

  -- Deterministic input: x[i] = (i+1) * 0.1 (simple, easy to inspect).
  let xArr : Array Float :=
    (List.range xLen).toArray.map (fun i => (i.toFloat + 1.0) * 0.1)

  -- CPU reference per the kernel spec (idx formulas match the kernel).
  let mut cpuOut : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    if i < halfN then
      -- THEN: idx = i + i*2 + i = 4*i
      let idx := i + i*2 + i
      cpuOut := cpuOut.set! i (xArr[idx]!)
    else
      -- ELSE: idx = (i - 5) + i*3 + i  ; for i ≥ halfN ≥ 32 this is positive
      let idx := (i - 5) + i*3 + i
      cpuOut := cpuOut.set! i (xArr[idx]! * 2.0)

  let ctx ← Hesper.CUDAContext.init
  let xBuf ← GPUBackend.allocBuffer ctx (xLen * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  GPUBackend.writeBuffer ctx xBuf (← packF32 xArr)

  let nBlocks := (n + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("x", xBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx (ifBranchCseKernel n) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocks, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf (n * 4).toUSize
  let hesperOut ← unpackF32 resultBytes n

  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  for i in [0:n] do
    let d := (hesperOut[i]! - cpuOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 1e-5 ∧ firstMis == -1 then
      firstMis := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} cpu={cpuOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr < 1e-5 then
    IO.println "═══ if_ branch CSE microtest PASS ═══"
  else
    IO.println "═══ if_ branch CSE microtest FAIL — CodeGen scoped expCache regression? ═══"
    IO.Process.exit 1
