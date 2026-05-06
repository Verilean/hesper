import Hesper.Circuit.Eval
import Hesper.WGSL.Eval

/-!
# Evaluator sanity tests

Pure-Lean (CPU, `native_decide`-evaluable) tests for
`Hesper.Circuit.Eval` and `Hesper.WGSL.Eval`. These don't need a
GPU and run as part of a normal `lake build`.

Float equality is not `Decidable` (NaN ≠ NaN), so we compare with
`==` (which returns `Bool`) and assert `... = true`.

The tests exercise the same constructors that BitNet's BitLinear
and attention kernels go through. Together they certify that the
evaluator's wiring is correct on the slice that downstream
equivalence proofs rely on.
-/

namespace Hesper.Proofs.EvalSanity

/-! ## Circuit IR sanity -/

section CircuitTests
open Hesper.Circuit

example :
    (ScalarExp.eval
      { inputs := #[#[1.0, 2.0, 3.0]], laneIdx := 1 }
      (.input 0) == 2.0) = true := by
  native_decide

example :
    (ScalarExp.eval
      { inputs := #[#[2.0, 3.0]], laneIdx := 1 }
      (.mul (.input 0) (.input 0)) == 9.0) = true := by
  native_decide

example :
    (evalPointwise
      (.add (.input 0) (.input 1))
      #[#[1.0, 2.0, 3.0], #[10.0, 20.0, 30.0]]
      3
      == #[11.0, 22.0, 33.0]) = true := by
  native_decide

example :
    (evalReduce ReduceOp.sum #[1.0, 2.0, 3.0, 4.0] == 10.0) = true := by
  native_decide

example :
    (evalReduce ReduceOp.sumOfSquares #[1.0, 2.0, 3.0] == 14.0) = true := by
  native_decide

end CircuitTests

/-! ## WGSL DSL sanity (Phase 1 slice) -/

section WGSLTests
open Hesper.WGSL

/-- Pull a Float result out of a typed expression. -/
def runF32 (env : EvalEnv) (e : Exp (.scalar .f32)) : Float :=
  Exp.eval env e

example :
    (runF32 default (.add (.litF32 2.0) (.litF32 3.0)) == 5.0) = true := by
  native_decide

example :
    (runF32 default (.mul (.litF32 4.0) (.litF32 0.25)) == 1.0) = true := by
  native_decide

example :
    (runF32 default (.exp (.litF32 0.0)) == 1.0) = true := by
  native_decide

example :
    (runF32 { f32_vars := [("x", 7.0)] } (.var "x") == 7.0) = true := by
  native_decide

/-- Indexing into a 4-element f32 array. -/
example :
    let arr : Exp (.array (.scalar .f32) 4) :=
      .var (t := .array (.scalar .f32) 4) "a"
    let env : EvalEnv := { f32_arrays := [("a", #[10.0, 20.0, 30.0, 40.0])] }
    (runF32 env (.index arr (.litU32 2)) == 30.0) = true := by
  native_decide

end WGSLTests

end Hesper.Proofs.EvalSanity
