import Hesper.Circuit.IR
import Hesper.Circuit.Passes

/-!
Unit test for `fusePointwise` — checks that a chain of pointwise ops
collapses into a single op with a composed ScalarExp body.

`BufT := Unit, CacheT := Unit` — fusion is buffer-agnostic, so this is
a pure IR test.
-/

open Hesper.Circuit

abbrev Buf := Unit
abbrev Cache := Unit

/-- Build `out3 = (in * 2) + 1` as a 3-op chain, then check
    `fusePointwise` collapses it. -/
def run : IO Unit := do
  let ((), st) := CircuitM.run (BufT := Buf) (CacheT := Cache) (do
    let a ← CircuitM.registerExternal (BufT := Buf) (CacheT := Cache)
              () #[16] .f32 .Global
    let doubled ← CircuitM.scale a 2.0
    let inc     ← CircuitM.map doubled (.add (.input 0) (.const 1.0))
    let _final  ← CircuitM.map inc (.mul (.input 0) (.input 0))
    pure ())

  IO.println s!"Before fuse: {st.ops.size} ops"
  for op in st.ops do
    match op.prim with
    | Prim.pointwise shape body =>
      IO.println s!"  pointwise shape={shape} body={repr body}"
    | _ => IO.println "  (other prim)"

  -- Protect only the final caller-facing buffer (the last produced id).
  let lastId := st.tensors.back!.id
  let fused := fusePointwise st.ops #[lastId]
  IO.println s!"After fuse (protecting id={lastId}): {fused.size} ops"
  for op in fused do
    match op.prim with
    | Prim.pointwise shape body =>
      IO.println s!"  pointwise shape={shape} body={repr body}"
    | _ => IO.println "  (other prim)"

  if fused.size == 1 then
    IO.println "✓ PASS: 3 pointwise ops collapsed to 1"
  else
    IO.println s!"✗ FAIL: expected 1 op after fusion, got {fused.size}"
    IO.Process.exit 1

def main : IO Unit := run
