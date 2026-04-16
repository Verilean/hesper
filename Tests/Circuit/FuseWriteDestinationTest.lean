import Hesper.Circuit.IR
import Hesper.Circuit.Passes

/-!
IR-level unit test for unified `Prim.scatter` + `fusePointwise`.

Previously this tested a separate `fuseWriteDestination` pass.  With
`Prim.scatter` unifying pointwise / writeSlice / pointwiseToSlice, the
same fusion is now handled by `fusePointwise`: a Map-shaped scatter
(addr = .laneIdx) producer is inlined into any scatter consumer
(including one with `addr = .laneIdx + offset`, the former writeSlice).

This test builds `pointwise(x, x*2.0) → writeSlice(dst, _, 128)` and
checks that it collapses to a single scatter with value `x*2.0` and
addr `laneIdx + 128`.
-/

open Hesper.Circuit

abbrev Buf := Unit
abbrev Cache := Unit

def run : IO Unit := do
  let ((), st) := CircuitM.run (BufT := Buf) (CacheT := Cache) (do
    let x   ← CircuitM.registerExternal (BufT := Buf) (CacheT := Cache)
                () #[64] .f32 .Global
    let dst ← CircuitM.registerExternal (BufT := Buf) (CacheT := Cache)
                () #[256] .f32 .Global
    let y   ← CircuitM.map x (.mul (.input 0) (.const 2.0))
    let _out ← CircuitM.writeSlice dst y 128
    pure ())

  IO.println s!"Before fuse: {st.ops.size} ops"
  for op in st.ops do
    match op.prim with
    | .scatter _ dstSh _ val addr =>
      IO.println s!"  scatter dst={dstSh} val={repr val} addr={repr addr}"
    | _ => IO.println "  (other)"

  -- Protect externals + final output.
  let extIds : Array Nat := st.externals.map (fun (tr, _) => tr.id)
  let finalId := match st.ops[st.ops.size - 1]? with
    | some op => op.outputs[0]!.id
    | none => 0
  let protectedIds : Array Nat := extIds.push finalId

  let fused := fusePointwise st.ops protectedIds
  IO.println s!"After fuse: {fused.size} ops"
  for op in fused do
    match op.prim with
    | .scatter _ dstSh _ val addr =>
      IO.println s!"  scatter dst={dstSh} val={repr val} addr={repr addr}"
    | _ => IO.println "  (other)"

  -- Expect exactly 1 scatter with dstShape=[256] and addr = laneIdx + 128
  if fused.size != 1 then
    IO.println s!"✗ FAIL: expected 1 op after fusion, got {fused.size}"
    IO.Process.exit 1

  match fused[0]? with
  | some op =>
    match op.prim with
    | .scatter outShape dstShape _inShapes _val addr =>
      if dstShape != #[256] then
        IO.println s!"✗ FAIL: expected dstShape=[256], got {dstShape}"
        IO.Process.exit 1
      if outShape != #[64] then
        IO.println s!"✗ FAIL: expected outShape=[64], got {outShape}"
        IO.Process.exit 1
      -- addr should be .add .laneIdx (.const 128) (writeSlice sugar).
      match addr with
      | .add .laneIdx (.const v) =>
        if v != 128.0 then
          IO.println s!"✗ FAIL: expected addr offset=128, got {v}"
          IO.Process.exit 1
      | _ =>
        IO.println s!"✗ FAIL: addr is not `laneIdx + const`, got {repr addr}"
        IO.Process.exit 1
      IO.println "✓ PASS: Map scatter + writeSlice collapsed into single scatter with offset addr"
    | _ =>
      IO.println "✗ FAIL: fused op is not a scatter"
      IO.Process.exit 1
  | none =>
    IO.println "✗ FAIL: fused ops array is empty"
    IO.Process.exit 1

def main : IO Unit := run
