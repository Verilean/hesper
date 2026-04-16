import Hesper.Circuit.IR
import Hesper.Circuit.Passes

/-!
IR-level unit test for `fuseWriteDestination`.

Constructs a synthetic `pointwise → writeSlice` chain and checks
that the pass collapses it into a single `Prim.pointwiseToSlice`
node that writes directly into the destination at the given offset.

No GPU dispatch needed — fusion passes are pure IR rewrites.
-/

open Hesper.Circuit

abbrev Buf := Unit
abbrev Cache := Unit

def run : IO Unit := do
  -- Build a circuit:
  --   x     : external [64] f32     (input data)
  --   dst   : external [256] f32    (destination buffer, e.g. KV cache)
  --   y     = pointwise(x, body = input 0 * 2.0)   -- compute
  --   out   = writeSlice(dst, y, offset=128)         -- splice y into dst[128..192]
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
    | .pointwise _ _ body => IO.println s!"  pointwise body={repr body}"
    | .writeSlice ds off ss => IO.println s!"  writeSlice dstShape={ds} offset={repr off} srcShape={ss}"
    | .pointwiseToSlice _ _ body ds off =>
      IO.println s!"  pointwiseToSlice dstShape={ds} offset={repr off} body={repr body}"
    | _ => IO.println "  (other)"

  -- Protect external inputs + final output.
  let extIds : Array Nat := st.externals.map (fun (tr, _) => tr.id)
  let finalId := match st.ops[st.ops.size - 1]? with
    | some op => op.outputs[0]!.id
    | none => 0
  let protectedIds : Array Nat := extIds.push finalId

  let fused := fuseWriteDestination st.ops protectedIds
  IO.println s!"After fuse: {fused.size} ops"
  for op in fused do
    match op.prim with
    | .pointwise _ _ body => IO.println s!"  pointwise body={repr body}"
    | .writeSlice ds off ss => IO.println s!"  writeSlice dstShape={ds} offset={repr off} srcShape={ss}"
    | .pointwiseToSlice _ _ body ds off =>
      IO.println s!"  pointwiseToSlice dstShape={ds} offset={repr off} body={repr body}"
    | _ => IO.println "  (other)"

  -- Expectation: 1 op, a `pointwiseToSlice` with offset=128, dstShape=[256]
  if fused.size != 1 then
    IO.println s!"✗ FAIL: expected 1 op after fusion, got {fused.size}"
    IO.Process.exit 1

  match fused[0]? with
  | some op =>
    match op.prim with
    | .pointwiseToSlice outShape _inShapes _body dstShape writeOffset =>
      if dstShape != #[256] then
        IO.println s!"✗ FAIL: expected dstShape=[256], got {dstShape}"
        IO.Process.exit 1
      if writeOffset != (128 : ScalarExp) then
        IO.println s!"✗ FAIL: expected writeOffset=128, got {repr writeOffset}"
        IO.Process.exit 1
      if outShape != #[64] then
        IO.println s!"✗ FAIL: expected outShape=[64], got {outShape}"
        IO.Process.exit 1
      IO.println "✓ PASS: pointwise + writeSlice collapsed into pointwiseToSlice"
    | _ =>
      IO.println "✗ FAIL: fused op is not pointwiseToSlice"
      IO.Process.exit 1
  | none =>
    IO.println "✗ FAIL: fused ops array is empty"
    IO.Process.exit 1

def main : IO Unit := run
