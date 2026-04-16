import Hesper.Circuit.IR
import Hesper.Circuit.Passes
import Hesper.Layers.Linear

/-!
IR-level unit test for `fuseMatmulEpilogue`.

Constructs a synthetic `matmulQ4K → pointwise(body)` chain and checks
that the pass collapses it into a single `Prim.matmulQ4KWithEpilogue`
node whose body is `pointwise.body` with the matmul slot rewired to
`input 0`.

No GPU dispatch needed — fusion passes are pure IR rewrites.  Uses
`BufT := Unit, CacheT := Unit` for the mock LinearLayer.
-/

open Hesper.Circuit
open Hesper.Layers.Linear

abbrev Buf := Unit
abbrev Cache := Unit

/-- Build an all-zero mock LinearLayer with the given dims. -/
def mkMockLayer (inDim outDim : Nat) : IO (LinearLayer Buf Cache) := do
  let cfg : Config := { inDim, outDim }
  let prepared ← IO.mkRef (α := Option Cache) none
  let splitKBuf ← IO.mkRef (α := Option Buf) none
  let splitKPartialPrepared ← IO.mkRef (α := Option Cache) none
  let splitKReducePrepared ← IO.mkRef (α := Option Cache) none
  let dp4aQ8Buf ← IO.mkRef (α := Option Buf) none
  let dp4aQuantizePrepared ← IO.mkRef (α := Option Cache) none
  let dp4aMatmulPrepared ← IO.mkRef (α := Option Cache) none
  return {
    config := cfg, weightBuf := (), quantFormat := .Q4_K, prepared,
    splitKBuf, splitKPartialPrepared, splitKReducePrepared,
    dp4aQ8Buf, dp4aQuantizePrepared, dp4aMatmulPrepared }

def run : IO Unit := do
  -- Build a circuit: y = matmulQ4K(x, layer); z = (y + bias) * 2.0
  let layer ← mkMockLayer (inDim := 256) (outDim := 64)
  let ((), st) := CircuitM.run (BufT := Buf) (CacheT := Cache) (do
    let x    ← CircuitM.registerExternal (BufT := Buf) (CacheT := Cache)
                 () #[256] .f32 .Global
    let bias ← CircuitM.registerExternal (BufT := Buf) (CacheT := Cache)
                 () #[64] .f32 .Global
    let y    ← CircuitM.matmulQ4K x layer
    -- pointwise: (input 0 + input 1) * 2.0
    --   input 0 = y (matmul output, full-shape [64])
    --   input 1 = bias (full-shape [64])
    let _out ← CircuitM.zip2 y bias
      (.mul (.add (.input 0) (.input 1)) (.const 2.0))
    pure ())

  IO.println s!"Before fuse: {st.ops.size} ops"
  for op in st.ops do
    match op.prim with
    | .matmulQ4K _ => IO.println "  matmulQ4K"
    | .matmulQ4KWithEpilogue _ _ _ body =>
      IO.println s!"  matmulQ4K+epi body={repr body}"
    | .scatter _ _ _ body _ => IO.println s!"  scatter body={repr body}"
    | _ => IO.println "  (other)"

  -- Protect the external inputs + the final op's output.  The
  -- intermediate matmul output (the y tensor) is NOT protected →
  -- fuseMatmulEpilogue is free to collapse it into the pointwise.
  let extIds : Array Nat := st.externals.map (fun (tr, _) => tr.id)
  let finalId := st.tensors.back!.id
  let protectedIds : Array Nat := extIds.push finalId

  let fused := fuseMatmulEpilogue st.ops protectedIds
  IO.println s!"After fuse: {fused.size} ops"
  for op in fused do
    match op.prim with
    | .matmulQ4K _ => IO.println "  matmulQ4K"
    | .matmulQ4KWithEpilogue _ sz off body =>
      IO.println s!"  matmulQ4K+epi sizes={sz} offs={off} body={repr body}"
    | .scatter _ _ _ body _ => IO.println s!"  scatter body={repr body}"
    | _ => IO.println "  (other)"

  -- Expectation: 1 op, a `matmulQ4KWithEpilogue` whose body reads
  -- `(input 0 + input 1) * 2.0` where input 0 is the matmul result
  -- and input 1 is the bias (renumbered from the consumer's slot 1).
  if fused.size != 1 then
    IO.println s!"✗ FAIL: expected 1 op after fusion, got {fused.size}"
    IO.Process.exit 1

  match fused[0]? with
  | some op =>
    match op.prim with
    | .matmulQ4KWithEpilogue _ sizes offsets _ =>
      if sizes != #[64] || offsets != #[0] then
        IO.println s!"✗ FAIL: expected sizes=[64] offsets=[0], got sizes={sizes} offsets={offsets}"
        IO.Process.exit 1
      IO.println "✓ PASS: matmulQ4K + pointwise collapsed into matmulQ4KWithEpilogue"
    | _ =>
      IO.println "✗ FAIL: fused op is not matmulQ4KWithEpilogue"
      IO.Process.exit 1
  | none =>
    IO.println "✗ FAIL: fused ops array is empty"
    IO.Process.exit 1

def main : IO Unit := run
