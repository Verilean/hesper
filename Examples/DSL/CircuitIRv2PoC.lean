import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering_v2
import Hesper.Layers.RMSNorm_v2
import Hesper.Layers.KVCache_v2
import Hesper.WGSL.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
Tiny PoC driver for `Hesper.Circuit.IRv2.fusePointwiseIntoReduce`.
Builds the two-block RMSNorm graph, runs the fusion pass, and checks
that the result collapses to exactly one Block whose body is a single
Reduce carrying the full `x * invRms` expression.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2

/-- Reference RMSNorm on CPU for the parity check.
    y[i] = x[i] * scale[i] / sqrt(mean(x²) + eps) -/
def cpuRMSNorm (x scale : Array Float) (eps : Float) : Array Float := Id.run do
  let n := x.size
  let mut sumSq : Float := 0.0
  for v in x do
    sumSq := sumSq + v * v
  let mean := sumSq / n.toFloat
  let invRms := 1.0 / Float.sqrt (mean + eps)
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let xi := x[i]!
    let si := scale[i]!
    out := out.push (xi * si * invRms)
  return out

/-- Pack an `Array Float` into a little-endian f32 `ByteArray`. -/
def arrToBytes (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

/-- Unpack a `ByteArray` of n f32 values back to `Array Float`. -/
def bytesToArr (bytes : ByteArray) (n : Nat) : IO (Array Float) := do
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

/-- Max absolute difference between two arrays (same length). -/
def maxAbsDiff (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def describeBlock (b : Block) : String :=
  let r := b.reads.map (·.tensorId) |>.toList
  let w := b.writes.map (·.tensorId) |>.toList
  let k := match b.body with
    | .Pointwise _ => "Pointwise"
    | .Reduce _ _ _ _ => "Reduce"
    | .Scatter _ _ => "Scatter"
    | .MatMul _ _ _ _ => "MatMul"
  "  Block kind=" ++ k ++ " reads=" ++ toString r ++ " writes=" ++ toString w

def describeGraph (label : String) (g : BlockGraph) : IO Unit := do
  IO.println s!"=== {label} ==="
  IO.println s!"tensors: {g.tensors.size}, blocks: {g.blocks.size}"
  for b in g.blocks do
    IO.println (describeBlock b)

def main : IO Unit := do
  let (fused, target) := fusePoCTest (N := 2560) (eps := 1e-6)
  let (_, unfused) := runBuilder (rmsNormTwoBlocks 2560 1e-6 100 101)
  describeGraph "Unfused" unfused
  describeGraph "Fused (after pass)" fused
  describeGraph "Reference (hand-fused)" target
  -- Simple structural assertions:
  if fused.blocks.size == 1 then
    IO.println "PASS: fused graph collapsed to 1 block"
  else
    IO.println s!"FAIL: expected 1 block, got {fused.blocks.size}"
    IO.Process.exit 1
  match fused.blocks[0]!.body with
  | .Reduce _ _ _ _ => IO.println "PASS: fused block is a Reduce"
  | _ =>
    IO.println "FAIL: fused block is not a Reduce"
    IO.Process.exit 1
  -- Tensor count: the intermediate `mid` scalar should have been
  -- dropped.  Unfused had tensors {mid}; fused graph declares none
  -- (the builder only allocated `mid`).
  if fused.tensors.size == 0 then
    IO.println "PASS: intermediate tensor eliminated"
  else
    IO.println s!"NOTE: fused tensor count = {fused.tensors.size} (not critical)"
  IO.println "PoC OK"
  IO.println ""
  IO.println "=== RMSNorm v2 end-to-end (build → fuse → lower → WGSL) ==="
  let (g, kernel) := Hesper.Layers.RMSNorm_v2.compile
    (dim := 32) (eps := 1e-6) (xId := 100) (scaleId := 102) (outId := 101)
    (workgroupSize := 64)
  IO.println s!"Fused graph blocks: {g.blocks.size}, tensors: {g.tensors.size}"
  let wgsl := Hesper.WGSL.CodeGen.generateWGSL "rmsnorm_v2" ⟨64, 1, 1⟩ [] [] kernel
  IO.println "--- generated WGSL ---"
  IO.println wgsl
  IO.println s!"--- WGSL byte size: {wgsl.length} ---"

  IO.println ""
  IO.println "=== Scatter PoC: rope + KV-write fusion ==="
  let kvDim0 : Nat := 16
  let rowStride0 : Nat := 16
  let (fusedG, scatterKernel) :=
    Hesper.Layers.KVCache_v2.compile kvDim0 rowStride0 200 201 202 16
  IO.println s!"Fused graph blocks: {fusedG.blocks.size}, tensors: {fusedG.tensors.size}"
  if fusedG.blocks.size == 1 then
    IO.println "PASS: rope+scatter collapsed to 1 block"
  else
    IO.println s!"FAIL: expected 1 block, got {fusedG.blocks.size}"
    IO.Process.exit 1
  match fusedG.blocks[0]!.body with
  | .Scatter _ _ => IO.println "PASS: fused body is Scatter"
  | _ =>
    IO.println "FAIL: fused body is not Scatter"
    IO.Process.exit 1
  let scatterWgsl :=
    Hesper.WGSL.CodeGen.generateWGSL "rope_scatter" ⟨16, 1, 1⟩ [] [] scatterKernel
  IO.println "--- Scatter WGSL (snippet) ---"
  IO.println (scatterWgsl.take 1400)
  IO.println s!"--- Scatter WGSL total bytes: {scatterWgsl.length} ---"

  IO.println ""
  IO.println "=== MatMul PoC: MatMul + GELU + Add fusion ==="
  -- Build: [MatMul → mid]; [Pointwise (mid, resid) → out]
  let mmBuilder : BuilderM Unit := do
    declareExternal 300 #[64] .f32 .Global     -- residual
    declareExternal 301 #[64] .f32 .Global     -- out
    let mid ← declareTensor #[64] .f32 .Register
    let residR : Region := { tensorId := 300 }
    let midR   : Region := { tensorId := mid.id }
    let outR   : Region := { tensorId := 301 }
    -- Block A: MatMul (layerId=42, out=64, in=128) → mid, identity epi.
    emitBlock
      { reads  := #[]
        writes := #[midR]
        body   := .MatMul 42 64 128 (.input 0) }
    -- Block B: Pointwise body = GELU(mid) + resid
    -- reads = [mid, resid]; mid at slot 0, resid at slot 1.
    emitBlock
      { reads  := #[midR, residR]
        writes := #[outR]
        body   := .Pointwise (.add (.gelu (.input 0)) (.input 1)) }
  let (_, mmUnfused) := runBuilder mmBuilder
  let mmFused := fusePointwiseIntoMatMul mmUnfused
  describeGraph "MatMul unfused" mmUnfused
  describeGraph "MatMul fused" mmFused
  if mmFused.blocks.size == 1 then
    IO.println "PASS: matmul+pointwise collapsed to 1 block"
  else
    IO.println s!"FAIL: expected 1 block, got {mmFused.blocks.size}"
    IO.Process.exit 1
  match mmFused.blocks[0]!.body with
  | .MatMul _ _ _ fusedEpi =>
    IO.println "PASS: fused body is MatMul"
    -- Expected epilogue shape: .add (.gelu (.input 0)) (.input 1)
    -- (slot 0 = dot, slot 1 = residual after dedup).
    IO.println s!"epilogue = {repr fusedEpi}"
    let want : ScalarExp := .add (.gelu (.input 0)) (.input 1)
    if fusedEpi == want then
      IO.println "PASS: epilogue = .add (.gelu (.input 0)) (.input 1)"
    else
      IO.println s!"FAIL: epilogue shape unexpected"
      IO.Process.exit 1
  | _ =>
    IO.println "FAIL: fused body is not MatMul"
    IO.Process.exit 1
  let mmShader := lowerBlockGraph mmFused 64
  let mmWgsl :=
    Hesper.WGSL.CodeGen.generateWGSL "matmul_gelu_add" ⟨64, 1, 1⟩ [] [] mmShader
  IO.println "--- MatMul fused WGSL (tail, epilogue region) ---"
  let startIdx := if mmWgsl.length > 1200 then mmWgsl.length - 1200 else 0
  IO.println (mmWgsl.extract ⟨startIdx⟩ ⟨mmWgsl.length⟩)
  IO.println s!"--- MatMul WGSL total bytes: {mmWgsl.length} ---"
  -- Structural check: the epilogue must reference both the dot product
  -- (through __total__/result) and the residual buffer `epi0`.
  let hasEpi0 : Bool := decide ((mmWgsl.splitOn "epi0").length > 1)
  let hasTanh : Bool := decide ((mmWgsl.splitOn "tanh").length > 1)
  let hasDp4a : Bool := decide ((mmWgsl.splitOn "dot4I8Packed").length > 1)
  if hasEpi0 && hasTanh && hasDp4a then
    IO.println "PASS: WGSL has dp4a matmul + GELU (tanh) + residual (epi0) all inlined"
  else
    IO.println s!"FAIL: dp4a={hasDp4a} gelu={hasTanh} epi0={hasEpi0}"
    IO.Process.exit 1

  -- ========================================================
  --   GPU execution test (CUDA backend)
  -- ========================================================
  let skipGpu := (← IO.getEnv "HESPER_SKIP_GPU").isSome
  if skipGpu then
    IO.println "[GPU test skipped via HESPER_SKIP_GPU]"
    return
  IO.println ""
  IO.println "=== GPU execution (CUDA) ==="
  let dim : Nat := 32
  let wg : Nat := 64
  let eps : Float := 1e-6
  -- Deterministic test data.
  let xArr : Array Float :=
    (List.range dim).toArray.map (fun i => (i.toFloat - 15.5) * 0.1)
  let scaleArr : Array Float :=
    (List.range dim).toArray.map (fun _ => 1.0)
  let expected := cpuRMSNorm xArr scaleArr eps

  -- Init CUDA, allocate, upload.
  let ctx ← Hesper.CUDAContext.init
  let xBytes ← arrToBytes xArr
  let scaleBytes ← arrToBytes scaleArr
  let nBytes : USize := (dim * 4).toUSize
  let xBuf ← GPUBackend.allocBuffer ctx nBytes
  let scaleBuf ← GPUBackend.allocBuffer ctx nBytes
  let outBuf ← GPUBackend.allocBuffer ctx nBytes
  GPUBackend.writeBuffer ctx xBuf xBytes
  GPUBackend.writeBuffer ctx scaleBuf scaleBytes

  -- Build the v2 kernel for this dim/eps and execute.
  let (_, v2Kernel) := Hesper.Layers.RMSNorm_v2.compile dim eps 100 102 101 wg
  GPUBackend.executeWithConfig ctx v2Kernel
    [("input", xBuf), ("scale", scaleBuf), ("output", outBuf)]
    { workgroupSize := { x := wg }, numWorkgroups := (1, 1, 1) }

  -- Read back + compare.
  let outBytes ← GPUBackend.readBuffer ctx outBuf nBytes
  let gotArr ← bytesToArr outBytes dim
  let err := maxAbsDiff gotArr expected
  IO.println s!"expected[0..4] = {expected.toList.take 4}"
  IO.println s!"got[0..4]      = {gotArr.toList.take 4}"
  IO.println s!"max |err|      = {err}"
  if err < 1.0e-4 then
    IO.println "PASS: v2 RMSNorm GPU output matches CPU reference (err < 1e-4)"
  else
    IO.println s!"FAIL: v2 RMSNorm GPU output differs (max err {err})"
    IO.Process.exit 1

  IO.println ""
  IO.println "=== Scatter PoC: rope + KV-write fusion ==="
  let kvDim : Nat := 16
  let rowStride : Nat := 16
  let (fusedG, scatterKernel) :=
    Hesper.Layers.KVCache_v2.compile kvDim rowStride 200 201 202 16
  IO.println s!"Fused graph blocks: {fusedG.blocks.size}, tensors: {fusedG.tensors.size}"
  if fusedG.blocks.size == 1 then
    IO.println "PASS: rope+scatter collapsed to 1 block"
  else
    IO.println s!"FAIL: expected 1 block, got {fusedG.blocks.size}"
    IO.Process.exit 1
  match fusedG.blocks[0]!.body with
  | .Scatter _ _ => IO.println "PASS: fused body is Scatter"
  | _ => IO.println "FAIL: fused body is not Scatter"; IO.Process.exit 1
  let scatterWgsl :=
    Hesper.WGSL.CodeGen.generateWGSL "rope_scatter" ⟨16, 1, 1⟩ [] [] scatterKernel
  IO.println "--- Scatter WGSL (snippet) ---"
  IO.println (scatterWgsl.take 1400)
  IO.println s!"--- Scatter WGSL total bytes: {scatterWgsl.length} ---"
