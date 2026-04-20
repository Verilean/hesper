import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.Basic
import Hesper.WebGPU.BufferOps
import Hesper.Layers.Linear
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader

/-!
# Hesper dp4a Q4_K vs llama.cpp PTX — per-row output comparison

Loads a SINGLE real Q4_K weight row from the Gemma 4 GGUF, runs both
hesper's dp4a kernel and llama.cpp's PTX kernel against the same Q8_1
input, and prints an element-by-element diff.

Expected (if ABI is correct): max abs difference < 1e-3.
Observed so far: garbage output in full pipeline, suggesting a
non-trivial mismatch somewhere.  This isolates the bug to the Q4_K
matmul itself (vs the Q8_1 quantize, which the ABITest already
validates).
-/

open Hesper
open Hesper.CUDA
open Hesper.LlamaCppPTX

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  -- Pick wQ.weight for layer 0 (shape: 2560 → 4096).  Small enough to
  -- inspect a single row.
  let tensorName := "blk.0.attn_q.weight"

  let ctx ← CUDAContext.init
  let k ← loadKernels

  IO.println s!"[Load] {ggufPath}"
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")
    | .ok g => pure g
  IO.println s!"[Load] {gguf.tensors.size} tensors"

  -- Find the tensor and extract first row (144 bytes per 256-block × blocksPerRow)
  let (weightBytes, numElements) ← Hesper.GGUF.Loader.extractQ4KTensor gguf tensorName
  -- First, inspect the tensor shape (should be inDim=2560, outDim=4096)
  let info ← match Hesper.GGUF.Loader.findTensor gguf tensorName with
    | .error e => throw (IO.userError e)
    | .ok x => pure x
  IO.println s!"[Load] shape = {info.shape}"
  -- GGUF stores weights row-major; for Q4_K a "row" = blocksPerRow*144 bytes.
  -- For ne[0]=2560, ne[1]=4096 (Q=out dim): each row is 2560 elements × Q4_K.
  let inDim : Nat := info.shape[0]!
  let outDim : Nat := info.shape[1]!
  let blocksPerRow := inDim / 256
  let bytesPerRow := blocksPerRow * 144
  IO.println s!"[Load] inDim={inDim} outDim={outDim} blocksPerRow={blocksPerRow} bytesPerRow={bytesPerRow}"

  -- Full-row comparison: take ALL rows (2048 for wQ).  Helps surface
  -- integration-scale bugs that the original 4-row test missed.
  let testOutDim : Nat := outDim
  let testBytes := testOutDim * bytesPerRow
  let mut wSlice : ByteArray := ByteArray.empty
  for i in [0:testBytes] do
    wSlice := wSlice.push (weightBytes.get! i)
  IO.println s!"[Test] Extracted {testBytes} bytes for {testOutDim} rows"

  -- Upload the 4-row slice as a Q4_K weight buffer.
  let wBuf ← createCUDABuffer testBytes.toUSize
  writeCUDABuffer wBuf wSlice

  -- Deterministic f32 input.  Use a RMSNorm-like distribution
  -- (post-norm values can reach ~±10 in real Gemma 4, not the [-1,1]
  -- of the original test).  Specifically: alternating signs, magnitude
  -- pattern that tests the Q8_1 amax reduction across all ranges.
  let xArr : Array Float := Array.ofFn (n := inDim) fun i =>
    let s : Float := if i.val % 2 == 0 then 1.0 else -1.0
    s * (1.0 + (i.val.toFloat / inDim.toFloat) * 5.0)
  let xBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let xBuf ← createCUDABuffer (4 * inDim).toUSize
  writeCUDABuffer xBuf xBytes
  IO.println s!"[Test] f32 input x[0..inDim) range [-1, 1]"

  -- llama.cpp Q8_1 quantize + Q4_K matmul.
  let nQ8Blocks := inDim / 32
  let q8Size := nQ8Blocks * 36
  let q8Buf ← createCUDABuffer q8Size.toUSize
  cuMemset q8Buf.ptr q8Size.toUSize
  launchQuantizeQ8_1 k xBuf.ptr q8Buf.ptr inDim
  let outLlama ← createCUDABuffer (testOutDim * 4).toUSize
  cuMemset outLlama.ptr (testOutDim * 4).toUSize
  launchMulMatVecQ4K k wBuf.ptr q8Buf.ptr outLlama.ptr inDim testOutDim
  let outLlamaBytes ← readCUDABuffer outLlama (testOutDim * 4).toUSize

  IO.println "[llama.cpp PTX output, first 4 rows]"
  for i in [0:(min 4 testOutDim)] do
    let v ← Hesper.Basic.bytesToFloat32 outLlamaBytes (i * 4)
    IO.println s!"  row[{i}] = {v}"

  -- Produce hesper's own Q8_1 buffer for comparison.
  let q8Hesper ← createCUDABuffer q8Size.toUSize
  cuMemset q8Hesper.ptr q8Size.toUSize
  GPUBackend.executeWithConfig ctx
    (Hesper.Layers.Linear.quantizeQ8_1Kernel inDim)
    [("input", xBuf), ("output", q8Hesper)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }

  -- hesper dp4a Q4_K matmul fed with hesper's own Q8_1 (self-consistent).
  let outHesper ← createCUDABuffer (testOutDim * 4).toUSize
  cuMemset outHesper.ptr (testOutDim * 4).toUSize
  let cfg : Hesper.Layers.Linear.Config := { inDim := inDim, outDim := testOutDim }
  GPUBackend.executeWithConfig ctx
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4AKernel cfg)
    [("weights", wBuf), ("input_q8", q8Hesper), ("output", outHesper)]
    { numWorkgroups := (testOutDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }

  let outHesperBytes ← readCUDABuffer outHesper (testOutDim * 4).toUSize

  IO.println "[hesper dp4a output, first 4 rows]"
  for i in [0:(min 4 testOutDim)] do
    let v ← Hesper.Basic.bytesToFloat32 outHesperBytes (i * 4)
    IO.println s!"  row[{i}] = {v}"

  -- Also inspect: first 8 bytes of each Q8_1 from each quantize path.
  IO.println "[Q8_1 block 0 header bytes]"
  let llamaQ8 ← readCUDABuffer q8Buf q8Size.toUSize
  let hesperQ8 ← readCUDABuffer q8Hesper q8Size.toUSize
  let b0L : Array UInt8 := Array.ofFn (n := 8) fun i => llamaQ8.get! i.val
  let b0H : Array UInt8 := Array.ofFn (n := 8) fun i => hesperQ8.get! i.val
  IO.println s!"  llama.cpp: {b0L}"
  IO.println s!"  hesper:    {b0H}"

  -- Diff llama-vs-hesper (full self-consistent runs).  Find max |diff|
  -- and the first row whose |diff| exceeds 1e-2 so we can see whether
  -- the bug is uniformly spread across rows or localised past some
  -- boundary.
  IO.println "[self-consistent llama vs self-consistent hesper — summary]"
  let mut maxDiff : Float := 0.0
  let mut maxDiffRow : Nat := 0
  let mut firstDivRow : Int := -1
  for i in [0:testOutDim] do
    let v1 ← Hesper.Basic.bytesToFloat32 outLlamaBytes (i * 4)
    let v2 ← Hesper.Basic.bytesToFloat32 outHesperBytes (i * 4)
    let d := v1 - v2
    let dAbs := if d < 0 then -d else d
    if dAbs > maxDiff then
      maxDiff := dAbs
      maxDiffRow := i
    if firstDivRow < 0 && dAbs > 0.01 then
      firstDivRow := i.toInt64.toInt
  IO.println s!"  max |diff| = {maxDiff} at row {maxDiffRow}"
  IO.println s!"  first row with |diff| > 0.01 = {firstDivRow}"
  -- Print first-4 and spot-check rows near firstDivRow if any.
  IO.println "[row-by-row around any divergence]"
  let spotRows : Array Nat :=
    if firstDivRow ≥ 0 then
      let r := firstDivRow.toNat
      #[r, r+1, r+2, r+3]
    else #[0, 1, 2, 3]
  for i in spotRows do
    if i < testOutDim then
      let v1 ← Hesper.Basic.bytesToFloat32 outLlamaBytes (i * 4)
      let v2 ← Hesper.Basic.bytesToFloat32 outHesperBytes (i * 4)
      let d := v1 - v2
      let dAbs := if d < 0 then -d else d
      IO.println s!"  row[{i}]: llama={v1} hesper={v2} |diff|={dAbs}"
