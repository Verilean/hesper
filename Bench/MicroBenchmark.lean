import Hesper
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.WGSL.Execute
import Hesper.WGSL.Elementwise
import Hesper.WGSL.MatMul
import Hesper.Layers.RMSNorm
import Hesper.Layers.BitLinear
import Hesper.Layers.Attention
import Hesper.Layers.TransformerBlock
import Hesper.Layers.Embedding
import Hesper.Layers.RoPE
import Hesper.Layers.Softmax
import Hesper.Models.BitNet
import Hesper.Logging

/-!
# Microbenchmark for BitNet Inference Pipeline

Measures per-operation latency to identify bottlenecks.
Tests each layer type with BitNet-2B dimensions.
-/

open Hesper.WebGPU
open Hesper.WGSL
open Hesper.Layers

/-- Measure average execution time in ms over multiple iterations -/
def timeMsAvg (iterations : Nat) (action : IO Unit) : IO Float := do
  -- Warmup
  action
  -- Timed iterations
  let start ← IO.monoNanosNow
  for _ in [0:iterations] do
    action
  let stop ← IO.monoNanosNow
  pure ((stop - start).toFloat / (iterations.toFloat * 1_000_000.0))

/-- Create a GPU buffer filled with test data -/
def createTestBuffer (device : Device) (numFloats : Nat) (fillVal : Float := 1.0) : IO Buffer := do
  let arr := Array.replicate numFloats fillVal
  let bytes := Hesper.WebGPU.floatArrayToBytes arr
  let buf ← createBuffer device { size := bytes.size.toUSize, usage := [.storage, .copyDst], mappedAtCreation := false }
  writeBuffer device buf 0 bytes
  pure buf

/-- Create a zero buffer for outputs -/
def createOutputBuffer (device : Device) (numFloats : Nat) : IO Buffer := do
  createBuffer device { size := (numFloats * 4).toUSize, usage := [.storage], mappedAtCreation := false }

/-- Create a ByteArray of given size filled with a byte value -/
def mkFilledByteArray (size : Nat) (val : UInt8) : ByteArray :=
  ⟨ByteArray.empty.data ++ (Array.replicate size val)⟩

def main : IO Unit := do
  -- Suppress verbose logging for clean benchmark output
  Hesper.Logging.setVerbose false

  IO.println "================================================================"
  IO.println "  Hesper MicroBenchmark - BitNet 2B Dimensions"
  IO.println "================================================================"
  IO.println ""

  -- Get GPU device
  let inst ← Hesper.init
  let device ← getDeviceWithFeatures inst
  IO.println "GPU device initialized."
  IO.println ""

  -- BitNet 2B dimensions
  let dim := 2560
  let ffnDim := 6912
  let vocabSize := 128256
  let iterations := 10

  let numElements := dim      -- batchSize=1, seqLen=1
  let ffnElements := ffnDim

  -- Create test buffers
  IO.println "Creating test buffers..."
  let inputBuf ← createTestBuffer device numElements 0.5
  let outputBuf ← createOutputBuffer device numElements
  let scaleData := Hesper.WebGPU.floatArrayToBytes (Array.replicate dim 1.0)

  IO.println ""
  IO.println s!"--- Per-Operation Latency (averaged over {iterations} iterations) ---"
  IO.println ""

  -- === 1. RMSNorm ===
  let rmsNormConfig : RMSNorm.Config := { dim }
  let rmsNorm ← RMSNorm.create device rmsNormConfig scaleData
  let rmsMs ← timeMsAvg iterations (RMSNorm.forward device rmsNorm inputBuf outputBuf 1)
  IO.println s!"RMSNorm ({dim} dim):                   {rmsMs} ms"

  -- === 2. BitLinear (Q projection: dim→dim) ===
  let packedBytes := dim * dim / 4  -- 2 bits per weight = 4 weights per byte
  let testWeights := mkFilledByteArray packedBytes 0x55  -- all zeros (01010101 = code 01 repeated)
  let qlConfig : BitLinear.Config := { inDim := dim, outDim := dim, batchSize := 1 }
  let qlLayer ← BitLinear.create device qlConfig testWeights 1.0
  let qlMs ← timeMsAvg iterations (BitLinear.forward device qlLayer inputBuf outputBuf 1)
  IO.println s!"BitLinear ({dim}->{dim}):               {qlMs} ms"

  -- === 3. BitLinear (gate: dim→ffnDim) ===
  let packedBytesGate := dim * ffnDim / 4
  let testWeightsGate := mkFilledByteArray packedBytesGate 0x55
  let gateConfig : BitLinear.Config := { inDim := dim, outDim := ffnDim, batchSize := 1 }
  let ffnOutputBuf ← createOutputBuffer device ffnElements
  let gateLayer ← BitLinear.create device gateConfig testWeightsGate 1.0
  let gateMs ← timeMsAvg iterations (BitLinear.forward device gateLayer inputBuf ffnOutputBuf 1)
  IO.println s!"BitLinear ({dim}->{ffnDim}):             {gateMs} ms"

  -- === 4. BitLinear (down: ffnDim→dim) ===
  let packedBytesDown := ffnDim * dim / 4
  let testWeightsDown := mkFilledByteArray packedBytesDown 0x55
  let downConfig : BitLinear.Config := { inDim := ffnDim, outDim := dim, batchSize := 1 }
  let ffnTmpBuf ← createTestBuffer device ffnElements 0.3
  let ffnDownOutputBuf ← createOutputBuffer device numElements
  let downLayer ← BitLinear.create device downConfig testWeightsDown 1.0
  let downMs ← timeMsAvg iterations (BitLinear.forward device downLayer ffnTmpBuf ffnDownOutputBuf 1)
  IO.println s!"BitLinear ({ffnDim}->{dim}):             {downMs} ms"

  -- === 5. Elementwise Add (residual) ===
  let addBuf ← createOutputBuffer device numElements
  let elemConfig : Elementwise.Config := { numElements }
  let addMs ← timeMsAvg iterations (Elementwise.executeAdd device inputBuf outputBuf addBuf elemConfig)
  IO.println s!"Elementwise Add ({numElements} elems):   {addMs} ms"

  -- === 6. ReLU²*Mul (gated activation) ===
  let hiddenBuf ← createOutputBuffer device ffnElements
  let ffnElemConfig : Elementwise.Config := { numElements := ffnElements }
  let ffnTestBuf1 ← createTestBuffer device ffnElements 0.5
  let ffnTestBuf2 ← createTestBuffer device ffnElements 0.3
  let reluMs ← timeMsAvg iterations (Elementwise.executeReluSqrMul device ffnTestBuf1 ffnTestBuf2 hiddenBuf ffnElemConfig)
  IO.println s!"ReLU-Sq-Mul ({ffnElements} elems):       {reluMs} ms"

  -- === 7. MatMul (LM head proxy: 1×dim @ dim×vocab) ===
  -- Note: Full vocab (128256) requires ~1.3GB CPU array to initialize.
  -- We use a zero-filled GPU buffer instead of createTestBuffer to avoid CPU OOM.
  let logitsBuf ← createBuffer device { size := (vocabSize * 4).toUSize, usage := [.storage, .copySrc], mappedAtCreation := false }
  let lmConfig : MatMul.Config := { M := 1, N := vocabSize, K := dim }
  let embTableBuf ← createOutputBuffer device (vocabSize * dim)
  let lmMs ← timeMsAvg iterations (MatMul.executeMatMulTranspose device inputBuf embTableBuf logitsBuf lmConfig)
  IO.println s!"MatMul LM Head (1x{dim} @ {dim}x{vocabSize}): {lmMs} ms"

  -- === 8. GPU Argmax ===
  let argmaxOutBuf ← createBuffer device { size := 4, usage := [.storage, .copySrc], mappedAtCreation := false }
  let argmaxMs ← timeMsAvg iterations (do
    let _ ← Hesper.Models.BitNet.gpuArgmax device logitsBuf argmaxOutBuf vocabSize
    pure ())
  IO.println s!"GPU Argmax ({vocabSize} logits):       {argmaxMs} ms"

  -- Compare with CPU download time
  let downloadMs ← timeMsAvg iterations (do
    let _ ← Hesper.WebGPU.BufferOps.downloadFloatArray device logitsBuf vocabSize
    pure ())
  IO.println s!"CPU Download ({vocabSize} floats):     {downloadMs} ms"
  IO.println s!"  GPU argmax saves: {downloadMs - argmaxMs} ms/token"

  IO.println ""

  -- === Summary: Estimate full forward pass ===
  let rmsPerLayer := 4.0 * rmsMs
  let bitlinearPerLayer := 4.0 * qlMs + 2.0 * gateMs + downMs
  let elemPerLayer := 2.0 * addMs + reluMs
  let layerEstimate := rmsPerLayer + bitlinearPerLayer + elemPerLayer
  let totalLayers := 30.0
  let layersTotal := totalLayers * layerEstimate
  let totalEstimate := layersTotal + lmMs

  IO.println "--- Estimated Per-Token Latency (30 layers, seqLen=1) ---"
  IO.println ""
  IO.println s!"  RMSNorm (4/layer):           {rmsPerLayer} ms/layer x {totalLayers} = {totalLayers * rmsPerLayer} ms"
  IO.println s!"  BitLinear (7/layer):          {bitlinearPerLayer} ms/layer x {totalLayers} = {totalLayers * bitlinearPerLayer} ms"
  IO.println s!"  Elementwise (3/layer):        {elemPerLayer} ms/layer x {totalLayers} = {totalLayers * elemPerLayer} ms"
  IO.println s!"  LM Head MatMul:               {lmMs} ms"
  IO.println s!"  NOTE: Excludes attention matmuls, reshape, softmax, RoPE"
  IO.println ""
  IO.println s!"  Estimated total:              {totalEstimate} ms/token"
  if totalEstimate > 0 then
    IO.println s!"  Projected TPS:                {1000.0 / totalEstimate} tokens/sec"
  IO.println ""

  -- === 8. Command Buffer Batching Comparison ===
  IO.println "--- Command Buffer Batching ---"
  IO.println ""

  -- Test: Run N elementwise add dispatches individually vs batched
  let batchN := 100
  let batchIter := 5

  -- Unbatched: each dispatch creates encoder + submits + waits
  let unbatchedMs ← timeMsAvg batchIter (do
    for _ in [0:batchN] do
      Elementwise.executeAdd device inputBuf outputBuf addBuf elemConfig)
  IO.println s!"  {batchN} dispatches (unbatched): {unbatchedMs} ms ({unbatchedMs / batchN.toFloat} ms/dispatch)"

  -- Batched: record all dispatches into one encoder, submit once
  let batchedMs ← timeMsAvg batchIter (do
    Hesper.WGSL.Execute.beginBatch device
    for _ in [0:batchN] do
      Elementwise.executeAdd device inputBuf outputBuf addBuf elemConfig
    Hesper.WGSL.Execute.endBatch device)
  IO.println s!"  {batchN} dispatches (batched):   {batchedMs} ms ({batchedMs / batchN.toFloat} ms/dispatch)"
  if unbatchedMs > 0 then
    IO.println s!"  Speedup:                        {unbatchedMs / batchedMs}x"
  IO.println ""

  -- === Pipeline cache stats ===
  let (cacheHits, cacheMisses) ← Hesper.WGSL.Execute.getPipelineCacheStats
  IO.println "--- Pipeline Cache Stats ---"
  IO.println s!"  Cache hits:    {cacheHits}"
  IO.println s!"  Cache misses:  {cacheMisses}"
  let total := cacheHits + cacheMisses
  if total > 0 then
    IO.println s!"  Hit rate:      {cacheHits.toFloat / total.toFloat * 100.0}%"
  IO.println ""

  -- === Bind group cache stats ===
  let (bgHits, bgMisses) ← Hesper.WGSL.Execute.getBindGroupCacheStats
  IO.println "--- Bind Group Cache Stats ---"
  IO.println s!"  Cache hits:    {bgHits}"
  IO.println s!"  Cache misses:  {bgMisses}"
  let bgTotal := bgHits + bgMisses
  if bgTotal > 0 then
    IO.println s!"  Hit rate:      {bgHits.toFloat / bgTotal.toFloat * 100.0}%"
  IO.println ""

  IO.println "================================================================"
  IO.println "  Benchmark complete."
  IO.println "================================================================"
