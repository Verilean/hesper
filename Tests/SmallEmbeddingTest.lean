import Hesper
import Hesper.Layers.Embedding
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Small Embedding Test

Test embedding layer with a small vocab size (< 128 MB) to verify if the
BindGroup error is due to buffer size limits.

Calculation:
- vocab_size = 10000 (instead of 50000)
- dim = 2560
- Total size = 10000 × 2560 × 4 bytes = **102.4 MB** (< 128 MB default limit)

If this test passes, it confirms the issue is the 128 MB `maxStorageBufferBindingSize` limit.
-/

def main : IO UInt32 := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Small Embedding Test (< 128 MB)"
  IO.println "═══════════════════════════════════════════════"

  -- Initialize WebGPU
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  IO.println "✓ Device initialized"

  -- Configuration: Small vocab size to stay under 128 MB
  let vocabSize := 10000  -- 5x smaller than BitNet
  let dim := 2560
  let bufferSize := vocabSize * dim * 4  -- 102.4 MB

  IO.println s!"  vocab_size: {vocabSize}"
  IO.println s!"  dim: {dim}"
  IO.println s!"  buffer_size: {bufferSize / (1024*1024)} MB"
  IO.println ""

  -- Create embedding config
  let config : Hesper.Layers.Embedding.Config := {
    vocabSize := vocabSize
    dim := dim
  }

  -- Create dummy F32 embedding data
  let mut embeddingData := ByteArray.empty
  for _ in [0:vocabSize * dim] do
    -- Fill with dummy float values (0.0)
    embeddingData := embeddingData.push 0
    embeddingData := embeddingData.push 0
    embeddingData := embeddingData.push 0
    embeddingData := embeddingData.push 0

  IO.println "[1/3] Creating embedding layer..."
  let embedding ← Hesper.Layers.Embedding.createFromFloat32 device config embeddingData
  IO.println "  ✓ Embedding layer created"

  -- Create test input tokens
  IO.println "[2/3] Preparing test input..."
  let inputTokens := #[1, 2, 3, 4, 5]  -- 5 tokens
  let tokensBuf ← Hesper.WebGPU.BufferOps.uploadTokens device inputTokens

  -- Create output buffer
  let seqLen := inputTokens.size
  let numElements := 1 * seqLen * dim
  let outputSize := (numElements * 4).toUSize
  let outputBuf ← Hesper.WebGPU.createBuffer device {
    size := outputSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  IO.println "  ✓ Buffers prepared"

  -- Execute embedding lookup
  IO.println "[3/3] Executing embedding lookup..."
  Hesper.Layers.Embedding.forward device embedding tokensBuf outputBuf 1 seqLen

  -- Download results
  let results ← Hesper.WebGPU.BufferOps.downloadFloatArray device outputBuf numElements
  IO.println s!"  ✓ Downloaded {results.size} elements"

  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "✅ TEST PASSED - Small embedding works!"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  IO.println "CONCLUSION: If this passes but 512 MB fails,"
  IO.println "the issue is the 128 MB maxStorageBufferBindingSize limit."

  pure 0
