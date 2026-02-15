import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Embedding

/-!
# Minimal Embedding Lookup Test

Tests embedding lookup with small buffers to isolate GPU validation issues.

Test Configuration:
- Vocabulary: 100 tokens (instead of 50,000)
- Dimension: 128 (instead of 2,560)
- Embedding table: 100 × 128 × 4 = 51,200 bytes (51 KB instead of 512 MB)
- Input: Single token ID
- Output: Single embedding vector

This minimal test helps identify if the GPU error is:
1. Buffer-size-related (512 MB exceeding limits)
2. General binding/dispatch issue
3. Shader compilation problem
-/

namespace Tests.Embedding

open Hesper.WebGPU
open Hesper.Layers.Embedding

def runMinimalTest : IO Unit := do
  IO.println "════════════════════════════════════════════════"
  IO.println "  Minimal Embedding Lookup Test"
  IO.println "════════════════════════════════════════════════"

  -- Initialize WebGPU
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "✓ Device created"

  -- Test configuration (small sizes)
  let vocabSize := 100
  let dim := 128
  let batchSize := 1
  let seqLen := 1

  IO.println s!"Configuration: vocab={vocabSize}, dim={dim}"
  IO.println s!"Embedding table size: {vocabSize * dim * 4} bytes (~51 KB)"

  -- Create embedding table with test data
  -- Fill with simple pattern: embedding[i][j] = i + j * 0.01
  let mut embeddingData := ByteArray.empty
  for tokenId in [0:vocabSize] do
    for d in [0:dim] do
      let value : Float := tokenId.toFloat + d.toFloat * 0.01
      let bits := value.toBits
      embeddingData := embeddingData.push (bits.toUInt8)
      embeddingData := embeddingData.push ((bits >>> 8).toUInt8)
      embeddingData := embeddingData.push ((bits >>> 16).toUInt8)
      embeddingData := embeddingData.push ((bits >>> 24).toUInt8)

  IO.println s!"✓ Generated {embeddingData.size} bytes of test data"

  -- Create embedding layer
  let config : Config := { vocabSize, dim }
  let layer ← createFromFloat32 device config embeddingData
  IO.println "✓ Embedding layer created"

  -- Create input buffer (single token ID = 42)
  let tokenId : UInt32 := 42
  let mut tokenData := ByteArray.empty
  tokenData := tokenData.push (tokenId.toUInt8)
  tokenData := tokenData.push ((tokenId >>> 8).toUInt8)
  tokenData := tokenData.push ((tokenId >>> 16).toUInt8)
  tokenData := tokenData.push ((tokenId >>> 24).toUInt8)

  let tokenBuf ← createBuffer device {
    size := 4
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  writeBuffer device tokenBuf 0 tokenData
  IO.println s!"✓ Token buffer created (token_id={tokenId})"

  -- Create output buffer
  let outputSize := (batchSize * seqLen * dim * 4).toUSize
  let outputBuf ← createBuffer device {
    size := outputSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  IO.println s!"✓ Output buffer created ({outputSize} bytes)"

  -- Run embedding lookup
  IO.println "\n[TEST] Running embedding lookup..."
  IO.println "  This will create BindGroup and dispatch compute..."

  try
    forward device layer tokenBuf outputBuf batchSize seqLen
    IO.println "✓ Embedding lookup succeeded!"

    -- Read back result
    IO.println "\n[TEST] Reading back results..."
    let resultData ← mapBufferRead device outputBuf 0 outputSize

    -- Parse first few values
    IO.println "First 5 embedding values:"
    for i in [0:min 5 dim] do
      if i * 4 + 4 <= resultData.size then
        let b0 := resultData.get! (i * 4)
        let b1 := resultData.get! (i * 4 + 1)
        let b2 := resultData.get! (i * 4 + 2)
        let b3 := resultData.get! (i * 4 + 3)
        let bits := b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
        let value := Float.ofBits bits
        let expected := 42.0 + i.toFloat * 0.01
        IO.println s!"  [{i}] = {value} (expected {expected})"

    IO.println "\n════════════════════════════════════════════════"
    IO.println "✓ TEST PASSED"
    IO.println "════════════════════════════════════════════════"

  catch e =>
    IO.println "\n════════════════════════════════════════════════"
    IO.println "✗ TEST FAILED"
    IO.println s!"Error: {e}"
    IO.println "════════════════════════════════════════════════"
    throw e

end Tests.Embedding

def main : IO Unit := Tests.Embedding.runMinimalTest
