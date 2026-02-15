import Hesper.WebGPU.Device
import Hesper.Models.BitNet
import Hesper.Inference.Sampling

/-!
# BitNet Inference Example

Complete end-to-end example of loading and running BitNet model inference.

## Quick Start

```bash
# Download a BitNet GGUF model
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf

# Run this example
lake env lean --run Examples/BitNetInference.lean
```

## Usage Examples

This file demonstrates:
1. Loading a BitNet model from GGUF
2. Running forward pass
3. Generating text with different sampling strategies
4. Model statistics and diagnostics
-/

namespace Hesper.Examples.BitNetInference

open Hesper.Models.BitNet
open Hesper.Inference.Sampling
open Hesper.WebGPU

/-! ## Example 1: Load Model and Print Statistics -/

def example1LoadModel (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 1: Load Model"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Initialize WebGPU device
  IO.println "Initializing WebGPU device..."
  let device ← initializeDevice

  -- Load BitNet model from GGUF
  IO.println s!"Loading model from: {ggufPath}"
  let model ← BitNet.fromGGUF device ggufPath none

  -- Print model statistics
  BitNet.printStats model

  IO.println "✓ Example 1 complete"
  IO.println ""

/-! ## Example 2: Generate Text (Greedy Decoding) -/

def example2GreedyGeneration (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 2: Greedy Generation"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let device ← initializeDevice
  let model ← BitNet.fromGGUF device ggufPath none

  -- Example prompt: "Hello world"
  -- (In real usage, these would come from a tokenizer)
  let promptTokens := #[1, 2, 3, 4, 5]

  IO.println "Input prompt tokens: [1, 2, 3, 4, 5]"
  IO.println "Generating with Greedy sampling (deterministic)..."
  IO.println ""

  -- Generate with greedy sampling
  let outputTokens ← BitNet.generate device model promptTokens 20 .Greedy

  IO.println s!"Generated {outputTokens.size - promptTokens.size} new tokens"
  IO.println s!"Output tokens: {outputTokens}"
  IO.println ""
  IO.println "✓ Example 2 complete"
  IO.println ""

/-! ## Example 3: Generate Text (Top-k Sampling) -/

def example3TopKGeneration (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 3: Top-k Sampling"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let device ← initializeDevice
  let model ← BitNet.fromGGUF device ggufPath none

  let promptTokens := #[1, 2, 3, 4, 5]

  -- Top-k sampling: sample from 40 most likely tokens
  -- Temperature 0.8: slightly more random than default
  let strategy := Strategy.TopK 40 0.8

  IO.println s!"Input prompt tokens: {promptTokens}"
  IO.println "Generating with Top-k sampling (k=40, temp=0.8)..."
  IO.println ""

  let outputTokens ← BitNet.generate device model promptTokens 20 strategy

  IO.println s!"Generated {outputTokens.size - promptTokens.size} new tokens"
  IO.println s!"Output tokens: {outputTokens}"
  IO.println ""
  IO.println "Note: Output will vary due to sampling randomness"
  IO.println "✓ Example 3 complete"
  IO.println ""

/-! ## Example 4: Generate Text (Nucleus Sampling) -/

def example4NucleusGeneration (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 4: Nucleus (Top-p) Sampling"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let device ← initializeDevice
  let model ← BitNet.fromGGUF device ggufPath none

  let promptTokens := #[1, 2, 3, 4, 5]

  -- Nucleus sampling: sample from tokens with cumulative prob 0.9
  -- Temperature 1.0: default randomness
  let strategy := Strategy.Nucleus 0.9 1.0

  IO.println s!"Input prompt tokens: {promptTokens}"
  IO.println "Generating with Nucleus sampling (p=0.9, temp=1.0)..."
  IO.println ""

  -- Stop at EOS token (ID 2 in this example)
  let outputTokens ← BitNet.generate device model promptTokens 20 strategy (some 2)

  IO.println s!"Generated {outputTokens.size - promptTokens.size} new tokens"
  IO.println s!"Output tokens: {outputTokens}"
  IO.println ""
  IO.println "Note: Generation may stop early if EOS token generated"
  IO.println "✓ Example 4 complete"
  IO.println ""

/-! ## Example 5: Compare Sampling Strategies -/

def example5CompareSamplingStrategies (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 5: Compare Sampling Strategies"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let device ← initializeDevice
  let model ← BitNet.fromGGUF device ggufPath none

  let promptTokens := #[1, 2, 3, 4, 5]
  let maxTokens := 10

  IO.println s!"Prompt: {promptTokens}"
  IO.println s!"Generating {maxTokens} tokens with each strategy..."
  IO.println ""

  -- Strategy 1: Greedy
  IO.println "1. Greedy (deterministic):"
  let output1 ← BitNet.generate device model promptTokens maxTokens .Greedy
  IO.println s!"   Output: {output1}"
  IO.println ""

  -- Strategy 2: Top-k with low temperature (more conservative)
  IO.println "2. Top-k (k=10, temp=0.5):"
  let strategy2 := Strategy.TopK 10 0.5
  let output2 ← BitNet.generate device model promptTokens maxTokens strategy2
  IO.println s!"   Output: {output2}"
  IO.println ""

  -- Strategy 3: Top-k with high temperature (more creative)
  IO.println "3. Top-k (k=40, temp=1.2):"
  let strategy3 := Strategy.TopK 40 1.2
  let output3 ← BitNet.generate device model promptTokens maxTokens strategy3
  IO.println s!"   Output: {output3}"
  IO.println ""

  -- Strategy 4: Nucleus
  IO.println "4. Nucleus (p=0.9, temp=1.0):"
  let strategy4 := Strategy.Nucleus 0.9 1.0
  let output4 ← BitNet.generate device model promptTokens maxTokens strategy4
  IO.println s!"   Output: {output4}"
  IO.println ""

  IO.println "✓ Example 5 complete"
  IO.println ""

/-! ## Example 6: Long-Form Generation -/

def example6LongFormGeneration (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 6: Long-Form Text Generation"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let device ← initializeDevice
  let model ← BitNet.fromGGUF device ggufPath none

  -- Longer prompt
  let promptTokens := #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  -- Generate longer sequence
  let maxTokens := 100

  -- Use nucleus sampling for quality
  let strategy := Strategy.Nucleus 0.9 1.0

  IO.println s!"Prompt length: {promptTokens.size} tokens"
  IO.println s!"Generating up to {maxTokens} new tokens..."
  IO.println ""

  let start ← IO.monoMsNow
  let outputTokens ← BitNet.generate device model promptTokens maxTokens strategy (some 2)
  let duration ← IO.monoMsNow

  let elapsed := duration - start
  let tokensGenerated := outputTokens.size - promptTokens.size
  let tokensPerSec := tokensGenerated.toFloat / (elapsed.toFloat / 1000.0)

  IO.println ""
  IO.println "Generation Statistics:"
  IO.println s!"  Tokens generated: {tokensGenerated}"
  IO.println s!"  Time elapsed: {elapsed} ms"
  IO.println s!"  Throughput: {tokensPerSec} tokens/sec"
  IO.println ""
  IO.println "✓ Example 6 complete"
  IO.println ""

/-! ## Example 7: Model Configuration Variants -/

def example7ConfigVariants : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 7: Model Configuration Variants"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- BitNet-3B configuration
  IO.println "BitNet-3B Configuration:"
  let config3B := Config.bitnet3B
  IO.println s!"  Vocabulary: {config3B.vocabSize}"
  IO.println s!"  Dimensions: {config3B.dim}"
  IO.println s!"  Layers: {config3B.numLayers}"
  IO.println s!"  Heads: {config3B.numHeads}"
  IO.println s!"  FFN: {config3B.ffnDim}"
  IO.println s!"  Max seq: {config3B.maxSeqLen}"
  IO.println ""

  -- BitNet-1.3B configuration
  IO.println "BitNet-1.3B Configuration:"
  let config1_3B := Config.bitnet1_3B
  IO.println s!"  Vocabulary: {config1_3B.vocabSize}"
  IO.println s!"  Dimensions: {config1_3B.dim}"
  IO.println s!"  Layers: {config1_3B.numLayers}"
  IO.println s!"  Heads: {config1_3B.numHeads}"
  IO.println s!"  FFN: {config1_3B.ffnDim}"
  IO.println s!"  Max seq: {config1_3B.maxSeqLen}"
  IO.println ""

  -- Custom configuration
  IO.println "Custom Configuration Example:"
  let customConfig : Config := {
    vocabSize := 32000,
    dim := 2048,
    numLayers := 24,
    numHeads := 16,
    ffnDim := 8192,
    maxSeqLen := 2048,
    temperature := 0.8,
    topK := 50,
    topP := 0.95
  }
  IO.println s!"  Vocabulary: {customConfig.vocabSize}"
  IO.println s!"  Dimensions: {customConfig.dim}"
  IO.println s!"  Layers: {customConfig.numLayers}"
  IO.println ""

  IO.println "✓ Example 7 complete"
  IO.println ""

/-! ## Main Entry Point -/

def main (args : List String) : IO Unit := do
  match args with
  | [] =>
    IO.println "Usage: lean --run Examples/BitNetInference.lean <gguf_path> [example_number]"
    IO.println ""
    IO.println "Examples:"
    IO.println "  1 - Load model and print statistics"
    IO.println "  2 - Greedy generation"
    IO.println "  3 - Top-k sampling"
    IO.println "  4 - Nucleus sampling"
    IO.println "  5 - Compare sampling strategies"
    IO.println "  6 - Long-form generation"
    IO.println "  7 - Model configuration variants"
    IO.println "  all - Run all examples"
    IO.println ""
    IO.println "Example:"
    IO.println "  lean --run Examples/BitNetInference.lean model.gguf 2"

  | [ggufPath] =>
    -- Run all examples
    example1LoadModel ggufPath
    example2GreedyGeneration ggufPath
    example3TopKGeneration ggufPath
    example4NucleusGeneration ggufPath
    example5CompareSamplingStrategies ggufPath
    example6LongFormGeneration ggufPath
    example7ConfigVariants

  | [ggufPath, exampleNum] =>
    match exampleNum with
    | "1" => example1LoadModel ggufPath
    | "2" => example2GreedyGeneration ggufPath
    | "3" => example3TopKGeneration ggufPath
    | "4" => example4NucleusGeneration ggufPath
    | "5" => example5CompareSamplingStrategies ggufPath
    | "6" => example6LongFormGeneration ggufPath
    | "7" => example7ConfigVariants
    | "all" =>
      example1LoadModel ggufPath
      example2GreedyGeneration ggufPath
      example3TopKGeneration ggufPath
      example4NucleusGeneration ggufPath
      example5CompareSamplingStrategies ggufPath
      example6LongFormGeneration ggufPath
      example7ConfigVariants
    | _ =>
      IO.println s!"Unknown example: {exampleNum}"
      IO.println "Valid examples: 1, 2, 3, 4, 5, 6, 7, all"

  | _ =>
    IO.println "Too many arguments"
    IO.println "Usage: lean --run Examples/BitNetInference.lean <gguf_path> [example_number]"

end Hesper.Examples.BitNetInference
