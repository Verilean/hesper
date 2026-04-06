import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Types

open Hesper.WebGPU

/-!
# Gemma 4 E2E Inference Test

Loads a Gemma 4 GGUF model and generates text.

Usage:
  lake exe gemma4-inference data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 10
-/

def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let prompt := args.getD 1 "Hello"
  let maxTokens := (args.getD 2 "10").toNat!

  IO.println "═══════════════════════════════════════════"
  IO.println "  Gemma 4 E2E Inference Test"
  IO.println "═══════════════════════════════════════════"
  IO.println s!"  Model: {ggufPath}"
  IO.println s!"  Prompt: \"{prompt}\""
  IO.println s!"  Max tokens: {maxTokens}"
  IO.println ""

  -- Step 1: Initialize WebGPU
  IO.println "[Init] Creating WebGPU device..."
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  -- Step 2: Load model
  IO.println "[Load] Loading Gemma 4 model..."
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF device ggufPath

  IO.println s!"[Config] hiddenSize={model.config.hiddenSize}, layers={model.config.numHiddenLayers}"
  IO.println s!"[Config] heads={model.config.numAttentionHeads}, kvHeads={model.config.numKeyValueHeadsFull}"
  IO.println s!"[Config] headDimFull={model.config.headDimFull}, headDimSWA={model.config.headDimSWA}"
  IO.println s!"[Config] ffnSize={model.config.intermediateSize}, vocabSize={model.config.vocabSize}"
  IO.println s!"[Config] experts={model.config.numExperts}, embdPerLayer={model.config.embdPerLayer}"
  IO.println s!"[Config] slidingWindow={model.config.slidingWindowSize}, kvShared={model.config.numKVSharedLayers}"

  -- Step 3: Simple token test (use BOS token as prompt for now)
  -- Gemma 4 tokenizer: BOS=2, EOS=1
  let promptTokens := #[2]  -- BOS token
  IO.println s!"[Tokens] Prompt tokens: {promptTokens}"

  -- Step 4: Generate
  let tokens ← Hesper.Models.Gemma4.generate device model promptTokens maxTokens (eosToken := some 1)

  IO.println ""
  IO.println s!"[Result] Generated tokens: {tokens}"
  IO.println "Done!"
