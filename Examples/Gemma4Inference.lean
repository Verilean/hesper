import Hesper
import Hesper.Models.Gemma4
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser
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
  -- If args[3] starts with "ids:" it's a comma-separated list of token IDs
  -- (e.g. "ids:2,105,2364,107,9259,106,107,105,4368,107"). Skips tokenization.
  let rawTokens : Option (Array Nat) :=
    match args[3]? with
    | some s =>
      if s.startsWith "ids:" then
        let rest := (s.toList.drop 4).asString
        some ((rest.splitOn ",").toArray.map (fun x => x.toNat!))
      else none
    | none => none

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

  -- Step 3: Load tokenizer from the same GGUF and encode the prompt
  IO.println "[Tokenize] Loading tokenizer from GGUF..."
  let ggufData ← IO.FS.readBinFile ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok g => pure g
    | .error e => throw <| IO.userError s!"Failed to parse GGUF: {e}"
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf
  let promptTokens ← match rawTokens with
    | some ids => do
      IO.println s!"[Tokens] Using raw token IDs: {ids}"
      pure ids
    | none => do
      let ts := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
      IO.println s!"[Tokens] Prompt '{prompt}' → {ts}"
      pure ts

  -- Step 4: Generate
  -- Gemma 4 uses <end_of_turn> (id 106) in chat format, plus <eos> (id 1).
  -- Stop on whichever comes first.
  let eosId := tokenizer.vocab.eosToken.getD 1
  let tokens ← Hesper.Models.Gemma4.generate device model promptTokens maxTokens
    (eosToken := some eosId) (extraEosTokens := #[106])

  IO.println ""
  IO.println s!"[Result] Generated tokens: {tokens}"
  -- Decode just the generated portion (skip the prompt tokens)
  let generated := tokens.extract promptTokens.size tokens.size
  let decoded := Hesper.Tokenizer.SentencePiece.decode tokenizer generated
  IO.println s!"[Result] Decoded: {decoded}"
  IO.println "Done!"
