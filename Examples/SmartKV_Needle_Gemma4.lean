import Hesper
import Hesper.Models.Gemma4
import Hesper.TTT.SmartKVCacheGemma4
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser
import Hesper.WebGPU.Device

/-!
# Gemma 4 Smart KV-Cache: Natural Language Needle Test

Uses the actual Gemma 4 tokenizer to encode natural language prompts,
avoiding the hidden-state divergence caused by synthetic raw token IDs.
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4
open Hesper.TTT.SmartKVGemma4
open Hesper.Tokenizer.SentencePiece

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 + Smart KV-Cache: Natural Language Needle     ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  let ggufPath ← match args with
    | path :: _ => pure path
    | [] => do
      IO.println "Usage: smart-kv-needle-gemma4 <gemma4.gguf>"
      IO.Process.exit 1
      pure ""

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Load GGUF for both model and tokenizer
  IO.println s!"[Load] Parsing GGUF: {ggufPath}..."
  let ggufData ← IO.FS.readBinFile ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")

  IO.println "[Load] Initializing tokenizer..."
  let tokenizer ← fromGGUF gguf (some true) (some false)  -- addBos=true, addEos=false
  IO.println s!"  Vocab: {tokenizer.vocab.vocabSize} tokens"

  IO.println "[Load] Loading model..."
  let model ← Gemma4Model.fromGGUF device ggufPath
  IO.println s!"[Model] Gemma 4: {model.config.hiddenSize}d, {model.config.numHiddenLayers}L"
  IO.println ""

  let smartConfig : SmartKVConfig := { windowSize := 256, tau := 5.0 }

  -- Natural language needle: a secret password buried in text
  -- Use a truly unpredictable needle: random hex string that no LLM
  -- could have seen in training data or predict from context.
  let needle := "The encryption key is: x7f2a9b4e1d6c8350."
  let needleAnswer := "x7f2a9b4e1d6c8350"

  -- Build prompts with increasing haystack
  -- Single test with verbose rank output for analysis
  let haystackTexts : Array (Nat × String) := #[
    (100, "The quick brown fox jumps over the lazy dog. " ++
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " ++
          "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " ++
          "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "),
    (300, "The quick brown fox jumps over the lazy dog. " ++
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " ++
          "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " ++
          "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. " ++
          "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore. " ++
          "Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt. " ++
          "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit. " ++
          "Sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. " ++
          "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet consectetur. " ++
          "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis. ")
  ]

  let query := "\n\nQuestion: What is the secret password?\nAnswer: The secret password is:"

  IO.println s!"Needle: \"{needle}\""
  IO.println s!"Query: \"{query}\""
  IO.println s!"Window: {smartConfig.windowSize}, Surprise tau: {smartConfig.tau}"
  IO.println ""

  for (label, haystack) in haystackTexts do
    let fullPrompt := needle ++ " " ++ haystack ++ query
    let promptTokensArr := encode tokenizer fullPrompt
    IO.println s!"═══ Haystack ~{label} words ({promptTokensArr.size} tokens) ═══"

    -- Run 1: Dumb window
    IO.println "  [Dumb Window]"
    let dumbTokens ← generateWithDumbWindow device model promptTokensArr 10 smartConfig.windowSize .Greedy
    let dumbGenIds := dumbTokens.extract promptTokensArr.size dumbTokens.size
    let dumbText := decode tokenizer dumbGenIds
    IO.println s!"    Generated: {dumbText}"

    -- Run 2: Smart KV
    IO.println "  [Smart KV]"
    let smartTokens ← generateWithSmartKV device model promptTokensArr 10 smartConfig .Greedy
      (verbose := label == 100)
    let smartGenIds := smartTokens.extract promptTokensArr.size smartTokens.size
    let smartText := decode tokenizer smartGenIds
    IO.println s!"    Generated: {smartText}"
    IO.println s!"    Sinks protected: (see prefill log)"

    -- Simple substring check
    let dumbHas := (dumbText.splitOn needleAnswer).length > 1
    let smartHas := (smartText.splitOn needleAnswer).length > 1

    IO.println s!"    Dumb contains '{needleAnswer}': {if dumbHas then "YES" else "no"}"
    IO.println s!"    Smart contains '{needleAnswer}': {if smartHas then "YES" else "no"}"

    if smartHas && !dumbHas then
      IO.println "    🎯 Smart KV wins!"
    else if smartHas && dumbHas then
      IO.println "    Both got it"
    else if !smartHas && !dumbHas then
      IO.println "    Both missed"
    else
      IO.println "    Dumb wins (unexpected)"
    IO.println ""

  IO.println "Done!"
