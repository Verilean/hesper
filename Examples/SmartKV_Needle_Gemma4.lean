import Hesper
import Hesper.Models.Gemma4
import Hesper.TTT.SmartKVCacheGemma4
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser
import Hesper.WebGPU.Device

/-!
# Gemma 4 Smart KV-Cache: 50K Token Endurance Test

Injects a random-word needle at the beginning of a ~50,000-token
prompt, then queries for it at the end. Smart KV-Cache should
protect the needle in permanent sinks while the 7,680-token sliding
window overwrites everything else.
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4
open Hesper.TTT.SmartKVGemma4
open Hesper.Tokenizer.SentencePiece

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 Smart KV-Cache: 50K Token Endurance Test      ║"
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
  IO.println "[GPU] Device initialized"

  IO.println s!"[Load] Parsing GGUF: {ggufPath}..."
  let ggufData ← IO.FS.readBinFile ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")

  IO.println "[Load] Initializing tokenizer..."
  let tokenizer ← fromGGUF gguf (some true) (some false)

  IO.println "[Load] Loading model..."
  let model ← Gemma4Model.fromGGUF device ggufPath
  IO.println s!"[Model] Gemma 4: {model.config.hiddenSize}d, {model.config.numHiddenLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"[Model] maxSeqLen={model.config.maxSeqLen}"
  IO.println ""

  -- Smart KV config: 512 sinks + 7680 window = 8192 total
  let smartConfig : SmartKVConfig := {
    windowSize := 7680
    tau := 0.5  -- rank-based: top-5 threshold
  }

  let needle := "The secret password is: banana dinosaur galaxy umbrella philosophy."
  let needleAnswer := "banana dinosaur"  -- check first 2 words (easier match)
  let query := "\n\nQuestion: What is the secret password?\nAnswer: The secret password is:"

  -- Build a large haystack by repeating diverse text
  let haystackBlock :=
    "The quick brown fox jumps over the lazy dog. " ++
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " ++
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " ++
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. " ++
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. " ++
    "Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia. " ++
    "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit. " ++
    "Sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. " ++
    "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis. " ++
    "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet consectetur. "

  -- Tokenize the haystack block to see how many tokens per repeat
  let blockTokens := encode tokenizer haystackBlock
  let tokensPerBlock := blockTokens.size
  IO.println s!"Haystack block: {tokensPerBlock} tokens per repeat"

  -- Target ~50,000 tokens — but capped at maxSeqLen
  let targetTokens := min 50000 (model.config.maxSeqLen - 200)
  let numRepeats := (targetTokens / tokensPerBlock) + 1

  -- Build full haystack
  let mut haystackFull := ""
  for _ in [0:numRepeats] do
    haystackFull := haystackFull ++ haystackBlock

  let fullPrompt := needle ++ " " ++ haystackFull ++ query
  let promptTokensArr := encode tokenizer fullPrompt
  let actualTokens := min promptTokensArr.size model.config.maxSeqLen

  IO.println s!"Needle: \"{needle}\""
  IO.println s!"Haystack: {numRepeats} repeats"
  IO.println s!"Full prompt: {promptTokensArr.size} tokens (capped to {actualTokens})"
  IO.println s!"Smart KV: window={smartConfig.windowSize}, top-K=5"
  IO.println s!"Physical KV slots: {smartConfig.windowSize} window (+ sinks)"
  IO.println ""

  -- Cap prompt to maxSeqLen
  let cappedPrompt := promptTokensArr.extract 0 actualTokens

  -- ═══════════════════════════════════════════
  -- Run 1: Smart KV (FIRST to avoid stale PreparedDispatch)
  -- ═══════════════════════════════════════════
  IO.println "═══ Run 1: Smart KV-Cache ═══"
  let startSmart ← IO.monoNanosNow
  let smartTokens ← generateWithSmartKV device model cappedPrompt 20 smartConfig .Greedy
    (verbose := false)
  let endSmart ← IO.monoNanosNow
  let smartMs := (endSmart - startSmart).toFloat / 1_000_000.0

  let smartGenIds := smartTokens.extract cappedPrompt.size smartTokens.size
  let smartText := decode tokenizer smartGenIds
  IO.println s!"  Generated: {smartText}"
  IO.println s!"  Time: {smartMs} ms ({smartMs / actualTokens.toFloat} ms/token)"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Run 2: Dumb Window
  -- ═══════════════════════════════════════════
  IO.println "═══ Run 2: Dumb Sliding Window ═══"
  let startDumb ← IO.monoNanosNow
  let dumbTokens ← generateWithDumbWindow device model cappedPrompt 20 smartConfig.windowSize .Greedy
  let endDumb ← IO.monoNanosNow
  let dumbMs := (endDumb - startDumb).toFloat / 1_000_000.0

  let dumbGenIds := dumbTokens.extract cappedPrompt.size dumbTokens.size
  let dumbText := decode tokenizer dumbGenIds
  IO.println s!"  Generated: {dumbText}"
  IO.println s!"  Time: {dumbMs} ms ({dumbMs / actualTokens.toFloat} ms/token)"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Comparison
  -- ═══════════════════════════════════════════
  let smartHas := (smartText.splitOn needleAnswer).length > 1
  let dumbHas := (dumbText.splitOn needleAnswer).length > 1

  IO.println "═══ Results ═══"
  IO.println s!"  Needle (first 2 words): '{needleAnswer}'"
  IO.println s!"  Smart KV contains needle: {if smartHas then "YES ✓" else "no ✗"}"
  IO.println s!"  Dumb window contains needle: {if dumbHas then "YES ✓" else "no ✗"}"
  IO.println ""

  if smartHas && !dumbHas then
    IO.println "╔══════════════════════════════════════════════════════════╗"
    IO.println "║  🎯 Smart KV-Cache WINS on Gemma 4 endurance test!     ║"
    IO.println "╚══════════════════════════════════════════════════════════╝"
  else if smartHas && dumbHas then
    IO.println "Both recalled the needle."
  else if !smartHas && !dumbHas then
    IO.println "Neither recalled the needle."
  else
    IO.println "Dumb window recalled but Smart KV didn't (unexpected)."

  IO.println ""
  IO.println "Done!"
