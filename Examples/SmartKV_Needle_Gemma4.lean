import Hesper
import Hesper.Models.Gemma4
import Hesper.TTT.SmartKVCacheGemma4
import Hesper.WebGPU.Device

/-!
# Gemma 4 Smart KV-Cache Needle Test

Cosine surprise sensor + KV eviction approach:
- Both models use full sequential positions (Gemma4 untouched)
- Dumb window: zeros out ALL KV rows older than windowSize
- Smart KV: zeros out old rows EXCEPT surprise-protected sinks
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4
open Hesper.TTT.SmartKVGemma4

def buildPrompt (haystackSize : Nat) (vocabSize : Nat)
    (needleKey needleValue : Nat) (seed : Nat := 42)
    : Array Nat := Id.run do
  let sep := min 2 (vocabSize - 1)
  let queryMark := min 3 (vocabSize - 1)
  let mut prompt : Array Nat := #[]
  for i in [0:5] do prompt := prompt.push (10 + i)
  for _ in [0:10] do
    prompt := prompt.push sep
    prompt := prompt.push needleKey
    prompt := prompt.push needleValue
  prompt := prompt.push sep
  let mut rng := seed
  for _ in [0:haystackSize] do
    rng := (rng * 1103515245 + 12345) % (2^31)
    prompt := prompt.push (100 + rng % 900)
  prompt := prompt.push sep
  prompt := prompt.push queryMark
  prompt := prompt.push needleKey
  prompt

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 + Cosine Surprise Smart KV-Cache              ║"
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

  IO.println s!"[Model] Loading {ggufPath}..."
  let model ← Gemma4Model.fromGGUF device ggufPath
  IO.println s!"[Model] Gemma 4: {model.config.hiddenSize}d, {model.config.numHiddenLayers}L, vocab={model.config.vocabSize}"
  IO.println ""

  let needleKey := min 77777 (model.config.vocabSize - 1)
  let needleValue := min 42424 (model.config.vocabSize - 1)

  let smartConfig : SmartKVConfig := {
    windowSize := 256
    tau := 0.05   -- cosine distance threshold
  }

  IO.println s!"Needle: Key {needleKey} → Value {needleValue}"
  IO.println s!"Window: {smartConfig.windowSize}, Surprise tau: {smartConfig.tau}"
  IO.println ""

  let haystackSizes : Array Nat := #[100, 300, 500, 1000]

  IO.println "┌───────────┬────────────────┬────────────────┬────────────────┬────────────────┐"
  IO.println "│ Haystack  │ Prompt Length   │ Dumb Window    │ Smart KV       │ Winner         │"
  IO.println "├───────────┼────────────────┼────────────────┼────────────────┼────────────────┤"

  let mut dumbTotal : Nat := 0
  let mut smartTotal : Nat := 0

  for hsSize in haystackSizes do
    let prompt := buildPrompt hsSize model.config.vocabSize needleKey needleValue

    IO.println s!"[Running haystack={hsSize}, prompt={prompt.size} tokens]"

    -- Dumb window: evicts everything older than windowSize
    let dumbTokens ← generateWithDumbWindow device model prompt 1 smartConfig.windowSize .Greedy
    let dumbGen := dumbTokens.getD prompt.size 0
    let dumbOk := dumbGen == needleValue

    -- Smart KV: protects surprise tokens from eviction
    let smartTokens ← generateWithSmartKV device model prompt 1 smartConfig .Greedy
      (verbose := hsSize == haystackSizes[0]!)
    let smartGen := smartTokens.getD prompt.size 0
    let smartOk := smartGen == needleValue

    if dumbOk then dumbTotal := dumbTotal + 1
    if smartOk then smartTotal := smartTotal + 1

    let dumbStr := if dumbOk then s!"✓ ({dumbGen})" else s!"✗ ({dumbGen})"
    let smartStr := if smartOk then s!"✓ ({smartGen})" else s!"✗ ({smartGen})"
    let winner := if smartOk && !dumbOk then "Smart wins! 🎯"
      else if dumbOk && smartOk then "Both ✓"
      else if !dumbOk && !smartOk then "Both ✗"
      else "Dumb wins"

    IO.println s!"│ {hsSize}       │ {prompt.size}            │ {dumbStr}     │ {smartStr}     │ {winner} │"

  IO.println "└───────────┴────────────────┴────────────────┴────────────────┴────────────────┘"
  IO.println ""
  IO.println s!"Dumb window: {dumbTotal}/{haystackSizes.size} correct"
  IO.println s!"Smart KV:    {smartTotal}/{haystackSizes.size} correct"
  IO.println ""

  if smartTotal > dumbTotal then
    IO.println "╔══════════════════════════════════════════════════════════╗"
    IO.println "║  🎯 Smart KV-Cache outperformed dumb window on Gemma4! ║"
    IO.println "╚══════════════════════════════════════════════════════════╝"
  else if smartTotal == dumbTotal then
    IO.println "Both performed equally. Try tuning tau."
  else
    IO.println "Dumb window outperformed Smart KV (unexpected)."
