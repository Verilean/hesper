import Hesper
import Hesper.Models.BitNet
import Hesper.TTT.SmartKVCache
import Hesper.WebGPU.Device

/-!
# Smart KV-Cache vs Dumb Sliding Window: Needle Test

Tests whether the Surprise-Gated Smart KV-Cache can recall a needle
from a 1000-token haystack when the total KV cache is limited to
288 slots (32 sinks + 256 window).

The dumb baseline uses a strict 288-slot sliding window, so the
needle (injected at the start) is physically overwritten after 288
tokens and must be forgotten.

The smart version detects the needle via MSE loss spikes and routes
its KV vectors to permanent sink slots, preserving them forever.
-/

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT.SmartKV

/-- Build a needle prompt with haystack -/
def buildPrompt (haystackSize : Nat) (vocabSize : Nat)
    (needleKey needleValue : Nat) (seed : Nat := 42)
    : Array Nat := Id.run do
  let sep := min 2 (vocabSize - 1)
  let queryMark := min 3 (vocabSize - 1)
  let mut prompt : Array Nat := #[]

  -- Background: 5 common tokens
  for i in [0:5] do
    prompt := prompt.push (10 + i)

  -- Inject needle 10 times
  for _ in [0:10] do
    prompt := prompt.push sep
    prompt := prompt.push needleKey
    prompt := prompt.push needleValue

  prompt := prompt.push sep

  -- Haystack
  let mut rng := seed
  for _ in [0:haystackSize] do
    rng := (rng * 1103515245 + 12345) % (2^31)
    prompt := prompt.push (100 + rng % 900)

  -- Query
  prompt := prompt.push sep
  prompt := prompt.push queryMark
  prompt := prompt.push needleKey

  prompt

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Smart KV-Cache vs Dumb Window: Needle Test            ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  let ggufPath ← match args with
    | path :: _ => pure path
    | [] => do
      IO.println "Usage: smart-kv-needle <model.gguf>"
      IO.Process.exit 1
      pure ""

  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "[GPU] Device initialized"

  IO.println s!"[Model] Loading from {ggufPath}..."
  let model ← fromGGUF device ggufPath
  IO.println s!"[Model] {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println ""

  let needleKey := min 77777 (model.config.vocabSize - 1)
  let needleValue := min 42424 (model.config.vocabSize - 1)

  let smartConfig : SmartKVConfig := {
    maxSinks := 64
    windowSize := 256
    tau := 0.003     -- lower threshold to catch needle tokens
  }

  IO.println s!"Needle: Key {needleKey} → Value {needleValue}"
  IO.println s!"KV Cache: {smartConfig.maxSinks} sinks + {smartConfig.windowSize} window = {smartConfig.maxSinks + smartConfig.windowSize} total"
  IO.println ""

  let haystackSizes : Array Nat := #[100, 300, 500, 1000]

  IO.println "┌───────────┬────────────────┬────────────────┬────────────────┬────────────────┐"
  IO.println "│ Haystack  │ Prompt Length   │ Dumb Window    │ Smart KV       │ Winner         │"
  IO.println "├───────────┼────────────────┼────────────────┼────────────────┼────────────────┤"

  let mut dumbTotal : Nat := 0
  let mut smartTotal : Nat := 0

  for hsSize in haystackSizes do
    let prompt := buildPrompt hsSize model.config.vocabSize needleKey needleValue
    let totalWindow := smartConfig.maxSinks + smartConfig.windowSize

    -- Run 1: Dumb sliding window
    let dumbTokens ← generateWithDumbWindow device model prompt 1 totalWindow .Greedy
    let dumbGen := dumbTokens.getD prompt.size 0
    let dumbOk := dumbGen == needleValue

    -- Run 2: Smart KV-Cache
    let smartTokens ← generateWithSmartKV device model prompt 1 smartConfig .Greedy
      (verbose := hsSize == haystackSizes[0]!)  -- verbose only for first
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
    IO.println "║  🎯 Smart KV-Cache outperformed dumb window!           ║"
    IO.println "╚══════════════════════════════════════════════════════════╝"
  else if smartTotal == dumbTotal then
    IO.println "Both performed equally."
  else
    IO.println "Dumb window outperformed Smart KV (unexpected)."
