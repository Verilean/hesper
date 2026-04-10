import Hesper
import Hesper.Models.BitNet
import Hesper.TTT.BitNetTTT
import Hesper.TTT.Types
import Hesper.WebGPU.Device

/-!
# BitNet + TTT: MQAR Proof-of-Concept

Demonstrates the Surprise-Gated Residual TTT architecture on a
Multi-Query Associative Recall (MQAR) task.

The test injects rare key-value pairs into a prompt that a standard
sliding-window model would struggle with. The TTT module should:
1. Open the surprise gate when encountering unexpected token patterns
2. Memorize the residual corrections into the TTT weights during prefill
3. Use the learned corrections to improve recall during decode

Usage:
  lake exe bitnet-ttt-mqar <model.gguf>
-/

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT
open Hesper.TTT.BitNetTTT

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + TTT: MQAR Proof-of-Concept                   ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Parse arguments
  let ggufPath ← match args with
    | path :: _ => pure path
    | [] => do
      IO.println "Usage: bitnet-ttt-mqar <model.gguf>"
      IO.println ""
      IO.println "Example:"
      IO.println "  lake exe bitnet-ttt-mqar data/bitnet-2b.gguf"
      IO.Process.exit 1
      pure ""

  -- Initialize GPU
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "[GPU] Device initialized"
  IO.println ""

  -- Load model
  IO.println s!"[Model] Loading from {ggufPath}..."
  let model ← fromGGUF device ggufPath
  IO.println s!"[Model] Loaded: {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println ""

  -- TTT configuration
  -- For LM-head TTT, hiddenDim = model.config.dim, vocabSize = model.config.vocabSize
  let tttConfig : TTTConfig := {
    hiddenDim := model.config.dim
    vocabSize := model.config.vocabSize
    innerLR := 0.01
    tau := 2.0  -- tune based on model's typical loss range
  }
  IO.println s!"[TTT] Config: dim={tttConfig.hiddenDim}, vocab={tttConfig.vocabSize}, lr={tttConfig.innerLR}, tau={tttConfig.tau}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- MQAR Prompt: key-value pairs the model must memorize
  -- ═══════════════════════════════════════════
  -- We use raw token IDs for this demo. In a real scenario, you'd
  -- tokenize text like "Apple=9912, Banana=3341" etc.
  -- For now, we construct a synthetic prompt with repeated patterns
  -- that include rare tokens the base model hasn't seen in context.

  -- Simple synthetic MQAR pattern:
  -- Phase 1 (background): common tokens the model predicts well
  -- Phase 2 (injection): rare key→value pairs repeated R times
  -- Phase 3 (query): re-present keys, expect values

  -- Using arbitrary token IDs within vocab range:
  let vocabSize := model.config.vocabSize
  let sep := min 2 (vocabSize - 1)         -- separator token
  let queryMark := min 3 (vocabSize - 1)   -- query marker

  -- 5 rare key-value pairs (using high token IDs unlikely to be common)
  let pairs : Array (Nat × Nat) := #[
    (min 50000 (vocabSize - 1), min 9912 (vocabSize - 1)),    -- key0 → val0
    (min 50001 (vocabSize - 1), min 3341 (vocabSize - 1)),    -- key1 → val1
    (min 50002 (vocabSize - 1), min 8810 (vocabSize - 1)),    -- key2 → val2
    (min 50003 (vocabSize - 1), min 1102 (vocabSize - 1)),    -- key3 → val3
    (min 50004 (vocabSize - 1), min 5543 (vocabSize - 1))     -- key4 → val4
  ]

  IO.println "═══ MQAR Task ═══"
  IO.println "Key-Value pairs to memorize:"
  for (k, v) in pairs do
    IO.println s!"  Key {k} → Value {v}"
  IO.println ""

  -- Build prompt: [background..., sep, (key, value, sep) × R, queryMark, key0]
  let mut prompt : Array Nat := #[]

  -- Background phase: 10 common tokens
  for i in [0:10] do
    prompt := prompt.push (1 + i % 100)  -- common low-range tokens

  -- Injection phase: repeat each pair 3 times
  let repetitions := 3
  for _ in [0:repetitions] do
    for (k, v) in pairs do
      prompt := prompt.push sep
      prompt := prompt.push k
      prompt := prompt.push v

  -- Query phase: ask for pair 2's value (key=50002, expected=8810)
  prompt := prompt.push sep
  prompt := prompt.push queryMark
  prompt := prompt.push (pairs[2]!.1)  -- key2 = 50002

  IO.println s!"Prompt: {prompt.size} tokens"
  IO.println s!"Query: What comes after key {pairs[2]!.1}? Expected: {pairs[2]!.2}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Run 1: Base model only (no TTT)
  -- ═══════════════════════════════════════════
  IO.println "═══ Run 1: Base Model Only (no TTT) ═══"
  let baseTokens ← generate device model prompt 5 .Greedy
  let generatedBase := baseTokens.extract prompt.size baseTokens.size
  IO.println s!"  Generated: {generatedBase}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Run 2: With TTT
  -- ═══════════════════════════════════════════
  IO.println "═══ Run 2: With TTT (Surprise-Gated Residual) ═══"
  let tttTokens ← generateWithTTT device model prompt 5 tttConfig .Greedy
  let generatedTTT := tttTokens.extract prompt.size tttTokens.size
  IO.println s!"  Generated: {generatedTTT}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Compare results
  -- ═══════════════════════════════════════════
  IO.println "═══ Comparison ═══"
  IO.println s!"  Expected first token: {pairs[2]!.2}"
  IO.println s!"  Base model generated: {generatedBase.getD 0 0}"
  IO.println s!"  TTT model generated:  {generatedTTT.getD 0 0}"

  let baseCorrect := generatedBase.getD 0 0 == pairs[2]!.2
  let tttCorrect := generatedTTT.getD 0 0 == pairs[2]!.2

  IO.println ""
  if tttCorrect && !baseCorrect then
    IO.println "╔══════════════════════════════════════════════════════════╗"
    IO.println "║  ✓ TTT SUCCEEDED where base model FAILED!              ║"
    IO.println "╚══════════════════════════════════════════════════════════╝"
  else if tttCorrect && baseCorrect then
    IO.println "Both models got it right (base model may already know this pattern)."
    IO.println "Try increasing tau or using rarer token IDs."
  else if !tttCorrect && !baseCorrect then
    IO.println "Neither model got it right."
    IO.println "The TTT module may need more injection repetitions, lower tau, or higher lr."
  else
    IO.println "Base model got it right but TTT didn't — unexpected!"
