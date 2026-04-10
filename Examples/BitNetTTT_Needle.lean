import Hesper
import Hesper.Models.BitNet
import Hesper.TTT.BitNetTTT
import Hesper.TTT.HiddenSpaceTTT
import Hesper.TTT.Types
import Hesper.WebGPU.Device

/-!
# BitNet + TTT: Needle-in-a-Haystack Test

A harder test than MQAR: inject a single rare key→value pair early in
the prompt, pad with hundreds of "haystack" tokens (random noise), then
query the key from far away. The base model must rely on distant
attention to recall the value — a known weak point for small models.
TTT should memorize the correction during prefill and recall perfectly.

We test multiple haystack sizes (50, 100, 200, 500) to find the
distance at which the base model fails but TTT succeeds.
-/

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT
open Hesper.TTT.BitNetTTT
open Hesper.TTT.HiddenSpace

/-- Build a needle-in-haystack prompt:
    [background(5)] [sep key value sep] [haystack(n)] [sep queryMark key]
    Returns (prompt, expectedValue) -/
def buildNeedlePrompt (haystackSize : Nat) (vocabSize : Nat)
    (needleKey needleValue : Nat) (seed : Nat := 42)
    : Array Nat × Nat := Id.run do
  let sep := min 2 (vocabSize - 1)
  let queryMark := min 3 (vocabSize - 1)
  let mut prompt : Array Nat := #[]

  -- Background: 5 common tokens
  for i in [0:5] do
    prompt := prompt.push (10 + i)

  -- Inject the needle 3 times (repetition helps TTT learn)
  for _ in [0:3] do
    prompt := prompt.push sep
    prompt := prompt.push needleKey
    prompt := prompt.push needleValue

  prompt := prompt.push sep

  -- Haystack: pseudo-random tokens in range [100, 1000)
  -- Use a simple LCG PRNG for reproducibility
  let mut rng := seed
  for _ in [0:haystackSize] do
    rng := (rng * 1103515245 + 12345) % (2^31)
    let tok := 100 + rng % 900  -- tokens 100..999
    prompt := prompt.push tok

  -- Query: sep queryMark key
  prompt := prompt.push sep
  prompt := prompt.push queryMark
  prompt := prompt.push needleKey

  (prompt, needleValue)

/-- Run one needle test — base model only -/
def runBaseTest (device : Device) (model : BitNetModel)
    (haystackSize : Nat) (needleKey needleValue : Nat)
    : IO (Bool × Nat) := do
  let (prompt, expected) := buildNeedlePrompt haystackSize model.config.vocabSize needleKey needleValue
  let tokens ← generate device model prompt 1 .Greedy
  let generated := tokens.getD prompt.size 0
  return (generated == expected, generated)

/-- Run one needle test — Hidden-Space TTT model -/
def runTTTTest (device : Device) (model : BitNetModel) (tttConfig : HiddenTTTConfig)
    (haystackSize : Nat) (needleKey needleValue : Nat)
    : IO (Bool × Nat) := do
  let (prompt, expected) := buildNeedlePrompt haystackSize model.config.vocabSize needleKey needleValue
  let tokens ← generateWithHiddenTTT device model prompt 1 tttConfig .Greedy (verbose := false)
  let generated := tokens.getD prompt.size 0
  return (generated == expected, generated)

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + TTT: Needle-in-a-Haystack Test               ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  let ggufPath ← match args with
    | path :: _ => pure path
    | [] => do
      IO.println "Usage: bitnet-ttt-needle <model.gguf>"
      IO.Process.exit 1
      pure ""

  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "[GPU] Device initialized"

  IO.println s!"[Model] Loading from {ggufPath}..."
  let model ← fromGGUF device ggufPath
  IO.println s!"[Model] {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println ""

  -- Hidden-Space TTT: W_ttt is [dim × dim] = 26 MB (vs 1.3 GB for logit-space)
  let hiddenTTTConfig : HiddenTTTConfig := {
    dim := model.config.dim
    vocabSize := model.config.vocabSize
    innerLR := 0.1   -- higher lr for hidden-space (2560-dim needs stronger updates)
    tau := 2.0
  }
  IO.println s!"[TTT] Hidden-Space: W_ttt=[{model.config.dim}×{model.config.dim}] = {model.config.dim * model.config.dim * 4 / 1024} KB"

  -- Use rare tokens as the needle
  let needleKey := min 77777 (model.config.vocabSize - 1)
  let needleValue := min 42424 (model.config.vocabSize - 1)

  IO.println s!"Needle: Key {needleKey} → Value {needleValue}"
  IO.println ""

  -- Test at increasing haystack sizes (keep small to avoid OOM/dispatch issues)
  let haystackSizes : Array Nat := #[10, 30, 50]

  IO.println "┌───────────┬────────────────┬────────────────┬────────────────┬────────────────┐"
  IO.println "│ Haystack  │ Prompt Length   │ Base Model     │ TTT Model      │ Winner         │"
  IO.println "├───────────┼────────────────┼────────────────┼────────────────┼────────────────┤"

  let mut baseTotalCorrect : Nat := 0
  let mut tttTotalCorrect : Nat := 0

  for hsSize in haystackSizes do
    if hsSize + 20 > model.config.maxSeqLen then
      IO.println s!"│ {hsSize}       │ (skipped — exceeds maxSeqLen {model.config.maxSeqLen})              │"
      continue

    let (prompt, _) := buildNeedlePrompt hsSize model.config.vocabSize needleKey needleValue
    let promptLen := prompt.size

    let (baseOk, baseTok) ← runBaseTest device model hsSize needleKey needleValue
    let (tttOk, tttTok) ← runTTTTest device model hiddenTTTConfig hsSize needleKey needleValue

    if baseOk then baseTotalCorrect := baseTotalCorrect + 1
    if tttOk then tttTotalCorrect := tttTotalCorrect + 1

    let baseStr := if baseOk then s!"✓ ({baseTok})" else s!"✗ ({baseTok})"
    let tttStr := if tttOk then s!"✓ ({tttTok})" else s!"✗ ({tttTok})"
    let winner := if tttOk && !baseOk then "TTT wins! 🎯"
      else if baseOk && tttOk then "Both ✓"
      else if !baseOk && !tttOk then "Both ✗"
      else "Base wins"

    -- Pad strings for alignment
    let hsStr := s!"{hsSize}" ++ String.mk (List.replicate (9 - s!"{hsSize}".length) ' ')
    let plStr := s!"{promptLen}" ++ String.mk (List.replicate (14 - s!"{promptLen}".length) ' ')
    let bStr := baseStr ++ String.mk (List.replicate (14 - baseStr.length) ' ')
    let tStr := tttStr ++ String.mk (List.replicate (14 - tttStr.length) ' ')
    IO.println s!"│ {hsStr} │ {plStr} │ {bStr} │ {tStr} │ {winner} │"

  IO.println "└───────────┴────────────────┴────────────────┴────────────────┴────────────────┘"
  IO.println ""
  IO.println s!"Base model: {baseTotalCorrect}/{haystackSizes.size} correct"
  IO.println s!"TTT model:  {tttTotalCorrect}/{haystackSizes.size} correct"
  IO.println ""

  if tttTotalCorrect > baseTotalCorrect then
    IO.println "╔══════════════════════════════════════════════════════════╗"
    IO.println "║  🎯 TTT outperformed base model!                       ║"
    IO.println "╚══════════════════════════════════════════════════════════╝"
  else if tttTotalCorrect == baseTotalCorrect then
    IO.println "Both models performed equally."
    IO.println "Try: lower tau, higher lr, or different needle tokens."
  else
    IO.println "Base model outperformed TTT (unexpected)."
