import Hesper
import Hesper.WebGPU.Device
import Hesper.Models.BitNet
import Hesper.Inference.Sampling
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Reader
import Hesper.Logging

/-!
# Complete BitNet Text Generation

Supports two modes:

## Single-shot mode
```bash
lake exe bitnet-complete data/gguf/ggml-model-i2_s.gguf "Hello world" 50
```

## Interactive mode
```bash
lake exe bitnet-complete data/gguf/ggml-model-i2_s.gguf --interactive
lake exe bitnet-complete data/gguf/ggml-model-i2_s.gguf -i
```
-/

namespace Hesper.Examples.BitNetComplete

open Hesper.Models.BitNet
open Hesper.Inference.Sampling
open Hesper.WebGPU
open Hesper.Tokenizer.SentencePiece
open Hesper.GGUF
open Hesper.Logging (setVerbose)

/-- Initialize WebGPU device -/
def initializeDevice : IO Device := do
  let inst ← Hesper.init
  Hesper.WebGPU.getDevice inst

/-- Load model and tokenizer from GGUF -/
def loadModel (ggufPath : String) : IO (BitNetModel × Tokenizer × Device × Option Nat) := do
  IO.println "[1/4] Loading GGUF model..."
  let gguf ← loadGGUF ggufPath
  IO.println s!"  Loaded {gguf.tensors.size} tensors"

  IO.println "[2/4] Initializing tokenizer..."
  let tokenizer ← fromGGUF gguf true false
  IO.println s!"  Vocabulary: {tokenizer.vocab.vocabSize} tokens"

  IO.println "[3/4] Initializing WebGPU device..."
  let device ← initializeDevice
  IO.println "  Device ready"

  IO.println "[4/4] Loading model to GPU..."
  let model ← fromGGUFObject device gguf none
  IO.println s!"  {model.config.numLayers} layers, {model.config.dim} dim"

  return (model, tokenizer, device, tokenizer.vocab.eosToken)

/-- Run single-shot generation -/
def runGeneration (args : List String) : IO Unit := do
  if args.length < 2 then
    IO.println "Usage: bitnet-complete <gguf_model> <prompt> [max_tokens] [--stats] [--verbose]"
    IO.println "       bitnet-complete <gguf_model> --interactive"
    IO.println "       bitnet-complete <gguf_model> -i"
    return

  let ggufPath := args[0]!
  let promptText := args[1]!
  let showStats := args.any (· == "--stats")
  let verbose := args.any (· == "--verbose")
  -- Filter out flags before parsing max_tokens
  let positionalArgs := args.filter (fun a => !a.startsWith "--")
  let maxTokens := if positionalArgs.length >= 3 then positionalArgs[2]!.toNat! else 20

  -- Disable verbose by default for clean output
  if !verbose then setVerbose false

  IO.println "═══════════════════════════════════════════════"
  IO.println "  BitNet Text Generation"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Model: {ggufPath}"
  IO.println s!"Prompt: \"{promptText}\""
  IO.println s!"Max tokens: {maxTokens}"
  IO.println ""

  let (model, tokenizer, device, eosToken) ← loadModel ggufPath

  let promptTokens := encode tokenizer promptText
  IO.println s!"Prompt tokens ({promptTokens.size}): {promptTokens}"
  IO.println ""

  let outputTokens ← generate device model promptTokens maxTokens .Greedy eosToken showStats

  let outputText := decode tokenizer outputTokens
  IO.println ""
  IO.println "─────────────────────────────────────────"
  IO.println outputText
  IO.println "─────────────────────────────────────────"

/-- Run interactive REPL -/
def runInteractive (ggufPath : String) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  BitNet Interactive Mode"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Model: {ggufPath}"
  IO.println ""

  let (model, tokenizer, device, eosToken) ← loadModel ggufPath

  -- Disable verbose logging for clean interactive output
  setVerbose false

  IO.println ""
  IO.println "Ready! Type your prompt and press Enter."
  IO.println "Commands: /quit to exit, /tokens <text> to show tokenization"
  IO.println "         /maxlen <n> to set max generation length (default: 50)"
  IO.println "         /verbose to toggle debug logging"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let mut maxTokens : Nat := 50
  let mut verbose := false
  let stdin ← IO.getStdin

  repeat do
    IO.print "> "
    let (stdout : IO.FS.Stream) ← IO.getStdout
    stdout.flush
    let line ← stdin.getLine
    let input := line.trim

    if input.isEmpty then
      continue

    if input == "/quit" || input == "/exit" || input == "/q" then
      IO.println "Goodbye!"
      break

    if input.startsWith "/tokens " then
      let text := input.drop 8
      let tokens := encode tokenizer text
      IO.println s!"Tokens ({tokens.size}): {tokens}"
      for i in [:tokens.size] do
        let tid := tokens[i]!
        if tid < tokenizer.vocab.tokens.size then
          let info := tokenizer.vocab.tokens[tid]!
          IO.println s!"  [{i}] {tid} = \"{info.piece}\""
      IO.println ""
      continue

    if input.startsWith "/maxlen " then
      let nStr := input.drop 8 |>.trim
      maxTokens := nStr.toNat!
      IO.println s!"Max generation length set to {maxTokens}"
      IO.println ""
      continue

    if input == "/verbose" then
      verbose := !verbose
      setVerbose verbose
      IO.println s!"Verbose logging: {if verbose then "ON" else "OFF"}"
      IO.println ""
      continue

    -- Encode prompt and generate
    let promptTokens := encode tokenizer input
    IO.println s!"[{promptTokens.size} tokens] Generating..."

    let outputTokens ← generate device model promptTokens maxTokens .Greedy eosToken

    let newTokenCount := outputTokens.size - promptTokens.size
    let outputText := decode tokenizer outputTokens
    IO.println ""
    IO.println outputText
    IO.println ""
    IO.println s!"({newTokenCount} tokens generated)"
    IO.println ""

end Hesper.Examples.BitNetComplete

open Hesper.Examples.BitNetComplete

def main (args : List String) : IO Unit := do
  if args.length < 2 then
    IO.println "Usage: bitnet-complete <gguf_model> <prompt> [max_tokens] [--stats] [--verbose]"
    IO.println "       bitnet-complete <gguf_model> --interactive"
    IO.println "       bitnet-complete <gguf_model> -i"
    return

  let arg1 := args[1]!
  if arg1 == "--interactive" || arg1 == "-i" then
    runInteractive args[0]!
  else
    runGeneration args
