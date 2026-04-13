import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser

/-!
# Gemma 4 CUDA PTX Inference

Same model, same code — CUDA backend via GPUBackend typeclass.

Usage:
  lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 10
-/

open Hesper
open Hesper.Models.Gemma4

def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let prompt := args.getD 1 "Hello"
  let maxTokens := (args.getD 2 "10").toNat!

  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║  Gemma 4 CUDA PTX Inference              ║"
  IO.println "╚══════════════════════════════════════════╝"
  IO.println s!"  Model: {ggufPath}"
  IO.println s!"  Prompt: \"{prompt}\""
  IO.println s!"  Max tokens: {maxTokens}"

  -- Initialize CUDA
  let ctx ← CUDAContext.init

  -- Load model on CUDA (fast file read via mmap)
  IO.println "[Load] Reading GGUF file..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  IO.println s!"[Load] Read {ggufData.size} bytes, loading model..."
  let model ← Gemma4Model.fromGGUFData ctx ggufData

  -- Load tokenizer (reuse already-read ggufData)
  IO.println "[Tokenize] Loading tokenizer..."
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf

  let promptTokens := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  IO.println s!"[Tokenize] Prompt: {promptTokens.size} tokens"

  -- Generate
  IO.println "[Generate] Starting CUDA inference..."
  let eosId := tokenizer.vocab.eosToken.getD 1
  let tokens ← generate ctx model promptTokens maxTokens
    (eosToken := some eosId) (extraEosTokens := #[106])

  let generated := tokens.extract promptTokens.size tokens.size
  let decoded := Hesper.Tokenizer.SentencePiece.decode tokenizer generated
  IO.println s!"[Result] Decoded: {decoded}"
