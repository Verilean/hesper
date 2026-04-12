import Hesper.Backend.CUDA
import Hesper.Models.BitNet
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Reader

/-!
# BitNet CUDA Inference

Runs BitNet text generation entirely on CUDA PTX JIT backend.
Same model, same code — different backend.

Usage:
  lake exe bitnet-cuda data/gguf/ggml-model-i2_s.gguf "Hello" 50
-/

open Hesper
open Hesper.CUDA
open Hesper.Models.BitNet

def main (args : List String) : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  BitNet CUDA PTX Inference                   ║"
  IO.println "╚══════════════════════════════════════════════╝"

  let (ggufPath, prompt, maxTokens) ← match args with
    | path :: prompt :: n :: _ => pure (path, prompt, n.toNat!)
    | path :: prompt :: _ => pure (path, prompt, 50)
    | path :: _ => pure (path, "The meaning of life is", 50)
    | [] => do
      IO.println "Usage: bitnet-cuda <model.gguf> [prompt] [max_tokens]"
      IO.Process.exit 1
      pure ("", "", 0)

  IO.println s!"Model: {ggufPath}"
  IO.println s!"Prompt: \"{prompt}\""
  IO.println s!"Max tokens: {maxTokens}"
  IO.println ""

  -- Initialize CUDA
  let ctx ← CUDAContext.init

  -- Load model on CUDA
  IO.println "[Load] Loading BitNet model on CUDA..."
  let model ← fromGGUF ctx ggufPath

  -- Load tokenizer
  IO.println "[Load] Loading tokenizer..."
  let ggufData ← IO.FS.readBinFile ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf (some true) (some false)

  -- Tokenize prompt
  let promptTokens := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  IO.println s!"[Tokenize] Prompt: {promptTokens.size} tokens"

  -- Generate 1 token to check logits
  IO.println "[Generate] 1 token test..."
  let testTokens ← generate ctx model promptTokens 1 .Greedy (some 2)
  -- Read logits after the 1-token generate
  IO.println s!"First generated token: {testTokens.getD promptTokens.size 0}"

  -- Full generate
  IO.println "[Generate] Starting CUDA inference..."
  let allTokens ← generate ctx model promptTokens maxTokens .Greedy (some 2) (showStats := true)

  -- Decode output
  let genTokens := allTokens.extract promptTokens.size allTokens.size
  let output := Hesper.Tokenizer.SentencePiece.decode tokenizer genTokens
  IO.println ""
  IO.println s!"Output: {output}"
