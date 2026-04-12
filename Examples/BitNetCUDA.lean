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

  -- Debug: compare logits after first token
  IO.println "[Debug] Forward single token for logit comparison..."
  let cacheState ← createKVCacheState ctx model
  forwardSingleToken ctx model promptTokens[0]! 0 cacheState
  let logitsBytes ← GPUBackend.readBuffer ctx cacheState.logitsBuf (min 40 (model.config.vocabSize * 4)).toUSize
  IO.println s!"[Debug] First 10 logits after token {promptTokens[0]!}:"
  for i in List.range 10 do
    let off := i * 4
    if off + 4 <= logitsBytes.size then
      let b0 : UInt32 := logitsBytes.get! off |>.toUInt32
      let b1 : UInt32 := logitsBytes.get! (off+1) |>.toUInt32
      let b2 : UInt32 := logitsBytes.get! (off+2) |>.toUInt32
      let b3 : UInt32 := logitsBytes.get! (off+3) |>.toUInt32
      let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
      let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
      let v := if e == 0 then 0.0 else
        let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
        if s == 1 then -fv else fv
      IO.println s!"  logit[{i}] = {v}"
  -- argmax
  let argResult ← gpuArgmax ctx cacheState.logitsBuf cacheState.argmaxBuf model.config.vocabSize
  IO.println s!"[Debug] argmax = {argResult}"

  -- Generate
  IO.println "[Generate] Starting CUDA inference..."
  let allTokens ← generate ctx model promptTokens maxTokens .Greedy (some 2) (showStats := true)

  -- Decode output
  let genTokens := allTokens.extract promptTokens.size allTokens.size
  let output := Hesper.Tokenizer.SentencePiece.decode tokenizer genTokens
  IO.println ""
  IO.println s!"Output: {output}"
