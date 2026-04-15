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

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let prompt := args.getD 1 "Hello"
  let maxTokens := (args.getD 2 "10").toNat!

  -- Enable dp4a path if HESPER_DP4A=1 is set in environment.
  match ← IO.getEnv "HESPER_DP4A" with
  | some "1" =>
    Hesper.Layers.Linear.dp4aEnabled.set true
    IO.println "[Config] dp4a Q4_K path: ENABLED"
  | _ => IO.println "[Config] dp4a Q4_K path: disabled (set HESPER_DP4A=1 to enable)"
  -- Q6_K lmHead dp4a is on by default when HESPER_DP4A=1; disable via
  -- HESPER_DP4A_Q6K=0 (for debugging only — correctness verified against
  -- nvcc reference kernel).
  match ← IO.getEnv "HESPER_DP4A_Q6K" with
  | some "0" =>
    Hesper.Layers.Linear.dp4aQ6KEnabled.set false
    IO.println "[Config] dp4a Q6_K lmHead: DISABLED (HESPER_DP4A_Q6K=0)"
  | _ =>
    let q6on ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
    let status := if q6on then "enabled (default)" else "disabled"
    IO.println s!"[Config] dp4a Q6_K lmHead: {status}"

  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║  Gemma 4 CUDA PTX Inference              ║"
  IO.println "╚══════════════════════════════════════════╝"
  IO.println s!"  Model: {ggufPath}"
  IO.println s!"  Prompt: \"{prompt}\""
  IO.println s!"  Max tokens: {maxTokens}"

  -- Initialize CUDA
  let ctx ← CUDAContext.init

  -- L2 persistence: bump the persisting cache limit to the device maximum
  -- (Ada / Ampere: typically 32 MB).  When HESPER_L2_PIN is set we'll install
  -- per-layer access windows in forwardBlock to keep the current layer's
  -- hot weights L2-resident across kernel boundaries (mirrors llama.cpp's
  -- back-to-back cuLaunchKernel pattern where L2 stays warm on its own).
  match ← IO.getEnv "HESPER_L2_LIMIT" with
  | some "0" => IO.println "[Config] L2 persist limit: default (no override)"
  | _ =>
    -- Default: ask the driver for the full device cap (clamped server-side).
    let requested : USize := 256 * 1024 * 1024  -- 256 MB; driver will clamp.
    let actual ← Hesper.CUDA.cuSetL2PersistLimit requested
    IO.println s!"[Config] L2 persist limit: {actual / 1024 / 1024} MB"

  -- L2 access-policy window: when HESPER_L2_PIN=<layer_idx>, pin that
  -- layer's FFN weight region (gate+up+down, ~47 MB for Gemma 4 E4B) in L2
  -- before generation starts.  For a single-layer test this lets us measure
  -- whether L2 residency helps at all — a sanity check before investing in
  -- per-layer rotation.  Also accepts "0" as "first layer".
  let pinLayerOpt ← IO.getEnv "HESPER_L2_PIN"
  match pinLayerOpt with
  | none => pure ()
  | some _ =>
    IO.println "[Config] HESPER_L2_PIN set — will install access window after model load"

  -- Load model on CUDA (fast file read via mmap)
  IO.println "[Load] Reading GGUF file..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  IO.println s!"[Load] Read {ggufData.size} bytes, loading model..."
  let model ← Gemma4Model.fromGGUFData ctx ggufData

  -- Install an L2 access window on the default stream if requested.  We pin
  -- the requested layer's ffn.gate weight buffer; its (ptr, size) defines
  -- the address range the driver will prefer to keep L2-resident.  This is
  -- a one-shot diagnostic — the full wire-up would rotate the window per
  -- layer in forwardBlock, but first we need to see any effect at all.
  match pinLayerOpt with
  | some s =>
    let idx := s.toNat!
    if h : idx < model.blocks.size then
      let block := model.blocks[idx]
      let rawBuf := block.ffn.gate.weightBuf
      let gateBuf : Hesper.CUDA.CUDABuffer := unsafeCast rawBuf
      Hesper.CUDA.cuSetL2AccessWindow gateBuf.ptr gateBuf.size
      IO.println s!"[Config] L2 access window: layer {idx} ffn_gate, {gateBuf.size / 1024 / 1024} MB"
    else
      IO.println s!"[Config] L2 pin: layer {idx} out of range (have {model.blocks.size} layers)"
  | none => pure ()

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
