import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.WGSL.Execute
import Hesper.Quantization.Q6_K
import Hesper.Quantization.Q4_K_M

/-!
# Gemma 4 Embedding Lookup Test

Standalone test of Q6_K embedding lookup without the full model.
Loads only token_embd and verifies dequant.
-/

open Hesper.WebGPU
open Hesper.WGSL

def main (args : List String) : IO Unit := do
  let modelPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let tokenId := (args.getD 1 "2").toNat!

  IO.println s!"[EmbdTest] Loading GGUF: {modelPath}"
  let ggufData ← IO.FS.readBinFile modelPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw $ IO.userError s!"Parse error: {e}"

  let embTensor ← match Hesper.GGUF.Loader.findTensor gguf "token_embd.weight" with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  IO.println s!"[EmbdTest] token_embd: shape={embTensor.shape}, type={toString embTensor.ggmlType}, size={embTensor.size}"

  let (_, embData) ← match Hesper.GGUF.Loader.getTensorData gguf "token_embd.weight" with
    | .ok r => pure r
    | .error e => throw $ IO.userError e
  IO.println s!"[EmbdTest] Data loaded: {embData.size} bytes"

  -- Get vocab/dim from tensor shape
  let dim := embTensor.shape[0]!  -- 2560
  let vocabSize := embTensor.shape[1]!  -- 262144
  IO.println s!"[EmbdTest] vocab={vocabSize}, dim={dim}"

  -- Initialize WebGPU
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  -- Upload embedding table
  let embBuf ← createBuffer device {
    size := embData.size.toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  writeBuffer device embBuf 0 embData

  -- Create token ID buffer
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  let tokenBuf ← createBuffer device {
    size := 4
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  writeBuffer device tokenBuf 0 tokenBytes

  -- Create output buffer
  let outBuf ← createBuffer device {
    size := (dim * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Run embedding lookup — branch on the tensor's actual quant format
  let namedBufs : List (String × Buffer) :=
    [("token_ids", tokenBuf), ("embedding_table", embBuf), ("output", outBuf)]
  match embTensor.ggmlType with
  | .Q4_K =>
    IO.println s!"[EmbdTest] Running Q4_K embedding lookup for token {tokenId}..."
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Quantization.Q4_K_M.q4kEmbeddingLookupKernel vocabSize dim)
      namedBufs
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D dim)
  | _ =>
    IO.println s!"[EmbdTest] Running Q6_K embedding lookup for token {tokenId}..."
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel vocabSize dim)
      namedBufs
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D dim)

  -- Download result
  let result ← Hesper.WebGPU.BufferOps.downloadFloatArray device outBuf dim
  IO.println s!"[EmbdTest] Downloaded {result.size} f32 values"

  -- Print first 16
  IO.println "[EmbdTest] First 16 values:"
  for i in [0:16] do
    IO.println s!"  [{i}] = {result[i]!}"

  -- Stats
  let mut sumAbs := 0.0
  let mut maxAbs := 0.0
  let mut nonzero := 0
  for v in result do
    let a := v.abs
    sumAbs := sumAbs + a
    if a > maxAbs then maxAbs := a
    if v != 0.0 then nonzero := nonzero + 1
  IO.println s!"[EmbdTest] Stats: nonzero={nonzero}/{result.size}, max_abs={maxAbs}, mean_abs={sumAbs / result.size.toFloat}"
