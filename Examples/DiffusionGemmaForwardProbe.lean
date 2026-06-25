import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Basic

/-!
# DiffusionGemma native forward probe (Metal)

First step of the native forward assembly: load the real 26B model and run
the first real forward ops on Metal with the loaded weights —
`RMSNorm.forward` (layer-0 attn_norm) then the Q4_K `wQ` projection — and
check the output is finite/correct shape.  Proves the native forward
pipeline (RMSNorm + Q4_K matmul) runs on Metal against the real model.

Run:  lake exe diffusiongemma-forward-probe
-/

open Hesper.WebGPU
open Hesper.Models.DiffusionGemma

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  IO.println "[forward-probe] init WebGPU (Metal) + load model..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  let dim := cfg.hiddenSize
  let some blk := model.inner.blocks[0]? | throw (IO.userError "no block 0")
  let qDim := cfg.numAttentionHeads * cfg.headDim 0

  let mkBuf (n : Nat) : IO Buffer :=
    createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

  -- synthetic input hidden vector
  let inArr := (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.01) * 0.5)
  let inBuf ← mkBuf dim
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes inArr)

  -- layer-0 attn pre-norm (RMSNorm) on Metal
  let normBuf ← mkBuf dim
  Hesper.Layers.RMSNorm.forward device blk.attnNorm inBuf normBuf

  -- layer-0 Q projection (Q4_K matmul) on Metal
  let qBuf ← mkBuf qDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wQ normBuf qBuf

  -- read back
  let normed ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device normBuf 0 (dim*4).toUSize)
  unmapBuffer normBuf
  let q ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device qBuf 0 (qDim*4).toUSize)
  unmapBuffer qBuf

  let normFinite := normed.all Float.isFinite
  let qFinite := q.all Float.isFinite
  IO.println s!"  dim={dim} qDim={qDim} (heads={cfg.numAttentionHeads} headDim0={cfg.headDim 0})"
  IO.println s!"  RMSNorm out: finite={normFinite}  [0..4]={(normed.extract 0 4).toList}"
  IO.println s!"  wQ (Q4_K) out: finite={qFinite}  size={q.size}  [0..4]={(q.extract 0 4).toList}"
  if normFinite && qFinite && q.size == qDim then
    IO.println "✓ native Metal forward ops (RMSNorm + Q4_K matmul) run on the real 26B model"
  else
    IO.println "✗ probe failed"
    throw (IO.userError "forward probe failed")
