import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Models.Gemma4.Kernels
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.RoPE
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

  -- layer-0 Q/K/V projections (Q4_K / Q6_K) on Metal
  let kvDim := cfg.numKVHeads 0 * cfg.headDim 0
  let qBuf ← mkBuf qDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wQ normBuf qBuf
  let kBuf ← mkBuf kvDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wK normBuf kBuf
  let vBuf ← mkBuf kvDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wV normBuf vBuf
  let kA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device kBuf 0 (kvDim*4).toUSize)
  unmapBuffer kBuf
  let vA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device vBuf 0 (kvDim*4).toUSize)
  unmapBuffer vBuf

  -- attention prologue: per-head qk-norm (Gemma) + RoPE on Q (Metal)
  let headDim0 := cfg.headDim 0
  let qNormBuf ← mkBuf qDim
  let qnKern := Hesper.Models.Gemma4.perHeadRMSNormKernel cfg.numAttentionHeads headDim0 cfg.rmsNormEps
  let qnBufs := ("input", qBuf) :: ("weight", blk.attention.qNormWeight) :: ("output", qNormBuf) :: List.nil
  let qnCfg : Hesper.ExecConfig := { numWorkgroups := (cfg.numAttentionHeads, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device qnKern qnBufs qnCfg
  let ropeBase := if cfg.isFullAttention 0 then cfg.ropeTheta else cfg.ropeThetaSWA
  let rope ← Hesper.Layers.RoPE.create { dim := qDim, maxSeqLen := 4096, base := ropeBase }
  let qRopedBuf ← mkBuf qDim
  Hesper.Layers.RoPE.forward device rope qNormBuf qRopedBuf 1 1 cfg.numAttentionHeads headDim0 0
  let qRoped ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device qRopedBuf 0 (qDim*4).toUSize)
  unmapBuffer qRopedBuf

  -- attention output projection: for seqLen=1 the softmax over the single key
  -- is 1, so attn_head = V[kv_head(h)] (GQA broadcast); concat heads → wO.
  -- (scores+softmax for seqLen>1 are validated separately in the GQA core test.)
  let grp := cfg.numAttentionHeads / cfg.numKVHeads 0
  let mut attnVec := Array.replicate qDim 0.0
  for h in [0:cfg.numAttentionHeads] do
    let kvh := h / grp
    for d in [0:headDim0] do
      attnVec := attnVec.set! (h*headDim0 + d) (vA[kvh*headDim0 + d]!)
  let attnVecBuf ← mkBuf qDim
  writeBuffer device attnVecBuf 0 (← Hesper.Basic.floatArrayToBytes attnVec)
  let woOutBuf ← mkBuf dim
  Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wO attnVecBuf woOutBuf
  let woOut ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device woOutBuf 0 (dim*4).toUSize)
  unmapBuffer woOutBuf

  -- read back
  let normed ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device normBuf 0 (dim*4).toUSize)
  unmapBuffer normBuf
  let q ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device qBuf 0 (qDim*4).toUSize)
  unmapBuffer qBuf

  -- dense FFN shared-expert matmuls (Q4_K) on Metal: ffn_norm → gate/up → down
  let ffnDim := cfg.intermediateSize
  let ffnNormBuf ← mkBuf dim
  Hesper.Layers.RMSNorm.forward device blk.ffnNorm inBuf ffnNormBuf
  let gateBuf ← mkBuf ffnDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.gate ffnNormBuf gateBuf
  let upBuf ← mkBuf ffnDim
  Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.up ffnNormBuf upBuf
  let gateA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device gateBuf 0 (ffnDim*4).toUSize)
  unmapBuffer gateBuf
  let upA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device upBuf 0 (ffnDim*4).toUSize)
  unmapBuffer upBuf
  -- dense shared-expert GeGLU: gelu(gate)*up  (geglu kernel validated separately;
  -- computed here on host for bring-up, down projection runs native Q8_0 on Metal)
  let mut geglu := Array.replicate ffnDim 0.0
  for i in [0:ffnDim] do
    let g := gateA[i]!
    let gl := 0.5*g*(1.0 + Float.tanh (0.7978845608 * (g + 0.044715*g*g*g)))
    geglu := geglu.set! i (gl * upA[i]!)
  let gegluBuf ← mkBuf ffnDim
  writeBuffer device gegluBuf 0 (← Hesper.Basic.floatArrayToBytes geglu)
  let downOutBuf ← mkBuf dim
  Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.down gegluBuf downOutBuf
  let downA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device downOutBuf 0 (dim*4).toUSize)
  unmapBuffer downOutBuf

  let normFinite := normed.all Float.isFinite
  let qFinite := q.all Float.isFinite
  let ffnFinite := gateA.all Float.isFinite && downA.all Float.isFinite
  IO.println s!"  dim={dim} qDim={qDim} ffnDim={ffnDim} (heads={cfg.numAttentionHeads} headDim0={cfg.headDim 0})"
  IO.println s!"  attn_norm RMSNorm: finite={normFinite}  [0..4]={(normed.extract 0 4).toList}"
  let kvFinite := kA.all Float.isFinite && vA.all Float.isFinite
  let qRopedFinite := qRoped.all Float.isFinite
  IO.println s!"  QKV: wQ(Q4_K) finite={qFinite} size={q.size}; wK finite={kA.all Float.isFinite} size={kA.size}; wV(Q6_K) finite={vA.all Float.isFinite} size={vA.size}"
  IO.println s!"  qk-norm + RoPE(Q): finite={qRopedFinite} size={qRoped.size} base={ropeBase}  [0..4]={(qRoped.extract 0 4).toList}"
  let woFinite := woOut.all Float.isFinite
  IO.println s!"  attn output wO(Q4_K): finite={woFinite} size={woOut.size}  [0..4]={(woOut.extract 0 4).toList}"
  IO.println s!"  dense FFN: gate/up(Q4_K)→GeGLU→down(Q8_0) finite={downA.all Float.isFinite} size={downA.size}  [0..4]={(downA.extract 0 4).toList}"
  if normFinite && qFinite && kvFinite && qRopedFinite && woFinite && ffnFinite && q.size == qDim && kA.size == kvDim && woOut.size == dim && gateA.size == ffnDim && downA.size == dim then
    IO.println "✓ native Metal: full attention path (norm→QKV→qk-norm→RoPE→V→wO) + FFN projections run on the real 26B model"
  else
    IO.println "✗ probe failed"
    throw (IO.userError "forward probe failed")
