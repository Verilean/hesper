import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Basic

/-!
# DiffusionGemma GPU-resident forward skeleton (milestone 1)

Follows BitNet.forward's Metal pattern: **preallocate buffers once**, wrap the
whole forward in `beginBatch`/`endBatch` (single command-buffer submission),
keep data **GPU-resident** (no per-op host round-trips), ping-pong buffers
across layers.  This fixes the per-op-dispatch resource accumulation that
crashed the host-round-trip probe at ~layer 3.

Structural skeleton (dense path: RMSNorm→gate→down→norm+residual per layer)
to prove the 30-layer GPU-resident loop runs in one batch without crashing.
Full attention V-reuse + GeGLU + MoE routing layer in next.

Run:  lake exe diffusiongemma-forward-gpu [path] [nLayers]
-/

open Hesper.WebGPU
open Hesper.Models.DiffusionGemma

abbrev B := Hesper.GPUBackend.Buf Device
abbrev C := Hesper.GPUBackend.CachedDispatch Device
abbrev Lin := Hesper.Layers.Linear.LinearLayer B C

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

def mkLin (device : Device) (wb : Buffer) (inDim outDim : Nat) (fmt : Hesper.Layers.Linear.QuantFormat) : IO Lin := do
  return {
    config := { inDim, outDim }, weightBuf := wb, quantFormat := fmt
    prepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    splitKBuf := ← IO.mkRef none
    splitKPartialPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    splitKReducePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    dp4aQ8Buf := ← IO.mkRef none
    dp4aQuantizePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    dp4aMatmulPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    dp4aBatchQuantizePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
    dp4aBatchMatmulPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
  }

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  let nLayers := (args.drop 1).head?.bind (·.toNat?) |>.getD 30
  IO.println "[dg-forward-gpu] init WebGPU (Metal) + load model..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  let dim := cfg.hiddenSize
  let ffn := cfg.intermediateSize
  let lmN := min cfg.vocabSize 32768

  -- preallocate ONCE (reused across all layers)
  let a ← mkBuf device dim
  let b ← mkBuf device dim
  let sN ← mkBuf device dim
  let sG ← mkBuf device ffn
  let sD ← mkBuf device dim
  let logitsBuf ← mkBuf device lmN
  let lmHead ← mkLin device model.inner.outputWeight dim lmN .Q6_K

  -- synthetic input → a (upload before the batch)
  let inArr := (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.01) * 0.5)
  writeBuffer device a 0 (← Hesper.Basic.floatArrayToBytes inArr)

  IO.println s!"[dg-forward-gpu] {nLayers}-layer GPU-resident forward (single beginBatch/endBatch)..."
  Hesper.GPUBackend.beginBatch device

  let mut cur := a
  let mut nxt := b
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError s!"no block {li}")
    -- dense path (structural): RMSNorm → gate → down → (postFFNNorm + residual)
    Hesper.Layers.RMSNorm.forward device blk.ffnNorm cur sN
    Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.gate sN sG
    Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.down sG sD
    let pref ← IO.mkRef none
    Hesper.Layers.RMSNorm.forwardNormThenAdd device blk.postFFNNorm sD cur nxt pref
    let t := cur; cur := nxt; nxt := t

  -- final norm + tied Q6_K lm_head (sliced)
  Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN
  Hesper.Layers.Linear.LinearLayer.forward device lmHead sN logitsBuf

  Hesper.GPUBackend.endBatch device
  IO.println "[dg-forward-gpu] batch submitted; reading logits..."

  let logits ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device logitsBuf 0 (lmN*4).toUSize)
  unmapBuffer logitsBuf
  let cap := cfg.logitSoftcapScale
  let mut top := 0
  let mut topV := -1e30
  for i in [0:lmN] do
    let l := cap * Float.tanh (logits[i]! / cap)
    if l > topV then topV := l; top := i
  let fin := logits.all Float.isFinite
  IO.println s!"[dg-forward-gpu] logits finite={fin} size={logits.size}  argmax(slice)={top}"
  if fin && logits.size == lmN then
    IO.println s!"✓ GPU-resident {nLayers}-layer forward ran in ONE batch on Metal (no per-op dispatch crash)"
  else
    IO.println "✗ failed"
    throw (IO.userError "gpu forward failed")
