import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Basic

/-!
# DiffusionGemma GPU-resident forward — Phase 1 (real block, dense path)

Real block (seqLen=1, canvas single position) GPU-resident in one beginBatch/endBatch:
  attention (V-reuse: softmax over 1 key = 1, so attn = wO(broadcast(wV(norm(x))))) + postAttnNorm residual
  dense FFN: ffnNorm → gate/up → GeGLU(clamped tanh) → down → moePostNorm1(=ffnPostNorm1)
  combine (dense only): postFFNNorm(curMlp) + pa  → × out_scale
MoE (router + experts) added next. out_scale is the per-layer scalar that bounds 30-layer growth.

Run:  lake exe diffusiongemma-forward-gpu [path] [nLayers]
-/

open Hesper.WebGPU
open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.Models.DiffusionGemma

/-- GeGLU gelu(gate)*up, clamped tanh arg (Metal tanh overflows→NaN for large args). -/
def geluMulV2 (n : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 n)
  let _g ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) n)
  let _u ← ShaderM.declareInputBuffer "up" (.array (.scalar .f32) n)
  let _o ← ShaderM.declareOutputBuffer "outp" (.array (.scalar .f32) n)
  let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "gate" i
  let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "up" i
  let g3 := Exp.mul g (Exp.mul g g)
  let inner := Exp.mul (Exp.litF32 0.7978845608) (Exp.add g (Exp.mul (Exp.litF32 0.044715) g3))
  let innerC := Exp.max (Exp.litF32 (-10.0)) (Exp.min (Exp.litF32 10.0) inner)
  let gl := Exp.mul (Exp.mul (Exp.litF32 0.5) g) (Exp.add (Exp.litF32 1.0) (Exp.tanh innerC))
  ShaderM.writeBuffer (ty := .scalar .f32) "outp" i (Exp.select inB (Exp.mul gl u) (Exp.litF32 0.0))

/-- GQA broadcast: vout[i] = vin[(i/hd/groupSize)*hd + i%hd]  (kvDim → qDim). -/
def broadcastK (qDim hd groupSize kvDim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 qDim)
  let _v ← ShaderM.declareReadOnlyBuffer "vin" (.array (.scalar .f32) kvDim)
  let _o ← ShaderM.declareOutputBuffer "vout" (.array (.scalar .f32) qDim)
  let h := Exp.div i (Exp.litU32 hd)
  let kvh := Exp.div h (Exp.litU32 groupSize)
  let d := Exp.sub i (Exp.mul h (Exp.litU32 hd))
  let src := Exp.add (Exp.mul kvh (Exp.litU32 hd)) d
  let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim) "vin" src
  ShaderM.writeBuffer (ty := .scalar .f32) "vout" i (Exp.select inB val (Exp.litF32 0.0))

/-- In-place scalar multiply: data[i] *= sc. -/
def scaleK (n : Nat) (sc : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 n)
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) n)
  let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "data" i
  ShaderM.writeBuffer (ty := .scalar .f32) "data" i (Exp.select inB (Exp.mul v (Exp.litF32 sc)) v)

abbrev B := Hesper.GPUBackend.Buf Device
abbrev C := Hesper.GPUBackend.CachedDispatch Device

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

/-- Dispatch a custom ShaderM kernel over `n` elements, GPU-resident (fresh ref records into the batch). -/
def disp (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (n : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  let c : Hesper.ExecConfig := { numWorkgroups := ((n + 255)/256, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.executeWithConfigCached device k bufs c key r

def readScale (device : Device) (b : Option Buffer) : IO Float := do
  match b with
  | none => return 1.0
  | some buf =>
    let a ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device buf 0 (4 : Nat).toUSize)
    unmapBuffer buf
    return a[0]!

def mkLmHead (device : Device) (wb : Buffer) (dim lmN : Nat) : IO (Hesper.Layers.Linear.LinearLayer B C) := do
  return { config := { inDim := dim, outDim := lmN }, weightBuf := wb, quantFormat := .Q6_K
           prepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           splitKBuf := ← IO.mkRef none
           splitKPartialPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           splitKReducePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           dp4aQ8Buf := ← IO.mkRef none
           dp4aQuantizePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           dp4aMatmulPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           dp4aBatchQuantizePrepared := ← Hesper.GPUBackend.newCacheRef (β := Device)
           dp4aBatchMatmulPrepared := ← Hesper.GPUBackend.newCacheRef (β := Device) }

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  let nLayers := (args.drop 1).head?.bind (·.toNat?) |>.getD 30
  IO.println "[dg-fwd-gpu] init WebGPU (Metal) + load model..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  let dim := cfg.hiddenSize
  let ffn := cfg.intermediateSize
  let lmN := min cfg.vocabSize 32768

  -- read per-layer out_scale (canvas) ONCE (setup; not in the batch)
  let mut scales : Array Float := #[]
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError s!"no block {li}")
    scales := scales.push (← readScale device blk.outScale)
  IO.println s!"[dg-fwd-gpu] out_scale[0..4]={(scales.extract 0 (min 4 scales.size)).toList}"

  -- preallocate ONCE
  let a ← mkBuf device dim
  let b ← mkBuf device dim
  let sN ← mkBuf device dim
  let sV ← mkBuf device 2048      -- max kvDim
  let sVc ← mkBuf device 8192     -- max qDim
  let sAO ← mkBuf device dim
  let sPA ← mkBuf device dim
  let sG ← mkBuf device ffn
  let sU ← mkBuf device ffn
  let sGeglu ← mkBuf device ffn
  let sD ← mkBuf device dim
  let logitsBuf ← mkBuf device lmN
  let lmHead ← mkLmHead device model.inner.outputWeight dim lmN

  -- synthetic canvas input → a
  let inArr := (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.01) * 0.5)
  writeBuffer device a 0 (← Hesper.Basic.floatArrayToBytes inArr)

  IO.println s!"[dg-fwd-gpu] {nLayers}-layer real block (attn + dense FFN + out_scale), GPU-resident..."
  Hesper.GPUBackend.beginBatch device

  let mut cur := a
  let mut nxt := b
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError s!"no block {li}")
    let qDim := blk.attention.wO.config.inDim
    let kvDim := blk.attention.wV.config.outDim
    let hd := qDim / cfg.numAttentionHeads
    let nKV := kvDim / hd
    let groupSize := cfg.numAttentionHeads / nKV
    -- attention (seqLen=1): attn = wO(broadcast(wV(norm(x)))) ; postAttnNorm + residual
    Hesper.Layers.RMSNorm.forward device blk.attnNorm cur sN
    Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wV sN sV
    disp device (broadcastK qDim hd groupSize kvDim) (("vin", sV) :: ("vout", sVc) :: List.nil) qDim (hash ("bc", li))
    Hesper.Layers.Linear.LinearLayer.forward device blk.attention.wO sVc sAO
    let pref1 ← IO.mkRef none
    Hesper.Layers.RMSNorm.forwardNormThenAdd device blk.postAttnNorm sAO cur sPA pref1
    -- dense FFN
    Hesper.Layers.RMSNorm.forward device blk.ffnNorm sPA sN
    Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.gate sN sG
    Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.up sN sU
    disp device (geluMulV2 ffn) (("gate", sG) :: ("up", sU) :: ("outp", sGeglu) :: List.nil) ffn (hash ("gg", li))
    Hesper.Layers.Linear.LinearLayer.forward device blk.ffn.down sGeglu sD
    let some mpn1 := blk.moePostNorm1 | throw (IO.userError "no moePostNorm1")
    Hesper.Layers.RMSNorm.forward device mpn1 sD sN              -- curMlp (= ffnPostNorm1)
    -- combine (dense only) + residual + out_scale
    let pref2 ← IO.mkRef none
    Hesper.Layers.RMSNorm.forwardNormThenAdd device blk.postFFNNorm sN sPA nxt pref2   -- resid = norm(curMlp)+pa
    disp device (scaleK dim (scales[li]!)) (("data", nxt) :: List.nil) dim (hash ("sc", li))
    let t := cur; cur := nxt; nxt := t

  Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN
  Hesper.Layers.Linear.LinearLayer.forward device lmHead sN logitsBuf

  Hesper.GPUBackend.endBatch device
  IO.println "[dg-fwd-gpu] batch submitted; reading logits..."

  let logits ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device logitsBuf 0 (lmN*4).toUSize)
  unmapBuffer logitsBuf
  let cap := cfg.logitSoftcapScale
  let mut top := 0
  let mut topV := -1e30
  for i in [0:lmN] do
    let l := cap * Float.tanh (logits[i]! / cap)
    if l > topV then topV := l; top := i
  let fin := logits.all Float.isFinite
  IO.println s!"[dg-fwd-gpu] logits finite={fin} size={logits.size}  argmax(slice)={top}"
  if fin && logits.size == lmN then
    IO.println s!"✓ real block (attn+dense+out_scale) {nLayers}-layer GPU-resident on Metal → finite logits"
  else
    IO.println "✗ failed"
    throw (IO.userError "gpu forward failed")
