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

/-- Per-head v-norm (RMS, no scale) + GQA broadcast: for output i (head h, dim d),
    src head kvh=h/groupSize; vout[i] = vin[kvh*hd+d] / rms(vin[kvh*hd .. +hd]). -/
def broadcastVNormK (qDim hd groupSize kvDim : Nat) (eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _v ← ShaderM.declareReadOnlyBuffer "vin" (.array (.scalar .f32) kvDim)
  let _o ← ShaderM.declareOutputBuffer "vout" (.array (.scalar .f32) qDim)
  ShaderM.if_ (Exp.lt i (Exp.litU32 qDim)) (do
    let h := Exp.div i (Exp.litU32 hd)
    let kvh := Exp.div h (Exp.litU32 groupSize)
    let d := Exp.sub i (Exp.mul h (Exp.litU32 hd))
    let base := Exp.mul kvh (Exp.litU32 hd)
    let src := Exp.add base d
    let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun j => do
      let vj := Exp.index (Exp.var "vin" : Exp (.array (.scalar .f32) kvDim)) (Exp.add base j)
      ShaderM.assign sumVar (Exp.add (Exp.var sumVar) (Exp.mul vj vj))
    let rms := Exp.sqrt (Exp.add (Exp.div (Exp.var sumVar) (Exp.litF32 hd.toFloat)) (Exp.litF32 eps))
    let val := Exp.index (Exp.var "vin" : Exp (.array (.scalar .f32) kvDim)) src
    ShaderM.assignIndex "vout" i (Exp.div val rms)) (pure ())

/-- In-place scalar multiply: data[i] *= sc. -/
def scaleK (n : Nat) (sc : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 n)
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) n)
  let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "data" i
  ShaderM.writeBuffer (ty := .scalar .f32) "data" i (Exp.select inB (Exp.mul v (Exp.litF32 sc)) v)

/-- Router prep: tmps[i] = (x[i]/rms(x)) * invSqrt * rscale[i]  (rmsNorm no-scale × 1/√d × router_scale). -/
def routerPrepK (dim : Nat) (invSqrt eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _x ← ShaderM.declareReadOnlyBuffer "xin" (.array (.scalar .f32) dim)
  let _rs ← ShaderM.declareReadOnlyBuffer "rscale" (.array (.scalar .f32) dim)
  let _o ← ShaderM.declareOutputBuffer "tmps" (.array (.scalar .f32) dim)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (do
    let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun j => do
      let xj := Exp.index (Exp.var "xin" : Exp (.array (.scalar .f32) dim)) j
      ShaderM.assign sumVar (Exp.add (Exp.var sumVar) (Exp.mul xj xj))
    let rms := Exp.sqrt (Exp.add (Exp.div (Exp.var sumVar) (Exp.litF32 dim.toFloat)) (Exp.litF32 eps))
    let xi := Exp.index (Exp.var "xin" : Exp (.array (.scalar .f32) dim)) i
    let rsi := Exp.index (Exp.var "rscale" : Exp (.array (.scalar .f32) dim)) i
    ShaderM.assignIndex "tmps" i (Exp.mul (Exp.mul (Exp.div xi rms) (Exp.litF32 invSqrt)) rsi)) (pure ())

/-- Router matmul (F32): rlogits[e] = Σ_k rw[e*dim+k] * tmps[k]. -/
def routerMatVecK (nExpert dim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let e := Exp.vec3X gid
  let _w ← ShaderM.declareReadOnlyBuffer "rw" (.array (.scalar .f32) (nExpert*dim))
  let _t ← ShaderM.declareReadOnlyBuffer "tmps" (.array (.scalar .f32) dim)
  let _o ← ShaderM.declareOutputBuffer "rlogits" (.array (.scalar .f32) nExpert)
  ShaderM.if_ (Exp.lt e (Exp.litU32 nExpert)) (do
    let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let base := Exp.mul e (Exp.litU32 dim)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun k => do
      let wv := Exp.index (Exp.var "rw" : Exp (.array (.scalar .f32) (nExpert*dim))) (Exp.add base k)
      let tv := Exp.index (Exp.var "tmps" : Exp (.array (.scalar .f32) dim)) k
      ShaderM.assign sumVar (Exp.add (Exp.var sumVar) (Exp.mul wv tv))
    ShaderM.assignIndex "rlogits" e (Exp.var sumVar : Exp (.scalar .f32))) (pure ())

/-- Softmax over nExpert + top-nUsed → idxs[nUsed] (u32) + wts[nUsed] (normalized). Single workgroup, thread 0. -/
def top8K (nExpert nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let _r ← ShaderM.declareReadOnlyBuffer "rlogits" (.array (.scalar .f32) nExpert)
  let _idxs ← ShaderM.declareOutputBuffer "idxs" (.array (.scalar .u32) nUsed)
  let _wts ← ShaderM.declareOutputBuffer "wts" (.array (.scalar .f32) nUsed)
  ShaderM.sharedNamed "sp" (.array (.scalar .f32) nExpert)
  ShaderM.if_ (Exp.lt tid (Exp.litU32 nExpert)) (do
    ShaderM.assignIndex "sp" tid (Exp.index (Exp.var "rlogits" : Exp (.array (.scalar .f32) nExpert)) tid)) (pure ())
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let mx ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      ShaderM.assign mx (Exp.max (Exp.var mx) (Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e))
    let sm ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      let ex := Exp.exp (Exp.sub (Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e) (Exp.var mx))
      ShaderM.assignIndex "sp" e ex
      ShaderM.assign sm (Exp.add (Exp.var sm) ex)
    let wsum ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nUsed) (Exp.litU32 1) fun p => do
      let bi ← ShaderM.var (.scalar .u32) (Exp.litU32 0)
      let bv ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
      ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
        let v := Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e
        ShaderM.if_ (Exp.gt v (Exp.var bv)) (do
          ShaderM.assign bv v
          ShaderM.assign bi e) (pure ())
      let prob := Exp.div (Exp.var bv : Exp (.scalar .f32)) (Exp.var sm : Exp (.scalar .f32))
      ShaderM.assignIndex "idxs" p (Exp.var bi : Exp (.scalar .u32))
      ShaderM.assignIndex "wts" p prob
      ShaderM.assign wsum (Exp.add (Exp.var wsum) prob)
      ShaderM.assignIndex "sp" (Exp.var bi) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nUsed) (Exp.litU32 1) fun p => do
      let w := Exp.index (Exp.var "wts" : Exp (.array (.scalar .f32) nUsed)) p
      ShaderM.assignIndex "wts" p (Exp.div w (Exp.var wsum))) (pure ())

/-- Merged-GeGLU: eh[i] = gelu(gu[i]) * gu[i+ff]  (clamped tanh). -/
def gegluMergedK (ff : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 ff)
  let _gu ← ShaderM.declareInputBuffer "gu" (.array (.scalar .f32) (2*ff))
  let _o ← ShaderM.declareOutputBuffer "eh" (.array (.scalar .f32) ff)
  let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2*ff) "gu" i
  let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2*ff) "gu" (Exp.add i (Exp.litU32 ff))
  let g3 := Exp.mul g (Exp.mul g g)
  let inner := Exp.mul (Exp.litF32 0.7978845608) (Exp.add g (Exp.mul (Exp.litF32 0.044715) g3))
  let innerC := Exp.max (Exp.litF32 (-10.0)) (Exp.min (Exp.litF32 10.0) inner)
  let gl := Exp.mul (Exp.mul (Exp.litF32 0.5) g) (Exp.add (Exp.litF32 1.0) (Exp.tanh innerC))
  ShaderM.writeBuffer (ty := .scalar .f32) "eh" i (Exp.select inB (Exp.mul gl u) (Exp.litF32 0.0))

/-- Weighted accumulate: acc[i] += wts[slot] * din[i]  (acc in-place). -/
def waccK (dim slot nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 dim)
  let _acc ← ShaderM.declareOutputBuffer "acc" (.array (.scalar .f32) dim)
  let _d ← ShaderM.declareReadOnlyBuffer "din" (.array (.scalar .f32) dim)
  let _w ← ShaderM.declareReadOnlyBuffer "wts" (.array (.scalar .f32) nUsed)
  let a ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "acc" i
  let d := Exp.index (Exp.var "din" : Exp (.array (.scalar .f32) dim)) i
  let w := Exp.index (Exp.var "wts" : Exp (.array (.scalar .f32) nUsed)) (Exp.litU32 slot)
  ShaderM.writeBuffer (ty := .scalar .f32) "acc" i (Exp.select inB (Exp.add a (Exp.mul w d)) a)

/-- Zero a buffer. -/
def zeroK (dim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) dim)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (ShaderM.assignIndex "data" i (Exp.litF32 0.0)) (pure ())

/-- Elementwise add: out[i] = a[i] + b[i]. -/
def addK (dim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 dim)
  let _a ← ShaderM.declareReadOnlyBuffer "ain" (.array (.scalar .f32) dim)
  let _b ← ShaderM.declareReadOnlyBuffer "bin" (.array (.scalar .f32) dim)
  let _o ← ShaderM.declareOutputBuffer "outc" (.array (.scalar .f32) dim)
  let av := Exp.index (Exp.var "ain" : Exp (.array (.scalar .f32) dim)) i
  let bv := Exp.index (Exp.var "bin" : Exp (.array (.scalar .f32) dim)) i
  ShaderM.writeBuffer (ty := .scalar .f32) "outc" i (Exp.select inB (Exp.add av bv) (Exp.litF32 0.0))

abbrev B := Hesper.GPUBackend.Buf Device
abbrev C := Hesper.GPUBackend.CachedDispatch Device

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

/-- Dispatch a custom ShaderM kernel over `n` elements, GPU-resident (fresh ref records into the batch). -/
def disp (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (n : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  let c : Hesper.ExecConfig := { numWorkgroups := ((n + 255)/256, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.executeWithConfigCached device k bufs c key r

/-- Dispatch with explicit workgroup count (expert kernels: one workgroup per output row). -/
def dispWG (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (nwg : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  let c : Hesper.ExecConfig := { numWorkgroups := (nwg, 1, 1), workgroupSize := { x := 256 } }
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
  -- MoE buffers
  let nExpert := cfg.numExperts
  let nUsed := cfg.numExpertsUsed
  let expFF := cfg.expertFFSize
  let sCurMlp ← mkBuf device dim
  let sMoeN ← mkBuf device dim
  let sTmpS ← mkBuf device dim
  let sRLogits ← mkBuf device nExpert
  let sIdxs ← mkBuf device nUsed       -- u32 indices
  let sWts ← mkBuf device nUsed
  let sMoeAcc ← mkBuf device dim
  let sGateUp ← mkBuf device (2*expFF)
  let sEh ← mkBuf device expFF
  let sDownE ← mkBuf device dim
  let sCurMoe ← mkBuf device dim
  let sComb ← mkBuf device dim
  let invSqrt := 1.0 / Float.sqrt dim.toFloat
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
    disp device (broadcastVNormK qDim hd groupSize kvDim 1e-6) (("vin", sV) :: ("vout", sVc) :: List.nil) qDim (hash ("bc", li))
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
    Hesper.Layers.RMSNorm.forward device mpn1 sD sCurMlp              -- curMlp (= ffnPostNorm1)
    -- MoE (dense + experts in parallel)
    let some mpn2pre := blk.moePreNorm2 | throw (IO.userError "no moePreNorm2")
    let some mpn2post := blk.moePostNorm2 | throw (IO.userError "no moePostNorm2")
    let some rW := blk.moeRouterWeight | throw (IO.userError "no router")
    let some rS := blk.moeRouterScale | throw (IO.userError "no routerScale")
    let some guExps := blk.moeGateUpExps | throw (IO.userError "no gateup exps")
    let some dnExps := blk.moeDownExps | throw (IO.userError "no down exps")
    Hesper.Layers.RMSNorm.forward device mpn2pre sPA sMoeN
    disp device (routerPrepK dim invSqrt 1e-6) (("xin", sPA) :: ("rscale", rS) :: ("tmps", sTmpS) :: List.nil) dim (hash ("rp", li))
    disp device (routerMatVecK nExpert dim) (("rw", rW) :: ("tmps", sTmpS) :: ("rlogits", sRLogits) :: List.nil) nExpert (hash ("rm", li))
    dispWG device (top8K nExpert nUsed) (("rlogits", sRLogits) :: ("idxs", sIdxs) :: ("wts", sWts) :: List.nil) 1 (hash ("t8", li))
    disp device (zeroK dim) (("data", sMoeAcc) :: List.nil) dim (hash ("z", li))
    for e in [0:nUsed] do
      dispWG device (Hesper.Layers.Linear.fusedQ4KMExpertKernel { inDim := dim, outDim := 2*expFF } nExpert nUsed e)
        (("weights", guExps) :: ("input", sMoeN) :: ("params", sIdxs) :: ("output", sGateUp) :: List.nil) (2*expFF) (hash ("eu", li, e))
      disp device (gegluMergedK expFF) (("gu", sGateUp) :: ("eh", sEh) :: List.nil) expFF (hash ("gm", li, e))
      dispWG device (Hesper.Layers.Linear.fusedQ8_0ExpertKernel { inDim := expFF, outDim := dim } nExpert nUsed e)
        (("weights", dnExps) :: ("input", sEh) :: ("params", sIdxs) :: ("output", sDownE) :: List.nil) dim (hash ("ed", li, e))
      disp device (waccK dim e nUsed) (("acc", sMoeAcc) :: ("din", sDownE) :: ("wts", sWts) :: List.nil) dim (hash ("wa", li, e))
    Hesper.Layers.RMSNorm.forward device mpn2post sMoeAcc sCurMoe
    -- combine: comb = curMlp + curMoe → postFFNNorm → +residual → ×out_scale
    disp device (addK dim) (("ain", sCurMlp) :: ("bin", sCurMoe) :: ("outc", sComb) :: List.nil) dim (hash ("ad", li))
    let pref2 ← IO.mkRef none
    Hesper.Layers.RMSNorm.forwardNormThenAdd device blk.postFFNNorm sComb sPA nxt pref2   -- resid = norm(comb)+pa
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
