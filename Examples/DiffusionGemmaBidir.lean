import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Quantization.Q6_K
import Hesper.Basic

/-!
# DiffusionGemma BIDIRECTIONAL BATCHED forward — Stage A (attn + dense FFN over N positions)

Assembles the validated batched kernels over N positions in one beginBatch/endBatch:
  per block: attnNorm → Q/K/V (batched Q4_K matmul) → qk-norm+RoPE (Q,K) / v-norm (V)
             → batched cross-position attention (region mask) → wO → postAttnNorm+residual
             → dense FFN (batched gate/up → GeGLU → down) → moePostNorm1 → postFFNNorm+residual → ×out_scale
MoE + real embedding + golden comparison are Stage B. Synthetic input; tests finiteness.

Run:  lake exe diffusiongemma-bidir [path] [nLayers] [N]
-/

open Hesper.WebGPU
open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.Models.DiffusionGemma

/-- GeGLU gelu(gate)*up, clamped tanh (works for any n = N*ffn — elementwise). -/
def geluMulB (n : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
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

/-- Elementwise add: out[i]=a[i]+b[i] (n=N*dim). -/
def addB (n : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 n)
  let _a ← ShaderM.declareReadOnlyBuffer "ain" (.array (.scalar .f32) n)
  let _b ← ShaderM.declareReadOnlyBuffer "bin" (.array (.scalar .f32) n)
  let _o ← ShaderM.declareOutputBuffer "outc" (.array (.scalar .f32) n)
  let av := Exp.index (Exp.var "ain" : Exp (.array (.scalar .f32) n)) i
  let bv := Exp.index (Exp.var "bin" : Exp (.array (.scalar .f32) n)) i
  ShaderM.writeBuffer (ty := .scalar .f32) "outc" i (Exp.select inB (Exp.add av bv) (Exp.litF32 0.0))

/-- In-place scalar multiply: data[i]*=sc (n=N*dim). -/
def scaleB (n : Nat) (sc : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let inB := Exp.lt i (Exp.litU32 n)
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) n)
  let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "data" i
  ShaderM.writeBuffer (ty := .scalar .f32) "data" i (Exp.select inB (Exp.mul v (Exp.litF32 sc)) v)

/-- Per-head qk-norm (RMS×weight) + RoPE; 1 thread per (pos=t/nHead, head). -/
def qkNormRopeB (N nHead hd : Nat) (theta eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let qDim := nHead*hd; let half := hd/2
  let coef := -2.0 * Float.log theta / hd.toFloat
  let _in ← ShaderM.declareReadOnlyBuffer "qin" (.array (.scalar .f32) (N*qDim))
  let _w ← ShaderM.declareReadOnlyBuffer "wnorm" (.array (.scalar .f32) hd)
  let _out ← ShaderM.declareOutputBuffer "qout" (.array (.scalar .f32) (N*qDim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*nHead))) (do
    let p := Exp.div t (Exp.litU32 nHead)
    let base := Exp.mul t (Exp.litU32 hd)
    let ss ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "qin" : Exp (.array (.scalar .f32) (N*qDim))) (Exp.add base d)
      ShaderM.assign ss (Exp.add (Exp.var ss) (Exp.mul v v))
    let inv := Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add (Exp.div (Exp.var ss) (Exp.litF32 hd.toFloat)) (Exp.litF32 eps)))
    let pf := Exp.toF32 p
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 half) (Exp.litU32 1) fun j => do
      let jh := Exp.add j (Exp.litU32 half)
      let qi : Exp (.array (.scalar .f32) (N*qDim)) := Exp.var "qin"
      let wn : Exp (.array (.scalar .f32) hd) := Exp.var "wnorm"
      let a := Exp.mul (Exp.mul (Exp.index qi (Exp.add base j)) inv) (Exp.index wn j)
      let b := Exp.mul (Exp.mul (Exp.index qi (Exp.add base jh)) inv) (Exp.index wn jh)
      let freq := Exp.exp (Exp.mul (Exp.litF32 coef) (Exp.toF32 j))
      let ang := Exp.mul pf freq
      ShaderM.assignIndex "qout" (Exp.add base j) (Exp.sub (Exp.mul a (Exp.cos ang)) (Exp.mul b (Exp.sin ang)))
      ShaderM.assignIndex "qout" (Exp.add base jh) (Exp.add (Exp.mul a (Exp.sin ang)) (Exp.mul b (Exp.cos ang)))) (pure ())

/-- Per-head v-norm (RMS, no weight, no rope); 1 thread per (pos, kvhead). -/
def vNormB (N nKV hd : Nat) (eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let kvDim := nKV*hd
  let _in ← ShaderM.declareReadOnlyBuffer "vin" (.array (.scalar .f32) (N*kvDim))
  let _out ← ShaderM.declareOutputBuffer "vout" (.array (.scalar .f32) (N*kvDim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*nKV))) (do
    let base := Exp.mul t (Exp.litU32 hd)
    let ss ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "vin" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add base d)
      ShaderM.assign ss (Exp.add (Exp.var ss) (Exp.mul v v))
    let inv := Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add (Exp.div (Exp.var ss) (Exp.litF32 hd.toFloat)) (Exp.litF32 eps)))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "vin" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add base d)
      ShaderM.assignIndex "vout" (Exp.add base d) (Exp.mul v inv)) (pure ())

/-- Cross-position attention: wg per (head=wid.x, query=wid.y); scores+region-mask+softmax+weighted-V. -/
def battnB (N P nHead hd nKV : Nat) (ws : Nat := 256) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let h := Exp.vec3X wid; let i := Exp.vec3Y wid; let tid := Exp.vec3X lid
  let qDim := nHead*hd; let kvDim := nKV*hd; let groupSize := nHead/nKV
  let _q ← ShaderM.declareReadOnlyBuffer "q" (.array (.scalar .f32) (N*qDim))
  let _k ← ShaderM.declareReadOnlyBuffer "k" (.array (.scalar .f32) (N*kvDim))
  let _v ← ShaderM.declareReadOnlyBuffer "v" (.array (.scalar .f32) (N*kvDim))
  let _o ← ShaderM.declareOutputBuffer "ctx" (.array (.scalar .f32) (N*qDim))
  ShaderM.sharedNamed "sS" (.array (.scalar .f32) N)
  let kvhE := Exp.div h (Exp.litU32 groupSize)
  let qBase := Exp.add (Exp.mul i (Exp.litU32 qDim)) (Exp.mul h (Exp.litU32 hd))
  let kvB0 := Exp.mul kvhE (Exp.litU32 hd)
  ShaderM.loop tid (Exp.litU32 N) (Exp.litU32 ws) fun j => do
    let allowed := Exp.or (Exp.ge i (Exp.litU32 P)) (Exp.ge i j)
    let kBase := Exp.add (Exp.mul j (Exp.litU32 kvDim)) kvB0
    let sc ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let qv := Exp.index (Exp.var "q" : Exp (.array (.scalar .f32) (N*qDim))) (Exp.add qBase d)
      let kv := Exp.index (Exp.var "k" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add kBase d)
      ShaderM.assign sc (Exp.add (Exp.var sc) (Exp.mul qv kv))
    ShaderM.assignIndex "sS" j (Exp.select allowed (Exp.var sc : Exp (.scalar .f32)) (Exp.litF32 (-1e30)))
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let mx ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assign mx (Exp.max (Exp.var mx) (Exp.index (Exp.var "sS" : Exp (.array (.scalar .f32) N)) j))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assignIndex "sS" j (Exp.exp (Exp.sub (Exp.index (Exp.var "sS" : Exp (.array (.scalar .f32) N)) j) (Exp.var mx)))
    let sm ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assign sm (Exp.add (Exp.var sm) (Exp.index (Exp.var "sS" : Exp (.array (.scalar .f32) N)) j))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assignIndex "sS" j (Exp.div (Exp.index (Exp.var "sS" : Exp (.array (.scalar .f32) N)) j) (Exp.var sm))) (pure ())
  ShaderM.barrier
  ShaderM.loop tid (Exp.litU32 hd) (Exp.litU32 ws) fun d => do
    let acc ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      let w := Exp.index (Exp.var "sS" : Exp (.array (.scalar .f32) N)) j
      let vv := Exp.index (Exp.var "v" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add (Exp.add (Exp.mul j (Exp.litU32 kvDim)) kvB0) d)
      ShaderM.assign acc (Exp.add (Exp.var acc) (Exp.mul w vv))
    ShaderM.writeBuffer (ty := .scalar .f32) "ctx" (Exp.add qBase d) (Exp.var acc : Exp (.scalar .f32))

abbrev B := Hesper.GPUBackend.Buf Device
abbrev C := Hesper.GPUBackend.CachedDispatch Device

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

/-- Elementwise/1D dispatch over n elements. -/
def disp (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (n : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device k bufs { numWorkgroups := ((n+255)/256,1,1), workgroupSize := {x:=256} } key r

/-- 2D dispatch (x, y). -/
def disp2 (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (nx ny : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device k bufs { numWorkgroups := (nx,ny,1), workgroupSize := {x:=256} } key r

/-- Batched matmul over N rows dispatching by the layer's quant format. -/
def bmm (device : Device) (layer : Hesper.Layers.Linear.LinearLayer B C) (inB outB : Buffer) (N : Nat) (key : UInt64) : IO Unit := do
  let cfg := layer.config
  let bufs := ("weights", layer.weightBuf)::("input", inB)::("output", outB)::List.nil
  let k := match layer.quantFormat with
    | .Q8_0 => Hesper.Layers.Linear.fusedQ8_0BatchKernel cfg N
    | .Q6_K => Hesper.Quantization.Q6_K.fusedQ6KBatchKernel cfg.inDim cfg.outDim N
    | _     => Hesper.Layers.Linear.fusedQ4KMBatchKernel cfg N
  disp2 device k bufs cfg.outDim N key

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  let nLayers := (args.drop 1).head?.bind (·.toNat?) |>.getD 3
  let N := (args.drop 2).head?.bind (·.toNat?) |>.getD 16
  let P := 3
  IO.println s!"[dg-bidir] init + load; N={N} P={P} layers={nLayers}"
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  let dim := cfg.hiddenSize; let ffn := cfg.intermediateSize; let nHead := cfg.numAttentionHeads
  let eps := 1e-6
  -- per-layer out_scale
  let mut scales : Array Float := #[]
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError "blk")
    let sc ← match blk.outScale with
      | none => pure 1.0
      | some bf => do let a ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device bf 0 (4:Nat).toUSize); unmapBuffer bf; pure a[0]!
    scales := scales.push sc
  -- preallocate [N, *]
  let a ← mkBuf device (N*dim); let b ← mkBuf device (N*dim)
  let sN ← mkBuf device (N*dim)
  let sQ ← mkBuf device (N*8192); let sK ← mkBuf device (N*2048); let sV ← mkBuf device (N*2048)
  let sQr ← mkBuf device (N*8192); let sKr ← mkBuf device (N*2048); let sVn ← mkBuf device (N*2048)
  let sCtx ← mkBuf device (N*8192); let sAO ← mkBuf device (N*dim); let sPA ← mkBuf device (N*dim)
  let sG ← mkBuf device (N*ffn); let sU ← mkBuf device (N*ffn); let sGe ← mkBuf device (N*ffn); let sD ← mkBuf device (N*dim)
  let sR ← mkBuf device (N*dim)
  -- synthetic input
  let inArr := (List.range (N*dim)).toArray.map (fun i => Float.sin (i.toFloat * 0.001) * 0.4)
  writeBuffer device a 0 (← Hesper.Basic.floatArrayToBytes inArr)
  IO.println "[dg-bidir] batched forward (attn + dense) ..."
  Hesper.GPUBackend.beginBatch device
  let mut cur := a; let mut nxt := b
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError "blk")
    let qDim := blk.attention.wO.config.inDim; let kvDim := blk.attention.wV.config.outDim
    let hd := qDim / nHead; let nKV := kvDim / hd
    let theta : Float := if li % 6 == 5 then 1000000.0 else 10000.0
    -- attention
    Hesper.Layers.RMSNorm.forward device blk.attnNorm cur sN N
    bmm device blk.attention.wQ sN sQ N (hash ("wq",li))
    bmm device blk.attention.wK sN sK N (hash ("wk",li))
    bmm device blk.attention.wV sN sV N (hash ("wv",li))
    disp device (qkNormRopeB N nHead hd theta eps) (("qin",sQ)::("wnorm",blk.attention.qNormWeight)::("qout",sQr)::List.nil) (N*nHead) (hash ("qn",li))
    disp device (qkNormRopeB N nKV hd theta eps) (("qin",sK)::("wnorm",blk.attention.kNormWeight)::("qout",sKr)::List.nil) (N*nKV) (hash ("kn",li))
    disp device (vNormB N nKV hd eps) (("vin",sV)::("vout",sVn)::List.nil) (N*nKV) (hash ("vn",li))
    disp2 device (battnB N P nHead hd nKV) (("q",sQr)::("k",sKr)::("v",sVn)::("ctx",sCtx)::List.nil) nHead N (hash ("at",li))
    bmm device blk.attention.wO sCtx sAO N (hash ("wo",li))
    Hesper.Layers.RMSNorm.forward device blk.postAttnNorm sAO sR N
    disp device (addB (N*dim)) (("ain",sR)::("bin",cur)::("outc",sPA)::List.nil) (N*dim) (hash ("ra",li))
    -- dense FFN
    Hesper.Layers.RMSNorm.forward device blk.ffnNorm sPA sN N
    bmm device blk.ffn.gate sN sG N (hash ("g",li))
    bmm device blk.ffn.up sN sU N (hash ("u",li))
    disp device (geluMulB (N*ffn)) (("gate",sG)::("up",sU)::("outp",sGe)::List.nil) (N*ffn) (hash ("gg",li))
    bmm device blk.ffn.down sGe sD N (hash ("dn",li))
    let some mpn1 := blk.moePostNorm1 | throw (IO.userError "mpn1")
    Hesper.Layers.RMSNorm.forward device mpn1 sD sN N
    Hesper.Layers.RMSNorm.forward device blk.postFFNNorm sN sR N
    disp device (addB (N*dim)) (("ain",sR)::("bin",sPA)::("outc",nxt)::List.nil) (N*dim) (hash ("rc",li))
    disp device (scaleB (N*dim) (scales[li]!)) (("data",nxt)::List.nil) (N*dim) (hash ("sc",li))
    let t := cur; cur := nxt; nxt := t
  Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN N
  Hesper.GPUBackend.endBatch device
  let outv ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sN 0 (N*dim*4).toUSize)
  unmapBuffer sN
  let fin := outv.all Float.isFinite
  IO.println s!"[dg-bidir] post-finalNorm finite={fin} size={outv.size}  [0..4]={(outv.extract 0 4).toList}"
  if fin then IO.println s!"✓ batched bidirectional forward (attn+dense) {nLayers}-layer N={N} on Metal → finite"
  else IO.println "✗ non-finite"; throw (IO.userError "fail")
