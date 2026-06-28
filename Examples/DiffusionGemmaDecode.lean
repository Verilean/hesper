import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Quantization.Q6_K
import Hesper.GGUF.Reader
import Hesper.Tokenizer.SentencePiece
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

set_option maxRecDepth 8000

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

/-- Identity copy out[i]=in[i]. -/
def copyB (n : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _a ← ShaderM.declareReadOnlyBuffer "cin" (.array (.scalar .f32) n)
  let _o ← ShaderM.declareOutputBuffer "cout" (.array (.scalar .f32) n)
  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (ShaderM.writeBuffer (ty := .scalar .f32) "cout" i (Exp.index (Exp.var "cin" : Exp (.array (.scalar .f32) n)) i)) (pure ())

/-- Activation Q8_K quant+dequant in-place (block 256 along K), matching ggml quantize_row_q8_K.
    One thread per (row,block). iscale=-128/signedMaxAbs; q=min(127, nearest_int(iscale*x)); x:=q/iscale. -/
def qActQ8K (N K : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let blk := Exp.vec3X gid
  let nB := K / 256
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) (N*K))
  ShaderM.if_ (Exp.lt blk (Exp.litU32 (N*nB))) (do
    let base := Exp.mul blk (Exp.litU32 256)
    let amax ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let vmax ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 256) (Exp.litU32 1) fun i => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*K) "data" (Exp.add base i)
      ShaderM.if_ (Exp.gt (Exp.abs v) (Exp.var amax)) (do
        ShaderM.assign amax (Exp.abs v); ShaderM.assign vmax v) (pure ())
    ShaderM.if_ (Exp.gt (Exp.var amax) (Exp.litF32 0.0)) (do
      let iscale := Exp.div (Exp.litF32 (-127.0)) (Exp.var vmax)
      let d := Exp.div (Exp.var vmax) (Exp.litF32 (-127.0))
      ShaderM.loop (Exp.litU32 0) (Exp.litU32 256) (Exp.litU32 1) fun i => do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*K) "data" (Exp.add base i)
        let y := Exp.mul iscale v
        let q := Exp.min (Exp.litF32 127.0) (Exp.mul (Exp.sign y) (Exp.floor (Exp.add (Exp.abs y) (Exp.litF32 0.5))))
        ShaderM.writeBuffer (ty := .scalar .f32) "data" (Exp.add base i) (Exp.mul q d)) (pure ())) (pure ())

/-- Activation Q8_0 quant+dequant in-place (block 32 along K), matching ggml quantize_row_q8_0.
    d=maxAbs/127; q=nearest_int(x*127/maxAbs); x:=q*d. -/
def qActQ80 (N K : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let blk := Exp.vec3X gid
  let nB := K / 32
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) (N*K))
  ShaderM.if_ (Exp.lt blk (Exp.litU32 (N*nB))) (do
    let base := Exp.mul blk (Exp.litU32 32)
    let amax ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun i => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*K) "data" (Exp.add base i)
      ShaderM.assign amax (Exp.max (Exp.var amax) (Exp.abs v))
    ShaderM.if_ (Exp.gt (Exp.var amax) (Exp.litF32 0.0)) (do
      let d := Exp.div (Exp.var amax) (Exp.litF32 127.0)
      let idv := Exp.div (Exp.litF32 127.0) (Exp.var amax)
      ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun i => do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*K) "data" (Exp.add base i)
        let y := Exp.mul idv v
        let q := Exp.mul (Exp.sign y) (Exp.floor (Exp.add (Exp.abs y) (Exp.litF32 0.5)))
        ShaderM.writeBuffer (ty := .scalar .f32) "data" (Exp.add base i) (Exp.mul q d)) (pure ())) (pure ())

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

/-- Region-aware per-layer scale: data[r,i] *= (r≥P ? outSc : encSc). -/
def scaleRegionB (N P dim : Nat) (outSc encSc : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) (N*dim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let r := Exp.div t (Exp.litU32 dim)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "data" t
    let factor := Exp.select (Exp.ge r (Exp.litU32 P)) (Exp.litF32 outSc) (Exp.litF32 encSc)
    ShaderM.writeBuffer (ty := .scalar .f32) "data" t (Exp.mul v factor)) (pure ())

/-- Soft-embedding for self-conditioning: sSC[(P+i),d] = embScale·Σ_{k<K} prob[i,k]·tempK[(i·K+k),d]
    over the previous step's top-K tokens (the softmax mass); prompt rows = 0.
    `tempK` is the gathered raw embeddings of the C·K top tokens; `prob` their softmax weights. -/
def softReduceB (N P C dim K : Nat) (embScale : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let _o ← ShaderM.declareOutputBuffer "ssc" (.array (.scalar .f32) (N*dim))
  let _tk ← ShaderM.declareReadOnlyBuffer "tempk" (.array (.scalar .f32) ((C*K)*dim))
  let _pr ← ShaderM.declareReadOnlyBuffer "prob" (.array (.scalar .f32) (C*K))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let r := Exp.div t (Exp.litU32 dim)
    let d := Exp.mod t (Exp.litU32 dim)
    ShaderM.if_ (Exp.ge r (Exp.litU32 P)) (do
      let i := Exp.sub r (Exp.litU32 P)
      let acc ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for k in [0:K] do
        let slot := Exp.add (Exp.mul i (Exp.litU32 K)) (Exp.litU32 k)
        let pv := Exp.index (Exp.var "prob" : Exp (.array (.scalar .f32) (N*K))) slot
        let ev := Exp.index (Exp.var "tempk" : Exp (.array (.scalar .f32) ((N*K)*dim))) (Exp.add (Exp.mul slot (Exp.litU32 dim)) d)
        ShaderM.assign acc (Exp.add (Exp.var acc) (Exp.mul pv ev))
      ShaderM.writeBuffer (ty := .scalar .f32) "ssc" t (Exp.mul (Exp.litF32 embScale) (Exp.var acc)))
      (ShaderM.writeBuffer (ty := .scalar .f32) "ssc" t (Exp.litF32 0.0))) (pure ())

/-- Self-conditioning add (canvas rows only): a[t] += scUse * sig[t] for r≥P. -/
def addCanvasB (N P dim : Nat) (scUse : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let _a ← ShaderM.declareOutputBuffer "acanvas" (.array (.scalar .f32) (N*dim))
  let _s ← ShaderM.declareReadOnlyBuffer "sig" (.array (.scalar .f32) (N*dim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let r := Exp.div t (Exp.litU32 dim)
    ShaderM.if_ (Exp.ge r (Exp.litU32 P)) (do
      let av ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "acanvas" t
      let sv := Exp.index (Exp.var "sig" : Exp (.array (.scalar .f32) (N*dim))) t
      ShaderM.writeBuffer (ty := .scalar .f32) "acanvas" t (Exp.add av (Exp.mul (Exp.litF32 scUse) sv))) (pure ())) (pure ())

/-- Per-head qk-norm (RMS×weight) + partial RoPE (rope only first `nRotHalf` pairs); 1 thread per (pos,head). -/
def qkNormRopeB (N nHead hd nRotHalf : Nat) (theta eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
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
      let ang0 := Exp.mul pf freq
      -- range-reduce to [-π,π] so f32 cos/sin stay precise at high positions (large p·freq)
      let twoPi := Exp.litF32 6.283185307179586
      let angRed := Exp.sub ang0 (Exp.mul twoPi (Exp.round (Exp.div ang0 twoPi)))
      -- partial rope: pairs j ≥ nRotHalf pass through unrotated (ggml global layers rope only n_rot/2 pairs)
      let ang := Exp.select (Exp.lt j (Exp.litU32 nRotHalf)) angRed (Exp.litF32 0.0)
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

/-- Cross-position attention: wg per (head=wid.x, query=wid.y); scores·scale+region-mask+softmax+weighted-V. -/
def battnB (N P nHead hd nKV : Nat) (scale : Float) (ws : Nat := 256) : Hesper.WGSL.Monad.ShaderM Unit := do
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
    ShaderM.assignIndex "sS" j (Exp.select allowed (Exp.mul (Exp.var sc : Exp (.scalar .f32)) (Exp.litF32 scale)) (Exp.litF32 (-1e30)))
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

/-- Router prep per row: tmps[r,i]=(x[r,i]/rms(x[r]))·invSqrt·rscale[i]. 1 thread/row. -/
def routerPrepB (N dim : Nat) (invSqrt eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let r := Exp.vec3X gid
  let _x ← ShaderM.declareReadOnlyBuffer "xin" (.array (.scalar .f32) (N*dim))
  let _rs ← ShaderM.declareReadOnlyBuffer "rscale" (.array (.scalar .f32) dim)
  let _o ← ShaderM.declareOutputBuffer "tmps" (.array (.scalar .f32) (N*dim))
  ShaderM.if_ (Exp.lt r (Exp.litU32 N)) (do
    let base := Exp.mul r (Exp.litU32 dim)
    let ss ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "xin" : Exp (.array (.scalar .f32) (N*dim))) (Exp.add base d)
      ShaderM.assign ss (Exp.add (Exp.var ss) (Exp.mul v v))
    let inv := Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add (Exp.div (Exp.var ss) (Exp.litF32 dim.toFloat)) (Exp.litF32 eps)))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "xin" : Exp (.array (.scalar .f32) (N*dim))) (Exp.add base d)
      let rs := Exp.index (Exp.var "rscale" : Exp (.array (.scalar .f32) dim)) d
      ShaderM.assignIndex "tmps" (Exp.add base d) (Exp.mul (Exp.mul (Exp.mul v inv) (Exp.litF32 invSqrt)) rs)) (pure ())

/-- Router matmul per row: rlogits[r,e]=Σ_k rw[e,k]·tmps[r,k]. 1 thread per (r,e). -/
def routerMatVecB (N nExpert dim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _w ← ShaderM.declareReadOnlyBuffer "rw" (.array (.scalar .f32) (nExpert*dim))
  let _t ← ShaderM.declareReadOnlyBuffer "tmps" (.array (.scalar .f32) (N*dim))
  let _o ← ShaderM.declareOutputBuffer "rlogits" (.array (.scalar .f32) (N*nExpert))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*nExpert))) (do
    let r := Exp.div t (Exp.litU32 nExpert)
    let e := Exp.sub t (Exp.mul r (Exp.litU32 nExpert))
    let wbase := Exp.mul e (Exp.litU32 dim); let tbase := Exp.mul r (Exp.litU32 dim)
    let sum ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun k => do
      let wv := Exp.index (Exp.var "rw" : Exp (.array (.scalar .f32) (nExpert*dim))) (Exp.add wbase k)
      let tv := Exp.index (Exp.var "tmps" : Exp (.array (.scalar .f32) (N*dim))) (Exp.add tbase k)
      ShaderM.assign sum (Exp.add (Exp.var sum) (Exp.mul wv tv))
    ShaderM.assignIndex "rlogits" t (Exp.var sum : Exp (.scalar .f32))) (pure ())

/-- Softmax + top-nUsed per row: 1 workgroup per row (wid.x=r). -/
def top8B (N nExpert nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId; let lid ← ShaderM.localId
  let r := Exp.vec3X wid; let tid := Exp.vec3X lid
  let _rl ← ShaderM.declareReadOnlyBuffer "rlogits" (.array (.scalar .f32) (N*nExpert))
  let _idxs ← ShaderM.declareOutputBuffer "idxs" (.array (.scalar .u32) (N*nUsed))
  let _wts ← ShaderM.declareOutputBuffer "wts" (.array (.scalar .f32) (N*nUsed))
  ShaderM.sharedNamed "sp" (.array (.scalar .f32) nExpert)
  let rbase := Exp.mul r (Exp.litU32 nExpert)
  let obase := Exp.mul r (Exp.litU32 nUsed)
  ShaderM.if_ (Exp.lt tid (Exp.litU32 nExpert)) (do
    ShaderM.assignIndex "sp" tid (Exp.index (Exp.var "rlogits" : Exp (.array (.scalar .f32) (N*nExpert))) (Exp.add rbase tid))) (pure ())
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let mx ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      ShaderM.assign mx (Exp.max (Exp.var mx) (Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      ShaderM.assignIndex "sp" e (Exp.exp (Exp.sub (Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e) (Exp.var mx)))
    let sm ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      ShaderM.assign sm (Exp.add (Exp.var sm) (Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e))
    let wsum ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nUsed) (Exp.litU32 1) fun p => do
      let bi ← ShaderM.var (.scalar .u32) (Exp.litU32 0)
      let bv ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
      ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
        let v := Exp.index (Exp.var "sp" : Exp (.array (.scalar .f32) nExpert)) e
        ShaderM.if_ (Exp.gt v (Exp.var bv)) (do
          ShaderM.assign bv v; ShaderM.assign bi e) (pure ())
      let prob := Exp.div (Exp.var bv : Exp (.scalar .f32)) (Exp.var sm : Exp (.scalar .f32))
      ShaderM.assignIndex "idxs" (Exp.add obase p) (Exp.var bi : Exp (.scalar .u32))
      ShaderM.assignIndex "wts" (Exp.add obase p) prob
      ShaderM.assign wsum (Exp.add (Exp.var wsum) prob)
      ShaderM.assignIndex "sp" (Exp.var bi : Exp (.scalar .u32)) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nUsed) (Exp.litU32 1) fun p => do
      let w := Exp.index (Exp.var "wts" : Exp (.array (.scalar .f32) (N*nUsed))) (Exp.add obase p)
      ShaderM.assignIndex "wts" (Exp.add obase p) (Exp.div w (Exp.var wsum))) (pure ())

/-- Merged-GeGLU per row: eh[r,i]=gelu(gu[r,i])·gu[r,i+ff]. 1 thread per (r,i). -/
def gegluMergedB (N ff : Nat) (inOffset guElems : Nat := 0) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gE := if guElems == 0 then N*2*ff else guElems
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _gu ← ShaderM.declareInputBuffer "gu" (.array (.scalar .f32) gE)
  let _o ← ShaderM.declareOutputBuffer "eh" (.array (.scalar .f32) (N*ff))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*ff))) (do
    let rr := Exp.div t (Exp.litU32 ff)
    let i := Exp.sub t (Exp.mul rr (Exp.litU32 ff))
    let gbase := Exp.add (Exp.litU32 inOffset) (Exp.mul rr (Exp.litU32 (2*ff)))
    let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := gE) "gu" (Exp.add gbase i)
    let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := gE) "gu" (Exp.add (Exp.add gbase i) (Exp.litU32 ff))
    let g3 := Exp.mul g (Exp.mul g g)
    let inner := Exp.mul (Exp.litF32 0.7978845608) (Exp.add g (Exp.mul (Exp.litF32 0.044715) g3))
    let iC := Exp.max (Exp.litF32 (-10.0)) (Exp.min (Exp.litF32 10.0) inner)
    let gl := Exp.mul (Exp.mul (Exp.litF32 0.5) g) (Exp.add (Exp.litF32 1.0) (Exp.tanh iC))
    ShaderM.assignIndex "eh" t (Exp.mul gl u)) (pure ())

/-- expert-grouping gather: gathered[k,j] = srcQ8[idx[k], j]. 1 thread per (k,j) u32. -/
def gatherQ8B (totalTok q8size srcRows : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let flat := Exp.vec3X gid
  let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .u32) (srcRows*q8size))
  let _idx ← ShaderM.declareReadOnlyBuffer "idx" (.array (.scalar .u32) totalTok)
  let _out ← ShaderM.declareOutputBuffer "gathered" (.array (.scalar .u32) (totalTok*q8size))
  ShaderM.if_ (Exp.lt flat (Exp.litU32 (totalTok*q8size))) (do
    let k := Exp.div flat (Exp.litU32 q8size)
    let j := Exp.sub flat (Exp.mul k (Exp.litU32 q8size))
    let srcRow ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "idx" k
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := srcRows*q8size) "src" (Exp.add (Exp.mul srcRow (Exp.litU32 q8size)) j)
    ShaderM.writeBuffer (ty := .scalar .u32) "gathered" flat v) (pure ())

/-- expert-grouping scatter: dst[slot[k]·N·outDim + pos[k]·outDim + o] = gathered[k,o]; skip dummies
    (slot ≥ nUsed). 1 thread per (k,o). -/
def scatterGUB (totalTok outDim N nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let flat := Exp.vec3X gid
  let _g ← ShaderM.declareReadOnlyBuffer "gathered" (.array (.scalar .f32) (totalTok*outDim))
  let _pos ← ShaderM.declareReadOnlyBuffer "pos" (.array (.scalar .u32) totalTok)
  let _slot ← ShaderM.declareReadOnlyBuffer "slot" (.array (.scalar .u32) totalTok)
  let _out ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (nUsed*N*outDim))
  ShaderM.if_ (Exp.lt flat (Exp.litU32 (totalTok*outDim))) (do
    let k := Exp.div flat (Exp.litU32 outDim)
    let o := Exp.sub flat (Exp.mul k (Exp.litU32 outDim))
    let slotK ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "slot" k
    ShaderM.if_ (Exp.lt slotK (Exp.litU32 nUsed)) (do
      let posK ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "pos" k
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalTok*outDim) "gathered" flat
      let dstIdx := Exp.add (Exp.add (Exp.mul slotK (Exp.litU32 (N*outDim))) (Exp.mul posK (Exp.litU32 outDim))) o
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v) (pure ())) (pure ())

/-- Weighted accumulate per row: acc[r,i]+=wts[r,slot]·din[r,i]. 1 thread per (r,i). -/
def waccB (N dim slot nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _acc ← ShaderM.declareOutputBuffer "acc" (.array (.scalar .f32) (N*dim))
  let _d ← ShaderM.declareReadOnlyBuffer "din" (.array (.scalar .f32) (N*dim))
  let _w ← ShaderM.declareReadOnlyBuffer "wts" (.array (.scalar .f32) (N*nUsed))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let rr := Exp.div t (Exp.litU32 dim)
    let a ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "acc" t
    let dv := Exp.index (Exp.var "din" : Exp (.array (.scalar .f32) (N*dim))) t
    let w := Exp.index (Exp.var "wts" : Exp (.array (.scalar .f32) (N*nUsed))) (Exp.add (Exp.mul rr (Exp.litU32 nUsed)) (Exp.litU32 slot))
    ShaderM.writeBuffer (ty := .scalar .f32) "acc" t (Exp.add a (Exp.mul w dv))) (pure ())

/-- Zero buffer. -/
def zeroB (n : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let i := Exp.vec3X gid
  let _d ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) n)
  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (ShaderM.assignIndex "data" i (Exp.litF32 0.0)) (pure ())

/-- Canvas embedding rms-norm (no scale): for rows p≥P, normalize emb[p] in place. 1 thread/row. -/
def embNormCanvasK (N P dim : Nat) (eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let p := Exp.vec3X gid
  let _e ← ShaderM.declareOutputBuffer "emb" (.array (.scalar .f32) (N*dim))
  ShaderM.if_ (Exp.and (Exp.lt p (Exp.litU32 N)) (Exp.ge p (Exp.litU32 P))) (do
    let base := Exp.mul p (Exp.litU32 dim)
    let ss ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun d => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "emb" (Exp.add base d)
      ShaderM.assign ss (Exp.add (Exp.var ss) (Exp.mul v v))
    let inv := Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add (Exp.div (Exp.var ss) (Exp.litF32 dim.toFloat)) (Exp.litF32 eps)))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 dim) (Exp.litU32 1) fun d => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "emb" (Exp.add base d)
      ShaderM.writeBuffer (ty := .scalar .f32) "emb" (Exp.add base d) (Exp.mul v inv)) (pure ())

/-- Logit softcap in place: logits[i] = cap·tanh(clamp(logits[i]/cap, -15, 15)) (clamp avoids Metal tanh overflow). -/
def softcapB (n : Nat) (cap : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let i := Exp.vec3X gid
  let _x ← ShaderM.declareOutputBuffer "logits" (.array (.scalar .f32) n)
  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "logits" i
    let arg := Exp.max (Exp.litF32 (-15.0)) (Exp.min (Exp.litF32 15.0) (Exp.div x (Exp.litF32 cap)))
    ShaderM.writeBuffer (ty := .scalar .f32) "logits" i (Exp.mul (Exp.litF32 cap) (Exp.tanh arg))) (pure ())

/-- Copy one lm_head chunk's CANVAS logits into the persistent full-vocab buffer.
    dst[i*vocab + chunkOff + o] = src[(P+i)*lmChunk + o], for canvas i, vocab-slot o. -/
def copyCanvasLogitsB (N C P vocab lmChunk chunkOff : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _s ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) (N*lmChunk))
  let _d ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (C*vocab))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (C*lmChunk))) (do
    let i := Exp.div t (Exp.litU32 lmChunk)
    let o := Exp.sub t (Exp.mul i (Exp.litU32 lmChunk))
    let v := Exp.index (Exp.var "src" : Exp (.array (.scalar .f32) (N*lmChunk))) (Exp.add (Exp.mul (Exp.add i (Exp.litU32 P)) (Exp.litU32 lmChunk)) o)
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" (Exp.add (Exp.add (Exp.mul i (Exp.litU32 vocab)) (Exp.litU32 chunkOff)) o) v) (pure ())

/-- GPU argmax/softmax/top-K reduction over the full vocab (one workgroup per canvas
    position).  Replaces the per-step Lean scan over 262144×C with on-device reductions:
    K sequential max-passes (excluding specials <5 and already-found tokens) → top-K
    tokens + logits; one sum-pass → softmax denom.  Outputs the K tokens + K softmax
    probs + the denom per position.  `ws`=256 threads. -/
def reduceTopKB (C vocab K ws : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId; let lid ← ShaderM.localId
  let i := Exp.vec3X wid; let tid := Exp.vec3X lid
  let _l ← ShaderM.declareReadOnlyBuffer "logits" (.array (.scalar .f32) (C*vocab))
  let _od ← ShaderM.declareOutputBuffer "odenom" (.array (.scalar .f32) C)
  let _ot ← ShaderM.declareOutputBuffer "otok" (.array (.scalar .u32) (C*K))
  let _op ← ShaderM.declareOutputBuffer "oprob" (.array (.scalar .f32) (C*K))
  ShaderM.sharedNamed "sVal" (.array (.scalar .f32) ws)
  ShaderM.sharedNamed "sTok" (.array (.scalar .u32) ws)
  ShaderM.sharedNamed "sFound" (.array (.scalar .u32) K)
  ShaderM.sharedNamed "sLog" (.array (.scalar .f32) K)
  let rowBase := Exp.mul i (Exp.litU32 vocab)
  -- K sequential max-passes → top-K (excluding specials <5 and previously-found tokens)
  for k in [0:K] do
    let nm := s!"lmax{k}"; let nt := s!"ltok{k}"; let ni := s!"isf{k}"
    ShaderM.varNamed nm (.scalar .f32) (Exp.litF32 (-1e30))
    ShaderM.varNamed nt (.scalar .u32) (Exp.litU32 0)
    ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
      let l := Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (C*vocab))) (Exp.add rowBase v)
      ShaderM.varNamed ni (.scalar .u32) (Exp.litU32 0)
      for kk in [0:k] do
        let fv ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := K) "sFound" (Exp.litU32 kk)
        ShaderM.if_ (Exp.eq v fv) (ShaderM.assign ni (Exp.litU32 1)) (pure ())
      ShaderM.if_ (Exp.and (Exp.and (Exp.ge v (Exp.litU32 5)) (Exp.eq (Exp.var ni) (Exp.litU32 0))) (Exp.gt l (Exp.var nm))) (do
        ShaderM.assign nm l; ShaderM.assign nt v) (pure ())
    ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.var nm)
    ShaderM.writeWorkgroup (ty := .scalar .u32) "sTok" tid (Exp.var nt)
    ShaderM.barrier
    let mut stride := ws / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" tid
        let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.add tid (Exp.litU32 stride))
        ShaderM.if_ (Exp.gt bb a) (do
          ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid bb
          let bt ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := ws) "sTok" (Exp.add tid (Exp.litU32 stride))
          ShaderM.writeWorkgroup (ty := .scalar .u32) "sTok" tid bt) (pure ())) (pure ())
      ShaderM.barrier
      stride := stride / 2
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      let mt ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := ws) "sTok" (Exp.litU32 0)
      let mv ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.litU32 0)
      ShaderM.writeWorkgroup (ty := .scalar .u32) "sFound" (Exp.litU32 k) mt
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sLog" (Exp.litU32 k) mv) (pure ())
    ShaderM.barrier
  -- denom: Σ_{v≥5} exp(logit - top0logit)
  let mx ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := K) "sLog" (Exp.litU32 0)
  ShaderM.varNamed "lsum" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let l := Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (C*vocab))) (Exp.add rowBase v)
    ShaderM.if_ (Exp.ge v (Exp.litU32 5)) (ShaderM.assign "lsum" (Exp.add (Exp.var "lsum") (Exp.exp (Exp.sub l mx)))) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.var "lsum")
  ShaderM.barrier
  let mut stride := ws / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.add a bb)) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let denom ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "odenom" i denom
    for k in [0:K] do
      let tk ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := K) "sFound" (Exp.litU32 k)
      let lg ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := K) "sLog" (Exp.litU32 k)
      ShaderM.writeBuffer (ty := .scalar .u32) "otok" (Exp.add (Exp.mul i (Exp.litU32 K)) (Exp.litU32 k)) tk
      ShaderM.writeBuffer (ty := .scalar .f32) "oprob" (Exp.add (Exp.mul i (Exp.litU32 K)) (Exp.litU32 k)) (Exp.div (Exp.exp (Exp.sub lg mx)) denom)) (pure ())

/-- u32 little-endian ByteArray from a Nat array. -/
def u32Bytes (xs : Array Nat) : ByteArray := Id.run do
  let mut b := ByteArray.empty
  for x in xs do
    b := b.push (UInt8.ofNat (x % 256)) |>.push (UInt8.ofNat (x / 256 % 256)) |>.push (UInt8.ofNat (x / 65536 % 256)) |>.push (UInt8.ofNat (x / 16777216 % 256))
  return b

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

def disp2w (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (nx ny ws : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device k bufs { numWorkgroups := (nx,ny,1), workgroupSize := {x:=ws} } key r

/-- Batched matmul over N rows dispatching by the layer's quant format. -/
def qK (device : Device) (buf : Buffer) (N K : Nat) (key : UInt64) : IO Unit :=
  disp device (qActQ8K N K) (("data",buf)::List.nil) (N*(K/256)) key
def q80 (device : Device) (buf : Buffer) (N K : Nat) (key : UInt64) : IO Unit :=
  disp device (qActQ80 N K) (("data",buf)::List.nil) (N*(K/32)) key

def bmm (device : Device) (layer : Hesper.Layers.Linear.LinearLayer B C) (inB outB : Buffer) (N : Nat) (key : UInt64) : IO Unit := do
  -- Q4_K via Hesper's CUDA-style tiled dp4a MMQ kernel (MMQ5 at seqLen≥32) — int8
  -- dot products + threadgroup tiling, far faster than the f32-dequant matmul.
  if layer.quantFormat == .Q4_K then
    Hesper.Layers.Linear.forwardBatchDP4A device layer inB outB N
    return
  let cfg := layer.config
  let bufs := ("weights", layer.weightBuf)::("input", inB)::("output", outB)::List.nil
  let k := match layer.quantFormat with
    | .Q8_0 => Hesper.Layers.Linear.fusedQ8_0BatchKernel cfg N
    | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchKernel cfg N
    | .Q6_K => Hesper.Quantization.Q6_K.fusedQ6KBatchKernel cfg.inDim cfg.outDim N
    | _     => Hesper.Layers.Linear.fusedQ4KMBatchKernel cfg N
  disp2 device k bufs cfg.outDim N key

/-- Tied Q6_K lm_head over the FULL vocab (tiled ≤65535-row chunks → logitsCanvas),
    then a GPU reduce (`reduceTopKB`, one workgroup/position) for argmax/softmax/top-K —
    replaces the per-step 262144×C Lean scan.  Reads back only the tiny top-K result.
    Returns (canvasIdx, predToken, confidence)·masked, top-K tokens [C·K], top-K probs [C·K]. -/
def lmHeadArgmaxFullVocab (device : Device) (outputWeight sN sLogits logitsCanvas outDenom outTok outProb : Buffer)
    (dim vocabSize N C P : Nat) (cap : Float) (masked : Array Bool) (K : Nat := 8)
    : IO (Array (Nat × Nat × Float) × Array Nat × Array Float) := do
  let lmChunk := 32768
  let nChunks := (vocabSize + lmChunk - 1) / lmChunk
  for c in [0:nChunks] do
    Hesper.GPUBackend.beginBatch device
    disp2w device (Hesper.Layers.Linear.fusedQ6KBatchF32WarpKernel dim lmChunk N (c*lmChunk) vocabSize)
      (("weights", outputWeight)::("input", sN)::("output", sLogits)::List.nil) lmChunk N 32 (hash ("lmheadw", c))
    disp device (softcapB (N*lmChunk) cap) (("logits", sLogits)::List.nil) (N*lmChunk) (hash ("softcap", c))
    disp device (copyCanvasLogitsB N C P vocabSize lmChunk (c*lmChunk)) (("src",sLogits)::("dst",logitsCanvas)::List.nil) (C*lmChunk) (hash ("cpcv", c))
    Hesper.GPUBackend.endBatch device
  Hesper.GPUBackend.beginBatch device
  disp2 device (reduceTopKB C vocabSize K 256)
    (("logits",logitsCanvas)::("odenom",outDenom)::("otok",outTok)::("oprob",outProb)::List.nil) C 1 (hash "reducetopk")
  Hesper.GPUBackend.endBatch device
  let tokBytes ← mapBufferRead device outTok 0 (C*K*4).toUSize
  let probFlat ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outProb 0 (C*K*4).toUSize)
  unmapBuffer outProb
  let denomA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outDenom 0 (C*4).toUSize)
  unmapBuffer outDenom
  unmapBuffer outTok
  let mut ktokFlat : Array Nat := Array.replicate (C*K) 0
  for j in [0:C*K] do
    let b := j*4
    ktokFlat := ktokFlat.set! j (tokBytes[b]!.toNat ||| (tokBytes[b+1]!.toNat <<< 8) ||| (tokBytes[b+2]!.toNat <<< 16) ||| (tokBytes[b+3]!.toNat <<< 24))
  let mut cand : Array (Nat × Nat × Float) := #[]
  for i in [0:C] do
    if masked[i]! then cand := cand.push (i, ktokFlat[i*K]!, 1.0 / denomA[i]!)
  return (cand, ktokFlat, probFlat)

/-- Phase-1 diagnostic: compute canvas-position-0 lm_head logits (chunk 0, tokens 0..32768)
    with BOTH the f32 Q6_K kernel and the dp4a Q6_K kernel for the SAME raw hidden, and
    compare — tells us whether the dp4a argmax-flip is a bug (systematically off) or precision
    (near-tie flips). Tokens of interest: 9079='▁Paris', 506='▁the'. -/
def lmHeadDiag (device : Device)
    (finalNorm : Hesper.Layers.RMSNorm.RMSNorm (Hesper.GPUBackend.Buf Device) (Hesper.GPUBackend.CachedDispatch Device))
    (outputWeight cur : Buffer) (dim vocabSize N P : Nat) : IO Unit := do
  let lmChunk := 32768
  let sN ← mkBuf device (N*dim)
  let sL1 ← mkBuf device (N*lmChunk)
  let sL2 ← mkBuf device (N*lmChunk)
  let sNQ8 ← mkBuf device (N*(dim/32)*9)
  Hesper.GPUBackend.beginBatch device
  Hesper.Layers.RMSNorm.forward device finalNorm cur sN N
  Hesper.GPUBackend.endBatch device
  Hesper.GPUBackend.beginBatch device
  disp2 device (Hesper.Quantization.Q6_K.fusedQ6KBatchKernel dim lmChunk N 256 0 vocabSize)
    (("weights",outputWeight)::("input",sN)::("output",sL1)::List.nil) lmChunk N (hash "diagf32")
  disp2w device (Hesper.Layers.Linear.fusedQ6KBatchF32WarpKernel dim lmChunk N 0 vocabSize)
    (("weights",outputWeight)::("input",sN)::("output",sL2)::List.nil) lmChunk N 32 (hash "diagwarp")
  Hesper.GPUBackend.endBatch device
  let f32L ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sL1 (P*lmChunk*4).toUSize (lmChunk*4).toUSize)
  unmapBuffer sL1
  let dpL ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sL2 (P*lmChunk*4).toUSize (lmChunk*4).toUSize)
  unmapBuffer sL2
  let argmaxEx (a : Array Float) : Nat × Float := Id.run do
    let mut mx := -1e30; let mut am := 0
    for v in [5:lmChunk] do if a[v]! > mx then mx := a[v]!; am := v
    return (am, mx)
  let (f32am, f32mx) := argmaxEx f32L
  let (dpam, dpmx) := argmaxEx dpL
  let mut em := 0.0; let mut vv := 0.0
  for v in [0:lmChunk] do em := em + (f32L[v]! - dpL[v]!)^2; vv := vv + f32L[v]!^2
  let rel := 100.0 * (em/lmChunk.toFloat).sqrt / ((vv/lmChunk.toFloat).sqrt + 1e-9)
  IO.println s!"[lmdiag] f32 argmax={f32am} ({f32mx}) | dp4a argmax={dpam} ({dpmx})"
  IO.println s!"[lmdiag] Paris(9079): f32={f32L[9079]!} dp4a={dpL[9079]!} | the(506): f32={f32L[506]!} dp4a={dpL[506]!}"
  IO.println s!"[lmdiag] chunk-0 logit relRMS(dp4a vs f32)={rel}% → big=BUG, small=precision"

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  let prompt := (args.drop 1).head?.getD "The capital of France is"
  let nLayers := (args.drop 2).head?.bind (·.toNat?) |>.getD 30
  IO.println s!"[dg-decode] init + load; layers={nLayers}  prompt={prompt.quote}"
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  -- tokenize the real prompt → P; canvas C from config.  This model's special
  -- turn tokens are non-standard ('<|turn>'=105), so use the raw prompt
  -- (completion-style: the model denoises the canvas as a continuation).
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF (← Hesper.GGUF.loadGGUFHeader path)
  let promptTokens := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  let P := promptTokens.size
  let C := model.dg.canvasLength
  let N := P + C
  IO.println s!"[dg-decode] P={P} (prompt tokens) C={C} (canvas) N={N}"
  IO.println s!"[dg-decode] prompt token ids: {(promptTokens.extract 0 (min 20 P)).toList}"
  let dim := cfg.hiddenSize; let ffn := cfg.intermediateSize; let nHead := cfg.numAttentionHeads
  let eps := 1e-6
  -- per-layer out_scale (canvas) + enc_out_scale (prompt)
  let rd1 (bo : Option Buffer) : IO Float := match bo with
    | none => pure 1.0
    | some bf => do let a ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device bf 0 (4:Nat).toUSize); unmapBuffer bf; pure a[0]!
  let mut scales : Array Float := #[]
  let mut encScales : Array Float := #[]
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError "blk")
    scales := scales.push (← rd1 blk.outScale)
    encScales := encScales.push (← rd1 blk.encOutScale)
  -- preallocate [N, *]
  let a ← mkBuf device (N*dim); let b ← mkBuf device (N*dim)
  let sN ← mkBuf device (N*dim)
  let sQ ← mkBuf device (N*8192); let sK ← mkBuf device (N*2048); let sV ← mkBuf device (N*2048)
  let sQr ← mkBuf device (N*8192); let sKr ← mkBuf device (N*2048); let sVn ← mkBuf device (N*2048)
  let sCtx ← mkBuf device (N*8192); let sAO ← mkBuf device (N*dim); let sPA ← mkBuf device (N*dim)
  let sG ← mkBuf device (N*ffn); let sU ← mkBuf device (N*ffn); let sGe ← mkBuf device (N*ffn); let sD ← mkBuf device (N*dim)
  let sR ← mkBuf device (N*dim)
  let nExpert := cfg.numExperts; let nUsed := cfg.numExpertsUsed; let expFF := cfg.expertFFSize
  let invSqrt := 1.0 / Float.sqrt dim.toFloat
  let sCurMlp ← mkBuf device (N*dim); let sMoeN ← mkBuf device (N*dim); let sTmpS ← mkBuf device (N*dim)
  let sRLogits ← mkBuf device (N*nExpert); let sIdxs ← mkBuf device (N*nUsed); let sWts ← mkBuf device (N*nUsed)
  let sMoeAcc ← mkBuf device (N*dim)
  let sGateUps ← (List.range nUsed).mapM (fun _ => mkBuf device (N*2*expFF))  -- per-expert (race fix)
  let sEhs ← (List.range nUsed).mapM (fun _ => mkBuf device (N*expFF))        -- per-expert (race fix)
  let sDownEs ← (List.range nUsed).mapM (fun _ => mkBuf device (N*dim))  -- per-expert (was 1 reused buf — race test)
  let sCurMoe ← mkBuf device (N*dim); let sComb ← mkBuf device (N*dim)
  let sMoeNQ8 ← mkBuf device (N*(dim/32)*9)  -- Q8_1 of the MoE input for the dp4a expert matmul
  -- expert-grouping (fused gate/up) buffers — FIXED maxPadded ⇒ one shader for the run
  let totalTok := N*nUsed
  let q8size := (dim/32)*9
  let maxPadded := totalTok + 32*nExpert
  let sSortedPos ← mkBuf device maxPadded
  let sSortedSlot ← mkBuf device maxPadded
  let sTileExpert ← mkBuf device (maxPadded/32)
  let sGatheredQ8 ← mkBuf device (maxPadded*q8size)
  let sGatheredGU ← mkBuf device (maxPadded*2*expFF)
  let sGateUpAll ← mkBuf device (nUsed*N*2*expFF)
  -- DECODE LOOP: real prompt + canvas of mask tokens; diffusion confidence-commit
  let decodeSteps := (args.drop 3).head?.bind (·.toNat?) |>.getD 13
  let mut toks : Array Nat := promptTokens ++ Array.replicate C model.dg.maskTokenId
  let mut masked : Array Bool := Array.replicate C true
  let scK := 8   -- top-K tokens carrying the softmax mass for the SC soft embedding
  let mut scTok : Array Nat := Array.replicate (C*scK) model.dg.maskTokenId  -- prev step's top-K tokens
  let mut scProb : Array Float := Array.replicate (C*scK) 0.0                  -- their softmax probs
  let tokBuf ← mkBuf device N
  let scTokBuf ← mkBuf device (C*scK)   -- self-cond top-K gather token ids
  let scProbBuf ← mkBuf device (C*scK)  -- self-cond top-K softmax probs
  let sTempK ← mkBuf device (C*scK*dim) -- gathered raw embeddings of the top-K tokens
  let sSC ← mkBuf device (N*dim)        -- self-cond soft embedding
  let embScale := Float.sqrt dim.toFloat
  let embTable := model.inner.embedding.embeddingTable
  let lmN := min cfg.vocabSize 32768
  let cap := cfg.logitSoftcapScale
  let sLogits ← mkBuf device (N*lmN)
  let logitsCanvas ← mkBuf device (C*cfg.vocabSize)  -- full-vocab canvas logits for the GPU reduce
  let outDenom ← mkBuf device C
  let outTok ← mkBuf device (C*scK)
  let outProb ← mkBuf device (C*scK)
  IO.println s!"[dg-decode] {decodeSteps} steps, N={N} P={P} C={C}"
  for step in [0:decodeSteps] do
    let remaining := masked.foldl (fun acc b => if b then acc+1 else acc) 0
    if remaining > 0 then
      let t0 ← IO.monoMsNow
      writeBuffer device tokBuf 0 (u32Bytes toks)
      if step > 0 then
        writeBuffer device scTokBuf 0 (u32Bytes scTok)
        writeBuffer device scProbBuf 0 (← Hesper.Basic.floatArrayToBytes scProb)
      Hesper.GPUBackend.beginBatch device
      disp device (Hesper.Quantization.Q6_K.q6kEmbedGatherKernel N cfg.vocabSize dim embScale) (("token_ids",tokBuf)::("embedding_table",embTable)::("output",a)::List.nil) (N*dim) (hash "emb")
      -- self-conditioning: soft-embed the previous step's top-K prediction → SC-MLP → add to canvas (step>0)
      if step > 0 then
        match model.scPreNorm, model.scGate, model.scUp, model.scDown with
        | some scPN, some scG, some scU, some scD =>
          -- gather raw embeddings of the C·K top tokens, weighted-reduce by softmax probs → sSC
          disp device (Hesper.Quantization.Q6_K.q6kEmbedGatherKernel (C*scK) cfg.vocabSize dim 1.0) (("token_ids",scTokBuf)::("embedding_table",embTable)::("output",sTempK)::List.nil) (C*scK*dim) (hash "scgath")
          disp device (softReduceB N P C dim scK embScale) (("ssc",sSC)::("tempk",sTempK)::("prob",scProbBuf)::List.nil) (N*dim) (hash "scsoft")
          Hesper.Layers.RMSNorm.forward device scPN sSC sN N
          qK device sN N dim (hash "scqk")
          bmm device scG sN sG N (hash "scg")
          bmm device scU sN sU N (hash "scu")
          disp device (geluMulB (N*ffn)) (("gate",sG)::("up",sU)::("outp",sGe)::List.nil) (N*ffn) (hash "scgg")
          q80 device sGe N ffn (hash "scq80")
          bmm device scD sGe sD N (hash "scd")
          disp device (addCanvasB N P dim 1.0) (("acanvas",a)::("sig",sD)::List.nil) (N*dim) (hash "scadd")
        | _, _, _, _ => pure ()
      disp device (embNormCanvasK N P dim eps) (("emb",a)::List.nil) N (hash "embn")
      Hesper.GPUBackend.endBatch device
      let mut cur := a; let mut nxt := b
      let lpb := ((← IO.getEnv "DG_LPB").bind (·.toNat?)).getD 30  -- layers per GPU submission (1 batch/forward; ~12% faster)
      for li in [0:nLayers] do
        if li % lpb == 0 then Hesper.GPUBackend.beginBatch device
        let some blk := model.inner.blocks[li]? | throw (IO.userError "blk")
        let qDim := blk.attention.wO.config.inDim; let kvDim := blk.attention.wV.config.outDim
        let hd := qDim / nHead; let nKV := kvDim / hd
        let theta : Float := if li % 6 == 5 then 1000000.0 else 10000.0
        let nRotHalf := if li % 6 == 5 then 64 else hd/2
        -- attention
        Hesper.Layers.RMSNorm.forward device blk.attnNorm cur sN N
        qK device sN N dim (hash ("qN",li))
        bmm device blk.attention.wQ sN sQ N (hash ("wq",li))
        bmm device blk.attention.wK sN sK N (hash ("wk",li))
        bmm device blk.attention.wV sN sV N (hash ("wv",li))
        disp device (qkNormRopeB N nHead hd nRotHalf theta eps) (("qin",sQ)::("wnorm",blk.attention.qNormWeight)::("qout",sQr)::List.nil) (N*nHead) (hash ("qn",li))
        disp device (qkNormRopeB N nKV hd nRotHalf theta eps) (("qin",sK)::("wnorm",blk.attention.kNormWeight)::("qout",sKr)::List.nil) (N*nKV) (hash ("kn",li))
        disp device (vNormB N nKV hd eps) (("vin",sV)::("vout",sVn)::List.nil) (N*nKV) (hash ("vn",li))
        disp2 device (battnB N P nHead hd nKV 1.0) (("q",sQr)::("k",sKr)::("v",sVn)::("ctx",sCtx)::List.nil) nHead N (hash ("at",li))
        qK device sCtx N qDim (hash ("qCtx",li))
        bmm device blk.attention.wO sCtx sAO N (hash ("wo",li))
        Hesper.Layers.RMSNorm.forward device blk.postAttnNorm sAO sR N
        disp device (addB (N*dim)) (("ain",sR)::("bin",cur)::("outc",sPA)::List.nil) (N*dim) (hash ("ra",li))
        -- dense FFN
        Hesper.Layers.RMSNorm.forward device blk.ffnNorm sPA sN N
        qK device sN N dim (hash ("qNf",li))
        bmm device blk.ffn.gate sN sG N (hash ("g",li))
        bmm device blk.ffn.up sN sU N (hash ("u",li))
        disp device (geluMulB (N*ffn)) (("gate",sG)::("up",sU)::("outp",sGe)::List.nil) (N*ffn) (hash ("gg",li))
        q80 device sGe N ffn (hash ("qGe",li))
        bmm device blk.ffn.down sGe sD N (hash ("dn",li))
        let some mpn1 := blk.moePostNorm1 | throw (IO.userError "mpn1")
        Hesper.Layers.RMSNorm.forward device mpn1 sD sCurMlp N        -- curMlp
        -- MoE (router top-8 per row + batched experts)
        let some mpn2pre := blk.moePreNorm2 | throw (IO.userError "mpn2pre")
        let some mpn2post := blk.moePostNorm2 | throw (IO.userError "mpn2post")
        let some rW := blk.moeRouterWeight | throw (IO.userError "rW")
        let some rS := blk.moeRouterScale | throw (IO.userError "rS")
        let some guE := blk.moeGateUpExps | throw (IO.userError "guE")
        let some dnE := blk.moeDownExps | throw (IO.userError "dnE")
        Hesper.Layers.RMSNorm.forward device mpn2pre sPA sMoeN N
        disp device (routerPrepB N dim invSqrt eps) (("xin",sPA)::("rscale",rS)::("tmps",sTmpS)::List.nil) N (hash ("rp",li))
        disp device (routerMatVecB N nExpert dim) (("rw",rW)::("tmps",sTmpS)::("rlogits",sRLogits)::List.nil) (N*nExpert) (hash ("rm",li))
        disp2 device (top8B N nExpert nUsed) (("rlogits",sRLogits)::("idxs",sIdxs)::("wts",sWts)::List.nil) N 1 (hash ("t8",li))
        if step == 0 && (li == 0 || li == 1 || li == 5 || li == 10 || li == 15 || li == 22 || li == 29) && (← IO.getEnv "DG_RDIAG").isSome then
          Hesper.GPUBackend.endBatch device
          let idxB ← mapBufferRead device sIdxs 0 (N*nUsed*4).toUSize
          let rdU32 (a : ByteArray) (j : Nat) : Nat :=
            a[j*4]!.toNat ||| (a[j*4+1]!.toNat <<< 8) ||| (a[j*4+2]!.toNat <<< 16) ||| (a[j*4+3]!.toNat <<< 24)
          let mut idxSum := 0
          for j in [0:N*nUsed] do idxSum := idxSum + rdU32 idxB j
          let mut c0 : Array Nat := #[]
          for k in [0:nUsed] do c0 := c0.push (rdU32 idxB (P*nUsed + k))
          unmapBuffer sIdxs
          let logF ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sRLogits 0 (N*nExpert*4).toUSize)
          unmapBuffer sRLogits
          let mut lsum := 0.0
          for v in logF do lsum := lsum + v
          IO.println s!"[rdiag] L{li} canvas0 top8={c0} | idxSum={idxSum} | rlogitSum={lsum}"
          Hesper.GPUBackend.beginBatch device
        disp device (zeroB (N*dim)) (("data",sMoeAcc)::List.nil) (N*dim) (hash ("z",li))
        -- Q8_1-quantize the MoE input once for the dp4a expert matmul (replaces qK)
        disp2w device (Hesper.Layers.Linear.quantizeQ8_1BatchKernel dim N) (("input",sMoeN)::("output",sMoeNQ8)::List.nil) (dim/32) N 32 (hash ("qmoeq8",li))
        if (← IO.getEnv "DG_GROUP").isSome then
          -- expert-grouping (fused): sync, bucket+pad by expert, gather, ONE tile-expert MMQ5, scatter
          Hesper.GPUBackend.endBatch device
          let idxBg ← mapBufferRead device sIdxs 0 (totalTok*4).toUSize
          let rdU32g (a : ByteArray) (j : Nat) : Nat :=
            a[j*4]!.toNat ||| (a[j*4+1]!.toNat <<< 8) ||| (a[j*4+2]!.toNat <<< 16) ||| (a[j*4+3]!.toNat <<< 24)
          let mut bk : Array (Array Nat) := Array.replicate nExpert #[]
          for ps in [0:totalTok] do
            let ex := rdU32g idxBg ps
            bk := bk.set! ex ((bk[ex]!).push ps)
          unmapBuffer sIdxs
          let mut sPos : Array Nat := #[]; let mut sSlot : Array Nat := #[]; let mut sTile : Array Nat := #[]
          for ex in [0:nExpert] do
            let b := bk[ex]!
            if b.size > 0 then
              for psv in b do sPos := sPos.push (psv / nUsed); sSlot := sSlot.push (psv % nUsed)
              while sPos.size % 32 != 0 do sPos := sPos.push 0; sSlot := sSlot.push nUsed
              for _x in [0:(sPos.size / 32) - sTile.size] do sTile := sTile.push ex
          while sPos.size < maxPadded do sPos := sPos.push 0; sSlot := sSlot.push nUsed
          while sTile.size < maxPadded/32 do sTile := sTile.push 0
          writeBuffer device sSortedPos 0 (u32Bytes sPos)
          writeBuffer device sSortedSlot 0 (u32Bytes sSlot)
          writeBuffer device sTileExpert 0 (u32Bytes sTile)
          Hesper.GPUBackend.beginBatch device
          disp device (gatherQ8B maxPadded q8size N) (("src",sMoeNQ8)::("idx",sSortedPos)::("gathered",sGatheredQ8)::List.nil) (maxPadded*q8size) (hash ("gthr",li))
          disp2 device (Hesper.Layers.Linear.q4kMatmulBatchMMQ5Kernel { inDim:=dim, outDim:=2*expFF } maxPadded 0 0 maxPadded true nExpert)
            (("weights",guE)::("input_q8",sGatheredQ8)::("output",sGatheredGU)::("tileExpert",sTileExpert)::List.nil) ((2*expFF)/64) (maxPadded/32) (hash ("emm",li))
          disp device (scatterGUB maxPadded (2*expFF) N nUsed) (("gathered",sGatheredGU)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sGateUpAll)::List.nil) (maxPadded*2*expFF) (hash ("sctr",li))
          for e in [0:nUsed] do
            let sEh := sEhs[e]?.getD sMoeN
            disp device (gegluMergedB N expFF (e*N*2*expFF) (nUsed*N*2*expFF)) (("gu",sGateUpAll)::("eh",sEh)::List.nil) (N*expFF) (hash ("gm",li,e))
            q80 device sEh N expFF (hash ("qEh",li,e))
            let downExpKernel := match blk.ffn.down.quantFormat with
              | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
              | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
            let sDownE := sDownEs[e]?.getD sEh
            disp2w device downExpKernel (("weights",dnE)::("input",sEh)::("idxs",sIdxs)::("output",sDownE)::List.nil) dim N 32 (hash ("ed",li,e))
            disp device (waccB N dim e nUsed) (("acc",sMoeAcc)::("din",sDownE)::("wts",sWts)::List.nil) (N*dim) (hash ("wa",li,e))
        else do
          for e in [0:nUsed] do
            let sGateUp := sGateUps[e]?.getD sMoeN
            let sEh := sEhs[e]?.getD sMoeN
            disp2w device (Hesper.Layers.Linear.fusedQ4KMBatchExpertDP4ATiledKernel { inDim:=dim, outDim:=2*expFF } nExpert N nUsed e 4) (("weights",guE)::("input_q8",sMoeNQ8)::("idxs",sIdxs)::("output",sGateUp)::List.nil) ((2*expFF)/4) N 32 (hash ("eut",li,e))
            disp device (gegluMergedB N expFF) (("gu",sGateUp)::("eh",sEh)::List.nil) (N*expFF) (hash ("gm",li,e))
            q80 device sEh N expFF (hash ("qEh",li,e))
            let downExpKernel := match blk.ffn.down.quantFormat with
              | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
              | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
            let sDownE := sDownEs[e]?.getD sEh
            disp2w device downExpKernel (("weights",dnE)::("input",sEh)::("idxs",sIdxs)::("output",sDownE)::List.nil) dim N 32 (hash ("ed",li,e))
            disp device (waccB N dim e nUsed) (("acc",sMoeAcc)::("din",sDownE)::("wts",sWts)::List.nil) (N*dim) (hash ("wa",li,e))
        Hesper.Layers.RMSNorm.forward device mpn2post sMoeAcc sCurMoe N
        -- combine: curMlp + curMoe → postFFNNorm → +residual → ×out_scale
        disp device (addB (N*dim)) (("ain",sCurMlp)::("bin",sCurMoe)::("outc",sComb)::List.nil) (N*dim) (hash ("ad",li))
        Hesper.Layers.RMSNorm.forward device blk.postFFNNorm sComb sR N
        disp device (addB (N*dim)) (("ain",sR)::("bin",sPA)::("outc",nxt)::List.nil) (N*dim) (hash ("rc",li))
        disp device (scaleRegionB N P dim (scales[li]!) (encScales[li]!)) (("data",nxt)::List.nil) (N*dim) (hash ("sc",li))
        if li % lpb == lpb-1 || li == nLayers-1 then Hesper.GPUBackend.endBatch device
        let t := cur; cur := nxt; nxt := t
      let tFwd ← IO.monoMsNow
      if step == 0 && (← IO.getEnv "DG_LMDIAG").isSome then
        lmHeadDiag device model.inner.finalNorm model.inner.outputWeight cur dim cfg.vocabSize N P
      -- final norm + Q8 quant, then full-vocab tiled lm_head (helper, keeps `main` small)
      Hesper.GPUBackend.beginBatch device
      Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN N
      -- lm_head reads the RAW f32 hidden (no qK): confirmed "Paris.", slightly more
      -- accurate than Q8_K, and the planned f32-warp lm_head kernel reads f32 directly.
      Hesper.GPUBackend.endBatch device
      let (cand, ktokFlat, probFlat) ← lmHeadArgmaxFullVocab device model.inner.outputWeight sN sLogits logitsCanvas outDenom outTok outProb dim cfg.vocabSize N C P cfg.logitSoftcapScale masked scK
      let tLm ← IO.monoMsNow
      scTok := ktokFlat; scProb := probFlat   -- feed this step's top-K soft prediction into next step's SC
      let want := if step+1 ≥ decodeSteps then remaining else max 1 ((C + decodeSteps - 1) / decodeSteps)
      let k := min want cand.size
      let mut picked := Array.replicate cand.size false
      for _ in [0:k] do
        let mut bi := 0; let mut bv := -1.0
        for c in [0:cand.size] do
          let (_,_,cf) := cand[c]!
          if !picked[c]! && cf > bv then bv := cf; bi := c
        picked := picked.set! bi true
        let (ci, pred, _) := cand[bi]!
        toks := toks.set! (P+ci) pred
        masked := masked.set! ci false
      let t1 ← IO.monoMsNow
      IO.println s!"[dg-decode] step {step}: committed {k} | total {t1-t0}ms = emb+fwd {tFwd-t0}ms + lmhead+reduce {tLm-tFwd}ms + commit {t1-tLm}ms"
  let outIds := toks.extract P (P+C)
  IO.println s!"[dg-decode] first canvas IDs: {(outIds.extract 0 (min 24 outIds.size)).toList}"
  IO.println s!"[dg-decode] TEXT: {Hesper.Tokenizer.SentencePiece.decode tokenizer outIds}"