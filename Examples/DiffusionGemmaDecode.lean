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
def gegluMergedB (N ff : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _gu ← ShaderM.declareInputBuffer "gu" (.array (.scalar .f32) (N*2*ff))
  let _o ← ShaderM.declareOutputBuffer "eh" (.array (.scalar .f32) (N*ff))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*ff))) (do
    let rr := Exp.div t (Exp.litU32 ff)
    let i := Exp.sub t (Exp.mul rr (Exp.litU32 ff))
    let gbase := Exp.mul rr (Exp.litU32 (2*ff))
    let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*2*ff) "gu" (Exp.add gbase i)
    let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*2*ff) "gu" (Exp.add (Exp.add gbase i) (Exp.litU32 ff))
    let g3 := Exp.mul g (Exp.mul g g)
    let inner := Exp.mul (Exp.litF32 0.7978845608) (Exp.add g (Exp.mul (Exp.litF32 0.044715) g3))
    let iC := Exp.max (Exp.litF32 (-10.0)) (Exp.min (Exp.litF32 10.0) inner)
    let gl := Exp.mul (Exp.mul (Exp.litF32 0.5) g) (Exp.add (Exp.litF32 1.0) (Exp.tanh iC))
    ShaderM.assignIndex "eh" t (Exp.mul gl u)) (pure ())

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

/-- Batched matmul over N rows dispatching by the layer's quant format. -/
def qK (device : Device) (buf : Buffer) (N K : Nat) (key : UInt64) : IO Unit :=
  disp device (qActQ8K N K) (("data",buf)::List.nil) (N*(K/256)) key
def q80 (device : Device) (buf : Buffer) (N K : Nat) (key : UInt64) : IO Unit :=
  disp device (qActQ80 N K) (("data",buf)::List.nil) (N*(K/32)) key

def bmm (device : Device) (layer : Hesper.Layers.Linear.LinearLayer B C) (inB outB : Buffer) (N : Nat) (key : UInt64) : IO Unit := do
  let cfg := layer.config
  let bufs := ("weights", layer.weightBuf)::("input", inB)::("output", outB)::List.nil
  let k := match layer.quantFormat with
    | .Q8_0 => Hesper.Layers.Linear.fusedQ8_0BatchKernel cfg N
    | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchKernel cfg N
    | .Q6_K => Hesper.Quantization.Q6_K.fusedQ6KBatchKernel cfg.inDim cfg.outDim N
    | _     => Hesper.Layers.Linear.fusedQ4KMBatchKernel cfg N
  disp2 device k bufs cfg.outDim N key

/-- Online-softmax + top-K merge of one lm_head chunk's logits for one position (pure, so
    the triple-nested decode loop doesn't blow the elaborator's recursion depth).
    Tracks the running (max, denom) AND the top-K (logit, token) sorted descending —
    the top-K carry the softmax mass for the self-conditioning soft embedding. -/
def mergeChunkTopK (logits : Array Float) (base oStart oEnd cbase K : Nat)
    (mx0 sm0 : Float) (klog0 : Array Float) (ktok0 : Array Nat)
    : (Float × Float × Array Float × Array Nat) := Id.run do
  let mut mx := mx0; let mut sm := sm0
  let mut klog := klog0; let mut ktok := ktok0
  for o in [oStart:oEnd] do
    let l := logits[base+o]!
    let tok := cbase + o
    if l > mx then sm := sm * Float.exp (mx - l) + 1.0; mx := l
    else sm := sm + Float.exp (l - mx)
    if l > klog[K-1]! then
      let mut j := K - 1
      while j > 0 && klog[j-1]! < l do
        klog := klog.set! j (klog[j-1]!); ktok := ktok.set! j (ktok[j-1]!); j := j - 1
      klog := klog.set! j l; ktok := ktok.set! j tok
  return (mx, sm, klog, ktok)

/-- Tied Q6_K lm_head over the FULL vocab, tiled into ≤65535-row chunks (WebGPU
    workgroup limit).  Per masked position keeps an online-softmax running
    (max, argmax, denom) over NON-special tokens (≥5), so the prediction can land
    anywhere in the 262144 vocab (e.g. '▁'=236743) — matching ggml.  `sN` must
    already be finalNorm'd + Q8-quantized.  Returns (canvasIdx, predToken, confidence). -/
def lmHeadArgmaxFullVocab (device : Device) (outputWeight sN : Buffer)
    (dim vocabSize N C P : Nat) (cap : Float) (masked : Array Bool) (dumpFirst : Bool := false) (K : Nat := 8)
    : IO (Array (Nat × Nat × Float) × Array Nat × Array Float) := do
  let lmChunk := 32768
  let nChunks := (vocabSize + lmChunk - 1) / lmChunk
  let sLogits ← mkBuf device (N*lmChunk)
  let mut mxArr : Array Float := Array.replicate C (-1e30)
  let mut smArr : Array Float := Array.replicate C 0.0
  let mut klogArr : Array (Array Float) := Array.replicate C (Array.replicate K (-1e30))
  let mut ktokArr : Array (Array Nat) := Array.replicate C (Array.replicate K 0)
  let mut p0logits : Array Float := if dumpFirst then Array.replicate vocabSize 0.0 else #[]
  for c in [0:nChunks] do
    Hesper.GPUBackend.beginBatch device
    disp2 device (Hesper.Quantization.Q6_K.fusedQ6KBatchKernel dim lmChunk N 256 (c*lmChunk) vocabSize)
      (("weights", outputWeight)::("input", sN)::("output", sLogits)::List.nil) lmChunk N (hash ("lmhead", c))
    disp device (softcapB (N*lmChunk) cap) (("logits", sLogits)::List.nil) (N*lmChunk) (hash ("softcap", c))
    Hesper.GPUBackend.endBatch device
    let logits ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sLogits 0 (N*lmChunk*4).toUSize)
    unmapBuffer sLogits
    let oStart := if c == 0 then 5 else 0   -- skip specials 0..4 (only in chunk 0)
    let oEnd := min lmChunk (vocabSize - c*lmChunk)
    if dumpFirst then  -- accumulate canvas-position-0 (token P) full logits for golden verify
      for o in [0:oEnd] do p0logits := p0logits.set! (c*lmChunk+o) logits[P*lmChunk+o]!
    for i in [0:C] do   -- softmax(max,denom) + top-K over ALL canvas positions
      let (mx, sm, kl, kt) := mergeChunkTopK logits ((P+i)*lmChunk) oStart oEnd (c*lmChunk) K mxArr[i]! smArr[i]! klogArr[i]! ktokArr[i]!
      mxArr := mxArr.set! i mx; smArr := smArr.set! i sm; klogArr := klogArr.set! i kl; ktokArr := ktokArr.set! i kt
  if dumpFirst then
    IO.FS.writeBinFile "/tmp/my_logits_p0.bin" (← Hesper.Basic.floatArrayToBytes p0logits)
    IO.println s!"[verify] dumped canvas-0 full logits ({vocabSize}) to /tmp/my_logits_p0.bin"
  -- flatten top-K tokens + softmax probs (over the full vocab denom) for the SC soft embedding
  let mut ktokFlat : Array Nat := Array.replicate (C*K) 0
  let mut probFlat : Array Float := Array.replicate (C*K) 0.0
  let mut cand : Array (Nat × Nat × Float) := #[]
  for i in [0:C] do
    for k in [0:K] do
      ktokFlat := ktokFlat.set! (i*K+k) (ktokArr[i]![k]!)
      probFlat := probFlat.set! (i*K+k) (Float.exp (klogArr[i]![k]! - mxArr[i]!) / smArr[i]!)
    if masked[i]! then cand := cand.push (i, ktokArr[i]![0]!, 1.0 / smArr[i]!)
  return (cand, ktokFlat, probFlat)

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
  let sMoeAcc ← mkBuf device (N*dim); let sGateUp ← mkBuf device (N*2*expFF); let sEh ← mkBuf device (N*expFF)
  let sDownE ← mkBuf device (N*dim); let sCurMoe ← mkBuf device (N*dim); let sComb ← mkBuf device (N*dim)
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
      for li in [0:nLayers] do
        Hesper.GPUBackend.beginBatch device   -- one submission per layer (bounds peak encoder memory)
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
        disp device (zeroB (N*dim)) (("data",sMoeAcc)::List.nil) (N*dim) (hash ("z",li))
        qK device sMoeN N dim (hash ("qMoeN",li))
        for e in [0:nUsed] do
          disp2 device (Hesper.Layers.Linear.fusedQ4KMBatchExpertKernel { inDim:=dim, outDim:=2*expFF } nExpert N nUsed e) (("weights",guE)::("input",sMoeN)::("idxs",sIdxs)::("output",sGateUp)::List.nil) (2*expFF) N (hash ("eu",li,e))
          disp device (gegluMergedB N expFF) (("gu",sGateUp)::("eh",sEh)::List.nil) (N*expFF) (hash ("gm",li,e))
          q80 device sEh N expFF (hash ("qEh",li,e))
          let downExpKernel := match blk.ffn.down.quantFormat with
            | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
            | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
          disp2 device downExpKernel (("weights",dnE)::("input",sEh)::("idxs",sIdxs)::("output",sDownE)::List.nil) dim N (hash ("ed",li,e))
          disp device (waccB N dim e nUsed) (("acc",sMoeAcc)::("din",sDownE)::("wts",sWts)::List.nil) (N*dim) (hash ("wa",li,e))
        Hesper.Layers.RMSNorm.forward device mpn2post sMoeAcc sCurMoe N
        -- combine: curMlp + curMoe → postFFNNorm → +residual → ×out_scale
        disp device (addB (N*dim)) (("ain",sCurMlp)::("bin",sCurMoe)::("outc",sComb)::List.nil) (N*dim) (hash ("ad",li))
        Hesper.Layers.RMSNorm.forward device blk.postFFNNorm sComb sR N
        disp device (addB (N*dim)) (("ain",sR)::("bin",sPA)::("outc",nxt)::List.nil) (N*dim) (hash ("rc",li))
        disp device (scaleRegionB N P dim (scales[li]!) (encScales[li]!)) (("data",nxt)::List.nil) (N*dim) (hash ("sc",li))
        Hesper.GPUBackend.endBatch device
        let t := cur; cur := nxt; nxt := t
      -- final norm + Q8 quant, then full-vocab tiled lm_head (helper, keeps `main` small)
      Hesper.GPUBackend.beginBatch device
      Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN N
      qK device sN N dim (hash "qFinal")
      Hesper.GPUBackend.endBatch device
      let (cand, ktokFlat, probFlat) ← lmHeadArgmaxFullVocab device model.inner.outputWeight sN dim cfg.vocabSize N C P cfg.logitSoftcapScale masked (step == 0 && (← IO.getEnv "DG_VERIFY").isSome) scK
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
      IO.println s!"[dg-decode] step {step}: committed {k}, remaining {remaining-k} | forward+decode {t1-t0}ms ({(259000)/(max 1 (t1-t0))} pos-layer/s over 30L)"
  let outIds := toks.extract P (P+C)
  IO.println s!"[dg-decode] first canvas IDs: {(outIds.extract 0 (min 24 outIds.size)).toList}"
  IO.println s!"[dg-decode] TEXT: {Hesper.Tokenizer.SentencePiece.decode tokenizer outIds}"