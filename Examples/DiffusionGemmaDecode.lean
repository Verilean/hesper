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
  -- GUARD the write, don't select the value: with n=N*ffn=262·2112 not divisible by 256, the 128
  -- excess threads' writes CLAMP onto the last element and race the owner (last canvas position's
  -- activation flips real/0.0 per run → chaotic layer amplification → non-deterministic decode).
  ShaderM.if_ inB (ShaderM.writeBuffer (ty := .scalar .f32) "outp" i (Exp.mul gl u)) (pure ())

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
  -- guard the write (same excess-thread clamp-race class as geluMulB; here n=N*dim is currently
  -- 256-divisible so it's latent, but don't rely on dimensional luck)
  ShaderM.if_ inB (ShaderM.writeBuffer (ty := .scalar .f32) "outc" i (Exp.add av bv)) (pure ())

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

/-- Full-vocab SC softmax (llama.cpp fidelity): probs[i,v] = S·exp((logits[i,v]−max)/t)/Z over the
    ENTIRE vocab, one workgroup per canvas position (ws threads stride the vocab). t is read from
    tbuf[0] (annealed 0.8→0.4 per llama.cpp). S=1024 keeps tail probs in f16-normal range for the
    WMMA A-load; the SC RMSNorm right after is scale-invariant so S (and embScale) cancel. -/
def scSoftmaxTempB (Cc vocab ws : Nat) (S : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId; let lid ← ShaderM.localId
  let i := Exp.vec3X wid; let tid := Exp.vec3X lid
  let _l ← ShaderM.declareReadOnlyBuffer "logits" (.array (.scalar .f32) (Cc*vocab))
  let _t ← ShaderM.declareReadOnlyBuffer "tbuf" (.array (.scalar .f32) 4)
  let _p ← ShaderM.declareOutputBuffer "probs" (.array (.scalar .f32) (Cc*vocab))
  ShaderM.sharedNamed "sRed" (.array (.scalar .f32) ws)
  let rowBase := Exp.mul i (Exp.litU32 vocab)
  let tv := Exp.index (Exp.var "tbuf" : Exp (.array (.scalar .f32) 4)) (Exp.litU32 0)
  ShaderM.varNamed "inv" (.scalar .f32) (Exp.div (Exp.litF32 1.0) tv)
  -- pass 1: row max
  ShaderM.varNamed "m" (.scalar .f32) (Exp.litF32 (-1e30))
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let l := Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (Cc*vocab))) (Exp.add rowBase v)
    ShaderM.if_ (Exp.gt l (Exp.var "m")) (ShaderM.assign "m" l) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sRed" tid (Exp.var "m")
  ShaderM.barrier
  let mut stride := ws / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" (Exp.add tid (Exp.litU32 stride))
      ShaderM.if_ (Exp.gt b a) (ShaderM.writeWorkgroup (ty := .scalar .f32) "sRed" tid b) (pure ())) (pure ())
    ShaderM.barrier
    stride := stride / 2
  let mx ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" (Exp.litU32 0)
  ShaderM.varNamed "mxv" (.scalar .f32) mx
  ShaderM.barrier   -- all threads have read sRed[0] before pass 2 reuses the shared array
  -- pass 2: Z = Σ exp((l−mx)·inv)
  ShaderM.varNamed "s" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let l := Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (Cc*vocab))) (Exp.add rowBase v)
    ShaderM.assign "s" (Exp.add (Exp.var "s") (Exp.exp (Exp.mul (Exp.sub l (Exp.var "mxv")) (Exp.var "inv"))))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sRed" tid (Exp.var "s")
  ShaderM.barrier
  stride := ws / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sRed" tid (Exp.add a b)) (pure ())
    ShaderM.barrier
    stride := stride / 2
  let z ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sRed" (Exp.litU32 0)
  ShaderM.varNamed "sOverZ" (.scalar .f32) (Exp.div (Exp.litF32 S) (Exp.max z (Exp.litF32 1e-30)))
  -- pass 3: write scaled probs
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let l := Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (Cc*vocab))) (Exp.add rowBase v)
    ShaderM.writeBuffer (ty := .scalar .f32) "probs" (Exp.add rowBase v)
      (Exp.mul (Exp.exp (Exp.mul (Exp.sub l (Exp.var "mxv")) (Exp.var "inv"))) (Exp.var "sOverZ"))

/-- One-time bit-level transpose of the f16 embed/lm_head table W[vocab, dim/2 u32-pairs] →
    WT[dim, vocab/2 u32-pairs], so the full-SC `probs @ W` runs on the existing transpose-B WMMA
    kernel (B must be [N=dim, K=vocab] row-major). Pure u32 shuffles, bounds-guarded, 2D grid. -/
def transposeEmbF16B (vocab dim gridXWidth : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let flat := Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridXWidth))
  let nOut := dim * (vocab/2)
  let _s ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .u32) (vocab*(dim/2)))
  let _d ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .u32) nOut)
  ShaderM.if_ (Exp.lt flat (Exp.litU32 nOut)) (do
    let d := Exp.div flat (Exp.litU32 (vocab/2))
    let vp := Exp.mod flat (Exp.litU32 (vocab/2))
    let v0 := Exp.mul vp (Exp.litU32 2)
    let sh := Exp.mul (Exp.mod d (Exp.litU32 2)) (Exp.litU32 16)
    let dHalf := Exp.div d (Exp.litU32 2)
    let w0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := vocab*(dim/2)) "src" (Exp.add (Exp.mul v0 (Exp.litU32 (dim/2))) dHalf)
    let w1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := vocab*(dim/2)) "src" (Exp.add (Exp.mul (Exp.add v0 (Exp.litU32 1)) (Exp.litU32 (dim/2))) dHalf)
    let h0 := Exp.bitAnd (Exp.shiftRight w0 sh) (Exp.litU32 0xFFFF)
    let h1 := Exp.bitAnd (Exp.shiftRight w1 sh) (Exp.litU32 0xFFFF)
    ShaderM.writeBuffer (ty := .scalar .u32) "dst" flat (Exp.bitOr h0 (Exp.shiftLeft h1 (Exp.litU32 16)))) (pure ())

/-- Spread the full-SC matmul output into the N-row sSC layout: sSC[r,d] = r≥P ? sSCC[r−P,d] : 0. -/
def scSpreadB (N P dim : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let _o ← ShaderM.declareOutputBuffer "ssc" (.array (.scalar .f32) (N*dim))
  let _i ← ShaderM.declareReadOnlyBuffer "sccc" (.array (.scalar .f32) ((N-P)*dim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let r := Exp.div t (Exp.litU32 dim)
    let d := Exp.mod t (Exp.litU32 dim)
    ShaderM.if_ (Exp.ge r (Exp.litU32 P))
      (do
        let src := Exp.add (Exp.mul (Exp.sub r (Exp.litU32 P)) (Exp.litU32 dim)) d
        let v := Exp.index (Exp.var "sccc" : Exp (.array (.scalar .f32) ((N-P)*dim))) src
        ShaderM.writeBuffer (ty := .scalar .f32) "ssc" t v)
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

/-- GPU grouping 0: clear sortedPos=0, sortedSlot=nUsed (dummy) over the padded buffer. -/
def clearSortedB (maxPadded nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let k := Exp.vec3X gid
  let _sp ← ShaderM.declareOutputBuffer "sp" (.array (.scalar .u32) maxPadded)
  let _ss ← ShaderM.declareOutputBuffer "ss" (.array (.scalar .u32) maxPadded)
  ShaderM.if_ (Exp.lt k (Exp.litU32 maxPadded)) (do
    ShaderM.writeBuffer (ty := .scalar .u32) "sp" k (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .u32) "ss" k (Exp.litU32 nUsed)) (pure ())

/-- GPU grouping 1: count tokens per expert. 1 thread per expert. -/
def countExpB (totalTok nExpert : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let e := Exp.vec3X gid
  let _idx ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) totalTok)
  let _cnt ← ShaderM.declareOutputBuffer "cnt" (.array (.scalar .u32) nExpert)
  ShaderM.if_ (Exp.lt e (Exp.litU32 nExpert)) (do
    ShaderM.varNamed "c" (.scalar .u32) (Exp.litU32 0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 totalTok) (Exp.litU32 1) fun ps => do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "idxs" ps
      ShaderM.if_ (Exp.eq v e) (ShaderM.assign "c" (Exp.add (Exp.var "c") (Exp.litU32 1))) (pure ())
    ShaderM.writeBuffer (ty := .scalar .u32) "cnt" e (Exp.var "c")) (pure ())

/-- GPU grouping 2: prefix-sum padded offsets + fill tileExpert. Single thread (cheap). -/
def offsetsExpB (nExpert maxPadded : Nat) (padTo : Nat := 32) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _cnt ← ShaderM.declareReadOnlyBuffer "cnt" (.array (.scalar .u32) nExpert)
  let _off ← ShaderM.declareOutputBuffer "off" (.array (.scalar .u32) nExpert)
  let _te ← ShaderM.declareOutputBuffer "te" (.array (.scalar .u32) (maxPadded/32))
  -- trs[tile] = REAL rows in the tile (1-32; 0 for sentinel tiles): lets the reg kernels skip
  -- fully-phantom 8-row WMMA sub-tiles (avg ~17 real rows/expert ⇒ ~46% of the MMA is phantom;
  -- ~33% recoverable at 8-row granularity with the per-expert B-panel reuse untouched).
  let _trs ← ShaderM.declareOutputBuffer "trs" (.array (.scalar .u32) (maxPadded/32))
  ShaderM.if_ (Exp.eq t (Exp.litU32 0)) (do
    ShaderM.varNamed "acc" (.scalar .u32) (Exp.litU32 0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 nExpert) (Exp.litU32 1) fun e => do
      let c ← ShaderM.readBuffer (ty := .scalar .u32) (n := nExpert) "cnt" e
      let pc := Exp.mul (Exp.div (Exp.add c (Exp.litU32 (padTo-1))) (Exp.litU32 padTo)) (Exp.litU32 padTo)
      ShaderM.writeBuffer (ty := .scalar .u32) "off" e (Exp.var "acc")
      ShaderM.varNamed "tt" (.scalar .u32) (Exp.div (Exp.var "acc") (Exp.litU32 32))
      ShaderM.loop (Exp.litU32 0) (Exp.div pc (Exp.litU32 32)) (Exp.litU32 1) fun z => do
        ShaderM.writeBuffer (ty := .scalar .u32) "te" (Exp.var "tt") e
        let zRows := Exp.sub c (Exp.mul z (Exp.litU32 32))   -- c > z*32 for z < pc/32 ⇒ no underflow
        let real := Exp.select (Exp.lt zRows (Exp.litU32 32)) zRows (Exp.litU32 32)
        ShaderM.writeBuffer (ty := .scalar .u32) "trs" (Exp.var "tt") real
        ShaderM.assign "tt" (Exp.add (Exp.var "tt") (Exp.litU32 1))
      ShaderM.assign "acc" (Exp.add (Exp.var "acc") pc)
    ShaderM.loop (Exp.div (Exp.var "acc") (Exp.litU32 32)) (Exp.litU32 (maxPadded/32)) (Exp.litU32 1) fun tt => do
      ShaderM.writeBuffer (ty := .scalar .u32) "te" tt (Exp.litU32 nExpert)
      ShaderM.writeBuffer (ty := .scalar .u32) "trs" tt (Exp.litU32 0)) (pure ())

/-- GPU grouping 3: scatter each (pos,slot) to off[expert]+rank (rank = #earlier same-expert). -/
def scatterRankB (totalTok nExpert nUsed maxPadded : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let ps := Exp.vec3X gid
  let _idx ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) totalTok)
  let _off ← ShaderM.declareReadOnlyBuffer "off" (.array (.scalar .u32) nExpert)
  let _sp ← ShaderM.declareOutputBuffer "sp" (.array (.scalar .u32) maxPadded)
  let _ss ← ShaderM.declareOutputBuffer "ss" (.array (.scalar .u32) maxPadded)
  ShaderM.if_ (Exp.lt ps (Exp.litU32 totalTok)) (do
    let e ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "idxs" ps
    ShaderM.varNamed "rank" (.scalar .u32) (Exp.litU32 0)
    ShaderM.loop (Exp.litU32 0) ps (Exp.litU32 1) fun ps2 => do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "idxs" ps2
      ShaderM.if_ (Exp.eq v e) (ShaderM.assign "rank" (Exp.add (Exp.var "rank") (Exp.litU32 1))) (pure ())
    let o ← ShaderM.readBuffer (ty := .scalar .u32) (n := nExpert) "off" e
    let si := Exp.add o (Exp.var "rank")
    let pos := Exp.div ps (Exp.litU32 nUsed)
    ShaderM.writeBuffer (ty := .scalar .u32) "sp" si pos
    ShaderM.writeBuffer (ty := .scalar .u32) "ss" si (Exp.sub ps (Exp.mul pos (Exp.litU32 nUsed)))) (pure ())

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

/-- f32 expert-grouping gather (for the fused Q4_K reg gate/up): gathered[k,j] = src[idx[k], j].
    maxPadded·dim ≈ 29M elements > 65535·256 ⇒ needs a 2D grid (same trap as the down scatter). -/
def gatherF32B (totalTok dim srcRows : Nat) (gridXWidth : Nat := 0) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let flat := if gridXWidth > 0
    then Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridXWidth))
    else Exp.vec3X gid
  let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) (srcRows*dim))
  let _idx ← ShaderM.declareReadOnlyBuffer "idx" (.array (.scalar .u32) totalTok)
  let _out ← ShaderM.declareOutputBuffer "gathered" (.array (.scalar .f32) (totalTok*dim))
  ShaderM.if_ (Exp.lt flat (Exp.litU32 (totalTok*dim))) (do
    let k := Exp.div flat (Exp.litU32 dim)
    let j := Exp.sub flat (Exp.mul k (Exp.litU32 dim))
    let srcRow ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTok) "idx" k
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := srcRows*dim) "src" (Exp.add (Exp.mul srcRow (Exp.litU32 dim)) j)
    ShaderM.writeBuffer (ty := .scalar .f32) "gathered" flat v) (pure ())

/-- per-64 tileExpert for the reg gate/up: te64[t] = te32[2t] (each 64-row reg M-tile = 2 same-expert
    32-tiles, since DG_MOERB pads experts to 64). 1 thread per 64-tile. -/
def deriveTE64 (nTiles64 : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _te ← ShaderM.declareReadOnlyBuffer "te" (.array (.scalar .u32) (nTiles64*2))
  let _te64 ← ShaderM.declareOutputBuffer "te64" (.array (.scalar .u32) nTiles64)
  ShaderM.if_ (Exp.lt t (Exp.litU32 nTiles64)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := nTiles64*2) "te" (Exp.mul t (Exp.litU32 2))
    ShaderM.writeBuffer (ty := .scalar .u32) "te64" t v) (pure ())

/-- expert-grouping scatter: dst[slot[k]·N·outDim + pos[k]·outDim + o] = gathered[k,o]; skip dummies
    (slot ≥ nUsed). 1 thread per (k,o). -/
def scatterGUB (totalTok outDim N nUsed : Nat) (gridXWidth : Nat := 0) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let flat := if gridXWidth > 0
    then Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridXWidth))
    else Exp.vec3X gid
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

/-- Q8_0 → f32 dequant (for the dense-down f16 pre-dequant): out[e] = d(block)·q(int8).
    Row-major [od, idim], Q8_0 row stride = (idim/32)·34 bytes (f16 scale + 32 int8 per block).
    1 thread per element; the write is bounds-guarded (grid-roundup excess threads must not
    clamp-write — today's race class). 2D-grid aware via gridXWidth. -/
def q8ToF32B (od idim : Nat) (gridXWidth : Nat := 0) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let e := if gridXWidth == 0 then Exp.vec3X gid
           else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridXWidth))
  let ne := od * idim
  let rowBytes := (idim / 32) * 34
  let nU32 := (od * rowBytes + 3) / 4
  let _w ← ShaderM.declareReadOnlyBuffer "data" (.array (.scalar .u32) nU32)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) ne)
  ShaderM.if_ (Exp.lt e (Exp.litU32 ne)) (do
    let r := Exp.div e (Exp.litU32 idim)
    let c := Exp.mod e (Exp.litU32 idim)
    let bb := Exp.add (Exp.mul r (Exp.litU32 rowBytes)) (Exp.mul (Exp.div c (Exp.litU32 32)) (Exp.litU32 34))
    let rd := fun (bo : Exp (.scalar .u32)) => do
      let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := nU32) "data" (Exp.shiftRight bo (Exp.litU32 2))
      pure (Exp.bitAnd (Exp.shiftRight w (Exp.mul (Exp.bitAnd bo (Exp.litU32 3)) (Exp.litU32 8))) (Exp.litU32 0xFF))
    let lo ← rd bb
    let hi ← rd (Exp.add bb (Exp.litU32 1))
    let d := Hesper.Quantization.Q4_K_M.fp16ToF32 (Exp.add lo (Exp.mul hi (Exp.litU32 256)))
    let qb ← rd (Exp.add (Exp.add bb (Exp.litU32 2)) (Exp.mod c (Exp.litU32 32)))
    let q := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 (Exp.shiftRight qb (Exp.litU32 7))))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" e (Exp.mul d q)) (pure ())

/-- Pack f32 → f16 half2 u32: out[i] = pack2x16float(fin[2i], fin[2i+1]).  Used to convert a
    dequantized-to-f32 weight into the reg-matmul B format [outDim, inDim/2] u32. nOut = total f32/2. -/
def packF32ToF16B (nOut : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let i := Exp.vec3X gid
  let _in ← ShaderM.declareReadOnlyBuffer "fin" (.array (.scalar .f32) (nOut*2))
  let _out ← ShaderM.declareOutputBuffer "fout" (.array (.scalar .u32) nOut)
  ShaderM.if_ (Exp.lt i (Exp.litU32 nOut)) (do
    let a := Exp.index (Exp.var "fin" : Exp (.array (.scalar .f32) (nOut*2))) (Exp.mul i (Exp.litU32 2))
    let b := Exp.index (Exp.var "fin" : Exp (.array (.scalar .f32) (nOut*2))) (Exp.add (Exp.mul i (Exp.litU32 2)) (Exp.litU32 1))
    ShaderM.writeBuffer (ty := .scalar .u32) "fout" i (Exp.pack2x16float (Exp.vec2 a b))) (pure ())

/-- Combined weighted-accumulate over ALL nUsed experts in ONE pass (no 8-way race on `acc`):
    acc[pos,o] = Σ_slot wts[pos,slot] · din[slot,pos,o].  din = sDownAll [nUsed,N,dim]. -/
def waccAllB (N dim nUsed : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _d ← ShaderM.declareReadOnlyBuffer "din" (.array (.scalar .f32) (nUsed*N*dim))
  let _w ← ShaderM.declareReadOnlyBuffer "wts" (.array (.scalar .f32) (N*nUsed))
  let _acc ← ShaderM.declareOutputBuffer "acc" (.array (.scalar .f32) (N*dim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let rr := Exp.div t (Exp.litU32 dim)
    ShaderM.varNamed "sum" (.scalar .f32) (Exp.litF32 0.0)
    for slot in [0:nUsed] do
      let dv := Exp.index (Exp.var "din" : Exp (.array (.scalar .f32) (nUsed*N*dim))) (Exp.add (Exp.litU32 (slot*N*dim)) t)
      let w := Exp.index (Exp.var "wts" : Exp (.array (.scalar .f32) (N*nUsed))) (Exp.add (Exp.mul rr (Exp.litU32 nUsed)) (Exp.litU32 slot))
      ShaderM.assign "sum" (Exp.add (Exp.var "sum" : Exp (.scalar .f32)) (Exp.mul w dv))
    ShaderM.writeBuffer (ty := .scalar .f32) "acc" t (Exp.var "sum")) (pure ())

def waccB (N dim slot nUsed : Nat) (dinOffset dinElems : Nat := 0) : Hesper.WGSL.Monad.ShaderM Unit := do
  let dE := if dinElems == 0 then N*dim else dinElems
  let gid ← ShaderM.globalId; let t := Exp.vec3X gid
  let _acc ← ShaderM.declareOutputBuffer "acc" (.array (.scalar .f32) (N*dim))
  let _d ← ShaderM.declareReadOnlyBuffer "din" (.array (.scalar .f32) dE)
  let _w ← ShaderM.declareReadOnlyBuffer "wts" (.array (.scalar .f32) (N*nUsed))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*dim))) (do
    let rr := Exp.div t (Exp.litU32 dim)
    let a ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*dim) "acc" t
    let dv := Exp.index (Exp.var "din" : Exp (.array (.scalar .f32) dE)) (Exp.add (Exp.litU32 dinOffset) t)
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

/-- DG_MODE=renoise: per-position FULL-VOCAB sampler (llama.cpp EB fidelity, examples/diffusion/
    diffusion.cpp:582-607). One workgroup per canvas position i, over z = logits[i,·]·(1/t):
    argmax(z) → oamax; H(softmax z) = lnZ − Σ(z−m)e^{z−m}/Z → oh; multinomial = first v in VOCAB
    ORDER with cumulative e^{z−m} ≥ u·Z → osamp ("uin"[i] pre-drawn CPU-side, clamped > 0).
    t comes from "params"[0] (written per step — the 0.8→0.4 anneal). FULL vocab, no specials
    exclusion (llama.cpp samples the whole row). Grid = C workgroups exactly; all writes indexed
    by i < C (bounds-safe). Phases A/B stride for coalescing; phase C uses CONTIGUOUS per-thread
    chunks so the cumulative walk matches llama.cpp's sequential vocab order. -/
def ebSampleFullB (Cc vocab ws : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId; let lid ← ShaderM.localId
  let i := Exp.vec3X wid; let tid := Exp.vec3X lid
  let _l ← ShaderM.declareReadOnlyBuffer "logits" (.array (.scalar .f32) (Cc*vocab))
  let _u ← ShaderM.declareReadOnlyBuffer "uin" (.array (.scalar .f32) Cc)
  let _pb ← ShaderM.declareReadOnlyBuffer "params" (.array (.scalar .f32) 4)
  let _oa ← ShaderM.declareOutputBuffer "oamax" (.array (.scalar .u32) Cc)
  let _os ← ShaderM.declareOutputBuffer "osamp" (.array (.scalar .u32) Cc)
  let _oh ← ShaderM.declareOutputBuffer "oh" (.array (.scalar .f32) Cc)
  ShaderM.sharedNamed "sVal" (.array (.scalar .f32) ws)
  ShaderM.sharedNamed "sTok" (.array (.scalar .u32) ws)
  ShaderM.sharedNamed "sSum" (.array (.scalar .f32) ws)
  ShaderM.sharedNamed "sPfx" (.array (.scalar .f32) ws)
  ShaderM.sharedNamed "sPick" (.array (.scalar .u32) ws)
  ShaderM.sharedNamed "sScal" (.array (.scalar .f32) 2)
  let rowBase := Exp.mul i (Exp.litU32 vocab)
  let tv := Exp.index (Exp.var "params" : Exp (.array (.scalar .f32) 4)) (Exp.litU32 0)
  ShaderM.varNamed "tinv" (.scalar .f32) (Exp.div (Exp.litF32 1.0) tv)
  let zAt := fun (v : Exp (.scalar .u32)) =>
    Exp.mul (Exp.index (Exp.var "logits" : Exp (.array (.scalar .f32) (Cc*vocab))) (Exp.add rowBase v)) (Exp.var "tinv")
  -- Phase A: max + argmax of z (strided, coalesced)
  ShaderM.varNamed "lm" (.scalar .f32) (Exp.litF32 (-1e30))
  ShaderM.varNamed "lt" (.scalar .u32) (Exp.litU32 0)
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let z := zAt v
    ShaderM.if_ (Exp.gt z (Exp.var "lm")) (do
      ShaderM.assign "lm" z; ShaderM.assign "lt" v) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.var "lm")
  ShaderM.writeWorkgroup (ty := .scalar .u32) "sTok" tid (Exp.var "lt")
  ShaderM.barrier
  let mut strideA := ws / 2
  while strideA > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 strideA)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.add tid (Exp.litU32 strideA))
      ShaderM.if_ (Exp.gt bb a) (do
        ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid bb
        let bt ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := ws) "sTok" (Exp.add tid (Exp.litU32 strideA))
        ShaderM.writeWorkgroup (ty := .scalar .u32) "sTok" tid bt) (pure ())) (pure ())
    ShaderM.barrier
    strideA := strideA / 2
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let mv ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.litU32 0)
    let mt ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := ws) "sTok" (Exp.litU32 0)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "sScal" (Exp.litU32 0) mv
    ShaderM.writeBuffer (ty := .scalar .u32) "oamax" i mt) (pure ())
  ShaderM.barrier
  let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 2) "sScal" (Exp.litU32 0)
  -- Phase B: Z = Σe^{z−m} and SZ = Σ(z−m)e^{z−m} (strided; H = lnZ − SZ/Z)
  ShaderM.varNamed "se" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "sz" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 vocab) (Exp.litU32 ws) fun v => do
    let d := Exp.sub (zAt v) m
    let e := Exp.exp d
    ShaderM.assign "se" (Exp.add (Exp.var "se") e)
    ShaderM.assign "sz" (Exp.add (Exp.var "sz") (Exp.mul d e))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.var "se")
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sSum" tid (Exp.var "sz")
  ShaderM.barrier
  let mut strideB := ws / 2
  while strideB > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 strideB)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.add tid (Exp.litU32 strideB))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.add a bb)
      let a2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sSum" tid
      let b2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sSum" (Exp.add tid (Exp.litU32 strideB))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sSum" tid (Exp.add a2 b2)) (pure ())
    ShaderM.barrier
    strideB := strideB / 2
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let zz ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" (Exp.litU32 0)
    let szz ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sSum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "oh" i (Exp.sub (Exp.log zz) (Exp.div szz zz))
    ShaderM.writeWorkgroup (ty := .scalar .f32) "sScal" (Exp.litU32 1) zz) (pure ())
  ShaderM.barrier
  let zAll ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 2) "sScal" (Exp.litU32 1)
  let target := Exp.mul (Exp.index (Exp.var "uin" : Exp (.array (.scalar .f32) Cc)) i) zAll
  -- Phase C: multinomial via contiguous chunks (vocab order = llama.cpp's sequential walk)
  let chunk := (vocab + ws - 1) / ws
  ShaderM.varNamed "cs" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 chunk) (Exp.litU32 1) fun j => do
    let v := Exp.add (Exp.mul tid (Exp.litU32 chunk)) j
    ShaderM.if_ (Exp.lt v (Exp.litU32 vocab))
      (ShaderM.assign "cs" (Exp.add (Exp.var "cs") (Exp.exp (Exp.sub (zAt v) m)))) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "sVal" tid (Exp.var "cs")
  ShaderM.writeWorkgroup (ty := .scalar .u32) "sPick" tid (Exp.litU32 0xFFFFFFFF)
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 ws) (Exp.litU32 1) fun k => do
      ShaderM.writeWorkgroup (ty := .scalar .f32) "sPfx" k (Exp.var "acc")
      let ck ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" k
      ShaderM.assign "acc" (Exp.add (Exp.var "acc") ck)) (pure ())
  ShaderM.barrier
  let myStart ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sPfx" tid
  let myCs ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := ws) "sVal" tid
  ShaderM.if_ (Exp.and (Exp.ge (Exp.add myStart myCs) target) (Exp.lt myStart target)) (do
    ShaderM.varNamed "cum" (.scalar .f32) myStart
    ShaderM.varNamed "done" (.scalar .u32) (Exp.litU32 0)
    ShaderM.varNamed "pv" (.scalar .u32) (Exp.litU32 (vocab - 1))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 chunk) (Exp.litU32 1) fun j => do
      let v := Exp.add (Exp.mul tid (Exp.litU32 chunk)) j
      ShaderM.if_ (Exp.and (Exp.lt v (Exp.litU32 vocab)) (Exp.eq (Exp.var "done") (Exp.litU32 0))) (do
        ShaderM.assign "cum" (Exp.add (Exp.var "cum") (Exp.exp (Exp.sub (zAt v) m)))
        ShaderM.if_ (Exp.ge (Exp.var "cum") target) (do
          ShaderM.assign "pv" v; ShaderM.assign "done" (Exp.litU32 1)) (pure ())) (pure ())
    ShaderM.writeWorkgroup (ty := .scalar .u32) "sPick" tid (Exp.var "pv")) (pure ())
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.varNamed "res" (.scalar .u32) (Exp.litU32 (vocab - 1))
    ShaderM.varNamed "found" (.scalar .u32) (Exp.litU32 0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 ws) (Exp.litU32 1) fun k => do
      let pk ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := ws) "sPick" k
      ShaderM.if_ (Exp.eq (Exp.var "found") (Exp.litU32 0))
        (ShaderM.if_ (Exp.eq pk (Exp.litU32 0xFFFFFFFF)) (pure ()) (do
          ShaderM.assign "res" pk; ShaderM.assign "found" (Exp.litU32 1))) (pure ())
    ShaderM.writeBuffer (ty := .scalar .u32) "osamp" i (Exp.var "res")) (pure ())

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

/-- Dispatch a subgroup-matrix (WMMA) kernel: 128-thread workgroups + the f16/subgroup_matrix
    extensions + the uniformity diagnostic the register-blocked matmul needs. -/
def dispRB (device : Device) (k : Hesper.WGSL.Monad.ShaderM Unit) (bufs : List (String × Buffer)) (nx ny : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device k bufs
    { numWorkgroups := (nx,ny,1), workgroupSize := {x:=128},
      extensions := ["f16","chromium_experimental_subgroup_matrix"],
      diagnostics := [("off","chromium.subgroup_matrix_uniformity")] } key r

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
def lmHeadArgmaxFullVocab (device : Device) (outputWeightF16 sN sLogits logitsCanvas outDenom outTok outProb : Buffer)
    (dim vocabSize N C P : Nat) (cap : Float) (masked : Array Bool) (K : Nat := 8)
    : IO (Array (Nat × Nat × Float) × Array Nat × Array Float) := do
  let lmChunk := 32768
  let nChunks := (vocabSize + lmChunk - 1) / lmChunk
  for c in [0:nChunks] do
    Hesper.GPUBackend.beginBatch device
    -- register-blocked WMMA matmul (f16 weight): logits[N, lmChunk] for this vocab chunk
    dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := ((N+63)/64)*64, N := lmChunk, K := dim } (c*lmChunk) vocabSize)
      (("a", sN)::("b", outputWeightF16)::("c", sLogits)::List.nil) ((lmChunk+31)/32) ((N+63)/64) (hash ("lmheadrb", c))
    -- batch split after the subgroup-matrix kernel: Dawn-on-Metal drops the inter-pass barrier after
    -- WMMA dispatches at scale (same pattern as the QKV/attnO/dense reg calls). Without it, softcap/
    -- copyCanvas read STALE sLogits → every canvas position argmaxes to the same garbage token
    -- (the <unused6226>×N failure). Hidden states were verified clean — the corruption was here.
    Hesper.WGSL.Execute.flushBatch device
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
  -- tokenize the real prompt → P; canvas C from config.
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF (← Hesper.GGUF.loadGGUFHeader path)
  -- DG_TEMPLATE=1: llama.cpp-parity input. llama-diffusion-cli applies the GGUF chat template
  -- (rendered, enable_thinking=true):  <bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{P}
  -- <turn|>\n<|turn>model\n   with specials parsed to single ids (<bos>=2 <|think|>=98 <|turn>=105
  -- <turn|>=106, from the GGUF vocab) and add_bos_token=true. Without it (default) we feed the raw
  -- prompt completion-style — which llama.cpp never does, and which makes the model commit
  -- confidently-wrong framings on knowledge prompts (Au/Shakespeare/Armstrong all fail raw,
  -- all pass templated in llama.cpp).
  let enc := Hesper.Tokenizer.SentencePiece.encode tokenizer
  -- segment encoder for the template: encode WITHOUT the tokenizer's automatic BOS (the template
  -- carries exactly one explicit BOS up front, like llama.cpp's add_special on the whole string)
  let encRaw := fun (s : String) =>
    let t := enc s
    if t.size > 0 && t[0]! == 2 then t.extract 1 t.size else t
  let promptTokens ←
    if (← IO.getEnv "DG_NOTEMPLATE").isNone then   -- DEFAULT ON (8/8 vs raw 5/8); DG_NOTEMPLATE=1 opts out
      pure <| #[2, 105] ++ encRaw "system\n" ++ #[98] ++ encRaw "\n" ++ #[106] ++ encRaw "\n"
        ++ #[105] ++ encRaw ("user\n" ++ prompt) ++ #[106] ++ encRaw "\n" ++ #[105] ++ encRaw "model\n"
    else
      pure <| enc prompt
  let P := promptTokens.size
  -- DG_CANVAS=<n>: override the canvas length (default = GGUF diffusion.canvas_length, 256).
  -- Bigger canvas ⇒ bigger M ⇒ matmuls move from the medium-M ceiling (~47%) into the large-M band
  -- (67-91%) and per-step weight reads amortize over more tokens — the in-step-parallel throughput
  -- lever behind the big diffusion-TPS demos. NOTE: logitsCanvas is [C, vocab] f32 (~1GB at C=1024).
  let C := ((← IO.getEnv "DG_CANVAS").bind (·.toNat?)).getD model.dg.canvasLength
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
  -- preallocate [N, *].  Buffers read ("a") or written ("c") by the WMMA reg-matmuls are padded to a
  -- 64-row multiple: the reg grid runs ceil(M/64)·64 rows and `subgroupMatrixStore` writes the tail
  -- tiles PAST row M — an OOB write into whatever Dawn allocated next (heap stomp → non-deterministic
  -- garbage). Padding keeps those tail-tile writes inside this allocation, where they're ignored.
  let nP := ((N + 63) / 64) * 64
  let a ← mkBuf device (nP*dim); let b ← mkBuf device (nP*dim)
  let sN ← mkBuf device (nP*dim)
  let sQ ← mkBuf device (nP*8192); let sK ← mkBuf device (nP*2048); let sV ← mkBuf device (nP*2048)
  let sQr ← mkBuf device (nP*8192); let sKr ← mkBuf device (nP*2048); let sVn ← mkBuf device (nP*2048)
  let sCtx ← mkBuf device (nP*8192); let sAO ← mkBuf device (nP*dim); let sPA ← mkBuf device (nP*dim)
  let sG ← mkBuf device (nP*ffn); let sU ← mkBuf device (nP*ffn); let sGe ← mkBuf device (nP*ffn); let sD ← mkBuf device (nP*dim)
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
  -- DG_MOERB: fused Q4_K reg-matmul for the MoE gate/up. The reg M-tile is 64 ⇒ each expert block must
  -- be 64-aligned ⇒ pad to 64 (vs 32 for MMQ). Gated: the default MMQ path keeps its 32-pad.
  let moeRB := (← IO.getEnv "DG_NOMOERB").isNone   -- DEFAULT ON (validated); DG_NOMOERB=1 opts out
  -- DG_MOENOFLUSH: drop the MoE chain's per-layer flushBatch splits. Those were added to fight a
  -- "Dawn drops barriers at scale" race — but that race may have been the kill-9 GPU wedge / swap
  -- corruption all along (post-reboot experiment). If correct without them: faster AND simpler.
  let moeNoFlush := (← IO.getEnv "DG_MOENOFLUSH").isSome
  -- DG_SKIP_*: drop one MoE component's dispatch(es) to measure its REAL cost = emb+fwd delta vs
  -- baseline (no endBatch sync, unlike DG_PROF). Output is garbage; we only read the timing.
  let skGrp := (← IO.getEnv "DG_SKIP_GROUP").isSome; let skGat := (← IO.getEnv "DG_SKIP_GATHER").isSome
  let skGU := (← IO.getEnv "DG_SKIP_GATEUP").isSome; let skGeg := (← IO.getEnv "DG_SKIP_GEGLU").isSome
  let skQ80 := (← IO.getEnv "DG_SKIP_Q80").isSome; let skDn := (← IO.getEnv "DG_SKIP_DOWN").isSome
  let skSc := (← IO.getEnv "DG_SKIP_SCATTER").isSome; let skWa := (← IO.getEnv "DG_SKIP_WACC").isSome
  -- DG_MOEDOWNRB: tile the MoE down with the fused Q8_0 reg-matmul (matrix units) instead of the
  -- warp-per-output kernel — the down was the biggest MoE cost (DG_SKIP: ~600ms, ~7× less efficient
  -- per FLOP than the tiled gate/up). Uses the grouped down structure (geglu grouped → reg → scatter).
  let moeDownRB := (← IO.getEnv "DG_NOMOEDOWNRB").isNone   -- DEFAULT ON (validated); DG_NOMOEDOWNRB=1 opts out
  -- DG_DENSEDOWNRB: tile the DENSE FFN down with the fused Q8_0 reg-matmul (single weight ⇒ nExpert=1,
  -- a zero tileExpert). The dense down was also on the warp kernel (~186ms). No grouping ⇒ no grouped-
  -- path issues; reuses the golden-validated q8MatmulGroupedRegKernel.
  let denseDownRB := (← IO.getEnv "DG_DENSEDOWNRB").isSome
  let sZeroTE ← mkBuf device 16
  writeBuffer device sZeroTE 0 (← Hesper.Basic.floatArrayToBytes (Array.replicate 16 (0.0:Float)))
  let padTo := 32   -- BM=32 fused reg aligns with the 32-pad grouping (no 64-pad penalty)
  let maxPadded := (((totalTok + padTo*nExpert) + (padTo-1))/padTo)*padTo
  let sSortedPos ← mkBuf device maxPadded
  let sSortedSlot ← mkBuf device maxPadded
  let sTileExpert ← mkBuf device (maxPadded/32)
  -- ragged 8-row sub-tile skip: per-tile REAL row count from offsetsExpB. DG_NORAGGED=1 binds the
  -- all-32 buffer instead → same kernel computes every row (the clean A/B for the ragged skip).
  let sTileRows ← mkBuf device (maxPadded/32)
  let sTileRowsFull ← mkBuf device (maxPadded/32)
  writeBuffer device sTileRowsFull 0 (u32Bytes (Array.replicate (maxPadded/32) 32))
  let raggedRows := if (← IO.getEnv "DG_NORAGGED").isSome then sTileRowsFull else sTileRows
  let sTileExpert64 ← mkBuf device (if moeRB then maxPadded/64 else 1)   -- per-64 tileExpert for the reg gate/up
  let sGatheredF32 ← mkBuf device (if moeRB then maxPadded*dim else 1)   -- f32 gathered MoE input (reg A)
  let sGatheredQ8 ← mkBuf device (maxPadded*q8size)
  let sGatheredGU ← mkBuf device (maxPadded*2*expFF)
  let sGateUpAll ← mkBuf device (nUsed*N*2*expFF)
  let sGatheredEh ← mkBuf device (maxPadded*expFF)   -- grouped geglu output (down input)
  let sGatheredDown ← mkBuf device (maxPadded*dim)   -- grouped down output
  let sDownAll ← mkBuf device (nUsed*N*dim)          -- scattered down [slot,N,dim] for wacc
  let sExpertCount ← mkBuf device nExpert
  let sExpertOffset ← mkBuf device nExpert
  -- DECODE LOOP: real prompt + canvas of mask tokens; diffusion confidence-commit
  -- default steps: 32 for the mask-commit schedule (gentle floor, CONF bulk-commit keeps EFFECTIVE
  -- steps low); 13 for the entropy-bound denoiser (its t-annealing spans S — S=13 converges clean,
  -- S=32 anneals too slowly, matching llama.cpp's --diffusion-steps 13 reference run).
  let decodeSteps := (args.drop 3).head?.bind (·.toNat?)
    |>.getD (if ((← IO.getEnv "DG_SCHED").getD "fixed") == "eb" || ((← IO.getEnv "DG_MODE").getD "renoise") == "renoise" then 48 else 32)
    -- eb/renoise anneal horizon S = 48 = llama.cpp's max_denoising_steps default (diffusion.h:68).
    -- S=20 annealed t 2.4× too fast → softmax over-sharpened → meanH crossed 0.005 prematurely on
    -- hard prompts (Jupiter stopped "confident" at step 7 in a WRONG canvas). The stop rule, not the
    -- cap, ends the run early (France ~10 steps); S only sets the anneal slope + worst-case bound.
  -- DG_SCHED=eb: llama.cpp's ENTROPY-BOUND denoiser (examples/diffusion/diffusion.cpp:442) — the
  -- canvas is RANDOM-initialized (not masks) and re-noised every step: per position compute the
  -- temperature-adjusted distribution, its entropy H, the argmax and a multinomial sample; ACCEPT the
  -- lowest-entropy positions whose strictly-earlier cumulative H ≤ entropy_bound (accepted keep their
  -- sampled token, the rest get a fresh RANDOM token); stop when the argmax canvas is stable for
  -- `stability` steps AND mean H < confidence threshold. Output = the argmax canvas. Nothing is ever
  -- irreversibly committed — llama.cpp finishes France in 8 steps with this. We approximate the
  -- per-position distribution from the GPU top-K (K=8) probs: q ∝ p^(1/t) renormalized (accepted
  -- positions are low-entropy ⇒ top-8 carries ≈all their mass; high-H positions are rejected anyway).
  -- DG_MODE=renoise: the FULL llama.cpp decode mode — the eb machinery below but with the
  -- per-position distribution/entropy/multinomial computed on GPU over the FULL vocab
  -- (ebSampleFullB) instead of the top-8+tail approximation, SC = full-vocab softmax(prev
  -- logits / prev t) (the FULLSC path with the ANNEALED prev t — verified: diffusion.cpp:557
  -- passes prev_temp_inv into set_sc, and diffusion-gemma.cpp:408 applies it), and llama.cpp's
  -- TRUE stop constants (meanH < 0.005, stability 1) — valid now that H is the real full-vocab H.
  let modeRenoise := ((← IO.getEnv "DG_MODE").getD "renoise") == "renoise"   -- DEFAULT: renoise (8/8 @ 6-11 steps, ~40 tok/s avg); DG_MODE=mask restores the mask-commit path
  let schedEB := (((← IO.getEnv "DG_SCHED").getD "fixed") == "eb") || modeRenoise
  let ebTmax := ((((← IO.getEnv "DG_EB_TMAX").bind (·.toNat?)).getD 80).toFloat) / 100.0   -- t at step 0
  let ebTmin := ((((← IO.getEnv "DG_EB_TMIN").bind (·.toNat?)).getD 40).toFloat) / 100.0   -- t at last step
  let ebBound := ((((← IO.getEnv "DG_EB_BOUND").bind (·.toNat?)).getD 10).toFloat) / 100.0 -- MI bound (0.1)
  -- stop rule, adapted to the top-K H scale: llama.cpp stops at meanH<0.005 (its full-vocab H → ~0
  -- when every position is ultra-confident); our top-8+tail H floors well above that, so the binding
  -- condition here is ARGMAX STABILITY (held ≥ 2) with meanH<0.9 as a sanity ceiling.
  let ebConfTh := ((((← IO.getEnv "DG_EB_CONF").bind (·.toNat?)).getD (if modeRenoise then 5 else 900)).toFloat) / 1000.0
  let ebStab := (((← IO.getEnv "DG_EB_STAB").bind (·.toNat?)).getD (if modeRenoise then 1 else 2))   -- stability steps
  -- deterministic LCG (llama.cpp uses mt19937(seed); exact stream differs, semantics match)
  let mut ebRng : UInt64 := (((← IO.getEnv "DG_SEED").bind (·.toNat?)).getD 12345).toUInt64
  let mut ebPrevArgmax : Array Nat := Array.replicate C 0
  let mut ebPrevT : Float := 1.0   -- renoise: prev step's anneal t, fed to the SC softmax (llama.cpp prev_temp_inv)
  let mut ebHeld : Nat := 0
  let mut ebFirstStep := true
  let mut toks : Array Nat := promptTokens ++ Array.replicate C model.dg.maskTokenId
  if schedEB then
    for i in [0:C] do
      ebRng := ebRng * 6364136223846793005 + 1442695040888963407
      toks := toks.set! (P+i) ((ebRng >>> 33).toNat % cfg.vocabSize)
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
  let outputWeightF16 ← match model.inner.outputWeightF16 with
    | some b => pure b
    | none => do
      -- dequant the tied Q6_K lm_head (= token_embd) → packed f16 [vocab, dim/2] once.
      -- totalBlocks (≈2.9M) exceeds the 65535 per-dimension grid limit → use a 2D grid.
      let bpr := dim / 256
      let totalBlocks := cfg.vocabSize * bpr
      let gx := 65535
      let gy := (totalBlocks + gx - 1) / gx
      let buf ← mkBuf device (cfg.vocabSize * (dim / 2))
      Hesper.GPUBackend.beginBatch device
      disp2w device (Hesper.Quantization.Q6_K.q6kToF16Kernel dim cfg.vocabSize gx)
        (("weights", model.inner.outputWeight)::("output", buf)::List.nil) gx gy 64 (hash "lmf16dq")
      Hesper.GPUBackend.endBatch device
      IO.println s!"[dg-decode] dequantized Q6_K lm_head → f16 ({(cfg.vocabSize * dim * 2)/(1024*1024)} MiB)"
      pure buf
  -- DG_QKVRB: dequant the Q4_K Q/K/V projection weights → f16 [outDim, dim/2] once, so attention
  -- can use the ~11.6× faster subgroup-matrix reg-matmul instead of MMQ5 dp4a.
  let qkvRB := (← IO.getEnv "DG_NOQKVRB").isNone   -- DEFAULT ON (validated); DG_NOQKVRB=1 opts out
  -- DG_DENSEF16: pre-dequant the DENSE ffn_down (Q8_0) → f16 and use the same WMMA reg matmul as the
  -- dense gate/up (the QKVRB rounding profile — DISTINCT from the rejected DG_DENSEDOWNRB int-dequant
  -- reg kernel). ~357MB f16 (dim·ffn·2B·30). Non-Q8_0 layers fall back to q80+bmm.
  let denseF16 := (← IO.getEnv "DG_NODENSEF16").isNone   -- DEFAULT ON (8/8 @ 802ms under the chat template — the raw-prompt misframing that blocked it is cured); DG_NODENSEF16=1 opts out
  let mut qF16s : Array Buffer := #[]; let mut kF16s : Array Buffer := #[]; let mut vF16s : Array Buffer := #[]
  let mut oF16s : Array Buffer := #[]; let mut gateF16s : Array Buffer := #[]; let mut upF16s : Array Buffer := #[]
  let mut downF16s : Array (Option Buffer) := #[]
  if qkvRB then
    let sDqTmp ← mkBuf device (8192*dim)   -- max f32 dequant scratch (≥ max outDim×inDim)
    -- branch on the weight's quant: Q4_K → dequantQ4KMKernel+pack; Q6_K → q6kToF16Kernel (direct).
    -- (SWA layers store wV as Q6_K, others Q4_K — dequanting Q6_K as Q4_K gave NaN.) idim = the
    -- weight's input dim (= dim for Q/K/V, = qDim for the O-proj).
    let dq := fun (wbuf : Buffer) (od idim : Nat) (qfmt : Hesper.Layers.Linear.QuantFormat) (key : UInt64) => do
      let ne := od * idim
      let f16buf ← mkBuf device (ne/2)
      match qfmt with
      | .Q6_K =>
        let totalBlocks := od * (idim / 256)
        let gx := min totalBlocks 65535; let gy := (totalBlocks + gx - 1) / gx
        Hesper.GPUBackend.beginBatch device
        disp2w device (Hesper.Quantization.Q6_K.q6kToF16Kernel idim od gx) (("weights",wbuf)::("output",f16buf)::List.nil) gx gy 64 key
        Hesper.GPUBackend.endBatch device
      | _ =>   -- Q4_K
        let wg := (ne+255)/256; let nx := min wg 65535; let ny := (wg+nx-1)/nx
        Hesper.GPUBackend.beginBatch device
        disp2 device (Hesper.Quantization.Q4_K_M.dequantQ4KMKernel ne (nx*256)) (("data",wbuf)::("output",sDqTmp)::List.nil) nx ny key
        Hesper.WGSL.Execute.flushBatch device
        disp device (packF32ToF16B (ne/2)) (("fin",sDqTmp)::("fout",f16buf)::List.nil) (ne/2) (key+1)
        Hesper.GPUBackend.endBatch device
      pure f16buf
    for li in [0:nLayers] do
      let some blk := model.inner.blocks[li]? | throw (IO.userError "blk-dq")
      qF16s := qF16s.push (← dq blk.attention.wQ.weightBuf blk.attention.wQ.config.outDim dim blk.attention.wQ.quantFormat (hash ("dqq",li)))
      kF16s := kF16s.push (← dq blk.attention.wK.weightBuf blk.attention.wK.config.outDim dim blk.attention.wK.quantFormat (hash ("dqk",li)))
      vF16s := vF16s.push (← dq blk.attention.wV.weightBuf blk.attention.wV.config.outDim dim blk.attention.wV.quantFormat (hash ("dqv",li)))
      oF16s := oF16s.push (← dq blk.attention.wO.weightBuf blk.attention.wO.config.outDim blk.attention.wO.config.inDim blk.attention.wO.quantFormat (hash ("dqo",li)))
      gateF16s := gateF16s.push (← dq blk.ffn.gate.weightBuf blk.ffn.gate.config.outDim blk.ffn.gate.config.inDim blk.ffn.gate.quantFormat (hash ("dqg",li)))
      upF16s := upF16s.push (← dq blk.ffn.up.weightBuf blk.ffn.up.config.outDim blk.ffn.up.config.inDim blk.ffn.up.quantFormat (hash ("dqu",li)))
      -- DG_DENSEF16: dense down (Q8_0) → f16 via q8ToF32B + pack (the same two-stage shape as Q4_K)
      if denseF16 && blk.ffn.down.quantFormat == .Q8_0 then
        let od := blk.ffn.down.config.outDim; let idim := blk.ffn.down.config.inDim
        let ne := od * idim
        let f16buf ← mkBuf device (ne/2)
        let wg := (ne+255)/256; let nx := min wg 65535; let ny := (wg+nx-1)/nx
        Hesper.GPUBackend.beginBatch device
        disp2 device (q8ToF32B od idim (nx*256)) (("data",blk.ffn.down.weightBuf)::("output",sDqTmp)::List.nil) nx ny (hash ("dqd",li))
        Hesper.WGSL.Execute.flushBatch device
        disp device (packF32ToF16B (ne/2)) (("fin",sDqTmp)::("fout",f16buf)::List.nil) (ne/2) (hash ("dqdp",li))
        Hesper.GPUBackend.endBatch device
        downF16s := downF16s.push (some f16buf)
      else
        downF16s := downF16s.push none
    let dfMsg := if denseF16 then s!" + dense down f16 ({(downF16s.filter (·.isSome)).size}/{nLayers} Q8_0 layers)" else ""
    IO.println s!"[dg-decode] dequantized Q4_K Q/K/V → f16 ({nLayers} layers){dfMsg}"
  if (← IO.getEnv "DG_DQDIAG").isSome then
    -- hidden row r = e_r ⇒ logits[r,v] = weight[v,r]; compare f32 Q6_K (outputWeight) vs reg (f16)
    let Md := 64
    let hid ← mkBuf device (Md*dim)
    let mut hidA : Array Float := Array.replicate (Md*dim) 0.0
    for r in [:Md] do hidA := hidA.set! (r*dim + r) 1.0
    writeBuffer device hid 0 (← Hesper.Basic.floatArrayToBytes hidA)
    let lF32 ← mkBuf device (Md*lmN); let lRB ← mkBuf device (Md*lmN)
    Hesper.GPUBackend.beginBatch device
    disp2 device (Hesper.Quantization.Q6_K.fusedQ6KBatchKernel dim lmN Md 256 0 cfg.vocabSize)
      (("weights",model.inner.outputWeight)::("input",hid)::("output",lF32)::List.nil) lmN Md (hash "dqf32")
    Hesper.GPUBackend.endBatch device
    Hesper.GPUBackend.beginBatch device
    dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := Md, N := lmN, K := dim } 0 cfg.vocabSize)
      (("a",hid)::("b",outputWeightF16)::("c",lRB)::List.nil) ((lmN+31)/32) ((Md+63)/64) (hash "dqrb")
    Hesper.GPUBackend.endBatch device
    let f32A ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device lF32 0 (Md*lmN*4).toUSize)
    let rbA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device lRB 0 (Md*lmN*4).toUSize)
    let mut maxDiff := 0.0; let mut nBad := 0
    for r in [:Md] do for v in [:min lmN 2000] do
      let d := (f32A.getD (r*lmN+v) 0.0 - rbA.getD (r*lmN+v) 0.0).abs
      if d > maxDiff then maxDiff := d
      if d > 0.05 then nBad := nBad+1
    IO.println s!"[dqdiag] f32 vs reg(f16): maxDiff={maxDiff} nBad(>0.05)={nBad}/{Md*(min lmN 2000)}; samples (col,tok)=(0,0):{f32A.getD 0 0.0}/{rbA.getD 0 0.0} (1,5):{f32A.getD (1*lmN+5) 0.0}/{rbA.getD (1*lmN+5) 0.0}"
  let sLogits ← mkBuf device ((((N + 63) / 64) * 64)*lmN)   -- 64-row padded: WMMA tail tiles write past row N
  let logitsCanvas ← mkBuf device (C*cfg.vocabSize)  -- full-vocab canvas logits for the GPU reduce
  -- renoise-mode sampler I/O (tiny; allocated unconditionally)
  let ebUBuf ← mkBuf device C        -- pre-drawn multinomial u per position
  let ebPBuf ← mkBuf device 4        -- [0] = anneal t for this step
  let ebAmaxBuf ← mkBuf device C     -- per-position argmax token
  let ebSampBuf ← mkBuf device C     -- per-position multinomial sample
  let ebHBuf ← mkBuf device C        -- per-position full-vocab entropy
  let outDenom ← mkBuf device C
  let outTok ← mkBuf device (C*scK)
  let outProb ← mkBuf device (C*scK)
  -- DG_FULLSC: full-vocab self-conditioning (llama.cpp fidelity). Instead of the top-8 soft-embed,
  -- softmax(prev step's FULL logitsCanvas / t) on GPU (t annealed 0.8→0.4) then ONE WMMA matmul
  -- probs[C,vocab] @ Wᵀ[dim,vocab] → sSC. Needs a one-time transpose of the f16 embed table (+1.5GB)
  -- and a probs buffer (+268MB). The top-8 sliver is a biased approximation at high-entropy positions —
  -- full-vocab SC is the denoiser's trained conditioning signal (llama.cpp converges in 8-11 steps).
  let fullSC := (← IO.getEnv "DG_FULLSC").isSome || modeRenoise
  let sScProbs ← mkBuf device (if fullSC then C*cfg.vocabSize else 1)
  let embTT ← mkBuf device (if fullSC then dim*(cfg.vocabSize/2) else 1)
  let sSCC ← mkBuf device (if fullSC then C*dim else 1)
  let scTBuf ← mkBuf device 4
  if fullSC then
    Hesper.GPUBackend.beginBatch device
    let nT := dim*(cfg.vocabSize/2)
    let wgT := (nT+255)/256; let gxT := min wgT 32768; let gyT := (wgT+gxT-1)/gxT
    disp2 device (transposeEmbF16B cfg.vocabSize dim (gxT*256)) (("src",outputWeightF16)::("dst",embTT)::List.nil) gxT gyT (hash "embtt")
    Hesper.GPUBackend.endBatch device
    IO.println s!"[dg-decode] full-vocab SC: transposed embT [dim={dim}, vocab={cfg.vocabSize}] ({dim*cfg.vocabSize*2/(1024*1024)} MiB)"
  let skipDense := (← IO.getEnv "DG_SKIPDENSE").isSome
  IO.println s!"[dg-decode] {decodeSteps} steps, N={N} P={P} C={C}  skipDense={skipDense}"
  -- DG_CONF=<percent>: confidence threshold for bulk-commit early stop (e.g. DG_CONF=90 → commit all
  -- candidates with confidence ≥ 0.90 each step; loop skips once every position is committed).
  let confThreshPct : Option Nat := some (((← IO.getEnv "DG_CONF").bind (·.toNat?)).getD 90)   -- DEFAULT 90%: bulk-commit confident tokens (llama.cpp-style); DG_CONF=101 disables
  -- DG_LAYERSUM: per-layer hidden checksum (Σ|h|, max|h|) printed each step — cross-run diff localizes
  -- the first non-deterministic layer. Use with DG_LPB=1 so each layer's batch closes before readback.
  let layerSum := (← IO.getEnv "DG_LAYERSUM").isSome
  -- DG_PROF: per-phase GPU time (endBatch syncs + timestamps at each phase boundary, accumulated).
  let prof := (← IO.getEnv "DG_PROF").isSome
  let rAttn ← IO.mkRef (0:Nat); let rDense ← IO.mkRef (0:Nat)
  let rMoe ← IO.mkRef (0:Nat); let rRest ← IO.mkRef (0:Nat); let tPrev ← IO.mkRef (0:Nat)
  let rMoeGrp ← IO.mkRef (0:Nat); let rMoeGU ← IO.mkRef (0:Nat)   -- MoE sub-timers: router+grouping / gate-up
  let rBattn ← IO.mkRef (0:Nat); let rAttnO ← IO.mkRef (0:Nat)   -- attention sub-phases: QK^T+softmax+V, and O-proj
  let rQkn ← IO.mkRef (0:Nat)   -- qk-norm + RoPE + v-norm (to split it from the Q/K/V matmuls)
  let pmark := fun (r : IO.Ref Nat) => do
    if prof then
      Hesper.GPUBackend.endBatch device
      let n ← IO.monoMsNow
      r.modify (· + (n - (← tPrev.get)))
      tPrev.set n
      Hesper.GPUBackend.beginBatch device
  -- end-of-generation token set: Gemma4 <turn|>=106 (end-of-turn) + the tokenizer's EOS if any.
  -- Used for the EOS early-stop (in the commit section) and the display trim (llama.cpp behavior).
  let eogIds : List Nat := 106 :: (match Hesper.Tokenizer.SentencePiece.eosToken tokenizer with
    | some e => [e] | none => [])
  let mut loopTotalMs : Nat := 0
  let mut effSteps : Nat := 0
  for step in [0:decodeSteps] do
    if prof then rAttn.set 0; rDense.set 0; rMoe.set 0; rRest.set 0; rBattn.set 0; rAttnO.set 0; rQkn.set 0; rMoeGrp.set 0; rMoeGU.set 0
    let remaining := masked.foldl (fun acc b => if b then acc+1 else acc) 0
    if remaining > 0 then
      let t0 ← IO.monoMsNow
      writeBuffer device tokBuf 0 (u32Bytes toks)
      if step > 0 then
        writeBuffer device scTokBuf 0 (u32Bytes scTok)
        writeBuffer device scProbBuf 0 (← Hesper.Basic.floatArrayToBytes scProb)
        if fullSC then
          -- SC softmax temperature: llama.cpp uses temp_inv = 1.0 (NO temperature — verified at every
          -- llama_diffusion_set_sc call site; the 0.8→0.4 anneal is the canvas RE-NOISING sampling
          -- temperature, a different mechanism). An annealed t here over-sharpens the SC → hard
          -- self-reinforcement loops ("thoughtthought", "and and and"). DG_SCTEMP=<percent> tunes.
          -- renoise mode: llama.cpp feeds softmax(prev logits · prev_temp_inv) — the ANNEALED prev t
          -- (diffusion.cpp:557 + diffusion-gemma.cpp:408). Mask-mode keeps DG_SCTEMP (default 1.0).
          let scTempEnv := (((← IO.getEnv "DG_SCTEMP").bind (·.toNat?)).getD 100).toFloat / 100.0
          let tSC := if modeRenoise then ebPrevT else scTempEnv
          writeBuffer device scTBuf 0 (← Hesper.Basic.floatArrayToBytes #[tSC])
      Hesper.GPUBackend.beginBatch device
      disp device (Hesper.Quantization.Q6_K.q6kEmbedGatherKernel N cfg.vocabSize dim embScale) (("token_ids",tokBuf)::("embedding_table",embTable)::("output",a)::List.nil) (N*dim) (hash "emb")
      -- self-conditioning: soft-embed the previous step's top-K prediction → SC-MLP → add to canvas (step>0)
      if step > 0 then
        match model.scPreNorm, model.scGate, model.scUp, model.scDown with
        | some scPN, some scG, some scU, some scD =>
          if fullSC then
            -- FULL-VOCAB SC (llama.cpp fidelity): softmax(prev logitsCanvas / t) on GPU, then one
            -- WMMA matmul probs[C,vocab] @ embTTᵀ[dim,vocab] → sSCC, spread into the N-row sSC.
            -- logitsCanvas still holds the PREVIOUS step's post-softcap logits (written at its end).
            disp2 device (scSoftmaxTempB C cfg.vocabSize 256 1024.0)
              (("logits",logitsCanvas)::("tbuf",scTBuf)::("probs",sScProbs)::List.nil) C 1 (hash "scsm")
            Hesper.WGSL.Execute.flushBatch device   -- softmax → WMMA: batch split
            dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := C, N := dim, K := cfg.vocabSize } 0 dim)
              (("a",sScProbs)::("b",embTT)::("c",sSCC)::List.nil) ((dim+31)/32) ((C+63)/64) (hash "scmm")
            Hesper.WGSL.Execute.flushBatch device   -- WMMA → spread (Dawn drops the inter-pass barrier after subgroup-matrix at scale)
            disp device (scSpreadB N P dim) (("ssc",sSC)::("sccc",sSCC)::List.nil) (N*dim) (hash "scspread")
          else
            -- gather raw embeddings of the C·K top tokens, weighted-reduce by softmax probs → sSC
            disp device (Hesper.Quantization.Q6_K.q6kEmbedGatherKernel (C*scK) cfg.vocabSize dim 1.0) (("token_ids",scTokBuf)::("embedding_table",embTable)::("output",sTempK)::List.nil) (C*scK*dim) (hash "scgath")
            disp device (softReduceB N P C dim scK embScale) (("ssc",sSC)::("tempk",sTempK)::("prob",scProbBuf)::List.nil) (N*dim) (hash "scsoft")
          if step == 1 && (← IO.getEnv "DG_SCDBG").isSome then
            Hesper.GPUBackend.endBatch device
            let sc ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sSC 0 (N*dim*4).toUSize)
            let mut sabs := 0.0
            for v in sc do sabs := sabs + v.abs
            IO.println s!"[scdbg] step1 Σ|sSC| = {sabs} (fullSC={fullSC})"
            Hesper.GPUBackend.beginBatch device
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
      if prof then tPrev.set (← IO.monoMsNow)
      for li in [0:nLayers] do
        if li % lpb == 0 then Hesper.GPUBackend.beginBatch device
        let some blk := model.inner.blocks[li]? | throw (IO.userError "blk")
        let qDim := blk.attention.wO.config.inDim; let kvDim := blk.attention.wV.config.outDim
        let hd := qDim / nHead; let nKV := kvDim / hd
        let theta : Float := if li % 6 == 5 then 1000000.0 else 10000.0
        let nRotHalf := if li % 6 == 5 then 64 else hd/2
        -- attention
        Hesper.Layers.RMSNorm.forward device blk.attnNorm cur sN N
        if qkvRB then
          Hesper.WGSL.Execute.flushBatch device   -- RMSNorm → reg-matmul: batch split
          -- reg-matmul QKV on f32 sN (real dequantized f16 weights). N=outDim, K=dim.
          let rb := fun (wf16 outB : Buffer) (od : Nat) (key : UInt64) =>
            dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := nP, N := od, K := dim } 0 od)
              (("a",sN)::("b",wf16)::("c",outB)::List.nil) ((od+31)/32) ((N+63)/64) key
          rb (qF16s[li]?.getD sQ) sQ qDim (hash ("rbq",li)); rb (kF16s[li]?.getD sK) sK kvDim (hash ("rbk",li)); rb (vF16s[li]?.getD sV) sV kvDim (hash ("rbv",li))
          Hesper.WGSL.Execute.flushBatch device   -- reg-matmul QKV → qknorm: batch split (Dawn drops the inter-pass barrier for subgroup_matrix at scale)
          if (li == 0 || li == 5) && (← IO.getEnv "DG_QKVDIAG").isSome then
            qK device sN N dim (hash ("dqk0",li))
            bmm device blk.attention.wQ sN sCtx N (hash ("dbmm0",li))   -- MMQ5 reference for wQ → sCtx
            Hesper.GPUBackend.endBatch device
            let regQ ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sQ 0 (N*qDim*4).toUSize)
            let refQ ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sCtx 0 (N*qDim*4).toUSize)
            let mut md := 0.0; let mut sr := 0.0; let mut sg := 0.0
            for i in [0:N*qDim] do
              let d := (regQ.getD i 0.0 - refQ.getD i 0.0).abs
              if d > md then md := d
              sg := sg + (regQ.getD i 0.0).abs; sr := sr + (refQ.getD i 0.0).abs
            IO.println s!"[qkvdiag] wQ reg vs MMQ5: maxDiff={md} Σ|reg|={sg} Σ|ref|={sr}; sample reg={regQ.getD 100 0.0} ref={refQ.getD 100 0.0}"
            Hesper.GPUBackend.beginBatch device
        else
          qK device sN N dim (hash ("qN",li))
          bmm device blk.attention.wQ sN sQ N (hash ("wq",li))
          bmm device blk.attention.wK sN sK N (hash ("wk",li))
          bmm device blk.attention.wV sN sV N (hash ("wv",li))
        pmark rAttn   -- attnNorm + qK + Q/K/V matmuls (Q4_K MMQ5)
        disp device (qkNormRopeB N nHead hd nRotHalf theta eps) (("qin",sQ)::("wnorm",blk.attention.qNormWeight)::("qout",sQr)::List.nil) (N*nHead) (hash ("qn",li))
        disp device (qkNormRopeB N nKV hd nRotHalf theta eps) (("qin",sK)::("wnorm",blk.attention.kNormWeight)::("qout",sKr)::List.nil) (N*nKV) (hash ("kn",li))
        disp device (vNormB N nKV hd eps) (("vin",sV)::("vout",sVn)::List.nil) (N*nKV) (hash ("vn",li))
        pmark rQkn    -- qk-norm + RoPE + v-norm
        disp2 device (battnB N P nHead hd nKV 1.0) (("q",sQr)::("k",sKr)::("v",sVn)::("ctx",sCtx)::List.nil) nHead N (hash ("at",li))
        pmark rBattn  -- battnB: QK^T + softmax + weighted-V (matrix-vector per query)
        if qkvRB then
          -- O-proj reg-matmul: A = f32 sCtx [N, qDim], B = wO f16 [dim, qDim/2], C = sAO. K=qDim.
          dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := nP, N := dim, K := qDim } 0 dim)
            (("a",sCtx)::("b",oF16s[li]?.getD sAO)::("c",sAO)::List.nil) ((dim+31)/32) ((N+63)/64) (hash ("rbo",li))
          Hesper.WGSL.Execute.flushBatch device
        else
          qK device sCtx N qDim (hash ("qCtx",li))
          bmm device blk.attention.wO sCtx sAO N (hash ("wo",li))
        Hesper.Layers.RMSNorm.forward device blk.postAttnNorm sAO sR N
        disp device (addB (N*dim)) (("ain",sR)::("bin",cur)::("outc",sPA)::List.nil) (N*dim) (hash ("ra",li))
        pmark rAttnO  -- O projection + post-norm + residual
        -- dense FFN
        if step == 0 && (← IO.getEnv "DG_QFMT").isSome then
          let qfs := fun (q : Hesper.Layers.Linear.QuantFormat) => match q with
            | .Q4_K => "Q4_K" | .Q8_0 => "Q8_0" | .Q5_0 => "Q5_0" | .Q6_K => "Q6_K"
          IO.println s!"[qfmt L{li}] wQ={qfs blk.attention.wQ.quantFormat} wK={qfs blk.attention.wK.quantFormat} wV={qfs blk.attention.wV.quantFormat} wO={qfs blk.attention.wO.quantFormat} | dense gate={qfs blk.ffn.gate.quantFormat} down={qfs blk.ffn.down.quantFormat}"
        Hesper.Layers.RMSNorm.forward device blk.ffnNorm sPA sN N
        unless qkvRB do qK device sN N dim (hash ("qNf",li))
        unless skipDense do
          if qkvRB then
            -- dense gate/up reg-matmul on f32 sN. N=ffn, K=dim.
            dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := nP, N := ffn, K := dim } 0 ffn)
              (("a",sN)::("b",gateF16s[li]?.getD sG)::("c",sG)::List.nil) ((ffn+31)/32) ((N+63)/64) (hash ("rbg",li))
            dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := nP, N := ffn, K := dim } 0 ffn)
              (("a",sN)::("b",upF16s[li]?.getD sU)::("c",sU)::List.nil) ((ffn+31)/32) ((N+63)/64) (hash ("rbu",li))
            Hesper.WGSL.Execute.flushBatch device
          else
            bmm device blk.ffn.gate sN sG N (hash ("g",li))
            bmm device blk.ffn.up sN sU N (hash ("u",li))
          disp device (geluMulB (N*ffn)) (("gate",sG)::("up",sU)::("outp",sGe)::List.nil) (N*ffn) (hash ("gg",li))
          if denseF16 && (downF16s[li]?.getD none).isSome then
            -- dense down f16 WMMA reg (QKVRB rounding profile): A = f32 geglu sGe [N,ffn], B = down
            -- f16 [dim,ffn/2], C = sD [N,dim] (64-row padded alloc). No q80 needed (A read as f32).
            let some dbuf := downF16s[li]?.getD none | throw (IO.userError "downF16")
            dispRB device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := nP, N := dim, K := ffn } 0 dim)
              (("a",sGe)::("b",dbuf)::("c",sD)::List.nil) ((dim+31)/32) ((N+63)/64) (hash ("rbd",li))
            Hesper.WGSL.Execute.flushBatch device   -- WMMA → next reader: batch split (Dawn barrier drop)
          else if denseDownRB && blk.ffn.down.quantFormat == .Q8_0 then
            -- TILED reg-matmul dense down (matrix units, in-kernel Q8_0 dequant). A=f32 geglu sGe [N,ffn],
            -- B=down Q8_0 [dim,ffn], C=sD [N,dim]. Single weight ⇒ nExpert=1, zero tileExpert.
            dispRB device (Hesper.Quantization.Q4_K_M.q8MatmulGroupedRegKernel N dim ffn 1)
              (("a",sGe)::("b",blk.ffn.down.weightBuf)::("c",sD)::("tileExpert",sZeroTE)::List.nil) ((dim+31)/32) ((N+31)/32) (hash ("ddrb",li))
          else
            q80 device sGe N ffn (hash ("qGe",li))
            bmm device blk.ffn.down sGe sD N (hash ("dn",li))
        pmark rDense
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
        if (← IO.getEnv "DG_NOGROUP").isNone then   -- grouped MoE DEFAULT ON (validated); DG_NOGROUP=1 opts out
          -- expert-grouping (fused, GPU-side counting-sort grouping — no readback, stays batched)
          unless skGrp do disp device (clearSortedB maxPadded nUsed) (("sp",sSortedPos)::("ss",sSortedSlot)::List.nil) maxPadded (hash ("clr",li))
          unless skGrp do disp device (countExpB totalTok nExpert) (("idxs",sIdxs)::("cnt",sExpertCount)::List.nil) nExpert (hash ("cntk",li))
          unless skGrp do disp device (offsetsExpB nExpert maxPadded padTo) (("cnt",sExpertCount)::("off",sExpertOffset)::("te",sTileExpert)::("trs",sTileRows)::List.nil) 1 (hash ("offk",li))
          if li == 0 && (← IO.getEnv "DG_TILEDIAG").isSome then
            Hesper.GPUBackend.endBatch device
            let teA ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sTileExpert 0 (maxPadded/32*4).toUSize)
            -- te is u32; reinterpret the f32 bytes as raw — count entries that are NOT the sentinel (nExpert)
            let teRaw ← mapBufferRead device sTileExpert 0 (maxPadded/32*4).toUSize
            let mut active := 0
            for tIdx in [0:maxPadded/32] do
              let b0 := teRaw.get! (tIdx*4); let b1 := teRaw.get! (tIdx*4+1)
              let v := b0.toNat + b1.toNat*256
              if v < nExpert then active := active+1
            IO.println s!"[tilediag] active tiles={active} / maxPadded tiles={maxPadded/32} (rows: {active*32} real / {maxPadded} dispatched); waste={maxPadded - active*32} rows ({100*(maxPadded-active*32)/maxPadded}%)"
            let _ := teA
            Hesper.GPUBackend.beginBatch device
          unless skGrp do disp device (scatterRankB totalTok nExpert nUsed maxPadded) (("idxs",sIdxs)::("off",sExpertOffset)::("sp",sSortedPos)::("ss",sSortedSlot)::List.nil) totalTok (hash ("srkk",li))
          pmark rMoeGrp   -- router + counting-sort grouping (clear/count/offsets/scatterRank)
          if moeRB then
            -- INDEXED Q4_K reg-matmul gate/up (mul_mat_id-style): the A-load reads token rows IN PLACE
            -- through sSortedPos — no physical gatherF32B pass. src=sMoeN [N,dim], B=guE Q4_K, C=sGatheredGU.
            dispRB device (Hesper.Quantization.Q4_K_M.q4kMatmulGroupedRegIndexedKernel maxPadded (2*expFF) dim nExpert N)
              (("src",sMoeN)::("idx",sSortedPos)::("b",guE)::("c",sGatheredGU)::("tileExpert",sTileExpert)::("tileRows",raggedRows)::List.nil) ((2*expFF+31)/32) ((maxPadded+31)/32) (hash ("emmrbi",li))
          else
            unless skGat do disp device (gatherQ8B maxPadded q8size N) (("src",sMoeNQ8)::("idx",sSortedPos)::("gathered",sGatheredQ8)::List.nil) (maxPadded*q8size) (hash ("gthr",li))
            unless skGU do
              disp2 device (Hesper.Layers.Linear.q4kMatmulBatchMMQ5Kernel { inDim:=dim, outDim:=2*expFF } maxPadded 0 0 maxPadded true nExpert)
                (("weights",guE)::("input_q8",sGatheredQ8)::("output",sGatheredGU)::("tileExpert",sTileExpert)::List.nil) ((2*expFF)/64) (maxPadded/32) (hash ("emm",li))
          -- BATCH SPLIT (cheap, no CPU wait): the grouped gate/up MMQ→scatter→geglu races in a too-
          -- large single encoder (Dawn-on-Metal drops an inter-pass barrier at scale); a no-wait
          -- submit + fresh encoder here keeps Dawn's barriers correct without a sync round-trip.
          unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device
          pmark rMoeGU   -- gather + gate/up matmul (MMQ5 or fused reg)
          if li == 0 && (← IO.getEnv "DG_GUDIAG").isSome then
            -- un-group the grouped gate/up + compute the per-slot reference, compare
            disp device (scatterGUB maxPadded (2*expFF) N nUsed) (("gathered",sGatheredGU)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sGateUpAll)::List.nil) (maxPadded*2*expFF) (hash ("gudsc",li))
            for e in [0:nUsed] do
              disp2w device (Hesper.Layers.Linear.fusedQ4KMBatchExpertDP4ATiledKernel { inDim:=dim, outDim:=2*expFF } nExpert N nUsed e 4) (("weights",guE)::("input_q8",sMoeNQ8)::("idxs",sIdxs)::("output",(sGateUps[e]?.getD sMoeN))::List.nil) ((2*expFF)/4) N 32 (hash ("gudref",li,e))
            Hesper.GPUBackend.endBatch device
            let gAll ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sGateUpAll 0 (nUsed*N*2*expFF*4).toUSize)
            let mut maxd := 0.0; let mut cnt := 0
            let mut s0g := 0.0; let mut s0r := 0.0
            for e in [0:nUsed] do
              let ref ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device (sGateUps[e]?.getD sMoeN) 0 (N*2*expFF*4).toUSize)
              for pos in [P:N] do
                for j in [0:2*expFF] do
                  let g := gAll.getD (e*N*2*expFF + pos*2*expFF + j) 0.0
                  let r := ref.getD (pos*2*expFF + j) 0.0
                  if e==0 && pos==P && j==0 then s0g := g
                  if e==0 && pos==P && j==0 then s0r := r
                  let d := (g - r).abs
                  if d > maxd then maxd := d
                  if d > 0.1 then cnt := cnt+1
            IO.println s!"[gudiag] grouped sGateUpAll vs per-slot ref: maxDiff={maxd} nBad(>0.1)={cnt}; sample e0p{P}j0 grouped={s0g} ref={s0r}"
            Hesper.GPUBackend.beginBatch device
          -- DEFAULT: gate/up grouped + per-slot down (CORRECT, "Paris", ~0.23s win). The TILED grouped
          -- down (DG_GROUPEDDOWN) is faster but currently emits 0 (sGatheredGU reads 0 in its geglu —
          -- an unresolved barrier/race when the gate/up scatter is skipped). See PERF_PLAN / commits.
          if (← IO.getEnv "DG_GROUPEDDOWN").isNone && !moeDownRB then
            -- ISOLATION: gate/up grouped, down per-slot (the pre-grouped-down state)
            unless skSc do disp device (scatterGUB maxPadded (2*expFF) N nUsed) (("gathered",sGatheredGU)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sGateUpAll)::List.nil) (maxPadded*2*expFF) (hash ("sctr",li))
            unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device   -- flush gate/up scatter→geglu (no-wait split)
            for e in [0:nUsed] do
              let sEh := sEhs[e]?.getD sMoeN
              unless skGeg do disp device (gegluMergedB N expFF (e*N*2*expFF) (nUsed*N*2*expFF)) (("gu",sGateUpAll)::("eh",sEh)::List.nil) (N*expFF) (hash ("gm",li,e))
              unless skQ80 do q80 device sEh N expFF (hash ("qEh",li,e))
              let downExpKernel := match blk.ffn.down.quantFormat with
                | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
                | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
              let sDownE := sDownEs[e]?.getD sEh
              unless skDn do disp2w device downExpKernel (("weights",dnE)::("input",sEh)::("idxs",sIdxs)::("output",sDownE)::List.nil) dim N 32 (hash ("ed",li,e))
              unless skWa do disp device (waccB N dim e nUsed) (("acc",sMoeAcc)::("din",sDownE)::("wts",sWts)::List.nil) (N*dim) (hash ("wa",li,e))
          else do
            -- GROUPED down: the FUSED single-kernel (geglu+down+scatter in one dispatch — no inter-pass
            -- flushes → no Dawn race, DG_MOEDOWNFUSED) OR the staged geglu→down→scatter chain.
            let moeDownFused := (← IO.getEnv "DG_MOEDOWNFUSED").isSome && blk.ffn.down.quantFormat == .Q8_0
            if moeDownFused then
              dispRB device (Hesper.Quantization.Q4_K_M.q8FusedGegluDownScatterKernel maxPadded dim expFF nExpert nUsed N)
                (("gu",sGatheredGU)::("b",dnE)::("tileExpert",sTileExpert)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sDownAll)::List.nil) ((dim+31)/32) ((maxPadded+31)/32) (hash ("fdg",li))
            else
              disp device (gegluMergedB maxPadded expFF 0 (maxPadded*2*expFF)) (("gu",sGatheredGU)::("eh",sGatheredEh)::List.nil) (maxPadded*expFF) (hash ("gmg",li))
            unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device
            if li == 0 && (← IO.getEnv "DG_MOEDOWNDIAG").isSome then
              Hesper.GPUBackend.endBatch device
              let eh ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sGatheredEh 0 (maxPadded*expFF*4).toUSize)
              let mut mx := 0.0
              for i in [0:maxPadded*expFF] do let v := (eh.getD i 0.0).abs; if v > mx then mx := v
              IO.println s!"[moedowndiag] max|geglu A (down input)| = {mx}  (f16 max = 65504 → overflow if larger)"
              Hesper.GPUBackend.beginBatch device
            if moeDownFused then pure ()   -- the fused kernel already did geglu+down+scatter → sDownAll
            else if moeDownRB && blk.ffn.down.quantFormat != .Q5_0 then
              -- INDEXED-SCATTER reg-matmul down (matrix units, in-kernel Q8_0 dequant): the C store
              -- scatters dst[slot,pos,col] IN-KERNEL — no 17.5M-element scatterGUB pass. The q80
              -- round-trip both matches the warp's Q8 rounding AND acts as the geglu→down sync.
              q80 device sGatheredEh maxPadded expFF (hash ("qgehrb",li))
              unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device
              dispRB device (Hesper.Quantization.Q4_K_M.q8MatmulGroupedRegIndexedScatterKernel maxPadded dim expFF nExpert nUsed N)
                (("a",sGatheredEh)::("b",dnE)::("tileExpert",sTileExpert)::("tileRows",raggedRows)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sDownAll)::List.nil) ((dim+31)/32) ((maxPadded+31)/32) (hash ("edrbi",li))
            else
              q80 device sGatheredEh maxPadded expFF (hash ("qgeh",li))   -- match the per-slot Q8 rounding
              let downGrpKernel := match blk.ffn.down.quantFormat with
                | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertF32WarpGroupedKernel { inDim:=expFF, outDim:=dim } nExpert maxPadded
                | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpGroupedKernel { inDim:=expFF, outDim:=dim } nExpert maxPadded
              unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device
              disp2w device downGrpKernel (("weights",dnE)::("input",sGatheredEh)::("tileExpert",sTileExpert)::("output",sGatheredDown)::List.nil) dim (maxPadded/32) 32 (hash ("edg",li))
            unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device
            -- down scatter is maxPadded*dim = ~17.5M elems → ~68k workgroups > 65535 limit (silently
            -- dropped → sDownAll stayed 0). Use a 2D grid: flat = gid.x + gid.y*(nx*256).
            let scN := maxPadded*dim
            let scWG := (scN + 255)/256
            let scNx := min scWG 32768
            let scNy := (scWG + scNx - 1)/scNx
            -- when fused OR indexed-scatter down ran, sDownAll is already written in-kernel — skip
            -- the staged scatter (it only remains for the Q5_0 warp-grouped fallback path).
            unless (moeDownFused || (moeDownRB && blk.ffn.down.quantFormat != .Q5_0)) do
              disp2 device (scatterGUB maxPadded dim N nUsed (scNx*256)) (("gathered",sGatheredDown)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sDownAll)::List.nil) scNx scNy (hash ("sctrd",li))
            -- NOTE: the grouped-down chain (geglu→q80→down→scatter→wacc) is NUMERICALLY correct (DG_MOEDIAG
            -- maxDiff 4e-6) but Dawn drops these no-wait flushes at batch scale → a routing-dependent RACE
            -- ("Paris" passes, harder prompts → garbage). endBatch here fixes ONE link but the chain has
            -- several races AND endBatch mid-batch is catastrophically expensive (3-4s/step) — so the grouped
            -- reg/warp down is NOT usable; the per-slot down (default) avoids the long racy chain.
            -- fused→wacc: the no-wait flush is dropped by Dawn for the big fused dispatch (the wacc reads
            -- sDownAll partial → garbage). A real barrier (endBatch) is the only reliable sync (cost TBD).
            if moeDownFused then
              Hesper.GPUBackend.endBatch device
              Hesper.GPUBackend.beginBatch device
            else
              unless moeNoFlush do Hesper.WGSL.Execute.flushBatch device   -- sync scatter→wacc
            if li == 0 && (← IO.getEnv "DG_MOEDIAG").isSome then
              -- compute the per-slot down reference (into sDownEs) and compare to the grouped sDownAll
              disp device (scatterGUB maxPadded (2*expFF) N nUsed) (("gathered",sGatheredGU)::("pos",sSortedPos)::("slot",sSortedSlot)::("dst",sGateUpAll)::List.nil) (maxPadded*2*expFF) (hash ("mdsc",li))
              for e in [0:nUsed] do
                let sEh := sEhs[e]?.getD sMoeN
                disp device (gegluMergedB N expFF (e*N*2*expFF) (nUsed*N*2*expFF)) (("gu",sGateUpAll)::("eh",sEh)::List.nil) (N*expFF) (hash ("mdgm",li,e))
                q80 device sEh N expFF (hash ("mdq",li,e))
                let dk := match blk.ffn.down.quantFormat with
                  | .Q5_0 => Hesper.Layers.Linear.fusedQ5_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
                  | _     => Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpKernel { inDim:=expFF, outDim:=dim } nExpert N nUsed e
                disp2w device dk (("weights",dnE)::("input",sEh)::("idxs",sIdxs)::("output",(sDownEs[e]?.getD sMoeN))::List.nil) dim N 32 (hash ("mdd",li,e))
              Hesper.GPUBackend.endBatch device
              let sgd ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sGatheredDown 0 (maxPadded*dim*4).toUSize)
              let mut sumGD := 0.0
              for v in sgd do sumGD := sumGD + v.abs
              IO.println s!"[moediag] Σ|sGatheredDown| (grouped down output, pre-scatter) = {sumGD}"
              let gAll ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device sDownAll 0 (nUsed*N*dim*4).toUSize)
              let mut maxd := 0.0; let mut cnt := 0; let mut sumG := 0.0; let mut sumR := 0.0
              for e in [0:nUsed] do
                let ref ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device (sDownEs[e]?.getD sMoeN) 0 (N*dim*4).toUSize)
                for pos in [P:N] do
                  for o in [0:dim] do
                    let g := gAll.getD (e*N*dim + pos*dim + o) 0.0
                    let r := ref.getD (pos*dim + o) 0.0
                    sumG := sumG + g.abs; sumR := sumR + r.abs
                    let d := (g - r).abs
                    if d > maxd then maxd := d
                    if d > 0.1 then cnt := cnt+1
              IO.println s!"[moediag] grouped sDownAll vs per-slot down: maxDiff={maxd} nBad={cnt} | Σ|grouped|={sumG} Σ|ref|={sumR}"
              Hesper.GPUBackend.beginBatch device
            -- single-pass weighted-accumulate (no 8-way read-modify-write race on sMoeAcc)
            disp device (waccAllB N dim nUsed) (("din",sDownAll)::("wts",sWts)::("acc",sMoeAcc)::List.nil) (N*dim) (hash ("waA",li))
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
        pmark rMoe
        -- combine: curMlp + curMoe → postFFNNorm → +residual → ×out_scale
        disp device (addB (N*dim)) (("ain",sCurMlp)::("bin",sCurMoe)::("outc",sComb)::List.nil) (N*dim) (hash ("ad",li))
        Hesper.Layers.RMSNorm.forward device blk.postFFNNorm sComb sR N
        disp device (addB (N*dim)) (("ain",sR)::("bin",sPA)::("outc",nxt)::List.nil) (N*dim) (hash ("rc",li))
        disp device (scaleRegionB N P dim (scales[li]!) (encScales[li]!)) (("data",nxt)::List.nil) (N*dim) (hash ("sc",li))
        pmark rRest
        if li % lpb == lpb-1 || li == nLayers-1 then Hesper.GPUBackend.endBatch device
        let t := cur; cur := nxt; nxt := t
        if layerSum && (li % lpb == lpb-1 || li == nLayers-1) then
          let h ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cur 0 (N*dim*4).toUSize)
          let mut s := 0.0; let mut mx := 0.0
          for v in h do
            let av := v.abs
            s := s + av
            if av > mx then mx := av
          IO.println s!"[lsum s{step} l{li}] sum={s} max={mx}"
      if prof then
        IO.println s!"[prof step {step}] qkv-mm={← rAttn.get}ms qknorm={← rQkn.get}ms battnB={← rBattn.get}ms attnO={← rAttnO.get}ms dense={← rDense.get}ms | MoE[router+grp={← rMoeGrp.get} gateup={← rMoeGU.get} down+rest={← rMoe.get}] rest={← rRest.get}ms"
      let tFwd ← IO.monoMsNow
      if step == 0 && (← IO.getEnv "DG_LMDIAG").isSome then
        lmHeadDiag device model.inner.finalNorm model.inner.outputWeight cur dim cfg.vocabSize N P
      -- final norm + Q8 quant, then full-vocab tiled lm_head (helper, keeps `main` small)
      Hesper.GPUBackend.beginBatch device
      Hesper.Layers.RMSNorm.forward device model.inner.finalNorm cur sN N
      -- lm_head reads the RAW f32 hidden (no qK): confirmed "Paris.", slightly more
      -- accurate than Q8_K, and the planned f32-warp lm_head kernel reads f32 directly.
      Hesper.GPUBackend.endBatch device
      let (cand, ktokFlat, probFlat) ← lmHeadArgmaxFullVocab device outputWeightF16 sN sLogits logitsCanvas outDenom outTok outProb dim cfg.vocabSize N C P cfg.logitSoftcapScale masked scK
      -- DG_LOGITDUMP=<path>: dump the step-0 full-canvas post-softcap logits [C, vocab] f32 for a
      -- golden diff vs llama-diffusion-gemma-eval on the same prompt+all-mask canvas.
      if step == 0 then
        if let some dumpPath ← IO.getEnv "DG_LOGITDUMP" then
          let bytes ← mapBufferRead device logitsCanvas 0 (C*cfg.vocabSize*4).toUSize
          IO.FS.writeBinFile dumpPath bytes
          unmapBuffer logitsCanvas
          -- also dump raw sLogits (holds the LAST vocab chunk) to separate WMMA-vs-copy for missing rows
          let sBytes ← mapBufferRead device sLogits 0 (N*32768*4).toUSize
          IO.FS.writeBinFile (dumpPath ++ ".slogits") sBytes
          unmapBuffer sLogits
          IO.println s!"[logitdump] step-0 logits [C={C} x vocab={cfg.vocabSize}] → {dumpPath} ({bytes.size} bytes); maskTokenId={model.dg.maskTokenId} N={N} P={P}"
      let tLm ← IO.monoMsNow
      if schedEB then
        -- ===== ENTROPY-BOUND step (llama.cpp port; see the schedEB comment at the canvas init) =====
        let S := decodeSteps
        let tCur := ebTmin + (ebTmax - ebTmin) * ((S - step).toFloat / S.toFloat)
        let tInv := 1.0 / tCur
        -- per position: distribution/entropy/argmax/multinomial — GPU full-vocab (renoise mode,
        -- llama.cpp fidelity) or the CPU top-K+tail approximation (DG_SCHED=eb).
        let mut entH : Array Float := Array.replicate C 0.0
        let mut argmaxT : Array Nat := Array.replicate C 0
        let mut sampledT : Array Nat := Array.replicate C 0
        let mut qFlat : Array Float := Array.replicate (C*scK) 0.0
        if modeRenoise then
          -- pre-draw u (llama.cpp pre-draws single-threaded for seed reproducibility); clamp > 0 so
          -- the sampler's "first chunk with cum ≥ u·Z" rule is well-defined at u=0
          let mut uArr : Array Float := Array.replicate C 0.0
          for pos in [0:C] do
            ebRng := ebRng * 6364136223846793005 + 1442695040888963407
            let u : Float := ((ebRng >>> 11).toNat.toFloat) / 9007199254740992.0
            uArr := uArr.set! pos (max u 1e-12)
          writeBuffer device ebUBuf 0 (← Hesper.Basic.floatArrayToBytes uArr)
          writeBuffer device ebPBuf 0 (← Hesper.Basic.floatArrayToBytes #[tCur, 0.0, 0.0, 0.0])
          disp2 device (ebSampleFullB C cfg.vocabSize 256)
            (("logits",logitsCanvas)::("uin",ebUBuf)::("params",ebPBuf)::("oamax",ebAmaxBuf)::("osamp",ebSampBuf)::("oh",ebHBuf)::List.nil) C 1 (hash "ebsample")
          let rdU := fun (a : ByteArray) (j : Nat) =>
            a[j*4]!.toNat ||| (a[j*4+1]!.toNat <<< 8) ||| (a[j*4+2]!.toNat <<< 16) ||| (a[j*4+3]!.toNat <<< 24)
          let amaxB ← mapBufferRead device ebAmaxBuf 0 (C*4).toUSize
          let sampB ← mapBufferRead device ebSampBuf 0 (C*4).toUSize
          let hArr ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device ebHBuf 0 (C*4).toUSize)
          for pos in [0:C] do
            argmaxT := argmaxT.set! pos (rdU amaxB pos)
            sampledT := sampledT.set! pos (rdU sampB pos)
            entH := entH.set! pos (hArr[pos]!)
        else for pos in [0:C] do
          let base := pos*scK
          -- full-vocab approximation from the top-K: model the tail (1-Σp) as nTail pseudo-tokens of
          -- mass p_K each (the K-th prob). Without this, renormalizing over K inflates uncertain
          -- positions' junk 20×+ in the SC and caps H at ln K (llama.cpp's uncertain H ≈ 10+).
          let mut sumP := 0.0
          let mut zTop := 0.0
          for j in [0:scK] do
            sumP := sumP + probFlat[base+j]!
            zTop := zTop + Float.pow (max (probFlat[base+j]!) 1e-30) tInv
          let pK := max (probFlat[base+scK-1]!) 1e-30
          let nTail := (max (1.0 - sumP) 0.0) / pK
          let qTail1 := Float.pow pK tInv
          let zAll := zTop + nTail * qTail1
          ebRng := ebRng * 6364136223846793005 + 1442695040888963407
          let u : Float := ((ebRng >>> 11).toNat.toFloat) / 9007199254740992.0
          let target := u * zAll
          let mut hh := 0.0
          let mut cum := 0.0
          let mut samp := ktokFlat[base]!
          let mut donePick := false
          for j in [0:scK] do
            let q := Float.pow (max (probFlat[base+j]!) 1e-30) tInv / zAll
            qFlat := qFlat.set! (base+j) q
            if q > 0.0 then hh := hh - q * Float.log q
            cum := cum + q * zAll
            if !donePick && cum ≥ target then samp := ktokFlat[base+j]!; donePick := true
          if !donePick then
            -- tail draw: proxy with a fresh random token (llama.cpp samples the true tail)
            ebRng := ebRng * 6364136223846793005 + 1442695040888963407
            samp := (ebRng >>> 33).toNat % cfg.vocabSize
          let qT := qTail1 / zAll
          if nTail > 0.0 && qT > 0.0 then hh := hh - nTail * qT * Float.log qT
          entH := entH.set! pos hh
          argmaxT := argmaxT.set! pos (ktokFlat[base]!)
          sampledT := sampledT.set! pos samp
        -- accept the lowest-entropy positions within the MI bound (strictly-earlier cum ≤ bound)
        let mut order : Array (Float × Nat) := Array.replicate C (0.0, 0)
        for pos in [0:C] do order := order.set! pos (entH[pos]!, pos)
        let sorted := order.qsort (fun a b => a.1 < b.1)
        let mut accepted : Array Bool := Array.replicate C false
        let mut cumE := 0.0
        let mut nAcc := 0
        for k2 in [0:C] do
          let (h, pos) := sorted[k2]!
          if cumE ≤ ebBound then accepted := accepted.set! pos true; nAcc := nAcc + 1
          cumE := cumE + h
        -- renoise: accepted → sampled, rest → fresh random; the OUTPUT canvas is the argmax
        let mut entSum := 0.0
        for pos in [0:C] do
          entSum := entSum + entH[pos]!
          if accepted[pos]! then
            toks := toks.set! (P+pos) (sampledT[pos]!)
          else
            ebRng := ebRng * 6364136223846793005 + 1442695040888963407
            toks := toks.set! (P+pos) ((ebRng >>> 33).toNat % cfg.vocabSize)
        -- SC = softmax(prev logits / prev t): feed the TEMPERED top-K as next step's soft prediction
        scTok := ktokFlat; scProb := qFlat
        -- adaptive stop: argmax stable ≥ stab steps AND mean entropy < threshold (llama.cpp rule)
        let stable := !ebFirstStep && argmaxT == ebPrevArgmax
        ebHeld := if stable then ebHeld + 1 else 0
        ebPrevArgmax := argmaxT
        ebFirstStep := false
        ebPrevT := tCur   -- renoise SC uses the PREV step's anneal t (llama.cpp prev_temp_inv)
        let meanH := entSum / C.toFloat
        let finish := (ebHeld ≥ ebStab && meanH < ebConfTh) || step+1 ≥ decodeSteps
        if finish then
          for i in [0:C] do
            toks := toks.set! (P+i) (argmaxT[i]!)
            masked := masked.set! i false
        let t1 ← IO.monoMsNow
        loopTotalMs := loopTotalMs + (t1 - t0)
        effSteps := effSteps + 1
        let stopStr := if finish then " | STOP" else ""
        IO.println s!"[dg-decode] step {step}: eb acc={nAcc} meanH={meanH} held={ebHeld} t={tCur} | total {t1-t0}ms = emb+fwd {tFwd-t0}ms + lmhead+reduce {tLm-tFwd}ms{stopStr}"
        continue
      scTok := ktokFlat; scProb := probFlat   -- feed this step's top-K soft prediction into next step's SC
      let want := if step+1 ≥ decodeSteps then remaining else max 1 ((C + decodeSteps - 1) / decodeSteps)
      -- DG_CONF=<percent>: confidence-based bulk commit (llama.cpp-style early stop). Commit every
      -- candidate whose confidence ≥ threshold (in addition to the fixed schedule's `want`). Once all
      -- positions commit, the remaining steps are skipped (the `remaining > 0` guard) → fewer effective
      -- steps on easy prompts, same quality floor via `want`.
      let nHigh := match confThreshPct with
        | some pct =>
          let th := pct.toFloat / 100.0
          cand.foldl (fun n (c : Nat × Nat × Float) => if c.2.2 ≥ th then n+1 else n) 0
        | none => 0
      let k := min (max want nHigh) cand.size
      let maskId := model.dg.maskTokenId
      -- DG_MINCONF=<percent>: near-tie floor for the FORCED commits. The fixed schedule (`want`)
      -- otherwise freezes low-confidence/near-tie predictions early — the root of the "the."-style
      -- degenerations AND of kernel-swap fragility (two independent dense-down rounding profiles
      -- flipped the same near-tie prompts). Picks are in descending confidence, so once a pick
      -- falls below the floor we defer the rest to a later step. The FIRST pick always commits
      -- (progress guarantee; the 32-step ceiling bounds total time); the LAST step commits all.
      let minConf : Float := (((← IO.getEnv "DG_MINCONF").bind (·.toNat?)).getD 0).toFloat / 100.0
      let confDbg := (← IO.getEnv "DG_CONFDBG").isSome
      let lastStep := step+1 ≥ decodeSteps
      let mut picked := Array.replicate cand.size false
      let mut nCommitted := 0
      let mut committedCfs : Array Float := #[]
      for i in [0:k] do
        let mut bi := 0; let mut bv := -1.0
        for c in [0:cand.size] do
          let (_, pr, cf) := cand[c]!
          -- never commit the MASK token itself: an argmax of "still masked" must stay uncommitted for
          -- a later step. Committing it freezes a mask into the text (掩 in the output) and the decode
          -- collapses — this is what degenerated the grouped-MoE path, whose few-% numeric drift ranked
          -- mask predictions into the top-k.
          if !picked[c]! && pr != maskId && cf > bv then bv := cf; bi := c
        if bv < 0.0 then break
        if i > 0 && !lastStep && bv < minConf then break
        picked := picked.set! bi true
        let (ci, pred, _) := cand[bi]!
        toks := toks.set! (P+ci) pred
        masked := masked.set! ci false
        nCommitted := nCommitted + 1
        committedCfs := committedCfs.push bv
      if confDbg && committedCfs.size > 0 then
        let sorted := committedCfs.qsort (· < ·)
        IO.println s!"[confdbg step {step}] n={sorted.size} min={sorted[0]!} med={sorted[sorted.size/2]!} max={sorted[sorted.size-1]!}"
      -- EOS early-stop: once everything BEFORE the first committed end-of-generation token is
      -- committed, the canvas positions after it are irrelevant — unmask them so the next step
      -- sees remaining=0 and the loop skips (llama.cpp stops at EOG the same way).
      let mut eogPos := C
      for i in [0:C] do
        if eogPos == C && !masked[i]! && eogIds.contains (toks[P+i]!) then eogPos := i
      if eogPos < C then
        let mut allBefore := true
        for j in [0:eogPos] do
          if masked[j]! then allBefore := false
        if allBefore then
          for j in [eogPos:C] do masked := masked.set! j false
          IO.println s!"[dg-decode] EOS early-stop: eog at canvas {eogPos}, tail unmasked"
      let t1 ← IO.monoMsNow
      loopTotalMs := loopTotalMs + (t1 - t0)
      effSteps := effSteps + 1
      IO.println s!"[dg-decode] step {step}: committed {nCommitted} | total {t1-t0}ms = emb+fwd {tFwd-t0}ms + lmhead+reduce {tLm-tFwd}ms + commit {t1-tLm}ms"
  let outIds := toks.extract P (P+C)
  -- Display trim (llama.cpp diffusion-cli behavior): cut at the first EOG token, else at the
  -- onset of a repetition loop (a token recurring at stride 1-2 for ≥6 reps — checkpoints often
  -- emit no stop token and pad the tail with spam).
  let mut cut := outIds.size
  for i in [0:outIds.size] do
    if cut == outIds.size && eogIds.contains (outIds[i]!) then cut := i
  let eogCut := cut
  for i in [0:eogCut] do
    if cut == eogCut then   -- only the FIRST loop onset
      for stride in [1:3] do
        -- FULL-PERIOD repetition: block [j..j+stride) must equal [j+stride..j+2·stride). A single
        -- repeating rail (e.g. the COMMA in "Mercury, Venus, Earth, …") is a legitimate list, not
        -- spam — the old one-rail check cut planet enumerations right before the answer.
        let mut reps := 0
        let mut j := i
        while j + 2*stride ≤ eogCut &&
              (List.range stride).all (fun k => outIds[j+k]! == outIds[j+k+stride]!) do
          reps := reps + 1; j := j + stride
        if reps ≥ 6 && cut == eogCut then cut := i
  let trimmed := outIds.extract 0 cut
  IO.println s!"[dg-decode] first canvas IDs: {(outIds.extract 0 (min 24 outIds.size)).toList}"
  let toks256 := (256.0 * 1000.0) / (max loopTotalMs 1).toFloat
  let toksUse := (cut.toFloat * 1000.0) / (max loopTotalMs 1).toFloat
  IO.println s!"[dg-decode] TPS: {effSteps} eff-steps × {loopTotalMs/(max effSteps 1)}ms avg = {loopTotalMs}ms total | canvas {toks256} tok/s | useful({cut} tok) {toksUse} tok/s"
  IO.println s!"[dg-decode] TEXT(raw {outIds.size} → cut {cut}): {Hesper.Tokenizer.SentencePiece.decode tokenizer trimmed}"