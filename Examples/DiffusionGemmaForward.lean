import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Models.DiffusionGemma.Loader
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Basic

/-!
# DiffusionGemma native 30-layer forward (single token) → first logits

Correct sequential `forwardBlock` (proper residual data flow) looped over all
30 layers, then tied LM head + softcap → logits, on the real 26B model on Metal.

Single-token (seqLen=1): attention softmax over one key = 1, so attn_out = V
→ wO (Q/K/qk-norm/RoPE don't affect the output and are skipped here; they're
validated separately for seqLen>1).  Per-op host round-trips for bring-up
correctness; batching is the performance step.

Run:  lake exe diffusiongemma-forward [path] [nLayers]
-/

open Hesper.WebGPU
open Hesper.Models.DiffusionGemma

abbrev B := Hesper.GPUBackend.Buf Device
abbrev C := Hesper.GPUBackend.CachedDispatch Device
abbrev Lin := Hesper.Layers.Linear.LinearLayer B C
abbrev Nrm := Hesper.Layers.RMSNorm.RMSNorm B C
abbrev Blk := Hesper.Models.Gemma4.Gemma4Block B C
abbrev Cfg := Hesper.Models.Gemma4.Config

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

def dlBuf (device : Device) (b : Buffer) (n : Nat) : IO (Array Float) := do
  let r ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device b 0 (n*4).toUSize)
  unmapBuffer b
  return r

def linF (device : Device) (layer : Lin) (x : Array Float) (outDim : Nat) : IO (Array Float) := do
  let ib ← mkBuf device x.size
  writeBuffer device ib 0 (← Hesper.Basic.floatArrayToBytes x)
  let ob ← mkBuf device outDim
  Hesper.Layers.Linear.LinearLayer.forward device layer ib ob
  dlBuf device ob outDim

def rmsF (device : Device) (norm : Nrm) (x : Array Float) : IO (Array Float) := do
  let ib ← mkBuf device x.size
  writeBuffer device ib 0 (← Hesper.Basic.floatArrayToBytes x)
  let ob ← mkBuf device x.size
  Hesper.Layers.RMSNorm.forward device norm ib ob
  dlBuf device ob x.size

/-- Dispatch an expert-indexed kernel (weights already on GPU) on host input. -/
def expF (device : Device) (weightsBuf : Buffer) (kern : Hesper.WGSL.Monad.ShaderM Unit) (x : Array Float) (outDim : Nat) : IO (Array Float) := do
  let ib ← mkBuf device x.size
  writeBuffer device ib 0 (← Hesper.Basic.floatArrayToBytes x)
  let ob ← mkBuf device outDim
  let bufs := ("weights", weightsBuf) :: ("input", ib) :: ("output", ob) :: List.nil
  let cfg : Hesper.ExecConfig := { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device kern bufs cfg
  dlBuf device ob outDim

@[inline] def geluH (g : Float) : Float := 0.5*g*(1.0 + Float.tanh (0.7978845608 * (g + 0.044715*g*g*g)))

def vadd (a b : Array Float) : Array Float := Id.run do
  let mut o := a
  for i in [0:a.size] do o := o.set! i (a[i]! + b[i]!)
  return o

/-- One transformer block, correct sequential data flow (single token). -/
def forwardBlock (device : Device) (cfg : Cfg) (blk : Blk) (x : Array Float) : IO (Array Float) := do
  let dim := cfg.hiddenSize
  let li := blk.layerIdx
  let headDim := cfg.headDim li
  let nKV := cfg.numKVHeads li
  let kvDim := nKV * headDim
  let qDim := cfg.numAttentionHeads * headDim
  let grp := cfg.numAttentionHeads / nKV
  -- attention (seqLen=1): attn_head = V[kv_head(h)] → wO
  let normed ← rmsF device blk.attnNorm x
  let v ← linF device blk.attention.wV normed kvDim
  let mut attnVec := Array.replicate qDim 0.0
  for h in [0:cfg.numAttentionHeads] do
    let kvh := h / grp
    for d in [0:headDim] do
      attnVec := attnVec.set! (h*headDim + d) (v[kvh*headDim + d]!)
  let wo ← linF device blk.attention.wO attnVec dim
  let pa ← rmsF device blk.postAttnNorm wo
  let r := vadd pa x                                   -- attention residual
  -- dense shared expert FFN on r
  let ffn := cfg.intermediateSize
  let m ← rmsF device blk.ffnNorm r
  let gate ← linF device blk.ffn.gate m ffn
  let up ← linF device blk.ffn.up m ffn
  let mut geg := Array.replicate ffn 0.0
  for i in [0:ffn] do geg := geg.set! i (geluH gate[i]! * up[i]!)
  let dmlp ← linF device blk.ffn.down geg dim
  let some pn1 := blk.moePostNorm1 | throw (IO.userError "no post_ffw_norm_1")
  let curMlp ← rmsF device pn1 dmlp
  -- MoE on r
  let some preN2 := blk.moePreNorm2 | throw (IO.userError "no pre_ffw_norm_2")
  let some gateUpExps := blk.moeGateUpExps | throw (IO.userError "no gate_up_exps")
  let some downExps := blk.moeDownExps | throw (IO.userError "no down_exps")
  let some routerWb := blk.moeRouterWeight | throw (IO.userError "no router weight")
  let some routerSb := blk.moeRouterScale | throw (IO.userError "no router scale")
  let nExp := cfg.numExperts
  let ffExp := cfg.expertFFSize
  let moeIn ← rmsF device preN2 r
  let routerWA ← dlBuf device routerWb (nExp*dim)
  let routerSA ← dlBuf device routerSb dim
  -- router prep: rmsnorm-noscale(r) × 1/√d × gate_inp_s
  let mut rss := 0.0
  for i in [0:dim] do rss := rss + r[i]! * r[i]!
  let invR := 1.0 / Float.sqrt (rss / dim.toFloat + cfg.rmsNormEps)
  let invSqrtD := 1.0 / Float.sqrt dim.toFloat
  let mut rtmp := Array.replicate dim 0.0
  for i in [0:dim] do rtmp := rtmp.set! i (r[i]! * invR * invSqrtD * routerSA[i]!)
  let mut rlog := Array.replicate nExp 0.0
  for e in [0:nExp] do
    let mut s := 0.0
    for i in [0:dim] do s := s + routerWA[e*dim + i]! * rtmp[i]!
    rlog := rlog.set! e s
  let mut rmx := rlog[0]!
  for e in [1:nExp] do if rlog[e]! > rmx then rmx := rlog[e]!
  let mut rsum := 0.0
  let mut rprob := Array.replicate nExp 0.0
  for e in [0:nExp] do
    let pe := Float.exp (rlog[e]! - rmx); rprob := rprob.set! e pe; rsum := rsum + pe
  for e in [0:nExp] do rprob := rprob.set! e (rprob[e]! / rsum)
  let mut chosen : Array (Nat × Float) := #[]
  let mut rused := Array.replicate nExp false
  for _ in [0:cfg.numExpertsUsed] do
    let mut bi := 0
    let mut bv := -1.0
    for e in [0:nExp] do
      if !rused[e]! && rprob[e]! > bv then bv := rprob[e]!; bi := e
    rused := rused.set! bi true
    chosen := chosen.push (bi, bv)
  let wSum := chosen.foldl (fun a y => a + y.2) 0.0
  let mut moeSum := Array.replicate dim 0.0
  for y in chosen do
    let e := y.1
    let gu ← expF device gateUpExps (Hesper.Layers.Linear.fusedQ4KMExpertKernel { inDim := dim, outDim := 2*ffExp } nExp e) moeIn (2*ffExp)
    let mut eg := Array.replicate ffExp 0.0
    for i in [0:ffExp] do eg := eg.set! i (geluH gu[i]! * gu[ffExp + i]!)
    let ed ← expF device downExps (Hesper.Layers.Linear.fusedQ8_0ExpertKernel { inDim := ffExp, outDim := dim } nExp e) eg dim
    let ww := y.2 / wSum
    for i in [0:dim] do moeSum := moeSum.set! i (moeSum[i]! + ww * ed[i]!)
  let some pn2 := blk.moePostNorm2 | throw (IO.userError "no post_ffw_norm_2")
  let curMoe ← rmsF device pn2 moeSum
  -- combine + ffn_post_norm + residual + per-layer scale
  let mut comb := Array.replicate dim 0.0
  for i in [0:dim] do comb := comb.set! i (curMlp[i]! + curMoe[i]!)
  let combN ← rmsF device blk.postFFNNorm comb
  let outScaleV ← match blk.outScale with
    | some b => do let a ← dlBuf device b 1; pure a[0]!
    | none => pure 1.0
  let resid := vadd combN r
  let mut blockOut := Array.replicate dim 0.0
  for i in [0:dim] do blockOut := blockOut.set! i (resid[i]! * outScaleV)
  return blockOut

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
  IO.println "[dg-forward] init WebGPU (Metal) + load model..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let cfg := model.inner.config
  let dim := cfg.hiddenSize

  -- single-token synthetic hidden input (bring-up); real path embeds a token ×√d
  let mut hidden := (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.01) * 0.5)
  IO.println s!"[dg-forward] running {nLayers}-layer forward (single token)..."
  for li in [0:nLayers] do
    let some blk := model.inner.blocks[li]? | throw (IO.userError s!"no block {li}")
    hidden ← forwardBlock device cfg blk hidden
    if li % 5 == 0 || li + 1 == nLayers then
      let fin := hidden.all Float.isFinite
      IO.println s!"  layer {li}: finite={fin}  hidden[0..3]={(hidden.extract 0 3).toList}"

  -- final norm + tied LM head (Q6_K) + softcap → logits.
  -- NOTE: WebGPU caps workgroups/dim at 65535; full vocab=262144 needs a
  -- tiled/2D dispatch (follow-up).  Bring-up computes a dispatchable slice.
  let x ← rmsF device model.inner.finalNorm hidden
  let lmN := min cfg.vocabSize 32768
  let lmHead ← mkLin device model.inner.outputWeight dim lmN .Q6_K
  IO.println s!"[dg-forward] lm_head Q6_K matmul → {lmN} logits (vocab slice; full {cfg.vocabSize} needs tiled dispatch)..."
  let logits ← linF device lmHead x lmN
  let cap := cfg.logitSoftcapScale
  let mut top := 0
  let mut topV := -1e30
  for i in [0:lmN] do
    let l := cap * Float.tanh (logits[i]! / cap)
    if l > topV then topV := l; top := i
  let logFinite := logits.all Float.isFinite
  IO.println s!"[dg-forward] logits finite={logFinite} size={logits.size}  argmax token={top} (softcapped logit={topV})"
  if logFinite && logits.size == lmN then
    IO.println s!"✓ native {nLayers}-layer DiffusionGemma forward → logits on Metal (real 26B model)"
  else
    IO.println "✗ forward failed"
    throw (IO.userError "forward failed")
