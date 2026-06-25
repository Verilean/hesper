/-!
# DiffusionGemma CPU reference forward + diffusion decode

A self-contained, GPU-free reference implementation of the DiffusionGemma
forward pass and the masked-diffusion decode loop, in plain `Float` arrays.

This is the *executable spec* for the architecture verified in
`refs/diffusiongemma/ARCH_NOTES.md` (derived from llama.cpp PR #24423,
`src/models/diffusion-gemma.cpp` + `gemma4-common.h`).  It is the
"UNIFIED, zero-self-conditioning, no-KV-cache" exactness path — a single
bidirectional forward over `[prompt | canvas]`.

Purpose:
- runs on a tiny random config with no GPU and no 16 GB model load, so the
  graph logic (region embeddings, bidirectional mask, V-reuse on global
  layers, q/k-norm, RoPE, dense + 128-expert-MoE FFN, dual per-layer
  scales, tied-LM-head + logit softcap, denoise loop) is testable locally;
- serves as the golden reference the eventual GPU port validates against
  (same pattern as `Tests/RTLCoSim.lean`).

Numerics that are structurally faithful but not yet bit-exact vs llama.cpp
(to revisit when validating the GPU port against the real weights):
- RoPE uses full-width NeoX rotation with `n_rot = headDim`; `rope_freqs`
  proportional scaling on global layers is treated as 1.0;
- RMSNorm multiplies by the raw weight (the Gemma `1 + w` convention, if
  present in these weights, would be folded in at load time).
-/

namespace Hesper.Models.DiffusionGemma.Reference

/-! ## Small linear-algebra helpers (row-major `Array Float`) -/

/-- `w` is row-major `[outDim, inDim]`; returns `w · x` of length `outDim`. -/
def matVec (w : Array Float) (x : Array Float) (outDim inDim : Nat) : Array Float := Id.run do
  let mut o := Array.replicate outDim 0.0
  for r in [0:outDim] do
    let base := r * inDim
    let mut s := 0.0
    for c in [0:inDim] do
      s := s + w[base + c]! * x[c]!
    o := o.set! r s
  return o

/-- RMSNorm over the whole vector; multiplies by `weight` if given. -/
def rmsNorm (x : Array Float) (weight : Option (Array Float)) (eps : Float) : Array Float := Id.run do
  let n := x.size
  if n == 0 then return x
  let mut ss := 0.0
  for i in [0:n] do
    ss := ss + x[i]! * x[i]!
  let inv := 1.0 / Float.sqrt (ss / n.toFloat + eps)
  let mut o := Array.replicate n 0.0
  for i in [0:n] do
    let w := match weight with | some w => w[i]! | none => 1.0
    o := o.set! i (x[i]! * inv * w)
  return o

/-- tanh-approximation GELU (matches ggml's `ggml_gelu`). -/
def gelu (x : Float) : Float :=
  let c := 0.7978845608028654  -- sqrt(2/pi)
  0.5 * x * (1.0 + Float.tanh (c * (x + 0.044715 * x * x * x)))

/-- Numerically-stable softmax. -/
def softmax (x : Array Float) : Array Float := Id.run do
  let n := x.size
  if n == 0 then return x
  let mut mx := x[0]!
  for i in [1:n] do
    if x[i]! > mx then mx := x[i]!
  let mut e := Array.replicate n 0.0
  let mut s := 0.0
  for i in [0:n] do
    let v := Float.exp (x[i]! - mx)
    e := e.set! i v
    s := s + v
  for i in [0:n] do
    e := e.set! i (e[i]! / s)
  return e

/-- Index of the maximum element. -/
def argmax (x : Array Float) : Nat := Id.run do
  let mut bi := 0
  let mut bv := x[0]!
  for i in [1:x.size] do
    if x[i]! > bv then bv := x[i]!; bi := i
  return bi

/-- Elementwise add. -/
def vadd (a b : Array Float) : Array Float := Id.run do
  let mut o := a
  for i in [0:a.size] do o := o.set! i (a[i]! + b[i]!)
  return o

/-- NeoX-style RoPE on a single head vector (`n_rot = hd`). -/
def ropeHead (v : Array Float) (pos : Nat) (hd : Nat) (theta : Float) : Array Float := Id.run do
  let half := hd / 2
  let logTheta := Float.log theta
  let mut o := v
  for j in [0:half] do
    let freq := Float.exp (-(2.0 * j.toFloat / hd.toFloat) * logTheta)
    let ang := pos.toFloat * freq
    let cosA := Float.cos ang
    let sinA := Float.sin ang
    let a := v[j]!
    let b := v[j + half]!
    o := o.set! j (a * cosA - b * sinA)
    o := o.set! (j + half) (a * sinA + b * cosA)
  return o

/-! ## Model structures -/

structure RefAttn where
  wq : Array Float            -- [nHead*hd, dim]
  wk : Array Float            -- [nKV*hd, dim]
  wv : Option (Array Float)   -- [nKV*hd, dim]; none on global layers (V reuses K proj)
  wo : Array Float            -- [dim, nHead*hd]
  qNorm : Array Float         -- [hd]
  kNorm : Array Float         -- [hd]
  deriving Inhabited

structure RefExpert where
  gate : Array Float          -- [expertFF, dim]
  up   : Array Float          -- [expertFF, dim]
  down : Array Float          -- [dim, expertFF]
  deriving Inhabited

structure RefBlock where
  isFull   : Bool
  headDim  : Nat
  nKVHeads : Nat
  ropeTheta : Float
  attnNorm : Array Float
  postAttnNorm : Array Float
  attn : RefAttn
  -- dense shared expert
  ffnNorm : Array Float
  gate : Array Float          -- [ffn, dim]
  up   : Array Float          -- [ffn, dim]
  down : Array Float          -- [dim, ffn]
  ffnPostNorm1 : Array Float
  -- MoE
  ffnPreNorm2 : Array Float
  routerW : Array Float       -- [nExpert, dim]
  routerScale : Array Float   -- [dim]
  experts : Array RefExpert
  ffnPostNorm2 : Array Float
  -- combine + residual
  ffnPostNorm : Array Float
  outScale : Float            -- canvas (decoder) per-layer scalar
  encOutScale : Float         -- prompt (encoder) per-layer scalar
  deriving Inhabited

structure RefModel where
  dim : Nat
  nHead : Nat
  ffn : Nat
  nExpert : Nat
  nExpertUsed : Nat
  expertFF : Nat
  vocab : Nat
  canvasLen : Nat
  maskToken : Nat
  slidingWindow : Nat
  eps : Float
  softcap : Float
  tokEmbd : Array Float        -- [vocab, dim]  (LM head is tied to this)
  blocks : Array RefBlock
  outputNorm : Array Float
  deriving Inhabited

/-! ## Forward pass -/

/-- One transformer block over the whole `[prompt | canvas]` sequence.
    `hs` is `n` rows of length `dim`; `P` = prompt length, `C` = canvas.
    Returns the new `n` rows (post per-layer scale). -/
def blockForward (m : RefModel) (blk : RefBlock) (hs : Array (Array Float))
    (P C : Nat) : Array (Array Float) := Id.run do
  let n := P + C
  let hd := blk.headDim
  let nKV := blk.nKVHeads
  let qDim := m.nHead * hd
  let kvDim := nKV * hd
  let scale := 1.0  -- f_attention_scale = 1.0 (no 1/sqrt(d) pre-scaling)
  let groupSize := m.nHead / nKV  -- GQA: query heads per kv head

  -- Per-row projections: Q heads (normed+roped), K heads (normed+roped), V heads (vnorm, no rope)
  let mut qH : Array (Array (Array Float)) := #[]  -- [row][head][hd]
  let mut kH : Array (Array (Array Float)) := #[]  -- [row][kvHead][hd]
  let mut vH : Array (Array (Array Float)) := #[]  -- [row][kvHead][hd]
  for r in [0:n] do
    let normed := rmsNorm hs[r]! (some blk.attnNorm) m.eps
    let q := matVec blk.attn.wq normed qDim m.dim
    let kRaw := matVec blk.attn.wk normed kvDim m.dim
    let vRaw := match blk.attn.wv with
      | some wv => matVec wv normed kvDim m.dim
      | none    => kRaw          -- global layers: V reuses the raw K projection
    -- Q heads: per-head q-norm then RoPE
    let mut qHeads : Array (Array Float) := #[]
    for h in [0:m.nHead] do
      let head := (q.extract (h*hd) (h*hd+hd))
      let head := rmsNorm head (some blk.attn.qNorm) m.eps
      let head := ropeHead head r hd blk.ropeTheta
      qHeads := qHeads.push head
    -- K heads: per-head k-norm then RoPE; V heads: v-norm (no scale), no RoPE
    let mut kHeads : Array (Array Float) := #[]
    let mut vHeads : Array (Array Float) := #[]
    for h in [0:nKV] do
      let kh := (kRaw.extract (h*hd) (h*hd+hd))
      let kh := rmsNorm kh (some blk.attn.kNorm) m.eps
      let kh := ropeHead kh r hd blk.ropeTheta
      kHeads := kHeads.push kh
      let vh := (vRaw.extract (h*hd) (h*hd+hd))
      let vh := rmsNorm vh none m.eps    -- v-norm: no scale, no rope
      vHeads := vHeads.push vh
    qH := qH.push qHeads
    kH := kH.push kHeads
    vH := vH.push vHeads

  -- Region-aware allow predicate (additive mask). swaClip omitted when the
  -- sliding window covers the whole sequence (n <= slidingWindow).
  let nSwa := m.slidingWindow
  let allow := fun (q k : Nat) =>
    let qCanvas := q ≥ P
    let kCanvas := k ≥ P
    if qCanvas then
      if blk.isFull then true
      else kCanvas || (k + nSwa ≥ P + 1)   -- k ≥ P-(nSwa-1)
    else
      (!kCanvas) && (k ≤ q) && (blk.isFull || (q < k + nSwa))

  -- Attention + output projection per row
  let mut attnOut : Array (Array Float) := #[]
  for q in [0:n] do
    let mut ctx := Array.replicate qDim 0.0
    for h in [0:m.nHead] do
      let kvh := h / groupSize
      let qv := qH[q]![h]!
      -- scores over allowed keys
      let mut scores : Array Float := #[]
      let mut keys : Array Nat := #[]
      for k in [0:n] do
        if allow q k then
          let kv := kH[k]![kvh]!
          let mut s := 0.0
          for d in [0:hd] do s := s + qv[d]! * kv[d]!
          scores := scores.push (s * scale)
          keys := keys.push k
      let w := softmax scores
      -- weighted sum of V
      for ki in [0:keys.size] do
        let k := keys[ki]!
        let vv := vH[k]![kvh]!
        let wk := w[ki]!
        for d in [0:hd] do
          ctx := ctx.set! (h*hd + d) (ctx[h*hd + d]! + wk * vv[d]!)
    let o := matVec blk.attn.wo ctx m.dim qDim
    attnOut := attnOut.push o

  -- post-attn norm + residual
  let mut postAttn : Array (Array Float) := #[]
  for r in [0:n] do
    let o := rmsNorm attnOut[r]! (some blk.postAttnNorm) m.eps
    postAttn := postAttn.push (vadd o hs[r]!)

  -- dense MLP + MoE + combine + ffn_post_norm + residual + region scale
  let mut out : Array (Array Float) := #[]
  for r in [0:n] do
    let x := postAttn[r]!
    -- dense shared expert
    let mlpN := rmsNorm x (some blk.ffnNorm) m.eps
    let g := matVec blk.gate mlpN m.ffn m.dim
    let u := matVec blk.up mlpN m.ffn m.dim
    let mut h := Array.replicate m.ffn 0.0
    for i in [0:m.ffn] do h := h.set! i (gelu g[i]! * u[i]!)
    let dMlp := matVec blk.down h m.dim m.ffn
    let curMlp := rmsNorm dMlp (some blk.ffnPostNorm1) m.eps
    -- MoE
    let moeN := rmsNorm x (some blk.ffnPreNorm2) m.eps
    -- router on the unnormed post-attn residual
    let tmp := rmsNorm x none m.eps
    let invSqrt := 1.0 / Float.sqrt m.dim.toFloat
    let mut tmpS := Array.replicate m.dim 0.0
    for i in [0:m.dim] do tmpS := tmpS.set! i (tmp[i]! * invSqrt * blk.routerScale[i]!)
    let rLogits := matVec blk.routerW tmpS m.nExpert m.dim
    let rProbs := softmax rLogits
    -- top-k experts by probability
    let mut chosen : Array (Nat × Float) := #[]
    let mut used := Array.replicate m.nExpert false
    for _ in [0:m.nExpertUsed] do
      let mut bi := 0
      let mut bv := -1.0
      for e in [0:m.nExpert] do
        if !used[e]! && rProbs[e]! > bv then bv := rProbs[e]!; bi := e
      used := used.set! bi true
      chosen := chosen.push (bi, bv)
    -- normalize selected gate weights (norm_w = true)
    let wSum := chosen.foldl (fun acc (_, w) => acc + w) 0.0
    let mut moeAcc := Array.replicate m.dim 0.0
    for (e, w) in chosen do
      let ge := blk.experts[e]!
      let eg := matVec ge.gate moeN m.expertFF m.dim
      let eu := matVec ge.up moeN m.expertFF m.dim
      let mut eh := Array.replicate m.expertFF 0.0
      for i in [0:m.expertFF] do eh := eh.set! i (gelu eg[i]! * eu[i]!)
      let ed := matVec ge.down eh m.dim m.expertFF
      let ww := w / wSum
      for i in [0:m.dim] do moeAcc := moeAcc.set! i (moeAcc[i]! + ww * ed[i]!)
    let curMoe := rmsNorm moeAcc (some blk.ffnPostNorm2) m.eps
    -- combine, post-norm, residual
    let mut comb := Array.replicate m.dim 0.0
    for i in [0:m.dim] do comb := comb.set! i (curMlp[i]! + curMoe[i]!)
    let combN := rmsNorm comb (some blk.ffnPostNorm) m.eps
    let resid := vadd combN x
    -- region-aware per-layer scalar
    let sc := if r ≥ P then blk.outScale else blk.encOutScale
    let mut scaled := Array.replicate m.dim 0.0
    for i in [0:m.dim] do scaled := scaled.set! i (resid[i]! * sc)
    out := out.push scaled
  return out

/-- Full forward over `[prompt | canvas]`.  `P` = prompt length; the rest of
    `tokens` is the canvas.  Returns `n` rows of `vocab` logits. -/
def forward (m : RefModel) (tokens : Array Nat) (P : Nat) : Array (Array Float) := Id.run do
  let n := tokens.size
  let C := n - P
  let embScale := Float.sqrt m.dim.toFloat
  -- region-aware input embeddings
  let mut hs : Array (Array Float) := #[]
  for r in [0:n] do
    let t := tokens[r]!
    let mut e := Array.replicate m.dim 0.0
    for i in [0:m.dim] do e := e.set! i (m.tokEmbd[t*m.dim + i]! * embScale)
    -- canvas rows: rms_norm (no scale) of the scaled embedding (zero-SC)
    let eFinal := if r ≥ P then rmsNorm e none m.eps else e
    hs := hs.push eFinal
  -- transformer blocks
  for blk in m.blocks do
    hs := blockForward m blk hs P C
  -- final norm + tied LM head + logit softcap
  let mut logits : Array (Array Float) := #[]
  for r in [0:n] do
    let x := rmsNorm hs[r]! (some m.outputNorm) m.eps
    let l := matVec m.tokEmbd x m.vocab m.dim   -- output tied to tok_embd
    let mut sc := Array.replicate m.vocab 0.0
    for i in [0:m.vocab] do
      sc := sc.set! i (m.softcap * Float.tanh (l[i]! / m.softcap))
    logits := logits.push sc
  return logits

/-! ## Masked-diffusion decode loop -/

/-- Confidence-based masked-diffusion decode.  Seeds the canvas with
    `maskToken`, then for `steps` iterations runs a full bidirectional
    forward, commits the highest-confidence masked canvas positions
    (linear unmask schedule), and repeats until none remain.  Returns the
    final canvas tokens. -/
def decode (m : RefModel) (promptTokens : Array Nat) (steps : Nat) : Array Nat := Id.run do
  let P := promptTokens.size
  let C := m.canvasLen
  let mut tokens := promptTokens ++ Array.replicate C m.maskToken
  let mut masked := Array.replicate C true
  let realSteps := max 1 (min steps C)
  for step in [0:realSteps] do
    let remaining := masked.foldl (fun acc b => if b then acc + 1 else acc) 0
    if remaining == 0 then
      pure ()
    else
      let logits := forward m tokens P
      -- confidence + prediction for each masked canvas position
      let mut cand : Array (Nat × Nat × Float) := #[]  -- (canvasIdx, pred, conf)
      for i in [0:C] do
        if masked[i]! then
          let probs := softmax logits[P + i]!
          let pred := argmax probs
          cand := cand.push (i, pred, probs[pred]!)
      -- how many to unmask this step (linear schedule, finish on last step)
      let want :=
        if step + 1 ≥ realSteps then remaining
        else max 1 ((C + realSteps - 1) / realSteps)
      let k := min want cand.size
      -- pick the k highest-confidence candidates
      let mut picked := Array.replicate cand.size false
      for _ in [0:k] do
        let mut bi := 0
        let mut bv := -1.0
        for c in [0:cand.size] do
          let (_, _, conf) := cand[c]!
          if !picked[c]! && conf > bv then bv := conf; bi := c
        picked := picked.set! bi true
        let (ci, pred, _) := cand[bi]!
        tokens := tokens.set! (P + ci) pred
        masked := masked.set! ci false
  return tokens.extract P (P + C)

end Hesper.Models.DiffusionGemma.Reference
