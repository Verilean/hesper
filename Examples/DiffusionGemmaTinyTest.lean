import Hesper.Models.DiffusionGemma.Reference

/-!
# DiffusionGemma tiny-config test

Builds a small DiffusionGemma model with deterministic pseudo-random
weights and exercises the CPU reference forward + masked-diffusion decode
end-to-end — no GPU, no 16 GB model.  It checks every code path runs and
produces finite results:
- dual attention geometry (SWA layers with V-proj, a global layer that
  omits V and reuses the K projection),
- per-head q/k-norm + RoPE, bidirectional region mask,
- dense shared-expert + top-k MoE, dual per-layer scales,
- tied LM head + logit softcap,
- the confidence-based denoise loop converging the canvas.

Run:  `lake exe diffusiongemma-tiny-test`
-/

open Hesper.Models.DiffusionGemma.Reference

/-- Deterministic LCG PRNG. -/
structure Rng where
  val : UInt64
  deriving Inhabited

def Rng.nextFloat (r : Rng) : Float × Rng :=
  let s := r.val * 6364136223846793005 + 1442695040888963407
  let u := (s >>> 40).toNat.toFloat / (16777216.0 : Float)  -- [0,1)
  (2.0 * u - 1.0, ⟨s⟩)                                       -- [-1,1)

def nextF (rng : IO.Ref Rng) : IO Float := do
  let (v, r') := (← rng.get).nextFloat
  rng.set r'
  return v

/-- Random array in `[-scale, scale)`. -/
def gArr (rng : IO.Ref Rng) (n : Nat) (scale : Float) : IO (Array Float) := do
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    a := a.set! i ((← nextF rng) * scale)
  return a

/-- Norm weight ~ `1 + small`. -/
def gNorm (rng : IO.Ref Rng) (n : Nat) : IO (Array Float) := do
  let a ← gArr rng n 0.1
  return a.map (1.0 + ·)

def gFloat (rng : IO.Ref Rng) (center scale : Float) : IO Float := do
  return center + (← nextF rng) * scale

def gExpert (rng : IO.Ref Rng) (dim expertFF : Nat) : IO RefExpert := do
  return {
    gate := ← gArr rng (expertFF * dim) 0.2
    up   := ← gArr rng (expertFF * dim) 0.2
    down := ← gArr rng (dim * expertFF) 0.2
  }

def gBlock (rng : IO.Ref Rng) (dim nHead ffn nExpert expertFF : Nat)
    (isFull : Bool) : IO RefBlock := do
  let hd := if isFull then 8 else 4
  let nKV := if isFull then 1 else 2
  let qDim := nHead * hd
  let kvDim := nKV * hd
  let wq ← gArr rng (qDim * dim) 0.2
  let wk ← gArr rng (kvDim * dim) 0.2
  let wv ← if isFull then pure none else (do return some (← gArr rng (kvDim * dim) 0.2))
  let wo ← gArr rng (dim * qDim) 0.2
  let qNorm ← gNorm rng hd
  let kNorm ← gNorm rng hd
  let attn : RefAttn := { wq, wk, wv, wo, qNorm, kNorm }
  let mut experts : Array RefExpert := #[]
  for _ in [0:nExpert] do
    experts := experts.push (← gExpert rng dim expertFF)
  return {
    isFull, headDim := hd, nKVHeads := nKV
    ropeTheta := if isFull then 1000000.0 else 10000.0
    attnNorm := ← gNorm rng dim
    postAttnNorm := ← gNorm rng dim
    attn
    ffnNorm := ← gNorm rng dim
    gate := ← gArr rng (ffn * dim) 0.2
    up   := ← gArr rng (ffn * dim) 0.2
    down := ← gArr rng (dim * ffn) 0.2
    ffnPostNorm1 := ← gNorm rng dim
    ffnPreNorm2 := ← gNorm rng dim
    routerW := ← gArr rng (nExpert * dim) 0.2
    routerScale := ← gNorm rng dim
    experts
    ffnPostNorm2 := ← gNorm rng dim
    ffnPostNorm := ← gNorm rng dim
    outScale := ← gFloat rng 1.0 0.1
    encOutScale := ← gFloat rng 1.0 0.1
  }

def buildModel (rng : IO.Ref Rng) : IO RefModel := do
  let dim := 16
  let nHead := 4
  let ffn := 24
  let nExpert := 4
  let nExpertUsed := 2
  let expertFF := 8
  let vocab := 32
  let layerIsFull := #[false, true, false]   -- SWA, FULL, SWA
  let mut blocks : Array RefBlock := #[]
  for isFull in layerIsFull do
    blocks := blocks.push (← gBlock rng dim nHead ffn nExpert expertFF isFull)
  return {
    dim, nHead, ffn, nExpert, nExpertUsed, expertFF, vocab
    canvasLen := 6
    maskToken := vocab - 1
    slidingWindow := 4096
    eps := 1e-6
    softcap := 30.0
    tokEmbd := ← gArr rng (vocab * dim) 0.2
    blocks
    outputNorm := ← gNorm rng dim
  }

def allFinite (rows : Array (Array Float)) : Bool :=
  rows.all (·.all Float.isFinite)

def main : IO Unit := do
  let rng ← IO.mkRef (⟨0x1234567⟩ : Rng)
  let m ← buildModel rng
  IO.println s!"[tiny] model: dim={m.dim} heads={m.nHead} layers={m.blocks.size} vocab={m.vocab} experts={m.nExpert}/{m.nExpertUsed} canvas={m.canvasLen}"
  let mut ok := true

  -- 1. Forward over [prompt | canvas]
  let prompt := #[1, 5, 9, 2]
  let P := prompt.size
  let tokens := prompt ++ Array.replicate m.canvasLen m.maskToken
  let logits := forward m tokens P
  IO.println s!"[tiny] forward: {logits.size} rows × {logits[0]!.size} vocab (expected {tokens.size} × {m.vocab})"
  let shapeOk := logits.size == tokens.size && logits.all (·.size == m.vocab)
  let finOk := allFinite logits
  IO.println s!"  shape ok = {shapeOk}   all-finite = {finOk}"
  ok := ok && shapeOk && finOk
  -- softcap bound: |logit| <= softcap
  let bounded := logits.all (·.all (fun x => x.abs ≤ m.softcap + 1e-3))
  IO.println s!"  logits within ±softcap = {bounded}"
  ok := ok && bounded

  -- 2. Determinism: same inputs → identical logits
  let logits2 := forward m tokens P
  let det := logits[0]! == logits2[0]!
  IO.println s!"[tiny] deterministic = {det}"
  ok := ok && det

  -- 3. Masked-diffusion decode converges the canvas
  let out := decode m prompt 4
  let noMask := out.all (· != m.maskToken)
  let inVocab := out.all (· < m.vocab)
  IO.println s!"[tiny] decode → canvas tokens = {out.toList}"
  IO.println s!"  size ok = {out.size == m.canvasLen}   no mask left = {noMask}   in vocab = {inVocab}"
  ok := ok && (out.size == m.canvasLen) && noMask && inVocab

  if ok then
    IO.println "✓ tiny-config test PASS — forward + decode exercise all paths with finite, bounded output."
  else
    IO.println "✗ tiny-config test FAIL"
    throw (IO.userError "tiny-config test failed")
