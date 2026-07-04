import Hesper.Models.DiffusionGemma.Reference
import Hesper.Basic

/-!
# DiffusionGemma attention-core parity vs ggml golden

Validates the scaled-dot-product attention core (scale=1.0, additive
region mask, single head) — `softmax(K^T·Q + mask) · V` — against the
ggml golden from `scripts/llama_parity/dump_dg_attn_golden`.  This is the
numeric heart of `build_attn`; the GQA head mapping, V-reuse on global
layers, and the region-mask *predicate* are structural (exercised by the
tiny-config test).

Layout (ggml, ne0 fastest): Q=[hd,nq] K=V=[hd,nk] mask=[nk,nq] out=[hd,nq].

Run:
  ./scripts/llama_parity/dump_dg_attn_golden /tmp/dg_golden/attn
  lake exe diffusiongemma-attn-parity
-/

open Hesper.Models.DiffusionGemma.Reference

def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)    ).toUInt32
    let b1 := (bytes.get! (i*4 + 1)).toUInt32
    let b2 := (bytes.get! (i*4 + 2)).toUInt32
    let b3 := (bytes.get! (i*4 + 3)).toUInt32
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)))
  return a

def main (args : List String) : IO Unit := do
  let dir := args.head?.getD "/tmp/dg_golden/attn"
  let hd := 8; let nq := 3; let nk := 5
  let Q ← readF32Bin s!"{dir}/q.bin"
  let K ← readF32Bin s!"{dir}/k.bin"
  let V ← readF32Bin s!"{dir}/v.bin"
  let mask ← readF32Bin s!"{dir}/mask.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"

  let mut out := Array.replicate (hd * nq) 0.0
  for q in [0:nq] do
    -- scores over keys: dot(Q[q], K[k]) * scale(1.0) + mask[k,q]
    let mut scores := Array.replicate nk 0.0
    for k in [0:nk] do
      let mut s := 0.0
      for d in [0:hd] do
        s := s + Q[q*hd + d]! * K[k*hd + d]!
      scores := scores.set! k (s + mask[q*nk + k]!)
    let probs := softmax scores
    -- weighted sum of V
    for d in [0:hd] do
      let mut acc := 0.0
      for k in [0:nk] do
        acc := acc + probs[k]! * V[k*hd + d]!
      out := out.set! (q*hd + d) acc

  let mut err := 0.0
  for i in [0:hd*nq] do
    let dd := (out[i]! - gold[i]!).abs
    if dd > err then err := dd
  IO.println s!"[dg-attn-parity] hd={hd} nq={nq} nk={nk} maxAbsErr={err}"
  IO.println s!"  ref[0..4]  = {(out.extract 0 4).toList}"
  IO.println s!"  gold[0..4] = {(gold.extract 0 4).toList}"
  if out.size == gold.size && err < 1e-4 then
    IO.println "✓ PASS"
  else
    IO.println "✗ FAIL"
    throw (IO.userError "attn parity failed")
