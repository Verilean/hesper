import Hesper.Models.DiffusionGemma.Reference
import Hesper.Basic

/-!
# DiffusionGemma RoPE (NeoX) parity vs ggml golden

Validates `Reference.ropeHead` against `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX)`
(the exact call gemma4-common.h uses).  Golden produced by
`scripts/llama_parity/dump_dg_rope_golden`.  Input layout is
`[hd, nHead, nTok]` (ne0=hd fastest), positions = 0..nTok-1.

Run:
  ./scripts/llama_parity/dump_dg_rope_golden /tmp/dg_golden/rope
  lake exe diffusiongemma-rope-parity
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
  let dir := args.head?.getD "/tmp/dg_golden/rope"
  IO.println s!"[dg-rope-parity] golden dir = {dir}"
  let hd := 8
  let nHead := 2
  let nTok := 4
  let theta := 10000.0
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"

  -- apply RoPE per (token, head) and assemble in the same [hd, nHead, nTok] layout
  let mut out := Array.replicate x.size 0.0
  for t in [0:nTok] do
    for h in [0:nHead] do
      let base := (t * nHead + h) * hd
      let head := x.extract base (base + hd)
      let r := ropeHead head t hd theta
      for i in [0:hd] do
        out := out.set! (base + i) r[i]!

  let mut err := 0.0
  for i in [0:x.size] do
    let d := (out[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  n={x.size}  maxAbsErr={err}"
  IO.println s!"  ref[0..6]  = {(out.extract 0 6).toList}"
  IO.println s!"  gold[0..6] = {(gold.extract 0 6).toList}"
  let tol := 1e-4
  if out.size == gold.size && err < tol then
    IO.println s!"✓ PASS (within f32 tolerance {tol})"
  else
    IO.println s!"✗ FAIL (err {err} ≥ tol {tol})"
    throw (IO.userError "rope parity failed")
