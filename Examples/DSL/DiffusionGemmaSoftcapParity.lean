import Hesper.Basic

/-!
# DiffusionGemma final logit-softcap parity vs ggml golden

`out = cap * tanh(x / cap)`, cap = 30 (`final_logit_softcapping`).
Golden from `scripts/llama_parity/dump_dg_softcap_golden`.

Run:
  ./scripts/llama_parity/dump_dg_softcap_golden /tmp/dg_golden/softcap
  lake exe diffusiongemma-softcap-parity
-/

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
  let dir := args.head?.getD "/tmp/dg_golden/softcap"
  IO.println s!"[dg-softcap-parity] golden dir = {dir}"
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"
  let cap := 30.0
  let mut out := Array.replicate x.size 0.0
  for i in [0:x.size] do
    out := out.set! i (cap * Float.tanh (x[i]! / cap))
  let mut err := 0.0
  for i in [0:x.size] do
    let d := (out[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  n={x.size}  maxAbsErr={err}"
  let tol := 1e-4
  if out.size == gold.size && err < tol then
    IO.println s!"✓ PASS (within f32 tolerance {tol})"
  else
    IO.println s!"✗ FAIL (err {err} ≥ tol {tol})"
    throw (IO.userError "softcap parity failed")
