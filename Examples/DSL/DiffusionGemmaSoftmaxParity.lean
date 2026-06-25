import Hesper.Models.DiffusionGemma.Reference
import Hesper.Basic
/-! DiffusionGemma softmax parity (router gating + attention) vs ggml_soft_max golden.
    Run: ./scripts/llama_parity/dump_dg_softmax_golden /tmp/dg_golden/softmax
         lake exe diffusiongemma-softmax-parity -/
open Hesper.Models.DiffusionGemma.Reference
def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)).toUInt32; let b1 := (bytes.get! (i*4+1)).toUInt32
    let b2 := (bytes.get! (i*4+2)).toUInt32; let b3 := (bytes.get! (i*4+3)).toUInt32
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 (b0 ||| (b1<<<8) ||| (b2<<<16) ||| (b3<<<24)))
  return a
def main (args : List String) : IO Unit := do
  let dir := args.head?.getD "/tmp/dg_golden/softmax"
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"
  let out := softmax x
  let mut err := 0.0
  for i in [0:x.size] do
    let d := (out[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"[dg-softmax-parity] n={x.size} maxAbsErr={err}"
  if out.size == gold.size && err < 1e-4 then IO.println "✓ PASS"
  else IO.println "✗ FAIL"; throw (IO.userError "softmax parity failed")
