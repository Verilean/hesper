import Hesper.CUDA.CodeGen
import Hesper.WGSL.MatMul

set_option maxRecDepth 2048

/-- Standalone PTX dumper for the lm_head f16 matmul kernel.
    Writes PTX to argv[1] (default `/tmp/hesper_lmhead.ptx`).
    Doesn't require a CUDA context. -/
def main (args : List String) : IO Unit := do
  let path := args.headD "/tmp/hesper_lmhead.ptx"
  let cfg : Hesper.WGSL.MatMul.Config := { M := 1, N := 262144, K := 2560 }
  let kernel := Hesper.WGSL.MatMul.matMulTransposeF16BlockCoopKernel cfg
  let ptx := Hesper.CUDA.CodeGen.generatePTX "matMulTransposeF16BlockCoop_lmhead"
    { x := 32, y := 1, z := 1 } kernel
  IO.FS.writeFile path ptx
  IO.println s!"[ok] wrote {ptx.length} chars to {path}"
