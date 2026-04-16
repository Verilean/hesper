import Hesper.Backend.LlamaCppPTX
import Hesper.Backend.CUDA

/-!
Smoke test: load llama.cpp's mmvq.ptx + quantize.ptx, resolve the three
target kernels (Q4_K matmul, Q6_K matmul, quantize_q8_1).  Confirms the
CUDA driver can JIT these PTX modules and find the mangled entry points.

No GPU execution — just module load + function lookup.
-/

open Hesper

unsafe def main : IO Unit := do
  let _ ← CUDAContext.init
  IO.println "[llamacpp-ptx-load-test] loading PTX modules..."
  let k ← Hesper.LlamaCppPTX.loadKernels
  IO.println s!"✓ Q4_K matmul @ 0x{Nat.toDigits 16 k.q4kMatmul.toNat |>.asString}"
  IO.println s!"✓ Q6_K matmul @ 0x{Nat.toDigits 16 k.q6kMatmul.toNat |>.asString}"
  IO.println s!"✓ quantize_q8_1 @ 0x{Nat.toDigits 16 k.q8_1Quantize.toNat |>.asString}"
  IO.println "PASS"
