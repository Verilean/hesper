#!/usr/bin/env python3
# Generate the DiffusionGemma full-forward golden (bidirectional canvas) for Phase 2 validation.
# Usage: python3 gen_dg_full_golden.py <model.gguf> <outdir>
# Produces prompt.i32, canvas.i32, logits.bin (256 x 262144 f32).
# Eval REQUIRES: non-empty prompt + canvas of EXACTLY diffusion.canvas_length (256).
import struct, subprocess, sys, os
model = sys.argv[1] if len(sys.argv) > 1 else "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dg_golden/full"
EVAL = os.path.expanduser("~/git/llama-dg/build/bin/llama-diffusion-gemma-eval")
os.makedirs(outdir, exist_ok=True)
prompt = [2, 651, 1437]          # bos + a couple tokens
canvas = [4] * 256               # all-mask step-0 canvas (mask_token_id=4)
open(f"{outdir}/prompt.i32","wb").write(struct.pack(f"<{len(prompt)}i", *prompt))
open(f"{outdir}/canvas.i32","wb").write(struct.pack(f"<{len(canvas)}i", *canvas))
subprocess.run([EVAL, model, f"{outdir}/prompt.i32", f"{outdir}/canvas.i32", f"{outdir}/logits.bin"], check=True)
print(f"golden: {outdir}/logits.bin  (256 x 262144 f32)")
