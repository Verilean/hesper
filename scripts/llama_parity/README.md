# llama.cpp parity goldens

Generates deterministic input + output of `ggml_im2col` and
`ggml_conv_transpose_1d` from llama.cpp's CPU backend, dumped as raw f32
bytes.  Lean tests then run hesper's GPU kernel on the **same** input and
compare against the golden output bit-for-bit.

This is the only way to verify that hesper's port reproduces llama.cpp's
behaviour exactly — a Lean-side CPU reference can match a buggy GPU
kernel if the bug is in our re-reading of the algorithm.

## Build

```bash
g++ -O2 -std=c++17 scripts/llama_parity/dump_im2col_golden.cpp \
  -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
  -L llama.cpp/build/bin -Wl,-rpath,$PWD/llama.cpp/build/bin \
  -lggml -lggml-base -lggml-cpu \
  -o scripts/llama_parity/dump_im2col_golden

g++ -O2 -std=c++17 scripts/llama_parity/dump_conv_transpose_1d_golden.cpp \
  -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
  -L llama.cpp/build/bin -Wl,-rpath,$PWD/llama.cpp/build/bin \
  -lggml -lggml-base -lggml-cpu \
  -o scripts/llama_parity/dump_conv_transpose_1d_golden
```

(Requires llama.cpp to be built; see top-level CLAUDE.md.)

## Run goldens + parity tests

```bash
mkdir -p /tmp/im2col_golden /tmp/conv_transpose_1d_golden
./scripts/llama_parity/dump_im2col_golden /tmp/im2col_golden
./scripts/llama_parity/dump_conv_transpose_1d_golden /tmp/conv_transpose_1d_golden

lake exe cuda-im2col-vs-llama
lake exe cuda-conv-transpose-1d-vs-llama
```

Both report `max |err| = 0.0` (bit-exact).

---

## DiffusionGemma per-module goldens

Same pattern as Gemma4/BitNet: ggml CPU op = golden, Lean parity test runs
the Hesper module on the **same** input and compares.  Validates the
DiffusionGemma CPU reference (`Hesper/Models/DiffusionGemma/Reference.lean`)
against llama.cpp's ops; the same goldens drive the WGSL/CUDA kernel parity
later.

Build all dumpers (LC = path to a built llama.cpp):

```bash
LC=~/git/llama.cpp
for m in rmsnorm rope geglu softcap matmul softmax attn; do
  g++ -O2 -std=c++17 scripts/llama_parity/dump_dg_${m}_golden.cpp \
    -I $LC/ggml/include -I $LC/ggml/src -L $LC/build/bin -Wl,-rpath,$LC/build/bin \
    -lggml -lggml-base -lggml-cpu -o scripts/llama_parity/dump_dg_${m}_golden
  mkdir -p /tmp/dg_golden/$m && ./scripts/llama_parity/dump_dg_${m}_golden /tmp/dg_golden/$m
done
```

Run parity tests (all report `maxAbsErr ≈ 0`, geglu ~4e-4 from ggml's table-gelu):

```bash
for m in rmsnorm rope geglu softcap matmul softmax attn; do
  lake exe diffusiongemma-${m}-parity
done
```
