# 43 — Q6_K real gap: fp16→f32 soft-impl + address arithmetic

*Written 2026-04-23, follow-up to doc 42.*

## Headline

Q6_K ffn_down matmul: **1.83 → 1.33 ms/decode** after replacing the
arithmetic `fp16ToF32` with PTX's native `cvt.f32.f16`.  Ratio vs
llama.cpp: **1.53× → 1.11×**.  Token parity preserved.

## How we found the gap

Used the llama.cpp PTX override (doc 41) as ground truth:

```bash
HESPER_USE_LLAMACPP_PTX=1 HESPER_LLAMACPP_Q4K=0 HESPER_LLAMACPP_Q6K=1 \
  lake exe gemma4-llama-prefill-skeleton data/gemma-4-e4b-it-Q4_K_M.gguf \
    "Hello world how are you" 10
```

Measured `void mul_mat_vec_q<(ggml_type)14, ...>` at **1.20 ms/decode**
from *inside* hesper.  This proved the 1.53× gap was compute-side, not
周辺overhead (argument packing, stream dispatch, etc).

Then counted PTX instructions in the inner loop (L12→bra L12):

| category               | hesper | llama | diff  |
|------------------------|-------:|------:|------:|
| add.u32/s32            |     58 |     4 |   +54 |
| shl                    |     42 |     1 |   +41 |
| mov.u32                |     23 |     1 |   +22 |
| bfe                    |     17 |     0 |   +17 |
| mov.f32 (const reload) |     12 |     0 |   +12 |
| setp + selp            |     19 |     1 |   +18 |
| div + ex2              |      3 |     0 |    +3 |
| **total inner loop**   |  **286** | **79** | **+207** |

The `mov.f32` + `div` + `ex2` + 4× `selp` + 2× `bfe` bracket was the
arithmetic fp16→f32 conversion.  `fp16ToF32` in
`Hesper/Quantization/Q4_K_M.lean:84` implements IEEE fp16 decoding via
Exp primitives — sign / exponent / mantissa extraction, subnormal
handling, `2^(exp-15)` via `ex2`.  PTX has `cvt.f32.f16` as a single
instruction for the normal case.

## The fix

hesper's `ShaderM` already lowers `Exp.vecX (Exp.unpack2x16float x)` to
`mov.b32 {lo,hi}, x; cvt.f32.f16 f, lo` (used by Q8_1 header reads).
We just reuse that path for the Q6_K block scale:

```lean
-- Before
let dBits ← readU16 blockByteBase (Exp.litU32 208)
let d := fp16ToF32 dBits

-- After
let dBitsName ← ShaderM.var (.scalar .u32)
  (← readU16 blockByteBase (Exp.litU32 208))
let dBits : Exp (.scalar .u32) := Exp.var dBitsName
let d := Exp.vecX (Exp.unpack2x16float dBits)
```

Result in PTX: `ld.global.nc.u16 %r, [%rd]; mov.b32 {%hLo,%hHi}, %r;
cvt.f32.f16 %f, %hLo` — 3 instructions instead of ~18.

Applied to all 3 Q6_K kernel variants (1-row, 2-row, 4-row) via a
single `replace_all` edit.

## Remaining 1.11× gap

Hesper still emits 258 inner-loop instructions vs llama.cpp's 79.  The
remaining excess is all **address arithmetic**:

- 54× `add.u32` (vs 4 in llama.cpp) — per-load address reassembly
- 41× `shl` (vs 1) — shift-by-constant
- 11× `mul.wide.u32` (vs 0) — u32→u64 extension
- 22× `mov.u32` — immediate reloads

Root cause: every `ShaderM.readBufferByte`/`readBufferU16`/`readBuffer`
call computes `mul.wide.u32 off, rIdx, stride; add.u64 addr, base, off`
from scratch.  The offsets share a common base (blockByteBase) but
hesper doesn't bind intermediate address expressions at the ShaderM
level, so every load re-emits `shl %rIdx, 2; add %blockByteBase, %off;
mul.wide.u32 %off, %addr, 1; add.u64 %addr, %base, %off`.

ptxas local CSE collapses some of these but not across basic blocks
within the unrolled loop.  llama.cpp's C++ compiler sees the whole
`block_q6_K *` struct at once and does aggressive common-address
extraction.

Possible fixes, in rough order of expected impact:

1. **Peephole in CodeGen**: when `mul.wide.u32` is emitted with literal
   multiplier 1, emit `cvt.u64.u32` instead (shorter, same semantics).
2. **Hoist address bases**: for loads from the same buffer within the
   same ShaderM block, compute the base u64 pointer once
   (`buffer_ptr + block_byte_base_u64`) and reuse.  Would eliminate
   most of the `mul.wide.u32` + inner `add.u64`.
3. **bfe instead of `(x >> k) & mask`**: hesper already has a `bfe`
   peephole but some bit extracts still use `shr` + `and`.

Not planned: kernel-body restructuring.  The ms/decode gap is now
11%, which is within the noise of other comparisons (Q4_K is at 12%
too).  Better ROI elsewhere (RMSNorm 2.98× is the new worst).

## Verification

```
hesper PTX: 434 lines (was 462)
hesper inner loop: 258 instructions (was 286)
ffn_down Q6_K decode: 1.33 ms (was 1.83 ms)
Token parity: "?" (canonical "Hello world how are you" 10-token run)
```
