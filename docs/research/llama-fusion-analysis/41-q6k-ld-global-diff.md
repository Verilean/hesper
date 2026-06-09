# 41 — Q6_K kernel PTX diff: hesper vs llama.cpp (sm_80)

*Written 2026-04-23.  nsys showed hesper Q6_K at 1.57× llama.cpp.
This doc pins the root cause to memory-access granularity after
comparing the two PTX inner loops.*

## How to reproduce the PTX dumps

llama.cpp (sm_80 PTX from multi-arch fatbin):

```bash
cuobjdump --dump-ptx llama.cpp/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o \
  > /tmp/lc_mmvq.ptx
# sm_80 section starts at line 1299515; Q6_K is ggml_type14, ncols=1, fusion=0, small_k=0
sed -n '148875,149190p' <(sed -n '1299515,1605577p' /tmp/lc_mmvq.ptx) > /tmp/lc_q6k_sm80.ptx
```

hesper (current 4-row kernel dumped via HESPER_PTX_DUMP=/tmp/ptx_dump):

```bash
HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0 HESPER_PTX_DUMP=/tmp/ptx_dump \
  lake exe gemma4-llama-prefill-skeleton data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 2
# grid=(640,1,1) is the Q6_K 4-row ffn_down kernel; hash appears via HESPER_KERNEL_TRACE=1.
```

## The diff that matters

Per-iter inner-loop instructions on sm_80:

| Metric (inner block `ShaderM.loop 0..blocksPerRow`) | llama.cpp | hesper |
|-----------------------------------------------------|----------:|-------:|
| PTX total lines                                     |       316 |    893 |
| `ld.global.nc` per iter                             |    **12** | **48** |
| `dp4a.s32.s32`                                      |         2 |      2 |
| `fma.rn.ftz.f32`                                    |         3 |      ~4 |

**Hesper issues 4× more global loads per iteration to read the same Q6_K block.**

### Why

A Q6_K block is 210 bytes:

```
offset   size  field
  0..128  128  ql (low 4 bits of quants, packed)
 128..192  64  qh (high 2 bits of quants, packed)
 192..208  16  scales (16 × i8)
 208..210   2  d (fp16 block scale)
```

**llama.cpp** indexes these via a `block_q6_K *` struct pointer and
lets the compiler emit the **natural load width** for each field:

```ptx
ld.global.nc.u32 %r94, [%rd36+4]     ; ql low 4-byte tile (4 bytes used)
ld.global.nc.u32 %r86, [%rd34]       ; ql low 4-byte tile
ld.global.nc.u16 %rs1, [%rd40+208]   ; d (2-byte fp16 load)
ld.global.nc.u8  %rs6, [%rd41+-4]    ; sc byte (1-byte load)
```

Each load uses the correct granularity — u8 for scales, u16 for the
half-precision block scale, u32 for packed quants.

**Hesper**'s `readByte` helper (`Hesper/Layers/Linear.lean:3135`):

```lean
let u32Idx    := Exp.shiftRight byteIdx (Exp.litU32 2)
let byteShift := Exp.mul (Exp.bitAnd byteIdx (Exp.litU32 3)) (Exp.litU32 8)
let u32 ← ShaderM.readBuffer (ty := .scalar .u32) "weights" u32Idx
pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))
```

It **always reads a full u32, then shift+masks a byte out**.  Reading
the 2-byte fp16 `d` (via two `readByte` calls) becomes two u32 loads
(8 bytes transferred for 2 bytes used).  Reading each of the 16 scale
bytes becomes one u32 load each.

`read4Bytes` (`Linear.lean:3142`) is worse — it **always issues two
u32 loads** to handle the unaligned case (byteOff != 0), then
shift-combines.  For the many byte-aligned offsets used in this
kernel that's 2× more global traffic than needed.

### Total per-iter bandwidth cost

Approximate bytes actually used per iter (by the compute):

| Data | Fields read per iter | Bytes used | hesper PTX bytes loaded | Waste |
|------|-----------------------|-----------:|-----------------------:|------:|
| ql   | 4 u32 (vl)            |         16 |      16 (2 × read4Bytes = 4 loads × 4) | 1× |
| qh   | 4 u32 (vh)            |         16 |      16                                 | 1× |
| scales | 2 u8 (sc0, sc1)     |          2 |       8 (2 × readByte = 2 loads × 4)   | 4× |
| d (fp16) | 2 u8 (dLo, dHi) |          2 |       8                                 | 4× |
| Q8_1 u input | 4 u32        |         16 |      16 (smem-staged)                  | 1× |
| Q8_1 d input | 2 u32        |          8 |       8                                 | 1× |
| **total**  |                   |     **60** |         **72** (1.2× from global)     |    |

But the 48 `ld.global.nc` count includes both the 4 byte-shifts on
`readByte` AND the 2-load pairs in `read4Bytes`.  Many are reading
cached u32 slots, so bandwidth isn't 4× — but the **instruction count
itself** bloats the kernel, hurts latency, and prevents ptxas from
hitting its unroll sweet spot.

## Proposed fix

Add `.u8` and `.u16` load support to hesper's `ShaderM.readBuffer` (or
a new `readBufferByte` / `readBufferU16` primitive), so `readByte` can
emit `ld.global.nc.u8` directly and `d = fp16ToF32(u16)` can emit one
`ld.global.nc.u16` instead of two `readByte` calls.

For `read4Bytes`: add an aligned-only fast path (when offset is
statically divisible by 4) that emits a single u32 load.  The outer
base here is `rowByteBase + blockIdx * 210 + fixed_offset` — not
aligned at compile time, but **at runtime** the base is always 4-byte
aligned for Q6_K ql/qh (they start at multiples of 4 within the 210-
byte block).

## Expected impact

Conservatively, if we halve the number of `ld.global.nc` ops:
- Fewer L1 tag lookups
- Lower register pressure (fewer intermediate u32 temps for shift+mask)
- Better unrolling headroom

Based on llama.cpp's 46 µs/call vs hesper's 73 µs/call, closing the
gap gives **-0.8 ms/decode** (ffn_down alone) and matches Q6_K lm_head
where the same load pattern would help even more.

Actual speedup TBD — depends on whether the current inefficiency is
BW-bound, latency-bound, or ptxas register-pressure bound.

## Status

Not yet implemented.  Requires extending hesper's ShaderM primitive
layer with u8/u16 typed loads, OR adding a peephole in CodeGen that
pattern-matches `bitAnd (shiftRight (u32_load x) shift) 0xFF` → u8
load at `x + shift/8`.  The latter is safer (no ShaderM type-system
churn) and reusable across all kernels.
