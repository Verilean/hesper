# 44 — Q6_K PTX address CSE: the law of diminishing returns

*Written 2026-04-23, follow-up to doc 43.*

## TL;DR

Two more PTX optimisations landed:

1. `mul.wide.u32 d, s, 1` → `cvt.u64.u32 d, s` peephole in the PTX
   printer.
2. Bind `byteIdx` / `u32Idx` / `byteOff` inside `read4Bytes` so each is
   emitted once per call instead of re-expanding the full address
   chain 6× per read.

Result:

| | ffn_down Q6_K ms/dec | inner-loop PTX |
|---|---:|---:|
| baseline | 1.83 | 286 inst |
| +fp16 cvt (doc 43) | 1.33 | 258 inst |
| **+addr CSE (doc 44)** | **1.32** | **203 inst** |
| llama.cpp (hesper override) | 1.20 | 79 inst |

Ratio vs llama.cpp: 1.53× → 1.11× → **1.10×**.  PTX instruction count
dropped 21% but wall-clock barely moved (0.01 ms).  ptxas's local CSE
was already collapsing most of the PTX-level redundancy into shared
SASS registers; we just made it work less hard.

## What the numbers tell us

PTX instruction count and wall-clock stopped correlating once we
crossed 1.11× ratio.  ptxas is doing its job; the remaining 10% is
probably in one of:

- **Grid occupancy**: hesper 4-row kernel launches `outDim/4 = 640`
  workgroups for ffn_down vs llama.cpp's 2560.  Fewer in-flight warps
  = less latency hiding.
- **Scale access pattern**: llama.cpp reads 16 scale bytes as 4×u32 +
  bitshift (amortised), hesper reads each scale byte with a separate
  ld.u8 (doc 42).
- **Register pressure at SASS level**: virtual u32 reg count
  301→239→ptxas-assigned, but hesper's wider working set may cause
  more spills than llama.cpp's.

All three are measurable only with ncu (Nsight Compute), not by PTX
inspection.  PTX-level optimisation is out of runway for this kernel.

## What shipped

**`Hesper/CUDA/PTX.lean`**: the `mul.wide.u32` printer now recognises
literal `1` and emits `cvt.u64.u32` instead (1 SASS op vs 1-2 for the
multiplier).  Applies to any `loadByteFromU32Buf`/`loadU16FromU32Buf`
call, which all use byte stride 1.

```lean
| .mul_wide_u32 d s n  =>
  if n == 1 then s!"  cvt.u64.u32 {d}, {s};"
  else s!"  mul.wide.u32 {d}, {s}, {n};"
```

**`Hesper/Layers/Linear.lean`**: `read4Bytes` now binds its
intermediate u32 expressions so the address chain emits once per
call:

```lean
let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx ...)
let u32Idx := Exp.var u32IdxName
let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx ...)
let byteOff := Exp.var byteOffName
-- downstream shiftLo/shiftHi/w0/w1/lo/hi all reference the bound vars
```

Before the bind, each of the 6 downstream references (`u32Idx`,
`byteOff` × 4) fully re-expanded `Exp.add blockBase offset` + the
shift/mask chain.  PTX had `shl.b32 %rX,2 ; add.u32 ; add.u32 ; shr`
repeated 6× per `read4Bytes` call.  After binding: each materialises
once.

## Where to look next

`docs/45-rmsnorm-gap.md` — the new top ratio is RMSNorm at 2.98×,
which would meaningfully move decode TPS.

Or `ncu --set detailed` on Q6_K to see if the remaining 10% is
occupancy, SASS spill, or something else.  No PTX-level changes
beyond this will help.
