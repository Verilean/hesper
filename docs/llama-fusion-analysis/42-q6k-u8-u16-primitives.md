# 42 — ShaderM u8/u16 primitives close Q6_K PTX gap

*Written 2026-04-23, follow-up to `41-q6k-ld-global-diff.md`.*

## Summary

Added native byte/half load primitives (`Exp.loadByteFromU32Buf`,
`Exp.loadU16FromU32Buf`) end-to-end through ShaderM → CodeGen → PTX
so Q6_K scale and fp16 `d` reads now emit single `ld.global.nc.u8`
and `ld.global.nc.u16` instructions instead of a u32 load + shift + mask.

Previously (hesper) vs now (inner loop of Q6_K 2-row matmul kernel):

| Metric                          | before | after | llama.cpp sm_80 |
|----------------------------------|-------:|------:|----------------:|
| Total PTX lines                  |    893 |   501 |             316 |
| `ld.global.nc.*` per iter        |     48 |    20 |              12 |
| `dp4a` per iter                  |      2 |     2 |               2 |

End-to-end kernel_compare:

| Class          | hs before | hs after | lc     | ratio before | ratio after |
|----------------|----------:|---------:|-------:|-------------:|------------:|
| Q6_K matmul    |  3.38 ms  | 3.12 ms  | 2.15 ms|    1.57×     |    1.45×    |
| Q4_K matmul    |  4.91 ms  | 4.91 ms  | 4.40 ms|    1.12×     |    1.12×    |

Token parity preserved — canonical "Hello world how are you" → 236881 "?".

## Implementation

### 1. PTX layer (`Hesper/CUDA/PTX.lean`)

Added two `Inst` variants:

```lean
| ld_u8  (space : AddrSpace) (dst : RegU32) (addr : RegU64) (nc : Bool := false)
| ld_u16 (space : AddrSpace) (dst : RegU32) (addr : RegU64) (nc : Bool := false)
```

Both load into a 32-bit register; PTX zero-extends per spec.  Pretty-
printer emits `ld.global.nc.u8 %rN, [%rdM]` / `.u16` (with `.nc` hint
for readonly buffers).

### 2. WGSL Exp layer (`Hesper/WGSL/Exp.lean`)

```lean
| loadByteFromU32Buf {n : Nat}
    : (bufName : String)
    → (byteIdx : Exp (.scalar .u32))
    → Exp (.scalar .u32)
| loadU16FromU32Buf  ... -- same shape
```

The buffer is declared as `array<u32, n>` but these primitives address
it by *byte* index.  On WGSL (no byte granularity in storage buffers)
they lower to the old `(buf[byteIdx >> 2] >> shift) & mask` emulation;
on CUDA they lower to a single `ld.u8` / `ld.u16` instruction.

### 3. CUDA CodeGen (`Hesper/CUDA/CodeGen.lean`)

```lean
| .loadByteFromU32Buf name byteIdx =>
    let rArr := (s.varMap.find? (·.1 == name)).map (·.2) |>.getD default
    let (rIdx, s) := expToPTX byteIdx s
    let (off, s)  := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 1)
    let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
    let (r, s)    := s.freshU32
    (.u32 r, s.emit (.ld_u8 .global r addr isRO))
```

Byte stride (×1) via `mul.wide.u32` — ptxas folds that into a plain
zero-extend.

### 4. ShaderM wrappers (`Hesper/WGSL/Monad.lean`)

```lean
def readBufferByte {n : Nat} (bufferName : String) (byteIdx : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .u32)) :=
  return Exp.loadByteFromU32Buf (n := n) bufferName byteIdx
def readBufferU16  ... -- same shape
```

### 5. Kernel rewrite (`Hesper/Layers/Linear.lean`)

Three Q6_K matmul variants (1-row, 2-row, 4-row) all shared the same
`readByte` / `read4Bytes` helpers.  Updated `readByte` and added a new
`readU16` helper:

```lean
let readByte (blockBase offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
  let byteIdx := Exp.add blockBase offset
  ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

let readU16  (blockBase offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
  let byteIdx := Exp.add blockBase offset
  ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx
```

The `dLo + dHi + shift + or` pattern for reading the fp16 block scale
collapses to one `readU16 blockByteBase (Exp.litU32 208)`.

`read4Bytes` is **unchanged** — the blockByteBase is 2-aligned (210 %
4 = 2), so we can't safely upgrade it to a single `ld.u32` without a
runtime branch on block parity.  Left as a future optimisation.

## Why we didn't close all the way to 1.00×

Remaining hesper-vs-llama.cpp delta (20 ld.global.nc per iter vs 12):

- `read4Bytes` still emits 2 × `ld.u32` (aligned + shifted), whereas
  llama.cpp emits 1 × `ld.u32` and relies on HW misalignment tolerance.
- Scale reads: hesper emits 1 × `ld.u8` per scale; llama.cpp reads 4
  scale bytes via 1 × `ld.u32` then shifts internally.  Batch-reading
  scales as u32 + bfe would reduce 4 ld.u8 → 1 ld.u32.

These are deferrable — the 8% wall-clock win unblocks moving to other
kernels (RMSNorm is now the biggest ratio at 3.01×).

## Next

1. RMSNorm (3.01×) — re-classify hesper kernels; the ratio is inflated
   by stubs being bucketed here.  Once clean, profile and compare.
2. FlashAttn (hesper currently "unmapped") — hesper's FlashAttn kernels
   aren't in the grid→class table yet.  Add classification so we see
   real numbers.
