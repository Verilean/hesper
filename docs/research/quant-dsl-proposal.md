# Quantization DSL Proposal

A proposal to add a typed, error-resistant DSL on top of `ShaperM` for
expressing quantization layouts and operations. Motivated by the bug
patterns we hit while implementing the Q4_K and Q6_K dp4a kernels.

This is **post-debug work** — the goal is to reduce the cost of writing
the next quantization kernel, not to refactor what already works.

---

## 1. Patterns of bugs we actually hit

Each row is a real mistake from the dp4a port (and the ones before it).
Frequency = how many times it bit us in this single feature.

| # | Bug pattern | Example (where) | Frequency |
|---|---|---|---|
| 1 | **Wrong byte/u32 unit conversion** | `q4 base = blockBase + 4 + 4*bq8Off + elemOff` (forgot whether unit was byte or u32) | 5 |
| 2 | **Truncate vs round in float→int** | `cvt.rzi.u32.f32` for Q8_1 quantize gave ±0.5 LSB error per element | 1 (catastrophic) |
| 3 | **Signed vs unsigned conversion** | `cvt.rn.f32.u32` on dp4a result interpreted negatives as 4 billion | 1 |
| 4 | **Off-by-2 lane mapping** | `pairIdx = tid/4 ∈ [0,8)` reading non-existent sub-blocks 8..14 | 1 |
| 5 | **Borrow propagation across byte boundaries** | `x - 0x20202020` for per-byte sub32 — fails when any byte < 32 | 1 (still WIP for Q6_K) |
| 6 | **Wrong bit field offset** | Q6_K vh_idx formula: `2*(iqs/8)+iqs%4` instead of `4*(iqs/8)+iqs%4` | 1 |
| 7 | **Identity-vs-bitcast confusion** | `Exp.toI32` had no codegen branch → 0 fallthrough; `bitcast u32→f32` reinterpreted bits when caller wanted identity | 2 |
| 8 | **f64 → f32 truncation in test harness** | Test data 0.1 stored as 0x3DCCCCCC instead of 0x3DCCCCCD | 1 (cost a day to find) |
| 9 | **Lane mapping vs sub-block index confusion** | bq8_offset for Q4_K (0,2,4,6) vs Q6_K (0,1,4,5) — same name, different formula | 1 |
| 10 | **Materialize-vs-inline of subgroup ops** | `ShaderM.if_` body inlined a `subgroupAdd` that should run for all lanes | 1 |

The shared root cause: **the DSL treats every value as `u32` and forces
the kernel author to manually keep track of: units (bytes / u32 / int8 /
int4 / fp16), signedness, what lanes are participating, what scope
(per-thread / per-warp / per-block) a value lives in.**

---

## 2. What a typed DSL should give us

### 2.1 Distinct types for distinct meanings

```lean
-- Current: everything is `Exp (.scalar .u32)`
-- Proposed:
abbrev ByteOff  := Exp UnitByte    -- offset in bytes
abbrev U32Off   := Exp UnitU32     -- offset in u32 elements
abbrev BitOff   := Exp UnitBit     -- bit offset within a u32

-- Conversion is explicit:
ByteOff.toU32Off : ByteOff → U32Off  -- panics if not aligned
U32Off.toByteOff : U32Off → ByteOff  -- multiply by 4
```

Now `q4BaseIdx + 4` won't typecheck unless `4` is also a `ByteOff` (or
the call site explicitly asks for the conversion).

### 2.2 Tagged numeric values

```lean
inductive Tag where
  | u32 | u16 | u8 | s32 | s16 | s8
  | i4Pair         -- two int4 packed into u8
  | f32 | f16
  | packed4xI8     -- four int8 packed into u32 (for dp4a)
  | packed4xU8     -- four uint8 packed into u32

abbrev Val (t : Tag) := Exp _   -- erased actual carrier

-- Operations only valid on matching tags:
Val.dp4a : Val .packed4xI8 → Val .packed4xI8 → Val .s32 → Val .s32

-- Float → int forced to choose rounding:
Val.roundToS32 : Val .f32 → Val .s32        -- cvt.rni
Val.truncToU32 : Val .f32 → Val .u32        -- cvt.rzi
-- Removing the easy default removes the bug we hit (#2).
```

### 2.3 Quant block layout descriptions

The bug-class around bytes-vs-u32 (#1, #6) goes away if the layout is
declared once and accessors are derived:

```lean
def q4_K_layout : QuantLayout where
  blockBytes := 144
  blockElems := 256
  fields := [
    .header  "d"      .f16  (offsetBytes := 0)
    .header  "dmin"   .f16  (offsetBytes := 2)
    .packed  "scales" .u8   (offsetBytes := 4)  (count := 12)
    .packed  "qs"     .i4   (offsetBytes := 16) (count := 256)
  ]

-- Auto-derived accessors:
q4_K_layout.read "scales" (subBlockIdx := is) → Val .u8
q4_K_layout.read "qs"     (groupIdx := iqs)   → Val .packed4xI4

-- Compiler / proof obligations:
-- - ensures field offsets + sizes don't overlap
-- - validates blockBytes matches the layout sum
-- - generates the byte ↔ u32 ↔ bit-shift code automatically
```

### 2.4 Lane / scope discipline

The lane mapping bugs (#4, #9, #10) come from the kernel author
manually reasoning "this value is per-lane vs broadcast vs reduced".
A scope-typed DSL can enforce this:

```lean
-- Per-thread (lane-private)
abbrev Lane (t : Tag) := ...

-- Same for all lanes (e.g. block scale d)
abbrev Uniform (t : Tag) := ...

-- A reduced value: only valid after a subgroup op
abbrev Reduced (t : Tag) := ...

-- Operations enforce scope:
Lane.subgroupAdd : Lane t → Reduced t   -- valid in any control flow
Lane.subgroupAddInBranch : ... → ⟨error: subgroupAdd cannot be in a divergent branch⟩

-- Reading a Uniform from inside `if tid == 0` is fine,
-- but writing back to it requires `if any-lane (...)` semantics.
```

### 2.5 Per-byte SIMD ops as primitives

The borrow-propagation bug (#5) goes away if per-byte arithmetic is its
own primitive that lowers to PTX `vsub4` / `vadd4`:

```lean
-- Direct, type-safe:
Val.vsubSatS8 : Val .packed4xI8 → Val .packed4xI8 → Val .packed4xI8

-- Codegen:
-- WGSL: subtract via per-byte unpack/pack
-- PTX:  vsub4.s32.s32.s32.sat
-- This kills the entire class of "did we get the borrow right?" bugs.
```

---

## 3. Migration plan (post-debug)

1. **Define `Tag` enum and `Val` wrapper** in a new
   `Hesper/WGSL/TypedDSL.lean`. Keep `Exp` underneath unchanged.
2. **Implement `QuantLayout` reader/writer** for one block type
   (probably Q8_0 — simplest). Verify against existing kernel.
3. **Re-implement Q4_K dp4a using the typed DSL**. Diff against the
   raw-bit version. Performance must match within noise.
4. **Add `Lane` / `Uniform` / `Reduced` scope tags**. This is the
   biggest change but catches the most insidious bugs.
5. **Implement Q6_K dp4a using the typed DSL** (the kernel that
   originally surfaced this whole proposal).

The point is: **stop debugging bit math by hand.** Each layout gets
described once; the DSL generates the addressing, units, shifts,
and bit-packing.

---

## 4. What this does NOT solve

- **PTX codegen bugs** (e.g. missing `Exp.toI32` branch) — those are in
  the lowering, not the DSL.
- **Algorithm bugs** (e.g. wrong dp4a chaining order) — DSL can't tell
  whether your dot-product algorithm is mathematically right.
- **Scheduling / occupancy issues** — those need profiling, not types.

DSL covers **layout and unit-discipline bugs**, which is ~70% of the
mistakes we made.

---

Last updated: 2026-04-14 (after Q6_K dp4a debug pass).
