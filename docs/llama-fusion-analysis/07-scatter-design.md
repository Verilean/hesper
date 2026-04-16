---
title: "07 — Dynamic offset / Scatter design discussion"
date: 2026-04-16
status: **IMPLEMENTED** (commit b515e13)
---

> **Update 2026-04-16**: this design item is now **implemented**.
> Implementation details and verification results are in
> [`08-scatter-impl-notes.md`](08-scatter-impl-notes.md). The text
> below is preserved as the original design discussion.

# Dynamic offset / Scatter design discussion

## What we found

While trying to bring Pattern E (`ROPE + VIEW + SET_ROWS`) into hesper's
Circuit DSL, a fundamental gap surfaced between **pure data-parallel Map
operations** and **Scatter operations with dynamic addressing**.

Concrete example: KV cache write addresses look like

```
out[kvHead * maxSeqLen * headDim + pos * headDim + d] = ...
```

where `pos` is a **runtime value read from a parameter buffer**. The
DSL as it stood (`Prim.pointwise` / `Prim.writeSlice`) couldn't express
this address calculation.

## DSL semantics at the time

### Map pattern (supported)

```
Prim.pointwise outShape inShapes body
  thread i:
    slots[k] = inputs[k][i]   -- broadcast if inShapes[k] == #[1]
    out[i]   = body(slots)
```

**Implicit assumption**: output index `i` equals input index `i`
(identity map). `ScalarExp.input k` reads "the k-th input at the same
lane".

### `Prim.writeSlice` (added during this session)

```
Prim.writeSlice dstShape dstOffset srcShape
  thread i:
    out[dstOffset + i] = src[i]
```

Even after we generalised `dstOffset` to `ScalarExp`, the `.input k`
inside it had no defined meaning — the slot array was empty.

## Where the gap really lived

| Operation | Output address | Input address | Status |
|---|---|---|---|
| Map (pointwise) | `i` (identity) | `i` (identity) | ✅ |
| Gather | `i` | `f(i, params)` | ❌ |
| **Scatter** | `f(i, params)` | `i` | ❌ **Pattern E is here** |
| General (MoE-like) | `g(i, params)` | `f(i, params)` | ❌ |

To express scatter we need **separate ScalarExps for "what to write" and
"where to write it"**, both with access to runtime parameters.

## What helped, what was missing

### ✅ Improvements made earlier in the session

- 9 new ScalarExp operators: `cos`, `sin`, `pow`, `lt`, `select`,
  `mod`, `idiv`, `toFloat`, `laneIdx`
- `laneIdx` exposes the thread index inside the body
- Operator overloading: `+`, `-`, `*`, `/`, `Neg`, `OfNat`, `OfScientific`, `BEq`
- `writeSlice` / `pointwiseToSlice` `dstOffset` generalised to `ScalarExp`
- `fuseWriteDestination` pass

### ❌ Still missing (at the time of this doc)

1. **Address expressions can't read parameter buffers.**
   `.input k` referred to the **pointwise body's slots**, not anything
   visible to `addrExpr`.
2. **Value and address are not separated.**
   `Prim.pointwise.body` was the only `ScalarExp`, and the output index
   was implicitly `laneIdx`.

## Proposal: `Prim.scatter`

Sketch from the design discussion:

```lean
/-- Scatter: compute value + destination address per lane.

    thread i:
      valueSlots[k]    = valueInputs[k][i]     (broadcast if [1])
      addrSlots[k]     = addrInputs[k][i]      (broadcast if [1])
      out[addrExpr]    = valueExpr             (with the slots above)

    `valueExpr`  : value to write (uses `.input k` for valueSlots)
    `addrExpr`   : destination index (uses `.input k` for addrSlots)
    `.laneIdx`   : thread global id, available in both
    `outShape`   : dispatch grid (one thread per write)
    `dstShape`   : destination buffer shape (addrExpr ∈ [0, dstShape.numel))

    Subsumes Map (addrExpr = laneIdx), writeSlice
    (addrExpr = laneIdx + const), and general Scatter
    (addrExpr uses parameter buffers + laneIdx arithmetic). -/
| scatter
    (outShape    : Shape)              -- dispatch grid
    (dstShape    : Shape)              -- destination buffer shape
    (valueInputs : Array Shape)        -- value-path input shapes
    (addrInputs  : Array Shape)        -- addr-path input shapes
    (valueExpr   : ScalarExp)
    (addrExpr    : ScalarExp)
```

Calling convention:
- `inputs` array layout: `[dst, valueInputs..., addrInputs...]`
- `valueExpr`'s `.input k` → `valueInputs[k]`
- `addrExpr`'s `.input k` → `addrInputs[k]`
- `.laneIdx` available in both

> **Implementation note (2026-04-16):** the actual implementation
> simplified this further by **sharing one `inputs` array** between
> `valueExpr` and `addrExpr`. See
> [`08-scatter-impl-notes.md`](08-scatter-impl-notes.md) for the
> rationale and final form.

### Reduction to existing Prims

| Old Prim | Equivalent `scatter` form |
|---|---|
| `pointwise outShape inShapes body` | `scatter outShape outShape inShapes #[] body .laneIdx` |
| `writeSlice dstShape dstOffset srcShape` | `scatter srcShape dstShape #[srcShape] #[] (.input 0) (.laneIdx + dstOffset)` |
| `pointwiseToSlice` | combination above (any body, addrExpr = laneIdx + offset) |

So `scatter` **is the generalisation that contains all of them**.

## How the KV cache write looks under `scatter`

V cache write (no RoPE):

```lean
-- external inputs
let vNew    ← registerExternal newVBuf       #[kvDim]                .f32 .Global
let vCache  ← registerExternal kvCacheV      #[numKVHeads * maxSeqLen * headDim] .f32 .Global
let params  ← registerExternal paramsBuf     #[1]                    .f32 .Global  -- pos

-- scatter: v_cache[kvHead * maxSeqLen * headDim + pos * headDim + d] = v_new[i]
--   where i = laneIdx, kvHead = i / headDim, d = i % headDim
open ScalarExp in
let i     := laneIdx
let d     := mod i (.const headDim.toFloat)
let kvH   := idiv i (.const headDim.toFloat)
let pos   := .input 0
let addr  := kvH * .const (maxSeqLen * headDim).toFloat
           + pos * .const headDim.toFloat
           + d
let _out ← CircuitM.scatter
  (outShape := #[kvDim])
  (dst := vCache)
  (valueInputs := #[vNew])
  (addrInputs  := #[params])
  (valueExpr := .input 0)
  (addrExpr := addr)
```

The K cache write with RoPE just changes `valueExpr` to a rotation
expression (`cos`/`sin`/`pow`).

## Lowering plan

- Read `valueInputs` and `addrInputs` into separate slot arrays
  - value path: lane-local indexing or broadcast
  - addr path: also broadcast (typically `[1]` shape)
- Compute `addrExpr` in f32, `Exp.toU32` at the end for indexing
- Output bounds check optional (debug only)

## Impact on the fusion pass

`fuseWriteDestination` would need extending:
- Detect `[A: pointwise] → [B: writeSlice with dynamic offset]`
- Merge B's addr inputs into A's, emit a single `scatter`

If `writeSlice` becomes a syntactic sugar over `scatter`, existing
fusion passes mostly continue to work as-is.

## Tasks for the next session

1. Add `Prim.scatter` to IR (or replace `Prim.writeSlice` /
   `pointwiseToSlice` outright)
2. Implement the lowering
3. Update `fuseWriteDestination` to operate on the new Prim
4. IR unit tests (static and dynamic offset)
5. Wire it into Gemma4.lean's KV cache write
   - V first (plain copy + dynamic offset)
   - K next (RoPE + plain scatter — exercises `valueExpr`'s complexity)
6. Confirm kernels-per-token and decode bit-identical

## Related files

- IR: [`Hesper/Circuit/IR.lean`](../../../Hesper/Circuit/IR.lean)
- Passes: [`Hesper/Circuit/Passes.lean`](../../../Hesper/Circuit/Passes.lean)
- Lowering: [`Hesper/Circuit/Lowering.lean`](../../../Hesper/Circuit/Lowering.lean)
- Existing test: [`Tests/Circuit/FuseWriteDestinationTest.lean`](../../../Tests/Circuit/FuseWriteDestinationTest.lean)
- llama.cpp Pattern E: [`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3762`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- Replacement target: [`Hesper/Models/Gemma4.lean:1682–1697`](../../../Hesper/Models/Gemma4.lean)
  (RoPE-K + KV write hand-coded kernel)

## Lesson learned

The investigation reaffirms the **first principle of the DSL**:

> **Data-parallel semantics separates "where to write" from "what to write".**

hesper's old DSL implicitly fixed the former to identity. Once we
introduce `scatter`, Map / Scatter / future Gather and atomic-accumulate
all live under the same framework.

Pausing here — instead of rushing to swap the hand-coded kernel — was
the right call: the discovery of the expressiveness gap was the actual
finding worth recording.
