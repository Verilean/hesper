---
title: "08 — Scatter unification: implementation notes and verification"
date: 2026-04-16
status: IMPLEMENTED (commit b515e13)
---

# Scatter unification: implementation notes and verification

The Map/Scatter semantic gap raised in [`07-scatter-design.md`](07-scatter-design.md)
was resolved by **making `Prim.scatter` the single write primitive**.
This document records the final design, implementation details, and
verification results.

## 1. Final design

### 1.1 One `Prim.scatter`

```lean
-- Hesper/Circuit/IR.lean
inductive Prim (BufT : Type) (CacheT : Type) where
  | matmulQ4K ...
  | matmulQ4KWithEpilogue ...
  | reduceLastAxis ...
  | reduceLastAxisWithEpilogue ...
  | scatter
      (outShape   : Shape)          -- dispatch grid
      (dstShape   : Shape)          -- destination buffer shape
      (inShapes   : Array Shape)    -- each input either outShape or #[1]
      (valueExpr  : ScalarExp)
      (addrExpr   : ScalarExp)
```

Removed: `pointwise`, `writeSlice`, `pointwiseToSlice`. The existing
builders (`CircuitM.pointwise`, `CircuitM.writeSlice`) survive as thin
sugar over `scatter`.

### 1.2 ScalarExp additions

In addition to the 9 operators added earlier:

| Constructor | Meaning | Used for |
|---|---|---|
| `laneIdx` | Thread global id as f32 | KV cache head/d decomposition |
| `indexed bufIdx addr` | `inputs[bufIdx][toU32(addr)]` (gather) | NeoX RoPE pair-index lookup |

Both `valueExpr` and `addrExpr` may reference `laneIdx`. `indexed` is
typically used in `valueExpr` (gather from a known buffer at a computed
position), but the grammar permits it in either.

### 1.3 Shared `inputs` array semantics

A single `inputs : Array TensorRef` is **shared by `valueExpr` and
`addrExpr`**. `.input k` refers to the same `inputs[k]` in both
expressions. This keeps slot meaning unambiguous and removes a class of
errors (mistaking value slots for addr slots).

`inShapes[k]` distinguishes lane-local (`= outShape`) from broadcast
(`= #[1]`). Lane-local inputs are pre-loaded into
`slots[k] = inputs[k][laneIdx]`; broadcast inputs into
`slots[k] = inputs[k][0]`.

### 1.4 Lowering: `lowerScalarExp` becomes monadic

Because `.indexed` needs to emit a `ShaderM.readBuffer` at runtime, the
return type of `lowerScalarExp` changes from `Exp (.scalar .f32)` to
`ShaderM (Exp (.scalar .f32))`. A `decls : Array InputDecl` argument
carries the per-input metadata (buffer name, length).

```lean
def lowerScalarExp
    (slot       : Array (Exp (.scalar .f32)))
    (laneIdxExp : Exp (.scalar .f32))
    (decls      : Array InputDecl)
    : ScalarExp → ShaderM (Exp (.scalar .f32))
  | .indexed i addr => do
    let ae ← lowerScalarExp slot laneIdxExp decls addr
    match decls[i]? with
    | some decl => ShaderM.readBuffer decl.name (Exp.toU32 ae)
    | none      => pure (Exp.litF32 0.0)
  | ...
```

## 2. Effect on fusion passes

### 2.1 `fuseWriteDestination` deleted

The old "fold `pointwise → writeSlice` into `pointwiseToSlice`" pass is
no longer needed:

- `CircuitM.writeSlice` directly emits a scatter (addrExpr = laneIdx + offset)
- The consumer is therefore a scatter, so the existing `fusePointwise`
  inlines a Map-shaped scatter (addrExpr = .laneIdx) producer into any
  scatter consumer, preserving the consumer's `addrExpr`
- One pass now does the work of two

### 2.2 Guards added to `fusePointwise`

Producer must satisfy `addrExpr = .laneIdx` — a scatter with dynamic
addressing has no notion of "value at lane i" because writes go to
arbitrary positions. The pass refuses such producers:

```lean
let .scatter aOutShape _aDstShape aInShapes aValue aAddr := A.prim | continue
if aAddr != .laneIdx then continue    -- Map producer only
```

### 2.3 `fuseMatmulEpilogue` / `fuseReduceEpilogue`

Both gain a guard that the consumer scatter has `addrExpr = .laneIdx`
(Map-shaped). Absorbing a dynamic scatter as an "epilogue" doesn't make
sense.

## 3. Wiring into Gemma 4

**Target**: `forwardBlock`'s call to `fusedRopeKAndCacheWriteKernel` —
the hand-coded kernel that writes K (with NeoX RoPE) and V (plain copy)
to the cache in a single dispatch.

**Switched by `HESPER_SCATTER_KV=1`** to the Circuit path:
- K scatter: NeoX RoPE + dynamic-addr scatter
- V scatter: plain copy + dynamic-addr scatter

K scatter as a `ScalarExp` (compact and readable):

```lean
let i        := .laneIdx
let headSE   := .idiv i (.const headDim.toFloat)
let d        := .mod  i (.const headDim.toFloat)
let dLow     := .lt d (.const halfDim.toFloat)
let pairD    := .select dLow (d + halfDim) (d - halfDim)
let pairIdx  := headSE * headDim + pairD
let xSelf    := .input 0
let xPair    := .indexed 0 pairIdx           -- gather
let freqFac  := .indexed 1 (.select dLow d (d - halfDim))
let theta    := .input 2 * .pow base (- (2*dimPair/headDim)) / freqFac
let cosT     := .cos theta
let sinT     := .sin theta
let x0       := .select dLow xSelf xPair
let x1       := .select dLow xPair xSelf
valueExpr    := .select dLow (x0*cosT - x1*sinT) (x0*sinT + x1*cosT)
addrExpr     := headSE * stride + .input 2 * headDim + d
```

This expresses the same semantics as llama.cpp's
`ROPE + VIEW + SET_ROWS` fusion (Pattern E).

## 4. Verification

### 4.1 IR fusion unit tests (all PASS)

- `FusePointwiseTest` — 3-stage Map scatter chain → 1 scatter
- `FuseMatmulEpilogueTest` — matmulQ4K + scatter → matmulQ4KWithEpilogue
- `FuseWriteDestinationTest` — Map scatter + writeSlice → single offset scatter

### 4.2 GPU correctness tests (PASS)

- `ScatterDynamicGPUTest` (512 cells)
  - KV-cache-style addressing: `head*stride + pos*headDim + d`
  - Only the `pos=3` slot is written; everything else stays zero

- `RopeKScatterGPUTest` (2048 cells)
  - Full NeoX RoPE-K + KV-cache-write expressed as one Circuit scatter
  - **Bit-identical** match against the hand-coded
    `fusedRopeKAndCacheWriteKernel`
  - All 2048/2048 cells satisfy `|diff| < 1e-4`

### 4.3 Gemma 4 in-situ verification

| Configuration | Output | TPS (3 tok) | Kernels/tok |
|---|---|---:|---:|
| Default (hand-coded) | `TheThe quick brownTheThe brown` | 48.6 | 957 |
| `HESPER_SCATTER_KV=1` | **bit-identical** | 48.2 | 961 |

- TPS delta: −0.4 (the scatter path splits the original fused K+V
  dispatch into two scatters, costing +4 kernels/tok)
- 10 tokens generated bit-identically across both paths

## 5. Design principles confirmed

The unification reinforced four DSL design rules:

1. **One semantic, not many.** "Compute a value, then write it
   somewhere" — once you separate value and address, Map / Slice /
   Scatter all collapse into one Prim.

2. **Use ScalarExp, not compile-time constants, for offsets.** The old
   `dstOffset : Nat` broke the moment dynamic addressing was needed.
   **Starting from `ScalarExp` makes the static-const case a special
   case (`.const n`) for free.**

3. **Operator overloading.** `+`, `-`, `*`, `/`, `OfNat`, etc. let
   `ScalarExp` code read like ordinary arithmetic, not like AST
   constructor soup.

4. **Fusion passes branch on shape, not Prim type.** `fusePointwise`
   only checks "is the producer a Map-shaped scatter?
   (`addrExpr = .laneIdx`)". It doesn't care that the consumer might be
   a dynamic scatter — same inlining works.

## 6. Follow-ups

### 6.1 Multi-output scatter (Plan 5 candidate)

Today `HESPER_SCATTER_KV=1` uses two dispatches (+4 kernels/tok).
llama.cpp does it in one. To regain that, we'd need a kernel that
writes to multiple destination buffers in one dispatch:

```lean
| scatterMulti
    (outShape : Shape)
    (inShapes : Array Shape)
    (outputs  : Array (Shape × ScalarExp × ScalarExp))
    -- outputs[k] = (dstShape_k, valueExpr_k, addrExpr_k)
```

### 6.2 matmul + scatter fusion (Plan 6 candidate)

Fuse `Prim.matmulQ4K → Prim.scatter` into one kernel (the equivalent of
`matmulQ4KWithEpilogue` for the scatter case). hesper-side analogue of
llama.cpp Patterns A/B/C.

### 6.3 reduce + scatter fusion

Fuse `Prim.reduceLastAxis → Prim.scatter` into one kernel (a scatter
counterpart of `reduceLastAxisWithEpilogue`). Enables RMSNorm + dynamic
write fusion.

## 7. Related commit and files

- **commit b515e13**: feat(circuit): unify Map/Scatter into Prim.scatter, wire KV cache write
- [`Hesper/Circuit/IR.lean`](../../Hesper/Circuit/IR.lean) — `Prim.scatter` definition
- [`Hesper/Circuit/Lowering.lean`](../../Hesper/Circuit/Lowering.lean) — `runScatterOp` / `lowerScatter`
- [`Hesper/Circuit/Passes.lean`](../../Hesper/Circuit/Passes.lean) — `fusePointwise` over scatter
- [`Tests/Circuit/ScatterDynamicGPUTest.lean`](../../Tests/Circuit/ScatterDynamicGPUTest.lean)
- [`Tests/Circuit/RopeKScatterGPUTest.lean`](../../Tests/Circuit/RopeKScatterGPUTest.lean)
- [`Hesper/Models/Gemma4.lean`](../../Hesper/Models/Gemma4.lean) — `HESPER_SCATTER_KV` path
