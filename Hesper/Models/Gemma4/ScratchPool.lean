import Hesper.Backend

/-!
# ScratchPool — per-forward reusable GPU buffers

Pool of GPU buffers that are reused across forward passes.  Inspired by
llama.cpp's `ggml_gallocr` (compute-buffer allocator), but built as a
thin layer on top of the existing `GPUBackend.allocBuffer` instead of
changing the backend typeclass.

## Invariant

Between `reset` calls, the pool hands out one buffer per `alloc` call in
insertion order.  Each buffer is of size `max(previousSize, thisSize)`
so growth is monotonic.  `reset` rewinds the counter to 0 without
freeing any buffer; subsequent `alloc` calls reuse the same underlying
`GPUBackend.Buf`.

This means:
  * First forward with N scratch calls: N `cuMemAlloc` device allocations.
  * Every subsequent forward with ≤ N scratch calls: **0 device allocs**,
    just handle reuse.
  * If a later forward needs MORE scratch slots than any prior forward,
    the pool grows once; that growth is permanent.

## Usage

```lean
let pool ← ScratchPool.new ctx
-- forward #1
let batchQ ← pool.alloc (qDim * seqLen)
let batchK ← pool.alloc (kvDim * seqLen)
... use batchQ, batchK ...
pool.reset
-- forward #2
let batchQ' ← pool.alloc (qDim * seqLen)  -- reuses the same underlying buffer
```
-/

namespace Hesper.Models.Gemma4

/-- Pool slot: a GPU buffer plus the number of f32 elements it holds. -/
private structure PoolSlot (BufT : Type) where
  buf     : BufT
  elems32 : Nat

/-- Scratch pool for reusable GPU buffers.
    Internals are `private` — users interact only via `alloc` / `reset`. -/
structure ScratchPool (β : Type) [GPUBackend β] where
  ctx   : β
  slots : IO.Ref (Array (PoolSlot (GPUBackend.Buf β)))
  cursor : IO.Ref Nat

namespace ScratchPool

variable {β : Type} [GPUBackend β]

/-- Create an empty pool.  No device allocations happen here. -/
def new (ctx : β) : IO (ScratchPool β) := do
  let slots ← IO.mkRef (Array.empty : Array (PoolSlot (GPUBackend.Buf β)))
  let cursor ← IO.mkRef 0
  return { ctx := ctx, slots := slots, cursor := cursor }

/-- Allocate (or reuse) a scratch buffer holding at least `elems32` f32 values.

    If the pool's cursor is within the existing slot array and the slot
    has capacity `≥ elems32`, reuse it as-is.  Otherwise, if the slot
    exists but is too small, free it and allocate a larger one.  If no
    slot exists at the cursor index, allocate a fresh one. -/
def alloc (pool : ScratchPool β) (elems32 : Nat) : IO (GPUBackend.Buf β) := do
  let idx ← pool.cursor.get
  let arr ← pool.slots.get
  let buf ←
    if h : idx < arr.size then
      let slot : PoolSlot (GPUBackend.Buf β) := arr[idx]
      if slot.elems32 >= elems32 then
        pure slot.buf
      else do
        GPUBackend.freeBuffer pool.ctx slot.buf
        let newBuf ← GPUBackend.allocBuffer pool.ctx (elems32 * 4).toUSize
        pool.slots.set (arr.set idx { buf := newBuf, elems32 := elems32 })
        pure newBuf
    else do
      let newBuf ← GPUBackend.allocBuffer pool.ctx (elems32 * 4).toUSize
      pool.slots.set (arr.push { buf := newBuf, elems32 := elems32 })
      pure newBuf
  pool.cursor.set (idx + 1)
  return buf

/-- Rewind the cursor so the next `alloc` reuses slot 0.  Does not
    free any device memory. -/
def reset (pool : ScratchPool β) : IO Unit :=
  pool.cursor.set 0

/-- Free all pool buffers and clear state.  Called at program end. -/
def shutdown (pool : ScratchPool β) : IO Unit := do
  let arr ← pool.slots.get
  for slot in arr do
    GPUBackend.freeBuffer pool.ctx slot.buf
  pool.slots.set Array.empty
  pool.cursor.set 0

/-- Diagnostic: number of slots currently allocated. -/
def size (pool : ScratchPool β) : IO Nat := do
  let arr ← pool.slots.get
  return arr.size

end ScratchPool

end Hesper.Models.Gemma4
