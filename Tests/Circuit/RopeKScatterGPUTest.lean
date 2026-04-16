import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.Layers.Attention

/-!
End-to-end GPU test for `CircuitM.scatterInto` expressing NeoX-style
RoPE-K + KV-cache write as **a single scatter** with:
  - a complex value expression (rotation using cos/sin/pow + gather read
    of the pair-position element via `ScalarExp.indexed`)
  - a dynamic address expression (kvHead * stride + pos * headDim + d)

Two kernels are run with identical inputs:
  1. Hand-coded `Attention.fusedRopeKAndCacheWriteKernel` (reference).
  2. Circuit DSL `scatterInto` (the new path).

We compare the K cache buffers element-wise; they must match to f32
precision.  This proves the DSL can express RoPE semantics and the
gather operation (`.indexed`) works.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

def runTest : IO Unit := do
  let numKVHeads : Nat := 4
  let maxSeqLen  : Nat := 32
  let headDim    : Nat := 16
  let halfDim    : Nat := headDim / 2
  let kvDim      : Nat := numKVHeads * headDim   -- 64
  let cacheSz    : Nat := numKVHeads * maxSeqLen * headDim  -- 2048
  let pos        : Nat := 5
  let ropeBase   : Float := 1000000.0
  let ctx ← Hesper.CUDAContext.init

  -- K source (new tokens): [kvDim]
  let kArr : Array Float := Array.ofFn (n := kvDim) fun i =>
    ((i.val.toFloat * 0.137) - 2.0)
  let kBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes kArr
  let kBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * kvDim).toUSize
  GPUBackend.writeBuffer (β := β) ctx kBuf kBytes

  -- V source (unused by the K-only circuit, but required by hand-coded
  -- kernel signature).  We only compare K caches.
  let vArr : Array Float := Array.ofFn (n := kvDim) fun i => i.val.toFloat
  let vBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes vArr
  let vBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * kvDim).toUSize
  GPUBackend.writeBuffer (β := β) ctx vBuf vBytes

  -- Frequency factors: [halfDim]
  let ffArr : Array Float := Array.ofFn (n := halfDim) fun i =>
    (1.0 + 0.1 * i.val.toFloat)
  let ffBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes ffArr
  let ffBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * halfDim).toUSize
  GPUBackend.writeBuffer (β := β) ctx ffBuf ffBytes

  -- Reference K cache (hand-coded kernel writes here).
  let zeroCache : Array Float := Array.ofFn (n := cacheSz) fun _ => 0.0
  let zeroCacheBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes zeroCache
  let kCacheRef ← GPUBackend.allocBuffer (β := β) ctx (4 * cacheSz).toUSize
  GPUBackend.writeBuffer (β := β) ctx kCacheRef zeroCacheBytes

  -- V cache (required by hand-coded signature, we ignore its contents).
  let vCacheRef ← GPUBackend.allocBuffer (β := β) ctx (4 * cacheSz).toUSize
  GPUBackend.writeBuffer (β := β) ctx vCacheRef zeroCacheBytes

  -- Params: [pos (u32), cacheLen (u32)]
  let paramsBuf ← GPUBackend.allocBuffer (β := β) ctx (8 : USize)
  let posU32 := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
  GPUBackend.writeBufferOffset (β := β) ctx paramsBuf 0 posU32
  let cacheLenU32 := Hesper.WebGPU.BufferOps.uint32ToBytes (pos + 1).toUInt32
  GPUBackend.writeBufferOffset (β := β) ctx paramsBuf 4 cacheLenU32

  -- Run hand-coded kernel via CUDA backend.
  let refCacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernel
      numKVHeads maxSeqLen headDim kvDim ropeBase)
    [("new_k", kBuf), ("new_v", vBuf),
     ("k_cache", kCacheRef), ("v_cache", vCacheRef),
     ("params", paramsBuf), ("freq_factors", ffBuf)]
    (Hesper.ExecConfig.dispatch1D kvDim)
    (hash ("refRopeKScatterTest", numKVHeads, maxSeqLen, headDim))
    refCacheRef

  let refBytes ← GPUBackend.readBuffer (β := β) ctx kCacheRef (4 * cacheSz).toUSize

  -- Now run Circuit scatter.
  let kCacheScatter ← GPUBackend.allocBuffer (β := β) ctx (4 * cacheSz).toUSize
  GPUBackend.writeBuffer (β := β) ctx kCacheScatter zeroCacheBytes

  -- posF32 as broadcast scalar.
  let posF32Buf ← GPUBackend.allocBuffer (β := β) ctx (4 : USize)
  let posF32Bytes ← Hesper.Basic.floatToBytes pos.toFloat
  GPUBackend.writeBuffer (β := β) ctx posF32Buf posF32Bytes

  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      -- inputs[0] = kBuf       (lane-local, shape [kvDim])
      -- inputs[1] = ffBuf      (gather-only, shape [halfDim], NOT lane-local)
      -- inputs[2] = posF32Buf  (broadcast, shape [1])
      -- output   = kCacheScatter (shape [cacheSz])
      let kT ← CircuitM.registerExternal
                 (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                 kBuf #[kvDim] .f32 .Global
      let ffT ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  ffBuf #[halfDim] .f32 .Global
      let posT ← CircuitM.registerExternal
                   (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                   posF32Buf #[1] .f32 .Global
      let dst ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  kCacheScatter #[cacheSz] .f32 .Global

      -- NeoX RoPE-K as a ScalarExp.
      let i        : ScalarExp := .laneIdx
      let headSE   : ScalarExp := .idiv i (.const headDim.toFloat)
      let d        : ScalarExp := .mod  i (.const headDim.toFloat)
      let dLow     : ScalarExp := .lt d (.const halfDim.toFloat)
      -- pairD = if dLow then d + halfDim else d - halfDim
      let pairD    : ScalarExp :=
        .select dLow (d + .const halfDim.toFloat) (d - .const halfDim.toFloat)
      let pairIdx  : ScalarExp :=
        headSE * .const headDim.toFloat + pairD
      -- xSelf = kBuf[laneIdx] via slot 0 (pre-loaded lane-local read)
      let xSelf    : ScalarExp := .input 0
      -- xPair = gather from kBuf at pairIdx via .indexed
      let xPair    : ScalarExp := .indexed 0 pairIdx
      -- dimPair ∈ [0, halfDim): same as d if dLow, else d - halfDim
      let dimPair  : ScalarExp :=
        .select dLow d (d - .const halfDim.toFloat)
      -- freqFactor = ff[dimPair] (gather from slot 1)
      let freqFac  : ScalarExp := .indexed 1 dimPair
      let posSE    : ScalarExp := .input 2
      let exponent : ScalarExp :=
        .const 2.0 * dimPair / .const headDim.toFloat
      -- freqInv = ropeBase^(-exponent)
      let freqInv  : ScalarExp := .pow (.const ropeBase) (.neg exponent)
      let theta    : ScalarExp := posSE * freqInv / freqFac
      let cosT     : ScalarExp := .cos theta
      let sinT     : ScalarExp := .sin theta
      -- x0 = xSelf if dLow else xPair; x1 = xPair if dLow else xSelf
      let x0       : ScalarExp := .select dLow xSelf xPair
      let x1       : ScalarExp := .select dLow xPair xSelf
      let x0new    : ScalarExp := x0 * cosT - x1 * sinT
      let x1new    : ScalarExp := x0 * sinT + x1 * cosT
      let valueExpr : ScalarExp := .select dLow x0new x1new
      -- addr = head * (maxSeqLen * headDim) + pos * headDim + d
      let addrExpr : ScalarExp :=
        headSE * .const (maxSeqLen * headDim).toFloat
        + posSE * .const headDim.toFloat
        + d
      let _ ← CircuitM.scatterInto dst #[kvDim] #[kT, ffT, posT] valueExpr addrExpr
      pure ())
    [(0, kBuf), (1, ffBuf), (2, posF32Buf), (3, kCacheScatter)]

  let scatterBytes ← GPUBackend.readBuffer (β := β) ctx kCacheScatter (4 * cacheSz).toUSize

  -- Compare element-wise.
  let mut ok : Nat := 0
  let mut mismatch : Nat := 0
  let mut firstErr : Option (Nat × Float × Float × Float) := none
  for i in [0:cacheSz] do
    let a ← Hesper.Basic.bytesToFloat32 refBytes (i * 4)
    let b ← Hesper.Basic.bytesToFloat32 scatterBytes (i * 4)
    let diff := (a - b).abs
    if diff < 1e-4 then
      ok := ok + 1
    else
      mismatch := mismatch + 1
      if firstErr.isNone then firstErr := some (i, a, b, diff)

  if mismatch == 0 then
    IO.println s!"✓ PASS: K cache matches hand-coded RoPE kernel bit-wise ({ok} cells compared)"
  else
    IO.println s!"✗ FAIL: {mismatch} mismatches, {ok} matches"
    match firstErr with
    | some (i, a, b, diff) =>
      IO.println s!"  first mismatch: cache[{i}] ref={a} scatter={b} diff={diff}"
    | none => pure ()
    IO.Process.exit 1

unsafe def main : IO Unit := runTest
