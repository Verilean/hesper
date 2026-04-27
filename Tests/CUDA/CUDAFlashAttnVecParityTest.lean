import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.FlashAttention
import Hesper.WGSL.FlashAttentionExperiments
import Hesper

/-!
# FlashAttention vec-params parity test

Compares `flashAttentionVecParamsKernel` (doc 60 Session 1, warp-shuffle
reduce) against `flashAttentionDynamicParamsKernel` (legacy, 256-thread
tree reduce) on identical Q/K/V/cacheLen inputs.  Both kernels share
the same input/output contract — Q from `q`, output to `output`,
cacheLen in `params[1]`.

Pass criterion: `max(|out_vec[i] - out_legacy[i]|) < 1e-4` for all
output positions, across multiple cacheLen sizes.

Run:
  lake exe cuda-flashattn-vec-parity
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else
    let e32 : Int := e.toNat - 1023 + 127
    if e32 <= 0 then 0
    else if e32 >= 255 then
      (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (s.toUInt32 <<< 31) |||
      (e32.toNat.toUInt32 <<< 23) |||
      ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let bits := f64ToF32Bits f
    acc.push bits.toUInt8
       |>.push (bits >>> 8).toUInt8
       |>.push (bits >>> 16).toUInt8
       |>.push (bits >>> 24).toUInt8) ByteArray.empty

/-- Convert f32 bits to IEEE 754 binary16 (half) bits.  Round-to-zero on
    mantissa truncation; subnormals/Inf/NaN folded to zero / signed inf. -/
private def f32ToF16Bits (b : UInt32) : UInt16 :=
  let s : UInt32 := (b >>> 31) &&& 1
  let e32 : UInt32 := (b >>> 23) &&& 0xFF
  let m32 : UInt32 := b &&& 0x7FFFFF
  if e32 == 0 then 0  -- f32 zero/subnormal → half zero
  else if e32 == 0xFF then
    -- f32 inf/nan → half inf (preserve sign)
    (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
  else
    let e32i : Int := e32.toNat - 127 + 15
    if e32i <= 0 then 0
    else if e32i >= 31 then
      (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
    else
      let m16 : UInt32 := m32 >>> 13
      (s.toUInt16 <<< 15) |||
      (e32i.toNat.toUInt16 <<< 10) |||
      m16.toUInt16

private def packHalfs (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let h := f32ToF16Bits (f64ToF32Bits f)
    acc.push h.toUInt8
       |>.push (h >>> 8).toUInt8) ByteArray.empty

private def unpackFloats (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut arr := #[]
  for i in [0:n] do
    let o := i * 4
    let b0 : UInt32 := ba.get! o |>.toUInt32
    let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
    let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
    let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    let e := (bits >>> 23) &&& 0xFF
    let m := bits &&& (0x7FFFFF : UInt32)
    let s := bits >>> 31
    let f := if e == 0 then 0.0 else
      let fv := (1.0 + m.toNat.toFloat / 8388608.0)
                * Float.pow 2.0 (e.toNat.toFloat - 127.0)
      if s == 1 then -fv else fv
    arr := arr.push f
  return arr

/-- Pack two Nat values as little-endian u32 followed by another u32. -/
private def packU32x2 (a b : Nat) : ByteArray :=
  let aU : UInt32 := a.toUInt32
  let bU : UInt32 := b.toUInt32
  ByteArray.empty
    |>.push aU.toUInt8 |>.push (aU >>> 8).toUInt8
    |>.push (aU >>> 16).toUInt8 |>.push (aU >>> 24).toUInt8
    |>.push bU.toUInt8 |>.push (bU >>> 8).toUInt8
    |>.push (bU >>> 16).toUInt8 |>.push (bU >>> 24).toUInt8

/-- Run one (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen) case
    on both kernels and return (legacyOut, vecOut). -/
def runCase
    (ctx : Hesper.CUDAContext)
    (numHeads numKVHeads headDim maxSeqLen cacheLen : Nat)
    (scale : Float) : IO (Array Float × Array Float) := do
  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let outSize := qSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let outBufLegacy ← GPUBackend.allocBuffer ctx outSize
  let outBufVec ← GPUBackend.allocBuffer ctx outSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)

  -- Deterministic test inputs
  let qData := Array.range (numHeads * headDim)
                |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
  let mut kData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for kv in [0:numKVHeads] do
    for pos in [0:cacheLen] do
      for d in [0:headDim] do
        let off := (kv * maxSeqLen + pos) * headDim + d
        kData := kData.set! off (((kv + 1).toFloat * 0.05) +
                                  ((pos + 1).toFloat * 0.013) +
                                  (d.toFloat / 53.0).cos)
        vData := vData.set! off (((kv + 1).toFloat * 0.07) +
                                  ((pos + 1).toFloat * 0.011) +
                                  (d.toFloat / 41.0).sin)

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)
  GPUBackend.writeBuffer ctx vBuf (packFloats vData)
  -- params = [pos=cacheLen-1, cacheLen]; the kernel only reads slot 1.
  GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

  -- Legacy: dynamic-params kernel (256-thread tree reduce)
  let legacyShader := Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
                       numHeads numKVHeads maxSeqLen headDim scale 256
  let legacyBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufLegacy)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx legacyShader legacyBufs
    { workgroupSize := { x := 256 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  -- Vec: doc 60 Session 1 kernel (128-thread, warp-shuffle reduce)
  let vecShader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                     numHeads numKVHeads maxSeqLen headDim scale
  let vecBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufVec)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx vecShader vecBufs
    { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  -- V2: per-thread accumulator + warp-only reduce (Session 2′).  Skipped
  -- for headDim < workgroupSize (V2 requires headDim % 128 == 0).
  let outBufV2 ← GPUBackend.allocBuffer ctx outSize
  if headDim % 128 == 0 then
    let v2Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV2
                       numHeads numKVHeads maxSeqLen headDim scale
    let v2Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",       qBuf)
      , ("k_cache", kBuf)
      , ("v_cache", vBuf)
      , ("output",  outBufV2)
      , ("params",  paramsBuf) ]
    GPUBackend.execute ctx v2Shader v2Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
        extensions := ["subgroups"] }

  -- V3: V2 + K cache as f16 (Session 3).  Needs separate f16-packed K buffer.
  let outBufV3 ← GPUBackend.allocBuffer ctx outSize
  let kBufF16  ← GPUBackend.allocBuffer ctx
                   ((numKVHeads * maxSeqLen * headDim * 2).toUSize)
  if headDim % 128 == 0 then
    -- Convert kData to f16 packed-pair u32 buffer.
    GPUBackend.writeBuffer ctx kBufF16 (packHalfs kData)
    let v3Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV3
                       numHeads numKVHeads maxSeqLen headDim scale
    let v3Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",            qBuf)
      , ("k_cache_f16",  kBufF16)
      , ("v_cache",      vBuf)
      , ("output",       outBufV3)
      , ("params",       paramsBuf) ]
    GPUBackend.execute ctx v3Shader v3Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
        extensions := ["subgroups"] }

  -- V6: K-parallel inner loop (true llama-pattern).  Same buffer layout
  -- as V2 (f32 K, f32 V).  Pre-condition: D % 32 == 0.
  let outBufV6 ← GPUBackend.allocBuffer ctx outSize
  if headDim % 32 == 0 ∧ headDim % 128 == 0 then
    let v6Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV6
                       numHeads numKVHeads maxSeqLen headDim scale
    let v6Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",       qBuf)
      , ("k_cache", kBuf)
      , ("v_cache", vBuf)
      , ("output",  outBufV6)
      , ("params",  paramsBuf) ]
    GPUBackend.execute ctx v6Shader v6Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
        extensions := ["subgroups"] }

  -- V7: V6 + K cache f16 + V cache f16.  Pre-cond: D % 64 == 0.
  let outBufV7 ← GPUBackend.allocBuffer ctx outSize
  let vBufF16  ← GPUBackend.allocBuffer ctx
                   ((numKVHeads * maxSeqLen * headDim * 2).toUSize)
  if headDim % 64 == 0 ∧ headDim % 128 == 0 then
    GPUBackend.writeBuffer ctx vBufF16 (packHalfs vData)
    let v7Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV7
                       numHeads numKVHeads maxSeqLen headDim scale
    let v7Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",            qBuf)
      , ("k_cache_f16",  kBufF16)
      , ("v_cache_f16",  vBufF16)
      , ("output",       outBufV7)
      , ("params",       paramsBuf) ]
    GPUBackend.execute ctx v7Shader v7Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
        extensions := ["subgroups", "f16"] }

  -- V9: V7 + split-K (Session 5).  Pre-cond: D % 64 == 0, cacheLen >= numSplits.
  -- Two-kernel pipeline: (V9 partial) → (combine).
  let numSplits : Nat := 8
  let outBufV9 ← GPUBackend.allocBuffer ctx outSize
  let partialOutBuf ← GPUBackend.allocBuffer ctx
                        ((numHeads * numSplits * headDim * 4).toUSize)
  let partialMetaBuf ← GPUBackend.allocBuffer ctx
                         ((numHeads * numSplits * 2 * 4).toUSize)
  if headDim % 64 == 0 ∧ headDim % 128 == 0 ∧ cacheLen >= numSplits then
    let v9Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV9
                       numHeads numKVHeads maxSeqLen headDim numSplits scale
    let v9Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",             qBuf)
      , ("k_cache_f16",   kBufF16)
      , ("v_cache_f16",   vBufF16)
      , ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("params",        paramsBuf) ]
    GPUBackend.execute ctx v9Shader v9Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, numSplits, 1),
        extensions := ["subgroups", "f16"] }
    let combineShader := Hesper.WGSL.FlashAttention.flashAttentionVecCombineKernel
                            numHeads headDim numSplits
    let combineBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("output",        outBufV9) ]
    GPUBackend.execute ctx combineShader combineBufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1) }

  -- V11: V9 + V8 sub-warp partition (nthreads_KQ=8).  Reuses V9's partial
  -- buffers (same partial_out / partial_meta layout) — runs second so they
  -- get overwritten before V11's combine.
  let outBufV11 ← GPUBackend.allocBuffer ctx outSize
  if headDim % 64 == 0 ∧ headDim % 128 == 0 ∧ cacheLen >= numSplits then
    let v11Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV11
                       numHeads numKVHeads maxSeqLen headDim numSplits scale
    let v11Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",             qBuf)
      , ("k_cache_f16",   kBufF16)
      , ("v_cache_f16",   vBufF16)
      , ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("params",        paramsBuf) ]
    GPUBackend.execute ctx v11Shader v11Bufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, numSplits, 1),
        extensions := ["subgroups", "f16"] }
    let combineShader := Hesper.WGSL.FlashAttention.flashAttentionVecCombineKernel
                            numHeads headDim numSplits
    let combineBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("output",        outBufV11) ]
    GPUBackend.execute ctx combineShader combineBufs
      { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1) }

  let legacyResultBytes ← GPUBackend.readBuffer ctx outBufLegacy outSize
  let vecResultBytes    ← GPUBackend.readBuffer ctx outBufVec outSize
  let legacyOut := unpackFloats legacyResultBytes (numHeads * headDim)
  let vecOut    := unpackFloats vecResultBytes (numHeads * headDim)
  if headDim % 128 == 0 then
    let v2ResultBytes ← GPUBackend.readBuffer ctx outBufV2 outSize
    let v2Out := unpackFloats v2ResultBytes (numHeads * headDim)
    let v3ResultBytes ← GPUBackend.readBuffer ctx outBufV3 outSize
    let v3Out := unpackFloats v3ResultBytes (numHeads * headDim)
    let v6ResultBytes ← GPUBackend.readBuffer ctx outBufV6 outSize
    let v6Out := unpackFloats v6ResultBytes (numHeads * headDim)
    let v7ResultBytes ← GPUBackend.readBuffer ctx outBufV7 outSize
    let v7Out := unpackFloats v7ResultBytes (numHeads * headDim)
    let v9Out ← if cacheLen >= numSplits then do
                  let v9ResultBytes ← GPUBackend.readBuffer ctx outBufV9 outSize
                  pure (unpackFloats v9ResultBytes (numHeads * headDim))
                else pure legacyOut  -- skip if precondition not met
    let v11Out ← if cacheLen >= numSplits then do
                    let v11ResultBytes ← GPUBackend.readBuffer ctx outBufV11 outSize
                    pure (unpackFloats v11ResultBytes (numHeads * headDim))
                 else pure legacyOut
    let mut maxAbsV2 := 0.0
    let mut maxAbsV3 := 0.0
    let mut maxAbsV6 := 0.0
    let mut maxAbsV7 := 0.0
    let mut maxAbsV9 := 0.0
    let mut maxAbsV11 := 0.0
    for i in [0:legacyOut.size] do
      let d2 := (legacyOut[i]! - v2Out[i]!).abs
      let d3 := (legacyOut[i]! - v3Out[i]!).abs
      let d6 := (legacyOut[i]! - v6Out[i]!).abs
      let d7 := (legacyOut[i]! - v7Out[i]!).abs
      let d9 := (legacyOut[i]! - v9Out[i]!).abs
      let d11 := (legacyOut[i]! - v11Out[i]!).abs
      if d2 > maxAbsV2 then maxAbsV2 := d2
      if d3 > maxAbsV3 then maxAbsV3 := d3
      if d6 > maxAbsV6 then maxAbsV6 := d6
      if d7 > maxAbsV7 then maxAbsV7 := d7
      if d9 > maxAbsV9 then maxAbsV9 := d9
      if d11 > maxAbsV11 then maxAbsV11 := d11
    let m2 := if maxAbsV2 < 1e-4 then "✓ V2" else "✗ V2"
    let m3 := if maxAbsV3 < 1e-2 then "✓ V3" else "✗ V3"
    let m6 := if maxAbsV6 < 1e-3 then "✓ V6" else "✗ V6"
    let m7 := if maxAbsV7 < 1e-2 then "✓ V7" else "✗ V7"
    let m9 := if maxAbsV9 < 1e-2 then "✓ V9" else "✗ V9"
    let m11 := if maxAbsV11 < 1e-2 then "✓ V11" else "✗ V11"
    IO.println s!"      V2={maxAbsV2}[{m2}] V3={maxAbsV3}[{m3}] V6={maxAbsV6}[{m6}] V7={maxAbsV7}[{m7}] V9={maxAbsV9}[{m9}] V11={maxAbsV11}[{m11}]"

  GPUBackend.freeBuffer ctx qBuf
  GPUBackend.freeBuffer ctx kBuf
  GPUBackend.freeBuffer ctx vBuf
  GPUBackend.freeBuffer ctx outBufLegacy
  GPUBackend.freeBuffer ctx outBufVec
  GPUBackend.freeBuffer ctx outBufV2
  GPUBackend.freeBuffer ctx outBufV3
  GPUBackend.freeBuffer ctx outBufV6
  GPUBackend.freeBuffer ctx outBufV7
  GPUBackend.freeBuffer ctx outBufV9
  GPUBackend.freeBuffer ctx outBufV11
  GPUBackend.freeBuffer ctx partialOutBuf
  GPUBackend.freeBuffer ctx partialMetaBuf
  GPUBackend.freeBuffer ctx kBufF16
  GPUBackend.freeBuffer ctx vBufF16
  GPUBackend.freeBuffer ctx paramsBuf

  return (legacyOut, vecOut)

/-- Benchmark a single kernel by launching it `iters` times in a tight
    loop, then `cuStreamSynchronize` once at the end.  Returns
    (avg_us_per_call, total_wall_ms).

    `warmup` calls run before timing to flush JIT / cubin cache.

    We use the CUDA-internal `cuStreamSynchronize` (stream 0) — the
    legacy / vec kernels both target the default stream when invoked
    via `GPUBackend.execute`. -/
def benchKernel
    (ctx : Hesper.CUDAContext)
    (kernelName : String)
    (shader : Hesper.WGSL.Monad.ShaderM Unit)
    (bufs : List (String × GPUBackend.Buf Hesper.CUDAContext))
    (workgroupX : Nat) (numHeads : Nat)
    (warmup iters : Nat)
    (numWGY : Nat := 1)
    (extensions : List String := ["subgroups"]) : IO (Float × Float) := do
  let _ := kernelName
  let cfg : Hesper.ExecConfig := {
    workgroupSize := { x := workgroupX, y := 1, z := 1 }
    numWorkgroups := (numHeads, numWGY, 1)
    extensions := extensions
  }
  for _ in [0:warmup] do
    GPUBackend.execute ctx shader bufs cfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let t0 ← IO.monoNanosNow
  for _ in [0:iters] do
    GPUBackend.execute ctx shader bufs cfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let t1 ← IO.monoNanosNow
  let totalNs := t1 - t0
  let totalMs := totalNs.toFloat / 1.0e6
  let avgUs := totalNs.toFloat / iters.toFloat / 1000.0
  return (avgUs, totalMs)

/-- Bench a split-K pair (partial then combine).  Times the two dispatches
    together as one logical call.  Used for V9/V11 which need a separate
    combine kernel. -/
def benchSplitK
    (ctx : Hesper.CUDAContext)
    (partialShader combineShader : Hesper.WGSL.Monad.ShaderM Unit)
    (partialBufs combineBufs : List (String × GPUBackend.Buf Hesper.CUDAContext))
    (numHeads numSplits : Nat)
    (warmup iters : Nat) : IO (Float × Float) := do
  let partialCfg : Hesper.ExecConfig := {
    workgroupSize := { x := 128, y := 1, z := 1 }
    numWorkgroups := (numHeads, numSplits, 1)
    extensions := ["subgroups", "f16"]
  }
  let combineCfg : Hesper.ExecConfig := {
    workgroupSize := { x := 128, y := 1, z := 1 }
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
  }
  for _ in [0:warmup] do
    GPUBackend.execute ctx partialShader partialBufs partialCfg
    GPUBackend.execute ctx combineShader combineBufs combineCfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let t0 ← IO.monoNanosNow
  for _ in [0:iters] do
    GPUBackend.execute ctx partialShader partialBufs partialCfg
    GPUBackend.execute ctx combineShader combineBufs combineCfg
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let t1 ← IO.monoNanosNow
  let totalNs := t1 - t0
  let totalMs := totalNs.toFloat / 1.0e6
  let avgUs := totalNs.toFloat / iters.toFloat / 1000.0
  return (avgUs, totalMs)

def compareCase (label : String) (legacyOut vecOut : Array Float)
    (tolerance : Float := 1e-4) : IO Bool := do
  let mut maxAbs := 0.0
  let mut maxRel := 0.0
  let mut firstMismatchIdx : Option Nat := none
  for i in [0:legacyOut.size] do
    let a := legacyOut[i]!
    let b := vecOut[i]!
    let d := (a - b).abs
    if d > maxAbs then maxAbs := d
    let denom := max a.abs b.abs
    if denom > 1e-6 then
      let r := d / denom
      if r > maxRel then maxRel := r
    if d > tolerance && firstMismatchIdx.isNone then
      firstMismatchIdx := some i
  let pass := maxAbs < tolerance
  let mark := if pass then "✓" else "✗"
  IO.println s!"  {mark} {label}: max abs diff = {maxAbs}, max rel diff = {maxRel}"
  match firstMismatchIdx with
  | some i => IO.println s!"      first mismatch at [{i}]: legacy={legacyOut[i]!} vec={vecOut[i]!}"
  | none => pure ()
  return pass

unsafe def main : IO Unit := do
  IO.println "═══ FlashAttention Vec-Params Parity Test ═══"
  let ctx ← Hesper.CUDAContext.init

  -- Cases match Gemma 4 (head_dim=256, num_heads=8, num_kv_heads=1)
  -- but kept small enough that PTX cache + run time stays under a
  -- few seconds.  Bisect by varying cacheLen.
  let cases : List (Nat × Nat × Nat × Nat × Nat) :=
    [ -- (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen)
      (1, 1, 64,  16, 4)
    , (1, 1, 64,  16, 8)
    , (1, 1, 128, 32, 16)
    , (1, 1, 256, 32, 8)
    , (2, 1, 256, 32, 16)
    , (2, 1, 256, 64, 32)
    , (4, 1, 256, 64, 32)
    -- Gemma 4 production geometry
    , (8, 1, 256, 64, 32)
    , (8, 1, 256, 128, 64)
    , (8, 1, 256, 256, 100)
    -- Gemma 4 decode-style cacheLen increments (1..40)
    , (8, 1, 256, 256, 5)
    , (8, 1, 256, 256, 6)
    , (8, 1, 256, 256, 7)
    , (8, 1, 256, 256, 10)
    , (8, 1, 256, 256, 33)
    , (8, 1, 256, 256, 35)
    , (8, 1, 256, 256, 40) ]

  let mut allPassed := true
  for (nh, nkv, hd, msl, cl) in cases do
    let scale : Float := 1.0 / (hd.toFloat.sqrt)
    let label := s!"nH={nh} nKV={nkv} D={hd} maxSeq={msl} cacheLen={cl}"
    try
      let (legacy, vec) ← runCase ctx nh nkv hd msl cl scale
      let ok ← compareCase label legacy vec
      if !ok then allPassed := false
    catch e =>
      IO.println s!"  ✗ {label}: exception {e}"
      allPassed := false

  IO.println ""
  if allPassed then
    IO.println "═══ ALL CASES PASS — vec kernel is bit-parity with legacy ═══"
  else
    IO.println "═══ SOME CASES FAILED — see above ═══"
    IO.Process.exit 1

  -- ────────────────────────────────────────────────────────────────
  -- Microbenchmark: real Gemma-4 geometry (numHeads=8, headDim=256,
  -- numKVHeads=1) at a few cacheLen points spanning the decode range.
  -- 200 iters per kernel after 20 warmup → total wall ~50 ms / point.
  -- Lets us iterate on kernel design with sub-second TAT.
  --
  -- Third column: llama.cpp's flash_attn_ext_vec<D=256, ncols=1, K=f16,
  -- V=f16> PTX, loaded via Hesper.LlamaCppPTX.  Requires the extracted
  -- PTX at /tmp/llamacpp_ptx/fattn_vec_f16f16.ptx (see scripts in
  -- docs/llama-fusion-analysis/60-flashattn-execution-plan.md).  If
  -- absent, the column is skipped with a one-line note.
  IO.println ""
  IO.println "═══ Microbenchmark (Gemma 4 geometry: nH=8, nKV=1, D=256) ═══"
  let llamaKernelOpt ← try
      let k ← Hesper.LlamaCppPTX.loadFattnVecF16F16Kernel
      pure (some k)
    catch e =>
      IO.println s!"  (llama.cpp PTX column skipped — {e})"
      pure none
  let benchCases : List (Nat × Nat) :=
    [ (256, 8), (256, 32), (256, 64), (256, 128), (256, 200) ]
  let warmup := 20
  let iters := 200
  for (maxSeq, cacheLen) in benchCases do
    let nh := 8
    let nkv := 1
    let hd := 256
    let scale : Float := 1.0 / hd.toFloat.sqrt
    -- Set up buffers once per cacheLen point.  Reuse across kernels
    -- for fair memory-state comparison.
    let qSize := (nh * hd * 4).toUSize
    let kvSize := (nkv * maxSeq * hd * 4).toUSize
    let kvSizeHalf := (nkv * maxSeq * hd * 2).toUSize
    let qBuf ← GPUBackend.allocBuffer ctx qSize
    let kBuf ← GPUBackend.allocBuffer ctx kvSize
    let vBuf ← GPUBackend.allocBuffer ctx kvSize
    let outBuf ← GPUBackend.allocBuffer ctx qSize
    let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
    -- f16 mirror buffers for llama.cpp's K=f16, V=f16 instantiation.
    let kBufHalf ← GPUBackend.allocBuffer ctx kvSizeHalf
    let vBufHalf ← GPUBackend.allocBuffer ctx kvSizeHalf
    let outBufLlama ← GPUBackend.allocBuffer ctx qSize
    let qData := Array.range (nh * hd)
                  |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
    GPUBackend.writeBuffer ctx qBuf (packFloats qData)
    -- KV: just zero-fill is enough for timing; the kernel reads the
    -- full row regardless of values.
    let kvZeros : Array Float := Array.replicate (nkv * maxSeq * hd) 0.0
    GPUBackend.writeBuffer ctx kBuf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx vBuf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx kBufHalf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx vBufHalf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

    let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",       qBuf)
      , ("k_cache", kBuf)
      , ("v_cache", vBuf)
      , ("output",  outBuf)
      , ("params",  paramsBuf) ]

    let legacyShader := Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
                         nh nkv maxSeq hd scale 256
    let vecShader    := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                         nh nkv maxSeq hd scale
    let v2Shader     := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV2
                         nh nkv maxSeq hd scale
    let v6Shader     := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV6
                         nh nkv maxSeq hd scale

    let (legacyUs, legacyMs) ← benchKernel ctx "legacy" legacyShader bufs 256 nh warmup iters
    let (vecUs,    vecMs)    ← benchKernel ctx "vec"    vecShader    bufs 128 nh warmup iters
    let (v2Us,     v2Ms)     ← benchKernel ctx "v2"     v2Shader     bufs 128 nh warmup iters
    let (v6Us,     v6Ms)     ← benchKernel ctx "v6"     v6Shader     bufs 128 nh warmup iters
    let speedup := legacyUs / vecUs
    let v2Speedup := vecUs / v2Us
    let v6Speedup := v2Us / v6Us

    -- V7/V9/V11 columns (require f16 K/V cache; D % 64 == 0 ∧ D % 128 == 0).
    let numSplits : Nat := 8
    let v7Shader  := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV7
                       nh nkv maxSeq hd scale
    let v9Shader  := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV9
                       nh nkv maxSeq hd numSplits scale
    let v11Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV11
                       nh nkv maxSeq hd numSplits scale
    let combineShader := Hesper.WGSL.FlashAttention.flashAttentionVecCombineKernel
                           nh hd numSplits

    let v7BufsBench : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",            qBuf)
      , ("k_cache_f16",  kBufHalf)
      , ("v_cache_f16",  vBufHalf)
      , ("output",       outBuf)
      , ("params",       paramsBuf) ]
    let partialOutBuf  ← GPUBackend.allocBuffer ctx ((nh * numSplits * hd * 4).toUSize)
    let partialMetaBuf ← GPUBackend.allocBuffer ctx ((nh * numSplits * 2 * 4).toUSize)
    let splitPartialBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("q",             qBuf)
      , ("k_cache_f16",   kBufHalf)
      , ("v_cache_f16",   vBufHalf)
      , ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("params",        paramsBuf) ]
    let combineBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
      [ ("partial_out",   partialOutBuf)
      , ("partial_meta",  partialMetaBuf)
      , ("output",        outBuf) ]

    let (v7Us, _) ← benchKernel ctx "v7" v7Shader v7BufsBench 128 nh warmup iters
                      (numWGY := 1) (extensions := ["subgroups", "f16"])
    let (v9Us, _) ← if cacheLen >= numSplits then
                      benchSplitK ctx v9Shader combineShader splitPartialBufs combineBufs
                                  nh numSplits warmup iters
                    else pure (0.0, 0.0)
    let (v11Us, _) ← if cacheLen >= numSplits then
                       benchSplitK ctx v11Shader combineShader splitPartialBufs combineBufs
                                   nh numSplits warmup iters
                     else pure (0.0, 0.0)

    -- Raw-launch helper: compile a single shader, pre-resolve args, time
    -- bare cuLaunchKernel.  Returns avg µs/call.
    let rawLaunch (label : String) (shader : Hesper.WGSL.Monad.ShaderM Unit)
                  (kernelBufs : List (String × GPUBackend.Buf Hesper.CUDAContext))
                  (gx gy : Nat) (bx : UInt32) : IO Float := do
      let ptx := Hesper.CUDA.CodeGen.generatePTX label
                   { x := bx.toNat, y := 1, z := 1 } shader
      match ← IO.getEnv "HESPER_PTX_DUMP" with
      | some dir => do
        IO.FS.createDirAll dir
        IO.FS.writeFile s!"{dir}/{label}.ptx" ptx
      | none => pure ()
      let cudaMod ← Hesper.CUDA.cuModuleLoadData ptx
      let f ← Hesper.CUDA.cuModuleGetFunction cudaMod label
      let state := Hesper.WGSL.Monad.ShaderM.exec shader
      let declaredNames := state.declaredBuffers.map (·.1)
      let args : Array USize ← declaredNames.foldlM (init := #[]) fun acc name => do
        match kernelBufs.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => throw (IO.userError s!"raw-launch {label}: missing buffer '{name}'")
      for _ in [0:warmup] do
        Hesper.CUDA.cuLaunchKernel f gx.toUInt32 gy.toUInt32 1 bx 1 1 0 args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t0 ← IO.monoNanosNow
      for _ in [0:iters] do
        Hesper.CUDA.cuLaunchKernel f gx.toUInt32 gy.toUInt32 1 bx 1 1 0 args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t1 ← IO.monoNanosNow
      pure ((t1 - t0).toFloat / iters.toFloat / 1000.0)

    -- Raw-launch column: same vec kernel but without GPUBackend.execute.
    let rawInfo ← do
      let funcName : String := "vec_raw_kernel"
      let ptx := Hesper.CUDA.CodeGen.generatePTX funcName
                   { x := 128, y := 1, z := 1 } vecShader
      let cudaMod ← Hesper.CUDA.cuModuleLoadData ptx
      let f ← Hesper.CUDA.cuModuleGetFunction cudaMod funcName
      -- Resolve buffer arg order from the ShaderM exec to match what
      -- GPUBackend.execute would have built.
      let state := Hesper.WGSL.Monad.ShaderM.exec vecShader
      let declaredNames := state.declaredBuffers.map (·.1)
      let args : Array USize ← declaredNames.foldlM (init := #[]) fun acc name => do
        match bufs.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => throw (IO.userError s!"raw-launch: missing buffer '{name}'")
      -- Warmup
      for _ in [0:warmup] do
        Hesper.CUDA.cuLaunchKernel f
          nh.toUInt32 1 1
          128 1 1
          0
          args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t0 ← IO.monoNanosNow
      for _ in [0:iters] do
        Hesper.CUDA.cuLaunchKernel f
          nh.toUInt32 1 1
          128 1 1
          0
          args
      Hesper.CUDA.cuStreamSynchronize (0 : USize)
      let t1 ← IO.monoNanosNow
      let rawUs := (t1 - t0).toFloat / iters.toFloat / 1000.0
      let vecVsRaw := vecUs / rawUs
      pure s!"  raw={rawUs} µs/call  vec/raw={vecVsRaw}×"

    -- Raw-launch GPU times for V7/V9/V11: same kernels as the "v7/v9/v11"
    -- columns but without GPUBackend.execute host-path cost.  V9/V11 each
    -- include partial + combine.
    let v7RawUs ← rawLaunch "v7_raw" v7Shader v7BufsBench nh 1 128
    let v9PartialRawUs ← if cacheLen >= numSplits then
                            rawLaunch "v9p_raw" v9Shader splitPartialBufs nh numSplits 128
                         else pure 0.0
    let v11PartialRawUs ← if cacheLen >= numSplits then
                             rawLaunch "v11p_raw" v11Shader splitPartialBufs nh numSplits 128
                          else pure 0.0
    let combineRawUs ← if cacheLen >= numSplits then
                           rawLaunch "comb_raw" combineShader combineBufs nh 1 128
                       else pure 0.0
    let v9RawUs := v9PartialRawUs + combineRawUs
    let v11RawUs := v11PartialRawUs + combineRawUs
    let rawV7V9V11 := s!"  raw_v7={v7RawUs}µs raw_v9={v9RawUs}µs (p={v9PartialRawUs} c={combineRawUs}) raw_v11={v11RawUs}µs (p={v11PartialRawUs} c={combineRawUs})"

    -- Fourth column: llama.cpp PTX (only if the PTX module loaded).
    let llamaInfo ← match llamaKernelOpt with
      | none => pure ""
      | some llamaK => do
        -- Warmup
        for _ in [0:warmup] do
          Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256 llamaK
            qBuf.ptr kBufHalf.ptr vBufHalf.ptr outBufLlama.ptr
            nh nkv hd cacheLen maxSeq scale
        Hesper.CUDA.cuStreamSynchronize (0 : USize)
        let t0 ← IO.monoNanosNow
        for _ in [0:iters] do
          Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256 llamaK
            qBuf.ptr kBufHalf.ptr vBufHalf.ptr outBufLlama.ptr
            nh nkv hd cacheLen maxSeq scale
        Hesper.CUDA.cuStreamSynchronize (0 : USize)
        let t1 ← IO.monoNanosNow
        let llamaUs := (t1 - t0).toFloat / iters.toFloat / 1000.0
        let llamaMs := (t1 - t0).toFloat / 1.0e6
        let vecVsLlama := vecUs / llamaUs
        pure s!"  llama={llamaUs} µs/call ({llamaMs} ms / {iters})  vec/llama={vecVsLlama}×"
    IO.println s!"  cacheLen={cacheLen}: legacy={legacyUs} vec={vecUs} v2={v2Us} v6={v6Us} v7={v7Us} v9={v9Us} v11={v11Us} µs/call  legacy/vec={speedup}× v2/v6={v6Speedup}×{rawInfo}{rawV7V9V11}{llamaInfo} (v6 {v6Ms}ms/{iters})"

    GPUBackend.freeBuffer ctx partialOutBuf
    GPUBackend.freeBuffer ctx partialMetaBuf
    GPUBackend.freeBuffer ctx qBuf
    GPUBackend.freeBuffer ctx kBuf
    GPUBackend.freeBuffer ctx vBuf
    GPUBackend.freeBuffer ctx outBuf
    GPUBackend.freeBuffer ctx paramsBuf
    GPUBackend.freeBuffer ctx kBufHalf
    GPUBackend.freeBuffer ctx vBufHalf
    GPUBackend.freeBuffer ctx outBufLlama

  IO.println ""
  IO.println "═══ DONE ═══"
