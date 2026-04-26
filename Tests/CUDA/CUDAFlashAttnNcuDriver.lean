import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad
import Hesper.WGSL.FlashAttention
import Hesper

/-!
# Slim FlashAttn ncu driver

Runs only one kernel variant (`vec` or `llama`) at a single cacheLen,
in a tight raw-launch loop with no wall-time printing or comparison.
Designed to be the target of `ncu --kernel-name regex --launch-skip N
--launch-count M` runs so we can isolate one kernel's compute/memory
profile.

Usage:
  cuda-flashattn-ncu-driver vec   <cacheLen> <iters>
  cuda-flashattn-ncu-driver llama <cacheLen> <iters>

`iters` includes a small warmup; ncu users typically pass
`--launch-skip 5 --launch-count 3` to skip JIT-warmup launches and
measure 3 steady-state launches.
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
    let b := f64ToF32Bits f
    acc.push b.toUInt8
       |>.push (b >>> 8).toUInt8
       |>.push (b >>> 16).toUInt8
       |>.push (b >>> 24).toUInt8) ByteArray.empty

private def f32ToF16Bits (b : UInt32) : UInt16 :=
  let s : UInt32 := (b >>> 31) &&& 1
  let e32 : UInt32 := (b >>> 23) &&& 0xFF
  let m32 : UInt32 := b &&& 0x7FFFFF
  if e32 == 0 then 0
  else if e32 == 0xFF then (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
  else
    let e32i : Int := e32.toNat - 127 + 15
    if e32i <= 0 then 0
    else if e32i >= 31 then (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
    else
      let m16 : UInt32 := m32 >>> 13
      (s.toUInt16 <<< 15) ||| (e32i.toNat.toUInt16 <<< 10) ||| m16.toUInt16

private def packHalfs (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let h := f32ToF16Bits f.toFloat32.toBits
    acc.push h.toUInt8 |>.push (h >>> 8).toUInt8) ByteArray.empty

private def packU32x2 (a b : Nat) : ByteArray :=
  let aU : UInt32 := a.toUInt32
  let bU : UInt32 := b.toUInt32
  ByteArray.empty
    |>.push aU.toUInt8 |>.push (aU >>> 8).toUInt8
    |>.push (aU >>> 16).toUInt8 |>.push (aU >>> 24).toUInt8
    |>.push bU.toUInt8 |>.push (bU >>> 8).toUInt8
    |>.push (bU >>> 16).toUInt8 |>.push (bU >>> 24).toUInt8

unsafe def main (argv : List String) : IO Unit := do
  let (tag, cacheLen, iters) ← match argv with
    | [t, cl, it] =>
      pure (t, cl.toNat!, it.toNat!)
    | _ =>
      IO.println "usage: cuda-flashattn-ncu-driver <vec|v2|v3|v6|v7|v8|v9|llama> <cacheLen> <iters>"
      IO.Process.exit 1

  let ctx ← Hesper.CUDAContext.init
  let nh := 8
  let nkv := 1
  let hd := 256
  let maxSeq := 256
  let scale : Float := 1.0 / hd.toFloat.sqrt

  let qSize := (nh * hd * 4).toUSize
  let kvSizeF32 := (nkv * maxSeq * hd * 4).toUSize
  let kvSizeF16 := (nkv * maxSeq * hd * 2).toUSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let outBuf ← GPUBackend.allocBuffer ctx qSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)

  let qData := Array.range (nh * hd) |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)
  let kvZeros : Array Float := Array.replicate (nkv * maxSeq * hd) 0.0

  match tag with
  | "v9" | "v11" =>
    -- V9 / V11 = split-K kernel + combine.  V11 uses sub-warp partition.
    let numSplits := 8
    let kF16Buf ← GPUBackend.allocBuffer ctx kvSizeF16
    let vF16Buf ← GPUBackend.allocBuffer ctx kvSizeF16
    GPUBackend.writeBuffer ctx kF16Buf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx vF16Buf (packHalfs kvZeros)
    let partialOutSize := (nh * numSplits * hd * 4).toUSize
    let partialMetaSize := (nh * numSplits * 2 * 4).toUSize
    let partialOutBuf ← GPUBackend.allocBuffer ctx partialOutSize
    let partialMetaBuf ← GPUBackend.allocBuffer ctx partialMetaSize
    let mainShader := match tag with
      | "v11" => Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV11
                    nh nkv maxSeq hd numSplits scale
      | _     => Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV9
                    nh nkv maxSeq hd numSplits scale
    let combineShader := Hesper.WGSL.FlashAttention.flashAttentionVecCombineKernel
                            nh hd numSplits
    let mainName := s!"{tag}_ncu"
    let combName := s!"{tag}_combine_ncu"
    let mainPtx := Hesper.CUDA.CodeGen.generatePTX mainName
                     { x := 128, y := 1, z := 1 } mainShader
                     (minnctapersm := some 1)
    let combPtx := Hesper.CUDA.CodeGen.generatePTX combName
                     { x := 128, y := 1, z := 1 } combineShader
                     (minnctapersm := some 1)
    IO.FS.writeFile s!"/tmp/v_{tag}.ptx" mainPtx
    IO.FS.writeFile s!"/tmp/v_{tag}_combine.ptx" combPtx
    let mainMod ← Hesper.CUDA.cuModuleLoadData mainPtx
    let combMod ← Hesper.CUDA.cuModuleLoadData combPtx
    let v9F ← Hesper.CUDA.cuModuleGetFunction mainMod mainName
    let combF ← Hesper.CUDA.cuModuleGetFunction combMod combName
    let v9State := Hesper.WGSL.Monad.ShaderM.exec mainShader
    let combState := Hesper.WGSL.Monad.ShaderM.exec combineShader
    let v9Names := v9State.declaredBuffers.map (·.1)
    let combNames := combState.declaredBuffers.map (·.1)
    let allBufs : List (String × Hesper.CUDA.CUDABuffer) :=
      [ ("q", qBuf), ("k_cache_f16", kF16Buf), ("v_cache_f16", vF16Buf)
      , ("partial_out", partialOutBuf), ("partial_meta", partialMetaBuf)
      , ("output", outBuf), ("params", paramsBuf) ]
    let v9Args : Array USize ← v9Names.foldlM (init := #[]) fun acc name => do
      match allBufs.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"{tag} missing buffer '{name}'")
    let combArgs : Array USize ← combNames.foldlM (init := #[]) fun acc name => do
      match allBufs.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"combine missing buffer '{name}'")
    IO.println s!"[ncu-driver] {tag} cacheLen={cacheLen} iters={iters} numSplits={numSplits} (symbols: {mainName} + {combName})"
    for _ in [0:iters] do
      Hesper.CUDA.cuLaunchKernel v9F nh.toUInt32 numSplits.toUInt32 1 128 1 1 0 v9Args
      Hesper.CUDA.cuLaunchKernel combF nh.toUInt32 1 1 128 1 1 0 combArgs
    Hesper.CUDA.cuStreamSynchronize (0 : USize)
  | "vec" | "v2" | "v3" | "v6" | "v7" | "v8" =>
    let vBuf ← GPUBackend.allocBuffer ctx kvSizeF32
    GPUBackend.writeBuffer ctx vBuf (packFloats kvZeros)
    -- K storage shape depends on tag
    let kF32Buf ← GPUBackend.allocBuffer ctx kvSizeF32
    let kF16Buf ← GPUBackend.allocBuffer ctx kvSizeF16
    GPUBackend.writeBuffer ctx kF32Buf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx kF16Buf (packHalfs kvZeros)
    -- For V7, we'll reuse the V/K f16 path by also creating an f16 V buffer.
    let vF16Buf ← GPUBackend.allocBuffer ctx kvSizeF16
    GPUBackend.writeBuffer ctx vF16Buf (packHalfs kvZeros)
    let (shader, funcName, kBufName, kBuf, vBufName, vBufActive) := match tag with
      | "v2" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV2
                   nh nkv maxSeq hd scale, "v2_ncu", "k_cache", kF32Buf, "v_cache", vBuf)
      | "v3" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV3
                   nh nkv maxSeq hd scale, "v3_ncu", "k_cache_f16", kF16Buf, "v_cache", vBuf)
      | "v6" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV6
                   nh nkv maxSeq hd scale, "v6_ncu", "k_cache", kF32Buf, "v_cache", vBuf)
      | "v7" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV7
                   nh nkv maxSeq hd scale, "v7_ncu", "k_cache_f16", kF16Buf, "v_cache_f16", vF16Buf)
      | "v8" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV8
                   nh nkv maxSeq hd scale, "v8_ncu", "k_cache", kF32Buf, "v_cache", vBuf)
      | _    => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                   nh nkv maxSeq hd scale, "vec_ncu", "k_cache", kF32Buf, "v_cache", vBuf)
    -- minnctapersm := 1 tells ptxas this kernel uses ≥ 1 CTA per SM, so
    -- it can use up to 64K reg / 128 thread = 512 reg/thread (capped at 255
    -- by hardware).  Without this, ptxas defaults to higher CTA-per-SM
    -- assumption → tighter register budget → spill in V6/V8.
    let ptx := Hesper.CUDA.CodeGen.generatePTX funcName
                 { x := 128, y := 1, z := 1 } shader
                 (minnctapersm := some 1)
    -- Dump for inspection
    IO.FS.writeFile s!"/tmp/v_{tag}.ptx" ptx
    let cudaMod ← Hesper.CUDA.cuModuleLoadData ptx
    let f ← Hesper.CUDA.cuModuleGetFunction cudaMod funcName
    let state := Hesper.WGSL.Monad.ShaderM.exec shader
    let declaredNames := state.declaredBuffers.map (·.1)
    let bufs : List (String × Hesper.CUDA.CUDABuffer) :=
      [ ("q", qBuf), (kBufName, kBuf), (vBufName, vBufActive)
      , ("output", outBuf), ("params", paramsBuf) ]
    let args : Array USize ← declaredNames.foldlM (init := #[]) fun acc name => do
      match bufs.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"missing buffer '{name}'")
    IO.println s!"[ncu-driver] {tag} cacheLen={cacheLen} iters={iters} (kernel symbol: {funcName})"
    for _ in [0:iters] do
      Hesper.CUDA.cuLaunchKernel f nh.toUInt32 1 1 128 1 1 0 args
    Hesper.CUDA.cuStreamSynchronize (0 : USize)
  | "llama" =>
    let kBuf ← GPUBackend.allocBuffer ctx kvSizeF16
    let vBuf ← GPUBackend.allocBuffer ctx kvSizeF16
    GPUBackend.writeBuffer ctx kBuf (packHalfs kvZeros)
    GPUBackend.writeBuffer ctx vBuf (packHalfs kvZeros)
    let llamaK ← Hesper.LlamaCppPTX.loadFattnVecF16F16Kernel
    IO.println s!"[ncu-driver] llama cacheLen={cacheLen} iters={iters} (kernel symbol: flash_attn_ext_vec<256,1,F16,F16,false>)"
    for _ in [0:iters] do
      Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256 llamaK
        qBuf.ptr kBuf.ptr vBuf.ptr outBuf.ptr
        nh nkv hd cacheLen maxSeq scale
    Hesper.CUDA.cuStreamSynchronize (0 : USize)
  | other =>
    IO.println s!"[ncu-driver] unknown tag '{other}', expected vec/v2/v3/v6/llama"
    IO.Process.exit 1
