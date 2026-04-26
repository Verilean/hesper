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
      IO.println "usage: cuda-flashattn-ncu-driver <vec|v2|llama> <cacheLen> <iters>"
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
  | "vec" | "v2" | "v3" =>
    let vBuf ← GPUBackend.allocBuffer ctx kvSizeF32
    GPUBackend.writeBuffer ctx vBuf (packFloats kvZeros)
    -- K storage shape depends on tag
    let kF32Buf ← GPUBackend.allocBuffer ctx kvSizeF32
    let kF16Buf ← GPUBackend.allocBuffer ctx kvSizeF16
    GPUBackend.writeBuffer ctx kF32Buf (packFloats kvZeros)
    GPUBackend.writeBuffer ctx kF16Buf (packHalfs kvZeros)
    let (shader, funcName, kBufName, kBuf) := match tag with
      | "v2" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV2
                   nh nkv maxSeq hd scale, "v2_ncu", "k_cache", kF32Buf)
      | "v3" => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV3
                   nh nkv maxSeq hd scale, "v3_ncu", "k_cache_f16", kF16Buf)
      | _    => (Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                   nh nkv maxSeq hd scale, "vec_ncu", "k_cache", kF32Buf)
    let ptx := Hesper.CUDA.CodeGen.generatePTX funcName
                 { x := 128, y := 1, z := 1 } shader
    let cudaMod ← Hesper.CUDA.cuModuleLoadData ptx
    let f ← Hesper.CUDA.cuModuleGetFunction cudaMod funcName
    let state := Hesper.WGSL.Monad.ShaderM.exec shader
    let declaredNames := state.declaredBuffers.map (·.1)
    let bufs : List (String × Hesper.CUDA.CUDABuffer) :=
      [ ("q", qBuf), (kBufName, kBuf), ("v_cache", vBuf)
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
    IO.println s!"[ncu-driver] unknown tag '{other}', expected 'vec', 'v2', or 'llama'"
    IO.Process.exit 1
