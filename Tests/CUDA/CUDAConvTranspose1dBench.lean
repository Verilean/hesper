import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Audio
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)
open Hesper.Layers.Audio

private structure Shape where
  IC : Nat
  OC : Nat
  KW : Nat
  IL : Nat
  s0 : Nat

private def packF32 (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

private unsafe def benchOne (ctx : Hesper.CUDAContext) (sh : Shape)
    (nIter nWarmup : Nat) : IO Float := do
  let { IC, OC, KW, IL, s0 } := sh
  let OL := (IL - 1) * s0 + KW
  let srcSize := IC * IL
  let wSize := IC * OC * KW
  let outSize := OC * OL

  let srcArr : Array Float :=
    (List.range srcSize).toArray.map (fun i => Float.sin (i.toFloat * 0.027) * 0.4)
  let wArr : Array Float :=
    (List.range wSize).toArray.map (fun i => Float.cos (i.toFloat * 0.039) * 0.3)
  let srcBuf ← GPUBackend.allocBuffer ctx (srcSize * 4).toUSize
  let wBuf   ← GPUBackend.allocBuffer ctx (wSize * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (outSize * 4).toUSize
  GPUBackend.writeBuffer ctx srcBuf (← packF32 srcArr)
  GPUBackend.writeBuffer ctx wBuf   (← packF32 wArr)

  let nBlocks := (outSize + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("w", wBuf), ("dst", dstBuf) ]
  let cfg : Hesper.ExecConfig :=
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocks, 1, 1) }
  let kernel := convTranspose1dF32Kernel IC OC KW IL OL s0
  let cacheKey := hash ("bench-conv-transpose-1d", IC, OC, KW, IL, s0)
  let cacheRef ← IO.mkRef (none : Option (GPUBackend.CachedDispatch Hesper.CUDAContext))

  for _ in [0:nWarmup] do
    GPUBackend.executeWithConfigCached ctx kernel bufs cfg cacheKey cacheRef
  let _ ← GPUBackend.readBuffer ctx dstBuf 4

  let t0 ← IO.monoNanosNow
  for _ in [0:nIter] do
    GPUBackend.executeWithConfigCached ctx kernel bufs cfg cacheKey cacheRef
  let _ ← GPUBackend.readBuffer ctx dstBuf 4
  let t1 ← IO.monoNanosNow

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx dstBuf

  let ns := (t1 - t0).toFloat
  return ns / 1000.0 / nIter.toFloat

unsafe def main : IO Unit := do
  IO.println "═══ hesper convTranspose1dF32 benchmark ═══"
  let ctx ← Hesper.CUDAContext.init

  let shapes : List Shape := [
    { IC := 4,   OC := 3,   KW := 4,  IL := 8,    s0 := 2 },   -- parity test
    { IC := 64,  OC := 64,  KW := 4,  IL := 512,  s0 := 2 },   -- VocoderS-like
    { IC := 128, OC := 64,  KW := 8,  IL := 256,  s0 := 4 },
    { IC := 512, OC := 256, KW := 16, IL := 128,  s0 := 8 },
    { IC := 32,  OC := 32,  KW := 3,  IL := 4096, s0 := 1 },   -- codec head
  ]
  IO.println "shape                                          hesper µs/call"
  IO.println "-------------------------------------------------------------"
  for sh in shapes do
    let t ← benchOne ctx sh 100 20
    let label := s!"IC={sh.IC} OC={sh.OC} KW={sh.KW} IL={sh.IL} s={sh.s0}"
    let padded := label ++ String.ofList (List.replicate (46 - label.length) ' ')
    IO.println s!"{padded}{t}"
