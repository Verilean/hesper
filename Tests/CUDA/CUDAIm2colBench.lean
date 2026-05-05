import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Microbenchmark: hesper im2colF32Kernel timing across realistic shapes

Mirrors the llama.cpp `scripts/llama_parity/bench_im2col` runner so the
two outputs can be diffed side-by-side.

NOTE: hesper's `GPUBackend.execute` includes some host-side overhead
(JIT lookup, dispatch building) on the first call.  We warm up first.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)
open Hesper.Layers.Vision

private structure Shape where
  N : Nat
  IC : Nat
  IH : Nat
  IW : Nat
  KH : Nat
  KW : Nat
  s0 : Nat
  s1 : Nat
  p0 : Nat
  p1 : Nat
  d0 : Nat
  d1 : Nat
  label : String

private def packF32 (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

private unsafe def benchOne (ctx : Hesper.CUDAContext) (sh : Shape)
    (nIter nWarmup : Nat) : IO Float := do
  let { N, IC, IH, IW, KH, KW, s0, s1, p0, p1, d0, d1, .. } := sh
  let OH := (IH + 2*p0 - KH) / s0 + 1
  let OW := (IW + 2*p1 - KW) / s1 + 1
  let IC_KH_KW := IC * KH * KW
  let srcSize := N * IC * IH * IW
  let dstSize := N * OH * OW * IC_KH_KW

  let srcArr : Array Float :=
    (List.range srcSize).toArray.map (fun i => Float.sin (i.toFloat * 0.013) * 0.4)
  let srcBuf ← GPUBackend.allocBuffer ctx (srcSize * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (dstSize * 4).toUSize
  GPUBackend.writeBuffer ctx srcBuf (← packF32 srcArr)

  let nBlocksX := (IC_KH_KW + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("dst", dstBuf) ]
  let cfg : Hesper.ExecConfig :=
    { workgroupSize := { x := IC_KH_KW.min 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, OW, N * OH) }
  let kernel := im2colF32Kernel IC IW IH OW OH KW KH s0 s1 p0 p1 d0 d1 N
  -- Reuse same cache ref across all calls so the dispatch is built once
  -- and replayed (same path as the production decode hot loop).
  let cacheKey := hash ("bench-im2col", IC, IH, IW, KH, KW, s0, p0, N)
  let cacheRef ← IO.mkRef (none : Option (GPUBackend.CachedDispatch Hesper.CUDAContext))

  -- Warmup.
  for _ in [0:nWarmup] do
    GPUBackend.executeWithConfigCached ctx kernel bufs cfg cacheKey cacheRef
  -- Drain.
  let _ ← GPUBackend.readBuffer ctx dstBuf 4

  let t0 ← IO.monoNanosNow
  for _ in [0:nIter] do
    GPUBackend.executeWithConfigCached ctx kernel bufs cfg cacheKey cacheRef
  let _ ← GPUBackend.readBuffer ctx dstBuf 4
  let t1 ← IO.monoNanosNow

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx dstBuf

  let ns := (t1 - t0).toFloat
  return ns / 1000.0 / nIter.toFloat   -- µs per call

unsafe def main : IO Unit := do
  IO.println "═══ hesper im2colF32 benchmark ═══"
  let ctx ← Hesper.CUDAContext.init

  let shapes : List Shape := [
    { N := 2, IC := 3,   IH :=   8, IW :=   8, KH := 3,  KW := 3,  s0 := 1, s1 := 1, p0 := 1, p1 := 1, d0 := 1, d1 := 1, label := "parity test" },
    { N := 1, IC := 3,   IH := 224, IW := 224, KH := 16, KW := 16, s0 := 16, s1 := 16, p0 := 0, p1 := 0, d0 := 1, d1 := 1, label := "SigLIP-patch" },
    { N := 1, IC := 3,   IH := 224, IW := 224, KH := 7,  KW := 7,  s0 := 4, s1 := 4, p0 := 3, p1 := 3, d0 := 1, d1 := 1, label := "CLIP-stem" },
    { N := 1, IC := 64,  IH :=  56, IW :=  56, KH := 3,  KW := 3,  s0 := 1, s1 := 1, p0 := 1, p1 := 1, d0 := 1, d1 := 1, label := "mid-channel" },
    { N := 1, IC := 256, IH :=  28, IW :=  28, KH := 3,  KW := 3,  s0 := 1, s1 := 1, p0 := 1, p1 := 1, d0 := 1, d1 := 1, label := "deep-channel" },
  ]

  IO.println "shape                                                    hesper µs/call"
  IO.println "-----------------------------------------------------------------------"
  for sh in shapes do
    let t ← benchOne ctx sh 50 10
    let label := s!"N={sh.N} IC={sh.IC} IH={sh.IH} IW={sh.IW} KH={sh.KH} KW={sh.KW} s={sh.s0} p={sh.p0}"
    -- Pad label to 56 chars
    let padded := label ++ String.ofList (List.replicate (58 - label.length) ' ')
    IO.println s!"{padded}{t}"
