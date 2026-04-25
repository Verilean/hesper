import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.Layers.Linear
import Hesper.WGSL.Monad

/-!
# Multi-layer Q4_K dp4a equivalence test

Runs the *single-layer* `fusedQ4KMLinearDP4A4WarpKernel` N times (one per
layer) and the *multi-layer* `fusedQ4KMLinearDP4A4WarpMultiLayerKernel`
once.  Verifies bit-identical outputs.

This is the correctness gate for Phase 2b: if both produce the same output,
we can collapse N dispatches into 1 in the decode path.
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.Layers.Linear

open Hesper (ExecConfig)

private def randByte (seed : UInt64) : UInt8 × UInt64 :=
  let s := seed * 6364136223846793005 + 1442695040888963407
  ((s >>> 33).toUInt8, s)

private def randomBytes (n : Nat) (seedStart : UInt64) : ByteArray := Id.run do
  let mut seed := seedStart
  let mut out := ByteArray.empty
  for _ in [0:n] do
    let (b, s) := randByte seed
    seed := s
    out := out.push b
  return out

unsafe def main : IO Unit := do
  IO.println "═══ Multi-layer Q4_K dp4a equivalence test ═══"
  let ctx ← CUDAContext.init

  -- Small test: inDim=512, outDim=128, numLayers=4.
  let inDim  := 512
  let outDim := 128
  let numLayers := 4
  let cfg : Config := { inDim, outDim }
  let blocksPerRow := inDim / 256
  let weightBytes := outDim * blocksPerRow * 144
  let q8BlocksPerRow := inDim / 32
  let q8Bytes := q8BlocksPerRow * 36
  let outBytes := outDim * 4
  IO.println s!"[Setup] inDim={inDim} outDim={outDim} numLayers={numLayers}"
  IO.println s!"[Setup] weight={weightBytes}B q8={q8Bytes}B out={outBytes}B per layer"

  -- Allocate N layers of (weights, q8 input, output).
  let mut wBufs : List CUDABuffer := []
  let mut qBufs : List CUDABuffer := []
  let mut oBufsSingle : List CUDABuffer := []
  let mut oBufsMulti : List CUDABuffer := []
  for li in [0:numLayers] do
    let w ← createCUDABuffer weightBytes.toUSize
    let q ← createCUDABuffer q8Bytes.toUSize
    let o1 ← createCUDABuffer outBytes.toUSize
    let o2 ← createCUDABuffer outBytes.toUSize
    -- Fill with deterministic pseudo-random bytes; different per layer.
    writeCUDABuffer w (randomBytes weightBytes (1000 + li).toUInt64)
    writeCUDABuffer q (randomBytes q8Bytes (2000 + li).toUInt64)
    cuMemset o1.ptr outBytes.toUSize
    cuMemset o2.ptr outBytes.toUSize
    wBufs := wBufs ++ [w]
    qBufs := qBufs ++ [q]
    oBufsSingle := oBufsSingle ++ [o1]
    oBufsMulti := oBufsMulti ++ [o2]

  -- Path A: run single-layer kernel N times.
  IO.println "[Path A] Running single-layer kernel N times..."
  let singleCfg : ExecConfig := {
    funcName := "q4kSingle"
    workgroupSize := { x := 128 }
    numWorkgroups := (outDim, 1, 1)
  }
  let mut li := 0
  let layerTuples := List.zip (List.zip wBufs qBufs) oBufsSingle
  for ((w, q), o) in layerTuples do
    let refA : IO.Ref (Option (GPUBackend.CachedDispatch CUDAContext)) ← IO.mkRef none
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4A4WarpKernel cfg)
      [("weights", w), ("input_q8", q), ("output", o)]
      singleCfg
      (hash ("q4k-single", li))
      refA
    li := li + 1

  -- Path B: run multi-layer kernel once.
  IO.println "[Path B] Running multi-layer kernel once (grid.y = N)..."
  let multiCfg : ExecConfig := {
    funcName := "q4kMulti"
    workgroupSize := { x := 128 }
    numWorkgroups := (outDim, numLayers, 1)
  }
  let refB : IO.Ref (Option (GPUBackend.CachedDispatch CUDAContext)) ← IO.mkRef none
  GPUBackend.executeWithConfigCachedArrays ctx
    (fusedQ4KMLinearDP4A4WarpMultiLayerKernel cfg numLayers)
    []
    [("weights", wBufs), ("input_q8", qBufs), ("output", oBufsMulti)]
    multiCfg
    (hash "q4k-multi")
    refB

  -- Compare byte-wise per layer.
  IO.println "[Compare] Checking byte equality..."
  let mut allOk := true
  let mut layerIdx := 0
  for (o1, o2) in List.zip oBufsSingle oBufsMulti do
    let b1 ← readCUDABuffer o1 outBytes.toUSize
    let b2 ← readCUDABuffer o2 outBytes.toUSize
    if b1 == b2 then
      IO.println s!"  layer {layerIdx}: ✓"
    else
      let mut firstMismatch := 0
      for i in [0:outBytes] do
        if b1.get! i != b2.get! i then
          firstMismatch := i
          break
      IO.println s!"  layer {layerIdx}: ✗ first mismatch at byte {firstMismatch}"
      allOk := false
    layerIdx := layerIdx + 1
  if allOk then
    IO.println "✓ PASS — multi-layer kernel matches single-layer ×N"
  else
    IO.println "✗ FAIL — divergence between paths"
    IO.Process.exit 1
