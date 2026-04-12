import Hesper.Backend.CUDA
import Hesper.Backend.WebGPU
import Hesper.Layers.BitLinear
import Hesper

/-!
# BitLinear WebGPU vs CUDA comparison

Runs the same BitLinear forward on both backends to find divergence.
-/

open Hesper
open Hesper.Layers.BitLinear

private def packF32 (v : Float) : ByteArray :=
  let b := v.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  let bits : UInt32 :=
    if e == 0 then 0
    else let e32 : Int := e.toNat - 1023 + 127
      if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else (s.toUInt32 <<< 31) ||| (e32.toNat.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
  ByteArray.mk #[bits.toUInt8, (bits>>>8).toUInt8, (bits>>>16).toUInt8, (bits>>>24).toUInt8]

private def unpackF32 (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -fv else fv

def runBitLinearTest [GPUBackend β] (ctx : β) (name : String)
    (config : Config) (packedWeights : ByteArray) (scale : Float)
    (inputData : ByteArray) : IO (Array Float) := do
  let layer ← create ctx config packedWeights scale
  let inBuf ← GPUBackend.allocBuffer ctx inputData.size.toUSize
  GPUBackend.writeBuffer ctx inBuf inputData
  let outBuf ← GPUBackend.allocBuffer ctx (config.outDim * 4).toUSize
  forward ctx layer inBuf outBuf
  let outBytes ← GPUBackend.readBuffer ctx outBuf (config.outDim * 4).toUSize
  let mut results : Array Float := #[]
  for i in List.range config.outDim do
    results := results.push (unpackF32 outBytes i)
  IO.println s!"  {name} output[0..4]: {results.extract 0 (min 5 results.size)}"
  return results

def main : IO Unit := do
  IO.println "═══ BitLinear WebGPU vs CUDA Comparison ═══"

  -- Small config: 32 → 8
  -- inDim must be multiple of 128 (i2_s group layout)
  let config : Config := { inDim := 128, outDim := 1, batchSize := 1 }

  -- Create test data: all-ones input, specific packed weights
  -- Packed weights: each u32 = 4 bytes = 16 ternary values
  -- Code: 00=−1, 01=0, 10=+1, 11 unused (maps to code-1=2→used as +2, but shouldn't happen)
  -- For simplicity: all weights = +1 → code = 10 = 0x2
  -- 4 codes per byte: 0x2 repeated → 10101010 = 0xAA per byte
  let numPackedBytes := config.outDim * config.inDim / 4  -- 32*8/4 = 64 bytes
  let packedWeights := ByteArray.mk (Array.replicate numPackedBytes 0xAA)
  let scale : Float := 1.0

  -- Input: [1.0, 1.0, ..., 1.0]
  let mut inputData := ByteArray.empty
  for _ in List.range config.inDim do
    inputData := inputData ++ packF32 1.0

  -- WebGPU
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let webgpuResults ← runBitLinearTest device "WebGPU" config packedWeights scale inputData

  -- CUDA
  let cudaCtx ← CUDAContext.init
  let cudaResults ← runBitLinearTest cudaCtx "CUDA" config packedWeights scale inputData

  -- Compare
  IO.println "\nComparison:"
  let mut maxDiff : Float := 0.0
  for i in List.range config.outDim do
    let w := webgpuResults.getD i 0.0
    let c := cudaResults.getD i 0.0
    let diff := (w - c).abs
    if diff > maxDiff then maxDiff := diff
    IO.println s!"  [{i}] WebGPU={w}, CUDA={c}, diff={diff}"

  IO.println s!"\nMax diff: {maxDiff}"
  if maxDiff < 0.01 then
    IO.println "✓ BitLinear outputs match!"
  else
    IO.println "✗ BitLinear outputs DIFFER"
