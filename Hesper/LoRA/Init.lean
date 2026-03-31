import Hesper.LoRA.Types
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# LoRA Weight Initialization

Creates and initializes LoRA adapter weights for BitNet finetuning.

## Initialization Strategy
- **A matrix**: Kaiming uniform initialization (preserves signal magnitude)
- **B matrix**: Zero initialization (LoRA output starts at zero, preserving base model behavior)

This ensures that at the start of training, the LoRA-augmented model
produces exactly the same output as the base model.
-/

namespace Hesper.LoRA

open Hesper.WebGPU
open Hesper.Logging

/-- Simple pseudo-random number generator (xoshiro128+) for weight initialization.
    Deterministic given a seed, which is important for reproducibility. -/
structure RNG where
  s0 : UInt64
  s1 : UInt64

namespace RNG

def create (seed : UInt64) : RNG :=
  -- SplitMix64 to generate two state words from a single seed
  let z1 := seed + 0x9e3779b97f4a7c15
  let z1 := (z1 ^^^ (z1 >>> 30)) * 0xbf58476d1ce4e5b9
  let z1 := (z1 ^^^ (z1 >>> 27)) * 0x94d049bb133111eb
  let z1 := z1 ^^^ (z1 >>> 31)
  let z2 := z1 + 0x9e3779b97f4a7c15
  let z2 := (z2 ^^^ (z2 >>> 30)) * 0xbf58476d1ce4e5b9
  let z2 := (z2 ^^^ (z2 >>> 27)) * 0x94d049bb133111eb
  let z2 := z2 ^^^ (z2 >>> 31)
  { s0 := z1, s1 := z2 }

/-- Generate next random UInt64 and advance state -/
def next (rng : RNG) : UInt64 × RNG :=
  let result := rng.s0 + rng.s1
  let s1 := rng.s0 ^^^ rng.s1
  let s0 := ((rng.s0 <<< 24) ||| (rng.s0 >>> 40)) ^^^ s1 ^^^ (s1 <<< 16)
  let s1 := (s1 <<< 37) ||| (s1 >>> 27)
  (result, { s0, s1 })

/-- Generate a Float in [0, 1) -/
def nextFloat (rng : RNG) : Float × RNG :=
  let (bits, rng') := rng.next
  let f := (bits >>> 11).toFloat / (1 <<< 53).toFloat
  (f, rng')

/-- Generate a Float in [-bound, bound) using uniform distribution.
    Used for Kaiming uniform initialization. -/
def nextUniform (rng : RNG) (bound : Float) : Float × RNG :=
  let (f, rng') := rng.nextFloat
  (f * 2.0 * bound - bound, rng')

end RNG

/-- Generate Kaiming uniform initialization values.
    bound = sqrt(3 / fanIn) where fanIn = inDim for the A matrix.
    This preserves the variance of activations through the network. -/
def kaimingUniformBound (fanIn : Nat) : Float :=
  Float.sqrt (3.0 / fanIn.toFloat)

/-- Convert Float64 to Float32 IEEE 754 bits -/
private def float64ToFloat32Bits (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 := (bits64 >>> 63) &&& 1
  let exp64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 := bits64 &&& 0x000FFFFFFFFFFFFF
  if exp64 == 0 then (0 : UInt32)
  else if exp64 == 0x7FF then
    (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
  else
    let exp32val : Int := exp64.toNat - 1023 + 127
    if exp32val <= 0 then (0 : UInt32)
    else if exp32val >= 255 then (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (sign64.toUInt32 <<< 31) ||| (exp32val.toNat.toUInt32 <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

/-- Convert a Float to 4 little-endian bytes (FP32) -/
private def floatToF32Bytes (f : Float) : ByteArray :=
  let bits := float64ToFloat32Bits f
  ByteArray.mk #[bits.toUInt8, (bits >>> 8).toUInt8, (bits >>> 16).toUInt8, (bits >>> 24).toUInt8]

/-- Create a ByteArray of FP32 values with Kaiming uniform initialization -/
def generateKaimingWeights (numElements : Nat) (fanIn : Nat) (seed : UInt64) : ByteArray :=
  let bound := kaimingUniformBound fanIn
  let (bytes, _) := Id.run do
    let mut rng := RNG.create seed
    let mut bytes := ByteArray.empty
    for _ in [:numElements] do
      let (val, rng') := rng.nextUniform bound
      rng := rng'
      bytes := bytes ++ floatToF32Bytes val
    pure (bytes, rng)
  bytes

/-- Create a ByteArray of zeros (numElements FP32 values) -/
def generateZeroWeights (numElements : Nat) : ByteArray :=
  ByteArray.mk (Array.replicate (numElements * 4) 0)

/-- Create a single LoRA weight pair for one projection.
    A is Kaiming initialized, B is zero initialized. -/
def createWeight (device : Device) (inDim outDim rank : Nat) (seed : UInt64) : IO Weight := do
  logVerbose s!"[LoRA] Creating weight: inDim={inDim}, outDim={outDim}, rank={rank}"

  -- A: [rank, inDim] FP32
  let aSize := (rank * inDim * 4).toUSize
  let aBuf ← createBuffer device { size := aSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let aData := generateKaimingWeights (rank * inDim) inDim seed
  writeBuffer device aBuf 0 aData

  -- B: [outDim, rank] FP32, zero initialized
  let bSize := (outDim * rank * 4).toUSize
  let bBuf ← createBuffer device { size := bSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let bData := generateZeroWeights (outDim * rank)
  writeBuffer device bBuf 0 bData

  pure { a := aBuf, b := bBuf, inDim, outDim, rank }

/-- Create gradient buffers for a single LoRA weight pair (initialized to zero) -/
def createWeightGrad (device : Device) (weight : Weight) : IO WeightGrad := do
  let mkZeroBuf := fun (numElements : Nat) => do
    let size := (numElements * 4).toUSize
    let buf ← createBuffer device { size := size, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
    writeBuffer device buf 0 (generateZeroWeights numElements)
    pure buf
  pure {
    dA := ← mkZeroBuf (weight.rank * weight.inDim)
    dB := ← mkZeroBuf (weight.outDim * weight.rank)
  }

/-- Create Adam optimizer state for a single LoRA weight pair (initialized to zero) -/
def createAdamState (device : Device) (weight : Weight) : IO AdamState := do
  let mkZeroBuf := fun (numElements : Nat) => do
    let size := (numElements * 4).toUSize
    let buf ← createBuffer device { size := size, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
    writeBuffer device buf 0 (generateZeroWeights numElements)
    pure buf
  pure {
    mA := ← mkZeroBuf (weight.rank * weight.inDim)
    vA := ← mkZeroBuf (weight.rank * weight.inDim)
    mB := ← mkZeroBuf (weight.outDim * weight.rank)
    vB := ← mkZeroBuf (weight.outDim * weight.rank)
  }

/-- Create a full LoRA adapter for a BitNet model.
    Applies LoRA to Q and V attention projections in all transformer layers.

    @param device GPU device
    @param config LoRA configuration
    @param numLayers Number of transformer layers (e.g., 30 for BitNet-2B)
    @param dim Model hidden dimension (e.g., 2560 for BitNet-2B)
    @param kvDim KV dimension for V projection (e.g., 640 for BitNet-2B with GQA 4:1)
    @param seed Random seed for weight initialization -/
def createAdapter (device : Device) (config : Config) (numLayers : Nat)
    (dim : Nat) (kvDim : Nat) (seed : UInt64 := 42) : IO Adapter := do
  IO.println s!"[LoRA] Creating adapter: rank={config.rank}, alpha={config.alpha}, layers={numLayers}"
  IO.println s!"[LoRA] Target modules: {config.targetModules}"

  let mut layers := #[]
  for i in [:numLayers] do
    -- Q projection: [dim, dim] → LoRA A: [rank, dim], B: [dim, rank]
    let loraQ ← createWeight device dim dim config.rank (seed + i.toUInt64 * 2)
    -- V projection: [dim, kvDim] → LoRA A: [rank, dim], B: [kvDim, rank]
    let loraV ← createWeight device dim kvDim config.rank (seed + i.toUInt64 * 2 + 1)
    layers := layers.push { loraQ, loraV }

  let totalParams := numLayers * 2 * config.rank * (dim + dim) +
                     numLayers * (config.rank * dim + config.rank * kvDim)
  IO.println s!"[LoRA] Total trainable parameters: {totalParams} ({totalParams * 4 / 1024} KB)"

  pure { config, layers }

/-- Create gradient buffers for the full adapter -/
def createAdapterGrad (device : Device) (adapter : Adapter) : IO AdapterGrad := do
  let mut layers := #[]
  for layer in adapter.layers do
    let gradQ ← createWeightGrad device layer.loraQ
    let gradV ← createWeightGrad device layer.loraV
    layers := layers.push { gradQ, gradV }
  pure { layers }

/-- Create Adam optimizer state for the full adapter -/
def createAdapterAdamState (device : Device) (adapter : Adapter) : IO AdapterAdamState := do
  let mut layers := #[]
  for layer in adapter.layers do
    let stateQ ← createAdamState device layer.loraQ
    let stateV ← createAdamState device layer.loraV
    layers := layers.push { stateQ, stateV }
  pure { layers, step := 0 }

/-- Create saved activation buffers for backward pass -/
def createSavedActivations (device : Device) (adapter : Adapter) (dim kvDim : Nat) : IO SavedActivations := do
  let mkBuf := fun (numElements : Nat) => do
    createBuffer device {
      size := (numElements * 4).toUSize
      usage := [.storage, .copySrc, .copyDst]
      mappedAtCreation := false
    }
  let mut layers := #[]
  for layer in adapter.layers do
    -- inputToQ: [dim], hQ: [rank], inputToV: [dim], hV: [rank]
    let inputToQ ← mkBuf dim
    let hQ ← mkBuf layer.loraQ.rank
    let inputToV ← mkBuf dim
    let hV ← mkBuf layer.loraV.rank
    layers := layers.push (inputToQ, hQ, inputToV, hV)
  pure { layers }

end Hesper.LoRA
