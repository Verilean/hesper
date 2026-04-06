import Hesper.LoRA.Types
import Hesper.LoRA.Init
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Training.SafeBuffer

/-!
# LoRA Weight Save/Load

Persists LoRA adapter weights to a simple binary format.

## File Format

```
Header (24 bytes):
  magic    : u32 = 0x4C4F5241 ("LORA")
  version  : u32 = 1
  rank     : u32
  alpha    : f32
  numLayers: u32
  reserved : u32 = 0

Per layer (2 * (A_size + B_size) bytes):
  Q_A data : f32[rank * dim]
  Q_B data : f32[dim * rank]
  V_A data : f32[rank * dim]
  V_B data : f32[kvDim * rank]
```
-/

namespace Hesper.LoRA.IO

open Hesper.WebGPU

private def magic : UInt32 := 0x4C4F5241  -- "LORA"
private def version : UInt32 := 1

/-- Write a UInt32 as 4 little-endian bytes -/
private def writeU32 (h : IO.FS.Handle) (v : UInt32) : IO Unit := do
  let bytes := ByteArray.empty
    |>.push v.toUInt8
    |>.push (v >>> 8).toUInt8
    |>.push (v >>> 16).toUInt8
    |>.push (v >>> 24).toUInt8
  h.write bytes

/-- Convert Float64 to Float32 IEEE 754 bits -/
private def float64ToFloat32Bits (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 := (bits64 >>> 63) &&& 1
  let exp64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 := bits64 &&& 0x000FFFFFFFFFFFFF
  -- Float64 bias=1023, Float32 bias=127
  if exp64 == 0 then (0 : UInt32)  -- zero/denorm → zero
  else if exp64 == 0x7FF then  -- inf/nan
    let sign32 := sign64.toUInt32 <<< 31
    let exp32 : UInt32 := (0xFF : UInt32) <<< 23
    let mant32 := (mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
    sign32 ||| exp32 ||| mant32
  else
    let exp32val : Int := exp64.toNat - 1023 + 127
    if exp32val <= 0 then (0 : UInt32)  -- underflow → zero
    else if exp32val >= 255 then  -- overflow → inf
      (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let sign32 := sign64.toUInt32 <<< 31
      let exp32 := exp32val.toNat.toUInt32 <<< 23
      let mant32 := (mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
      sign32 ||| exp32 ||| mant32

/-- Write a Float as 4 little-endian bytes (FP32) -/
private def writeF32 (h : IO.FS.Handle) (f : Float) : IO Unit := do
  let bits := float64ToFloat32Bits f
  let bytes := ByteArray.empty
    |>.push bits.toUInt8
    |>.push (bits >>> 8).toUInt8
    |>.push (bits >>> 16).toUInt8
    |>.push (bits >>> 24).toUInt8
  h.write bytes

/-- Read a UInt32 from 4 little-endian bytes (bounds-checked) -/
private def readU32 (bytes : ByteArray) (offset : Nat) : UInt32 :=
  Hesper.Training.SafeBuffer.readU32 bytes offset

/-- Read a Float from 4 little-endian bytes (bounds-checked) -/
private def readF32 (bytes : ByteArray) (offset : Nat) : Float :=
  Hesper.Training.SafeBuffer.readF32 bytes offset

/-- Save LoRA adapter weights to a binary file -/
def saveAdapter (device : Device) (adapter : Adapter) (path : String) : IO Unit := do
  IO.println s!"[LoRA] Saving adapter to {path}..."
  let h ← IO.FS.Handle.mk path .write

  -- Write header
  writeU32 h magic
  writeU32 h version
  writeU32 h adapter.config.rank.toUInt32
  writeF32 h adapter.config.alpha
  writeU32 h adapter.layers.size.toUInt32
  writeU32 h 0  -- reserved

  -- Write per-layer data
  for layer in adapter.layers do
    -- Read GPU buffers back to CPU and write
    let qASize := (layer.loraQ.rank * layer.loraQ.inDim * 4).toUSize
    let qBSize := (layer.loraQ.outDim * layer.loraQ.rank * 4).toUSize
    let vASize := (layer.loraV.rank * layer.loraV.inDim * 4).toUSize
    let vBSize := (layer.loraV.outDim * layer.loraV.rank * 4).toUSize

    let qAData ← mapBufferRead device layer.loraQ.a 0 qASize
    h.write qAData
    let qBData ← mapBufferRead device layer.loraQ.b 0 qBSize
    h.write qBData
    let vAData ← mapBufferRead device layer.loraV.a 0 vASize
    h.write vAData
    let vBData ← mapBufferRead device layer.loraV.b 0 vBSize
    h.write vBData

  IO.println s!"[LoRA] Adapter saved ({adapter.layers.size} layers, rank={adapter.config.rank})"

/-- Load LoRA adapter weights from a binary file -/
def loadAdapter (device : Device) (path : String) (dim kvDim : Nat) : IO Adapter := do
  IO.println s!"[LoRA] Loading adapter from {path}..."
  let bytes ← IO.FS.readBinFile path

  -- Parse header
  if bytes.size < 24 then
    throw (IO.userError "LoRA file too small for header")
  let fileMagic := readU32 bytes 0
  if fileMagic != magic then
    throw (IO.userError s!"Invalid LoRA file magic: 0x{String.ofList (Nat.toDigits 16 fileMagic.toNat)}")
  let fileVersion := readU32 bytes 4
  if fileVersion != version then
    throw (IO.userError s!"Unsupported LoRA file version: {fileVersion}")

  let rank := (readU32 bytes 8).toNat
  let alpha := readF32 bytes 12
  let numLayers := (readU32 bytes 16).toNat

  let config : Config := { rank, alpha }
  IO.println s!"[LoRA] Config: rank={rank}, alpha={alpha}, layers={numLayers}"

  -- Parse per-layer data
  let mut offset := 24
  let mut layers := #[]

  for _ in [:numLayers] do
    let qASize := rank * dim * 4
    let qBSize := dim * rank * 4
    let vASize := rank * dim * 4
    let vBSize := kvDim * rank * 4

    -- Create GPU buffers and upload data
    let mkBufWithData := fun (numBytes : Nat) => do
      let buf ← createBuffer device {
        size := numBytes.toUSize
        usage := [.storage, .copySrc, .copyDst]
        mappedAtCreation := false
      }
      let data := bytes.extract offset (offset + numBytes)
      writeBuffer device buf 0 data
      pure buf

    let qA ← mkBufWithData qASize
    offset := offset + qASize
    let qB ← mkBufWithData qBSize
    offset := offset + qBSize
    let vA ← mkBufWithData vASize
    offset := offset + vASize
    let vB ← mkBufWithData vBSize
    offset := offset + vBSize

    let loraQ : Weight := { a := qA, b := qB, inDim := dim, outDim := dim, rank }
    let loraV : Weight := { a := vA, b := vB, inDim := dim, outDim := kvDim, rank }
    layers := layers.push { loraQ, loraV }

  IO.println s!"[LoRA] Loaded {layers.size} layers"
  pure { config, layers }

end Hesper.LoRA.IO
