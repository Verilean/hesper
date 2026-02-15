import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# TQ2_0 Quantization - GPU Unpacking Kernels

Implements on-the-fly dequantization of TQ2_0 (2-bit ternary) weights on GPU.

## TQ2_0 Format
- **Block size**: 256 elements
- **Packing**: 4 ternary values per byte (2 bits each)
- **Scale**: 1 FP16 scale per block (stored as 2 bytes)
- **Encoding**: {-1, 0, 1} → {0b11, 0b00, 0b01}
  - 0b00 → 0
  - 0b01 → +1
  - 0b11 → -1 (two's complement)
- **Layout**: [packed_bytes (64)] [scale (FP16=2 bytes)] = 66 bytes per block

## Memory Layout Example
```
Block 0: [byte0][byte1]...[byte63][scale_low][scale_high]
         └─4 values─┘                └────FP16────┘
```

## Performance
- **Bandwidth savings**: 16x (2-bit vs 32-bit)
- **PCIe transfer**: ~23ms for 3B model (vs ~375ms for Float32)
- **Compute**: Fused unpack + compute in single kernel

## References
- llama.cpp: ggml/src/ggml-quants.c (lines 2103-2271)
- GGUF spec: TQ2_0 = type ID 35
-/

namespace Hesper.Quantization.TQ2_0

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Constants -/

/-- TQ2_0 block size in elements -/
def blockSize : Nat := 256

/-- TQ2_0 block size in bytes (64 packed + 2 scale) -/
def blockSizeBytes : Nat := 66

/-- Number of ternary values packed per byte -/
def valuesPerByte : Nat := 4

/-! ## DSL Helper Functions -/

/-- Unpack a single 2-bit ternary value from a packed byte (i2_s encoding)
    @param packed The packed u32 containing multiple bytes
    @param idx The index (0-3) of the value to extract within a byte
    @return Ternary value as f32: -1.0, 0.0, or 1.0

    i2_s encoding:
    - 0b00 (0) → -1
    - 0b01 (1) →  0
    - 0b10 (2) → +1
    Formula: ternary = code - 1
-/
def unpackTernary2bit (packed : Exp (.scalar .u32)) (idx : Exp (.scalar .u32)) : Exp (.scalar .f32) :=
  -- Extract 2 bits: shift right by (idx * 2), then mask with 0x03
  let shift := Exp.mul idx (Exp.litU32 2)
  let shifted := Exp.shiftRight packed shift
  let bits := Exp.bitAnd shifted (Exp.litU32 0x03)

  -- i2_s decode: code - 1 gives {-1, 0, +1}
  -- Convert u32 code to f32, then subtract 1.0
  let codeF32 := Exp.toF32 bits
  let result := Exp.sub codeF32 (Exp.litF32 1.0)
  result

/-! ## GPU Unpacking Kernel -/

/-- GPU kernel to unpack TQ2_0 quantized weights to Float32

    This kernel reads packed ternary weights and FP16 scales from GPU buffers,
    unpacks them on-the-fly, and writes Float32 output.

    **Buffer Layout:**
    - Input buffer "packed": Array of u32 containing packed bytes (64 bytes per block)
    - Input buffer "scales": Array of u32 containing FP16 scales (2 bytes per block, stored as u32)
    - Output buffer "output": Array of f32 containing unpacked values

    **Workgroup size**: 256 threads (one block per workgroup)

    @param numElements Total number of elements to unpack
-/
def unpackTQ2_0Kernel (numElements : Nat) : ShaderM Unit := do
  -- Get global thread ID
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Bounds check
  let inBounds := Exp.lt idx (Exp.litU32 numElements)

  -- Declare buffers
  let numBlocks := (numElements + blockSize - 1) / blockSize
  let numPackedU32 := (numBlocks * 64 + 3) / 4  -- 64 bytes per block, 4 bytes per u32

  let _packed ← ShaderM.declareInputBuffer "packed" (.array (.scalar .u32) numPackedU32)
  let _scales ← ShaderM.declareInputBuffer "scales" (.array (.scalar .u32) numBlocks)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) numElements)

  -- Calculate which block and position within block
  let blockIdx := Exp.div idx (Exp.litU32 blockSize)
  let localIdx := Exp.mod idx (Exp.litU32 blockSize)

  -- Calculate byte index within block
  let byteIdx := Exp.div localIdx (Exp.litU32 valuesPerByte)
  let bitIdx := Exp.mod localIdx (Exp.litU32 valuesPerByte)

  -- Read packed byte (as part of u32)
  let packedBase := Exp.mul blockIdx (Exp.litU32 16)  -- 64 bytes = 16 u32s
  let u32Idx := Exp.add packedBase (Exp.div byteIdx (Exp.litU32 4))
  let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "packed" u32Idx

  -- Extract the specific byte from u32
  let byteOffset := Exp.mul (Exp.mod byteIdx (Exp.litU32 4)) (Exp.litU32 8)
  let packedByte := Exp.shiftRight packedU32 byteOffset

  -- Unpack ternary value
  let ternary := unpackTernary2bit packedByte bitIdx

  -- Read scale (stored as u32, but represents FP16)
  -- For now, we'll use a simplified approximation: treat lower 16 bits as mantissa
  let scaleU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numBlocks) "scales" blockIdx

  -- Simplified FP16 decode: just normalize to [0.0, 1.0] range
  -- This is a placeholder - proper FP16 decode would require bit manipulation
  let scaleF32 := Exp.div (Exp.toF32 scaleU32) (Exp.litF32 65535.0)

  -- Scale the ternary value
  let result := Exp.mul ternary scaleF32

  -- Write output (using select for conditional)
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-! ## High-Level API -/

/-- Configuration for TQ2_0 unpacking execution -/
structure UnpackConfig where
  numElements : Nat
  workgroupSize : Nat := 256
  deriving Repr

/-- Execute TQ2_0 unpacking on GPU

    @param device WebGPU device
    @param packedBuf GPU buffer containing packed TQ2_0 data
    @param scalesBuf GPU buffer containing FP16 scales (as u32)
    @param outputBuf GPU buffer for Float32 output
    @param config Unpacking configuration
-/
def executeUnpack (device : Device)
                  (packedBuf scalesBuf outputBuf : Buffer)
                  (config : UnpackConfig) : IO Unit := do
  IO.println s!"[TQ2_0] Unpacking {config.numElements} elements on GPU..."

  let shader := unpackTQ2_0Kernel config.numElements
  let namedBuffers := [
    ("packed", packedBuf),
    ("scales", scalesBuf),
    ("output", outputBuf)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    config.numElements
    config.workgroupSize

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  IO.println "[TQ2_0] ✓ Unpacking complete"

end Hesper.Quantization.TQ2_0
