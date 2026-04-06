import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Q6_K Quantization - GPU Dequantization Kernels

Implements on-the-fly dequantization of Q6_K (6-bit K-quantization) weights on GPU.

## Q6_K Block Format (256 elements, 210 bytes)

```
Offset  Size   Description
0       128    ql[128] - quants, lower 4 bits (2 values per byte: low + high nibble)
128     64     qh[64]  - quants, upper 2 bits (4 values per byte: 2 bits each)
192     16     scales[16] - signed 8-bit scales (1 per 16 elements)
208     2      d (FP16) - super-block scale
```

Total: 128 + 64 + 16 + 2 = 210 bytes per 256 elements

## Dequantization Formula

Blocks of 256 elements processed in 2 chunks of 128 elements.
Each chunk of 128 has 4 groups of 32 elements.

For element in group:
  q6 = (ql_nibble | (qh_2bits << 4)) - 32   (signed 6-bit: range [-32, 31])
  y = d * scales[group] * q6

## References
- llama.cpp: ggml/src/ggml-quants.c (dequantize_row_q6_K)
- GGUF spec: Q6_K = type ID 14
-/

namespace Hesper.Quantization.Q6_K

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Constants -/

def blockSize : Nat := 256
def blockSizeBytes : Nat := 210  -- 128 + 64 + 16 + 2

/-! ## Fused Q6_K Linear Kernel -/

/-- Fused Q6_K dequant + matrix-vector multiply kernel.
    One workgroup per output element. Reads packed Q6_K weights,
    dequantizes on-the-fly, and accumulates dot product.

    Block layout (210 bytes = 52.5 u32s, padded to 53):
    - u32[0..31]:  ql[128] = 32 u32s
    - u32[32..47]: qh[64] = 16 u32s
    - u32[48..51]: scales[16] = 4 u32s
    - u32[52]:     d (FP16 in lower 16 bits, upper 16 bits = next block's ql or padding)

    Actually, blocks are packed contiguously in bytes. So we read as u32 with byte offsets.

    @param inDim Input dimension
    @param outDim Output dimension
    @param workgroupSize Threads per workgroup
-/
def fusedQ6KLinearKernel (inDim outDim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / blockSize
  -- Total weight data as u32 array
  -- Each block = 210 bytes. Total bytes = outDim * blocksPerRow * 210
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt outIdx (Exp.litU32 outDim)) (do
    -- Row byte offset for this output element
    let rowByteBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes))

    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    -- Each thread processes blocks in a strided pattern
    ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
      let blockByteOffset := Exp.add rowByteBase (Exp.mul blockLocalIdx (Exp.litU32 blockSizeBytes))
      let elemBase := Exp.mul blockLocalIdx (Exp.litU32 blockSize)

      -- Read d (FP16) at byte offset 208 within block
      let dByteOffset := Exp.add blockByteOffset (Exp.litU32 208)
      let dU32Idx := Exp.div dByteOffset (Exp.litU32 4)
      let dByteInU32 := Exp.mul (Exp.mod dByteOffset (Exp.litU32 4)) (Exp.litU32 8)
      let dU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" dU32Idx
      let dBits := Exp.bitAnd (Exp.shiftRight dU32 dByteInU32) (Exp.litU32 0xFFFF)
      -- FP16 to F32 (arithmetic conversion)
      let sign := Exp.shiftRight dBits (Exp.litU32 15)
      let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
      let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
      let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
      let mantF := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
      let expF := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
      let d := Exp.select (Exp.eq exp5 (Exp.litU32 0)) (Exp.litF32 0.0) (Exp.mul signF (Exp.mul mantF expF))

      -- Process 2 chunks of 128 elements each (n = 0, 128)
      for chunk in [0:2] do
        let chunkOffset := chunk * 128
        -- ql base: blockByteOffset + chunk * 64 (64 bytes of ql per chunk)
        let qlByteBase := Exp.add blockByteOffset (Exp.litU32 (chunk * 64))
        -- qh base: blockByteOffset + 128 + chunk * 32
        let qhByteBase := Exp.add blockByteOffset (Exp.litU32 (128 + chunk * 32))
        -- scales base: blockByteOffset + 192 + chunk * 8
        let scByteBase := Exp.add blockByteOffset (Exp.litU32 (192 + chunk * 8))

        -- Process 4 groups of 32 elements
        for group in [0:4] do
          let groupOffset := group * 32
          -- Read scale for this group (signed 8-bit)
          let scByteIdx := Exp.add scByteBase (Exp.litU32 (group * 2))  -- 2 scales per group (is + 0, is + 2)
          let scU32Idx := Exp.div scByteIdx (Exp.litU32 4)
          let scByteInU32 := Exp.mul (Exp.mod scByteIdx (Exp.litU32 4)) (Exp.litU32 8)
          let scU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" scU32Idx
          let scByte := Exp.bitAnd (Exp.shiftRight scU32 scByteInU32) (Exp.litU32 0xFF)
          -- Sign extend 8-bit to i32: if bit 7 set, subtract 256
          let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
            (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
            (Exp.toF32 scByte)
          let dsc := Exp.mul d scSigned

          -- Process 32 elements in this group
          -- For groups 0,1: use low nibble of ql, qh bits 0-1 or 2-3
          -- For groups 2,3: use high nibble of ql, qh bits 4-5 or 6-7
          for l in [0:8] do  -- 8 u32s = 32 bytes for ql (but we read 4 bytes at a time)
            -- Each iteration processes 4 elements
            let qlByteIdx := Exp.add qlByteBase (Exp.litU32 (if group < 2 then l * 4 else l * 4))
            let qlOffset := if group < 2 then groupOffset else groupOffset - 64  -- ql reuses same bytes for groups 2,3
            let _ := qlOffset  -- suppress unused warning

            -- Simplified: process one element at a time for correctness
            for sub in [0:4] do
              let elemIdx := chunkOffset + groupOffset + l * 4 + sub
              if elemIdx < blockSize then
                -- Read ql byte
                let qlByte_offset := if group < 2
                  then Exp.add qlByteBase (Exp.litU32 (l * 4 + sub))
                  else Exp.add qlByteBase (Exp.litU32 (l * 4 + sub))  -- same ql bytes, different nibble
                let qlU32Idx := Exp.div qlByte_offset (Exp.litU32 4)
                let qlByteInU32 := Exp.mul (Exp.mod qlByte_offset (Exp.litU32 4)) (Exp.litU32 8)
                let qlU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" qlU32Idx
                let qlByte := Exp.bitAnd (Exp.shiftRight qlU32 qlByteInU32) (Exp.litU32 0xFF)
                let qlNibble := if group < 2
                  then Exp.bitAnd qlByte (Exp.litU32 0xF)        -- low nibble
                  else Exp.shiftRight qlByte (Exp.litU32 4)       -- high nibble

                -- Read qh 2 bits
                let qhByteIdx := Exp.add qhByteBase (Exp.litU32 (l * 4 + sub))
                let qhU32Idx := Exp.div qhByteIdx (Exp.litU32 4)
                let qhByteInU32 := Exp.mul (Exp.mod qhByteIdx (Exp.litU32 4)) (Exp.litU32 8)
                let qhU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" qhU32Idx
                let qhByte := Exp.bitAnd (Exp.shiftRight qhU32 qhByteInU32) (Exp.litU32 0xFF)
                let qhShift := group * 2  -- bits 0-1, 2-3, 4-5, 6-7 for groups 0,1,2,3
                let qhBits := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 qhShift)) (Exp.litU32 0x3)

                -- Combine: q6 = (qlNibble | (qhBits << 4)) - 32
                let q6 := Exp.sub (Exp.toF32 (Exp.bitOr qlNibble (Exp.shiftLeft qhBits (Exp.litU32 4)))) (Exp.litF32 32.0)

                -- Dequant: y = d * sc * q6
                let weight := Exp.mul dsc q6

                -- FMA with input
                let inputIdx := Exp.add elemBase (Exp.litU32 elemIdx)
                let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inputIdx
                ShaderM.assign "acc" (Exp.add acc (Exp.mul weight inVal))

    -- Tree reduction
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
    ShaderM.barrier
    let mut stride := workgroupSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2

    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
    ) (pure ())
  ) (pure ())

end Hesper.Quantization.Q6_K
