import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Basic

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

    Supports 2D workgroup grids (wid.x + wid.y * gridX) so that outDim can
    exceed WebGPU's per-dimension limit of 65535 (e.g. vocabSize=262144 for
    the Gemma 4 LM head). Pass `gridX=0` to use a 1D grid (outIdx = wid.x).
-/
def fusedQ6KLinearKernel (inDim outDim : Nat) (workgroupSize : Nat := 256)
    (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / blockSize
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt outIdx (Exp.litU32 outDim)) (do
    let rowByteBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes))

    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    -- Helper to read a byte at a (compile-time-relative) offset within the block
    let readByte (blockBase : Exp (.scalar .u32)) (offset : Nat) : ShaderM (Exp (.scalar .u32)) := do
      let byteIdx := Exp.add blockBase (Exp.litU32 offset)
      let u32Idx := Exp.div byteIdx (Exp.litU32 4)
      let byteShift := Exp.mul (Exp.mod byteIdx (Exp.litU32 4)) (Exp.litU32 8)
      let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
      pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

    -- Each thread processes blocks in a strided pattern
    ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
      let blockByteBase := Exp.add rowByteBase (Exp.mul blockLocalIdx (Exp.litU32 blockSizeBytes))
      let elemBase := Exp.mul blockLocalIdx (Exp.litU32 blockSize)

      -- Read d (FP16) at byte offset 208
      let dLoByte ← readByte blockByteBase 208
      let dHiByte ← readByte blockByteBase 209
      let dBits := Exp.bitOr dLoByte (Exp.shiftLeft dHiByte (Exp.litU32 8))
      -- FP16 → F32 (with subnormal support)
      let sign := Exp.shiftRight dBits (Exp.litU32 15)
      let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
      let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
      let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
      let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
      let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
      let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
      let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
      let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
      let expFSubnormal := Exp.litF32 6.103515625e-5  -- 2^(-14)
      let expF := Exp.select isSubnormal expFSubnormal expFNormal
      let d := Exp.mul signF (Exp.mul mantF expF)

      -- Helper: read sign-extended int8 scale at byte offset within block
      let readScaleI8 (offset : Nat) : ShaderM (Exp (.scalar .f32)) := do
        let scByte ← readByte blockByteBase offset
        let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
          (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
          (Exp.toF32 scByte)
        pure scSigned

      -- Faithful translation of dequantize_row_q6_K from llama.cpp:
      --   for n in [0, 128]: chunk
      --     for l in 0..31:
      --       is = l/16
      --       q1 = ((ql[l]      & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
      --       q2 = ((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
      --       q3 = ((ql[l]      >> 4)  | ((qh[l] >> 4) & 3) << 4) - 32
      --       q4 = ((ql[l + 32] >> 4)  | ((qh[l] >> 6) & 3) << 4) - 32
      --       y[l +  0] = d * sc[is+0] * q1
      --       y[l + 32] = d * sc[is+2] * q2
      --       y[l + 64] = d * sc[is+4] * q3
      --       y[l + 96] = d * sc[is+6] * q4
      --     ql += 64; qh += 32; sc += 8; y += 128
      for chunk in [0:2] do
        -- Per-chunk byte offsets within the block:
        let qlBaseOff := chunk * 64        -- ql offset: 0 or 64
        let qhBaseOff := 128 + chunk * 32  -- qh offset: 128 or 160
        let scBaseOff := 192 + chunk * 8   -- sc offset: 192 or 200
        let chunkOutBase := chunk * 128
        for l in [0:32] do
          let isIdx := l / 16  -- 0 or 1
          -- Read scales (8 of them per chunk: indices 0..7)
          -- but we only need is+0, is+2, is+4, is+6 for this l
          let sc0F ← readScaleI8 (scBaseOff + isIdx)
          let sc2F ← readScaleI8 (scBaseOff + isIdx + 2)
          let sc4F ← readScaleI8 (scBaseOff + isIdx + 4)
          let sc6F ← readScaleI8 (scBaseOff + isIdx + 6)
          let dsc0 := Exp.mul d sc0F
          let dsc2 := Exp.mul d sc2F
          let dsc4 := Exp.mul d sc4F
          let dsc6 := Exp.mul d sc6F

          -- Read ql and qh bytes
          let qlByte0 ← readByte blockByteBase (qlBaseOff + l)        -- ql[l]
          let qlByte32 ← readByte blockByteBase (qlBaseOff + l + 32)  -- ql[l + 32]
          let qhByte ← readByte blockByteBase (qhBaseOff + l)         -- qh[l]

          let qlLow0 := Exp.bitAnd qlByte0 (Exp.litU32 0xF)
          let qlLow32 := Exp.bitAnd qlByte32 (Exp.litU32 0xF)
          let qlHigh0 := Exp.shiftRight qlByte0 (Exp.litU32 4)
          let qlHigh32 := Exp.shiftRight qlByte32 (Exp.litU32 4)

          let qhBits0 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 0)) (Exp.litU32 3)
          let qhBits2 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 2)) (Exp.litU32 3)
          let qhBits4 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 4)) (Exp.litU32 3)
          let qhBits6 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 6)) (Exp.litU32 3)

          -- Compute q1..q4 (signed 6-bit, range [-32, 31])
          let q1Raw := Exp.bitOr qlLow0  (Exp.shiftLeft qhBits0 (Exp.litU32 4))
          let q2Raw := Exp.bitOr qlLow32 (Exp.shiftLeft qhBits2 (Exp.litU32 4))
          let q3Raw := Exp.bitOr qlHigh0  (Exp.shiftLeft qhBits4 (Exp.litU32 4))
          let q4Raw := Exp.bitOr qlHigh32 (Exp.shiftLeft qhBits6 (Exp.litU32 4))
          let q1 := Exp.sub (Exp.toF32 q1Raw) (Exp.litF32 32.0)
          let q2 := Exp.sub (Exp.toF32 q2Raw) (Exp.litF32 32.0)
          let q3 := Exp.sub (Exp.toF32 q3Raw) (Exp.litF32 32.0)
          let q4 := Exp.sub (Exp.toF32 q4Raw) (Exp.litF32 32.0)

          let w1 := Exp.mul dsc0 q1
          let w2 := Exp.mul dsc2 q2
          let w3 := Exp.mul dsc4 q3
          let w4 := Exp.mul dsc6 q4

          -- Output positions in the block: chunkOutBase + (l, l+32, l+64, l+96)
          let inIdx1 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l))
          let inIdx2 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 32))
          let inIdx3 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 64))
          let inIdx4 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 96))

          let in1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx1
          let in2 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx2
          let in3 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx3
          let in4 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx4

          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.add (Exp.mul w1 in1) (Exp.mul w2 in2))
                                                      (Exp.add (Exp.mul w3 in3) (Exp.mul w4 in4))))

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


/-- Batched Q6_K matmul: [M,inDim] x dequant(Q6_K weights) -> [M,outDim]. Dispatch (outDim,M,1). -/
def fusedQ6KBatchKernel (inDim outDim M : Nat) (workgroupSize : Nat := 256)
    (rowOffset : Nat := 0) (weightRows : Nat := 0) : ShaderM Unit := do
  -- `rowOffset`/`weightRows` enable tiling a wide weight (e.g. 262144-vocab lm_head)
  -- past the 65535-workgroup limit: bind the FULL weight (weightRows rows), dispatch
  -- `outDim` (≤65535) rows per chunk, the kernel reads weight row (outIdx+rowOffset)
  -- and writes chunk-local output[row*outDim + outIdx].  Default (0,0) = old behavior.
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let inRowBase := Exp.mul row (Exp.litU32 inDim)

  let blocksPerRow := inDim / blockSize
  let wRows := if weightRows == 0 then outDim else weightRows
  let totalWeightBytes := wRows * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (M * inDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (M * outDim))

  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt outIdx (Exp.litU32 outDim)) (do
    let weightRow := Exp.add outIdx (Exp.litU32 rowOffset)
    let rowByteBase := Exp.mul weightRow (Exp.litU32 (blocksPerRow * blockSizeBytes))

    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    -- Helper to read a byte at a (compile-time-relative) offset within the block
    let readByte (blockBase : Exp (.scalar .u32)) (offset : Nat) : ShaderM (Exp (.scalar .u32)) := do
      let byteIdx := Exp.add blockBase (Exp.litU32 offset)
      let u32Idx := Exp.div byteIdx (Exp.litU32 4)
      let byteShift := Exp.mul (Exp.mod byteIdx (Exp.litU32 4)) (Exp.litU32 8)
      let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
      pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

    -- Each thread processes blocks in a strided pattern
    ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
      let blockByteBase := Exp.add rowByteBase (Exp.mul blockLocalIdx (Exp.litU32 blockSizeBytes))
      let elemBase := Exp.mul blockLocalIdx (Exp.litU32 blockSize)

      -- Read d (FP16) at byte offset 208
      let dLoByte ← readByte blockByteBase 208
      let dHiByte ← readByte blockByteBase 209
      let dBits := Exp.bitOr dLoByte (Exp.shiftLeft dHiByte (Exp.litU32 8))
      -- FP16 → F32 (with subnormal support)
      let sign := Exp.shiftRight dBits (Exp.litU32 15)
      let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
      let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
      let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
      let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
      let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
      let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
      let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
      let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
      let expFSubnormal := Exp.litF32 6.103515625e-5  -- 2^(-14)
      let expF := Exp.select isSubnormal expFSubnormal expFNormal
      let d := Exp.mul signF (Exp.mul mantF expF)

      -- Helper: read sign-extended int8 scale at byte offset within block
      let readScaleI8 (offset : Nat) : ShaderM (Exp (.scalar .f32)) := do
        let scByte ← readByte blockByteBase offset
        let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
          (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
          (Exp.toF32 scByte)
        pure scSigned

      -- Faithful translation of dequantize_row_q6_K from llama.cpp:
      --   for n in [0, 128]: chunk
      --     for l in 0..31:
      --       is = l/16
      --       q1 = ((ql[l]      & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
      --       q2 = ((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
      --       q3 = ((ql[l]      >> 4)  | ((qh[l] >> 4) & 3) << 4) - 32
      --       q4 = ((ql[l + 32] >> 4)  | ((qh[l] >> 6) & 3) << 4) - 32
      --       y[l +  0] = d * sc[is+0] * q1
      --       y[l + 32] = d * sc[is+2] * q2
      --       y[l + 64] = d * sc[is+4] * q3
      --       y[l + 96] = d * sc[is+6] * q4
      --     ql += 64; qh += 32; sc += 8; y += 128
      for chunk in [0:2] do
        -- Per-chunk byte offsets within the block:
        let qlBaseOff := chunk * 64        -- ql offset: 0 or 64
        let qhBaseOff := 128 + chunk * 32  -- qh offset: 128 or 160
        let scBaseOff := 192 + chunk * 8   -- sc offset: 192 or 200
        let chunkOutBase := chunk * 128
        for l in [0:32] do
          let isIdx := l / 16  -- 0 or 1
          -- Read scales (8 of them per chunk: indices 0..7)
          -- but we only need is+0, is+2, is+4, is+6 for this l
          let sc0F ← readScaleI8 (scBaseOff + isIdx)
          let sc2F ← readScaleI8 (scBaseOff + isIdx + 2)
          let sc4F ← readScaleI8 (scBaseOff + isIdx + 4)
          let sc6F ← readScaleI8 (scBaseOff + isIdx + 6)
          let dsc0 := Exp.mul d sc0F
          let dsc2 := Exp.mul d sc2F
          let dsc4 := Exp.mul d sc4F
          let dsc6 := Exp.mul d sc6F

          -- Read ql and qh bytes
          let qlByte0 ← readByte blockByteBase (qlBaseOff + l)        -- ql[l]
          let qlByte32 ← readByte blockByteBase (qlBaseOff + l + 32)  -- ql[l + 32]
          let qhByte ← readByte blockByteBase (qhBaseOff + l)         -- qh[l]

          let qlLow0 := Exp.bitAnd qlByte0 (Exp.litU32 0xF)
          let qlLow32 := Exp.bitAnd qlByte32 (Exp.litU32 0xF)
          let qlHigh0 := Exp.shiftRight qlByte0 (Exp.litU32 4)
          let qlHigh32 := Exp.shiftRight qlByte32 (Exp.litU32 4)

          let qhBits0 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 0)) (Exp.litU32 3)
          let qhBits2 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 2)) (Exp.litU32 3)
          let qhBits4 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 4)) (Exp.litU32 3)
          let qhBits6 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 6)) (Exp.litU32 3)

          -- Compute q1..q4 (signed 6-bit, range [-32, 31])
          let q1Raw := Exp.bitOr qlLow0  (Exp.shiftLeft qhBits0 (Exp.litU32 4))
          let q2Raw := Exp.bitOr qlLow32 (Exp.shiftLeft qhBits2 (Exp.litU32 4))
          let q3Raw := Exp.bitOr qlHigh0  (Exp.shiftLeft qhBits4 (Exp.litU32 4))
          let q4Raw := Exp.bitOr qlHigh32 (Exp.shiftLeft qhBits6 (Exp.litU32 4))
          let q1 := Exp.sub (Exp.toF32 q1Raw) (Exp.litF32 32.0)
          let q2 := Exp.sub (Exp.toF32 q2Raw) (Exp.litF32 32.0)
          let q3 := Exp.sub (Exp.toF32 q3Raw) (Exp.litF32 32.0)
          let q4 := Exp.sub (Exp.toF32 q4Raw) (Exp.litF32 32.0)

          let w1 := Exp.mul dsc0 q1
          let w2 := Exp.mul dsc2 q2
          let w3 := Exp.mul dsc4 q3
          let w4 := Exp.mul dsc6 q4

          -- Output positions in the block: chunkOutBase + (l, l+32, l+64, l+96)
          let inIdx1 := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (chunkOutBase + l)))
          let inIdx2 := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 32)))
          let inIdx3 := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 64)))
          let inIdx4 := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 96)))

          let in1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * inDim) "input" inIdx1
          let in2 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * inDim) "input" inIdx2
          let in3 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * inDim) "input" inIdx3
          let in4 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * inDim) "input" inIdx4

          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.add (Exp.mul w1 in1) (Exp.mul w2 in2))
                                                      (Exp.add (Exp.mul w3 in3) (Exp.mul w4 in4))))

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
      ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 outDim)) outIdx) total
    ) (pure ())
  ) (pure ())

/-! ## Q6_K MatVec with Subgroup Reduction -/

/-- Subgroup-reduction variant of `fusedQ6KLinearKernel` for decode (M=1).

    Same math as the tree-reduction kernel, but uses 32 threads per
    workgroup and a single `subgroupAdd` instead of a 256-thread
    shared-memory tree reduction. Matches the pattern used by
    `BitLinear.fusedBitLinearM1Kernel` and
    `Linear.fusedQ4KMLinearSubgroupKernel`.

    Supports 2D workgroup grids via the optional `gridX` parameter so
    that `outDim` can exceed WebGPU's 65535 per-dimension workgroup
    limit (e.g. the Gemma 4 LM head at vocabSize=262144). With
    `gridX = 0` (the default), the kernel uses a 1D grid and computes
    `outIdx = workgroup_id.x`; with `gridX > 0`, it computes
    `outIdx = workgroup_id.x + workgroup_id.y * gridX`.

    @param inDim  Input dimension (must be multiple of blockSize=256)
    @param outDim Output dimension
    @param gridX  Set > 0 to enable 2D dispatch; must divide the dispatch
                  configuration such that wid.x + wid.y*gridX covers [0, outDim).
-/
def fusedQ6KLinearSubgroupKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / blockSize
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  -- NOTE: no barriers, no shared memory. outIdx bounds check is fine as a
  -- uniform branch since all 32 lanes share the same workgroup_id.
  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  let rowByteBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes))
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let readByte (blockBase : Exp (.scalar .u32)) (offset : Nat) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase (Exp.litU32 offset)
    let u32Idx := Exp.div byteIdx (Exp.litU32 4)
    let byteShift := Exp.mul (Exp.mod byteIdx (Exp.litU32 4)) (Exp.litU32 8)
    let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 32) fun blockLocalIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockLocalIdx (Exp.litU32 blockSizeBytes))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 blockSize)

    -- d (FP16) at byte offset 208
    let dLoByte ← readByte blockByteBase 208
    let dHiByte ← readByte blockByteBase 209
    let dBits := Exp.bitOr dLoByte (Exp.shiftLeft dHiByte (Exp.litU32 8))
    let sign := Exp.shiftRight dBits (Exp.litU32 15)
    let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
    let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
    let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
    let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
    let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
    let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
    let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
    let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
    let expFSubnormal := Exp.litF32 6.103515625e-5
    let expF := Exp.select isSubnormal expFSubnormal expFNormal
    let d := Exp.mul signF (Exp.mul mantF expF)

    let readScaleI8 (offset : Nat) : ShaderM (Exp (.scalar .f32)) := do
      let scByte ← readByte blockByteBase offset
      let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
        (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
        (Exp.toF32 scByte)
      pure scSigned

    for chunk in [0:2] do
      let qlBaseOff := chunk * 64
      let qhBaseOff := 128 + chunk * 32
      let scBaseOff := 192 + chunk * 8
      let chunkOutBase := chunk * 128
      for l in [0:32] do
        let isIdx := l / 16
        let sc0F ← readScaleI8 (scBaseOff + isIdx)
        let sc2F ← readScaleI8 (scBaseOff + isIdx + 2)
        let sc4F ← readScaleI8 (scBaseOff + isIdx + 4)
        let sc6F ← readScaleI8 (scBaseOff + isIdx + 6)
        let dsc0 := Exp.mul d sc0F
        let dsc2 := Exp.mul d sc2F
        let dsc4 := Exp.mul d sc4F
        let dsc6 := Exp.mul d sc6F

        let qlByte0 ← readByte blockByteBase (qlBaseOff + l)
        let qlByte32 ← readByte blockByteBase (qlBaseOff + l + 32)
        let qhByte ← readByte blockByteBase (qhBaseOff + l)

        let qlLow0 := Exp.bitAnd qlByte0 (Exp.litU32 0xF)
        let qlLow32 := Exp.bitAnd qlByte32 (Exp.litU32 0xF)
        let qlHigh0 := Exp.shiftRight qlByte0 (Exp.litU32 4)
        let qlHigh32 := Exp.shiftRight qlByte32 (Exp.litU32 4)

        let qhBits0 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 0)) (Exp.litU32 3)
        let qhBits2 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 2)) (Exp.litU32 3)
        let qhBits4 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 4)) (Exp.litU32 3)
        let qhBits6 := Exp.bitAnd (Exp.shiftRight qhByte (Exp.litU32 6)) (Exp.litU32 3)

        let q1Raw := Exp.bitOr qlLow0  (Exp.shiftLeft qhBits0 (Exp.litU32 4))
        let q2Raw := Exp.bitOr qlLow32 (Exp.shiftLeft qhBits2 (Exp.litU32 4))
        let q3Raw := Exp.bitOr qlHigh0  (Exp.shiftLeft qhBits4 (Exp.litU32 4))
        let q4Raw := Exp.bitOr qlHigh32 (Exp.shiftLeft qhBits6 (Exp.litU32 4))
        let q1 := Exp.sub (Exp.toF32 q1Raw) (Exp.litF32 32.0)
        let q2 := Exp.sub (Exp.toF32 q2Raw) (Exp.litF32 32.0)
        let q3 := Exp.sub (Exp.toF32 q3Raw) (Exp.litF32 32.0)
        let q4 := Exp.sub (Exp.toF32 q4Raw) (Exp.litF32 32.0)

        let w1 := Exp.mul dsc0 q1
        let w2 := Exp.mul dsc2 q2
        let w3 := Exp.mul dsc4 q3
        let w4 := Exp.mul dsc6 q4

        let inIdx1 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l))
        let inIdx2 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 32))
        let inIdx3 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 64))
        let inIdx4 := Exp.add elemBase (Exp.litU32 (chunkOutBase + l + 96))

        let in1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx1
        let in2 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx2
        let in3 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx3
        let in4 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx4

        ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.add (Exp.mul w1 in1) (Exp.mul w2 in2))
                                                    (Exp.add (Exp.mul w3 in3) (Exp.mul w4 in4))))

  -- Subgroup reduction: hardware-accelerated sum across 32 lanes.
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Q6_K MatVec, Block-Cooperative + Software Pipelining -/

/-- Block-cooperative + software-pipelined Q6_K mat-vec kernel.
    Mirrors the Q4_K `fusedQ4KMLinearBlockCoopKernel` design:

      * 1 workgroup per output row (or `outIdx = wid.x + wid.y * gridX`
        for 2D dispatch so `outDim` can exceed WebGPU's 65535 per-dim
        limit, which the Gemma 4 lm_head needs at vocabSize=262144),
        32 threads per workgroup
      * Each lane owns index `l = tid` in each of the 2 chunks of a
        Q6_K block and produces 8 dequantised weights (4 per chunk),
        so all 32 lanes are always active regardless of blocksPerRow
      * `d` (fp16) and `scales[0..15]` are read redundantly by all 32
        lanes — hardware broadcast coalesces them into one transaction
      * Depth-1 software pipelining of the heavy per-lane byte reads
        (qlByte0/32/64/96 + qhByte/32 = 6 bytes per lane per block)
        into `next*` mutable vars so load latency overlaps with the
        dequant + FMA chain of the previous iteration. The cheap
        shared header bytes (d, scales) are reloaded each iteration
        because broadcast loads are already fast.

    With `gridX = 0` (default) uses a 1D grid `outIdx = wid.x`.
    With `gridX > 0`, uses a 2D grid: `outIdx = wid.x + wid.y * gridX`.
-/
def fusedQ6KLinearBlockCoopKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / blockSize
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowByteBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes))

  -- Helper: read a single byte at `blockByteBase + offsetExp` where the
  -- offset is a runtime Exp (not a Lean literal).
  let readByteExp (blockByteBase : Exp (.scalar .u32)) (offsetExp : Exp (.scalar .u32))
      : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockByteBase offsetExp
    let u32Idx := Exp.div byteIdx (Exp.litU32 4)
    let byteShift := Exp.mul (Exp.sub byteIdx (Exp.mul u32Idx (Exp.litU32 4))) (Exp.litU32 8)
    let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

  -- Same helper but offset is a Lean-level Nat (compile-time literal).
  let readByteLit (blockByteBase : Exp (.scalar .u32)) (offset : Nat)
      : ShaderM (Exp (.scalar .u32)) :=
    readByteExp blockByteBase (Exp.litU32 offset)

  -- Per-lane byte offsets inside a block:
  --   chunk 0: ql[tid] at `tid`, ql[tid+32] at `tid+32`, qh[tid] at `128+tid`
  --   chunk 1: ql[tid+64] at `tid+64`, ql[tid+96] at `tid+96`, qh[tid+32] at `160+tid`
  let offQl0  := tid                                    -- chunk0 ql[l]
  let offQl32 := Exp.add tid (Exp.litU32 32)            -- chunk0 ql[l+32]
  let offQh0  := Exp.add tid (Exp.litU32 128)           -- chunk0 qh[l]
  let offQl64 := Exp.add tid (Exp.litU32 64)            -- chunk1 ql[l]
  let offQl96 := Exp.add tid (Exp.litU32 96)            -- chunk1 ql[l+32]
  let offQh32 := Exp.add tid (Exp.litU32 160)           -- chunk1 qh[l]

  -- ## Pre-loop prefetch of block 0's per-lane bytes.
  let block0Base := rowByteBase
  let nextQl0Init  ← readByteExp block0Base offQl0
  let nextQl32Init ← readByteExp block0Base offQl32
  let nextQh0Init  ← readByteExp block0Base offQh0
  let nextQl64Init ← readByteExp block0Base offQl64
  let nextQl96Init ← readByteExp block0Base offQl96
  let nextQh32Init ← readByteExp block0Base offQh32
  ShaderM.varNamed "nextQl0"  (.scalar .u32) nextQl0Init
  ShaderM.varNamed "nextQl32" (.scalar .u32) nextQl32Init
  ShaderM.varNamed "nextQh0"  (.scalar .u32) nextQh0Init
  ShaderM.varNamed "nextQl64" (.scalar .u32) nextQl64Init
  ShaderM.varNamed "nextQl96" (.scalar .u32) nextQl96Init
  ShaderM.varNamed "nextQh32" (.scalar .u32) nextQh32Init

  -- Runtime block loop (avoids compile-time unroll → register spill on large inDim)
  ShaderM.varNamed "currQl0"  (.scalar .u32) (Exp.var "nextQl0"  : Exp (.scalar .u32))
  ShaderM.varNamed "currQl32" (.scalar .u32) (Exp.var "nextQl32" : Exp (.scalar .u32))
  ShaderM.varNamed "currQh0"  (.scalar .u32) (Exp.var "nextQh0"  : Exp (.scalar .u32))
  ShaderM.varNamed "currQl64" (.scalar .u32) (Exp.var "nextQl64" : Exp (.scalar .u32))
  ShaderM.varNamed "currQl96" (.scalar .u32) (Exp.var "nextQl96" : Exp (.scalar .u32))
  ShaderM.varNamed "currQh32" (.scalar .u32) (Exp.var "nextQh32" : Exp (.scalar .u32))

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes))
    let elemBase := Exp.mul blockIdx (Exp.litU32 blockSize)

    ShaderM.assign "currQl0"  (Exp.var (t := .scalar .u32) "nextQl0")
    ShaderM.assign "currQl32" (Exp.var (t := .scalar .u32) "nextQl32")
    ShaderM.assign "currQh0"  (Exp.var (t := .scalar .u32) "nextQh0")
    ShaderM.assign "currQl64" (Exp.var (t := .scalar .u32) "nextQl64")
    ShaderM.assign "currQl96" (Exp.var (t := .scalar .u32) "nextQl96")
    ShaderM.assign "currQh32" (Exp.var (t := .scalar .u32) "nextQh32")
    let qlByte0  : Exp (.scalar .u32) := Exp.var "currQl0"
    let qlByte32 : Exp (.scalar .u32) := Exp.var "currQl32"
    let qhByte0  : Exp (.scalar .u32) := Exp.var "currQh0"
    let qlByte64 : Exp (.scalar .u32) := Exp.var "currQl64"
    let qlByte96 : Exp (.scalar .u32) := Exp.var "currQl96"
    let qhByte32 : Exp (.scalar .u32) := Exp.var "currQh32"

    -- Prefetch next block
    let nextBlockIdx := Exp.add blockIdx (Exp.litU32 1)
    ShaderM.if_ (Exp.lt nextBlockIdx (Exp.litU32 blocksPerRow)) (do
      let nbBase := Exp.add rowByteBase (Exp.mul nextBlockIdx (Exp.litU32 blockSizeBytes))
      ShaderM.assign "nextQl0"  (← readByteExp nbBase offQl0)
      ShaderM.assign "nextQl32" (← readByteExp nbBase offQl32)
      ShaderM.assign "nextQh0"  (← readByteExp nbBase offQh0)
      ShaderM.assign "nextQl64" (← readByteExp nbBase offQl64)
      ShaderM.assign "nextQl96" (← readByteExp nbBase offQl96)
      ShaderM.assign "nextQh32" (← readByteExp nbBase offQh32)
    ) (pure ())

    -- Shared block header: d (fp16 at byte 208), scales[0..15] at bytes 192..207.
    -- All 32 lanes read the same addresses → hardware broadcast.
    let dLoByte ← readByteLit blockByteBase 208
    let dHiByte ← readByteLit blockByteBase 209
    let dBits := Exp.bitOr dLoByte (Exp.shiftLeft dHiByte (Exp.litU32 8))
    let sign := Exp.shiftRight dBits (Exp.litU32 15)
    let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
    let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
    let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
    let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
    let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
    let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
    let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
    let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
    let expFSubnormal := Exp.litF32 6.103515625e-5
    let expF := Exp.select isSubnormal expFSubnormal expFNormal
    let d := Exp.mul signF (Exp.mul mantF expF)

    let readScaleI8 (scaleOff : Nat) : ShaderM (Exp (.scalar .f32)) := do
      let scByte ← readByteLit blockByteBase scaleOff
      let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
        (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
        (Exp.toF32 scByte)
      pure scSigned

    -- Per-lane `l = tid`, `isIdx = l / 16` is runtime-dependent. Since
    -- `is ∈ {0, 1}` and is used to pick between scales[chunk*8 + is]
    -- vs scales[chunk*8 + is + {2,4,6}], we compute both possibilities
    -- (is=0 and is=1) and select at runtime.
    let isIdx := Exp.div tid (Exp.litU32 16)

    -- For each chunk, read 8 scales (only the 4 we need given is), do
    -- the l-specific dequant for 4 output positions, and FMA.
    for chunk in [0:2] do
      let scBaseOff := 192 + chunk * 8
      let chunkOutBase := chunk * 128
      -- All 8 scales for this chunk — 4 needed for is=0, 4 for is=1.
      -- We select dynamically per lane.
      let sc0_0 ← readScaleI8 (scBaseOff + 0)
      let sc0_1 ← readScaleI8 (scBaseOff + 1)
      let sc0_2 ← readScaleI8 (scBaseOff + 2)
      let sc0_3 ← readScaleI8 (scBaseOff + 3)
      let sc0_4 ← readScaleI8 (scBaseOff + 4)
      let sc0_5 ← readScaleI8 (scBaseOff + 5)
      let sc0_6 ← readScaleI8 (scBaseOff + 6)
      let sc0_7 ← readScaleI8 (scBaseOff + 7)
      let isZero := Exp.eq isIdx (Exp.litU32 0)
      let scFor0 := Exp.select isZero sc0_0 sc0_1  -- sc[is+0]
      let scFor2 := Exp.select isZero sc0_2 sc0_3  -- sc[is+2]
      let scFor4 := Exp.select isZero sc0_4 sc0_5  -- sc[is+4]
      let scFor6 := Exp.select isZero sc0_6 sc0_7  -- sc[is+6]
      let dsc0 := Exp.mul d scFor0
      let dsc2 := Exp.mul d scFor2
      let dsc4 := Exp.mul d scFor4
      let dsc6 := Exp.mul d scFor6

      -- Pick which qlByte and qhByte to use for this chunk.
      let ql0  := if chunk == 0 then qlByte0  else qlByte64
      let ql32 := if chunk == 0 then qlByte32 else qlByte96
      let qhL  := if chunk == 0 then qhByte0  else qhByte32

      let qlLow0  := Exp.bitAnd ql0  (Exp.litU32 0xF)
      let qlLow32 := Exp.bitAnd ql32 (Exp.litU32 0xF)
      let qlHigh0  := Exp.shiftRight ql0  (Exp.litU32 4)
      let qlHigh32 := Exp.shiftRight ql32 (Exp.litU32 4)

      let qhBits0 := Exp.bitAnd qhL (Exp.litU32 0x3)
      let qhBits2 := Exp.bitAnd (Exp.shiftRight qhL (Exp.litU32 2)) (Exp.litU32 0x3)
      let qhBits4 := Exp.bitAnd (Exp.shiftRight qhL (Exp.litU32 4)) (Exp.litU32 0x3)
      let qhBits6 := Exp.bitAnd (Exp.shiftRight qhL (Exp.litU32 6)) (Exp.litU32 0x3)

      let q1Raw := Exp.bitOr qlLow0  (Exp.shiftLeft qhBits0 (Exp.litU32 4))
      let q2Raw := Exp.bitOr qlLow32 (Exp.shiftLeft qhBits2 (Exp.litU32 4))
      let q3Raw := Exp.bitOr qlHigh0  (Exp.shiftLeft qhBits4 (Exp.litU32 4))
      let q4Raw := Exp.bitOr qlHigh32 (Exp.shiftLeft qhBits6 (Exp.litU32 4))
      let q1 := Exp.sub (Exp.toF32 q1Raw) (Exp.litF32 32.0)
      let q2 := Exp.sub (Exp.toF32 q2Raw) (Exp.litF32 32.0)
      let q3 := Exp.sub (Exp.toF32 q3Raw) (Exp.litF32 32.0)
      let q4 := Exp.sub (Exp.toF32 q4Raw) (Exp.litF32 32.0)

      let w1 := Exp.mul dsc0 q1
      let w2 := Exp.mul dsc2 q2
      let w3 := Exp.mul dsc4 q3
      let w4 := Exp.mul dsc6 q4

      -- Input addresses: elemBase + (chunkOutBase + l + {0,32,64,96}), l = tid.
      let chunkBaseU := Exp.add elemBase (Exp.litU32 chunkOutBase)
      let inIdx1 := Exp.add chunkBaseU tid
      let inIdx2 := Exp.add chunkBaseU (Exp.add tid (Exp.litU32 32))
      let inIdx3 := Exp.add chunkBaseU (Exp.add tid (Exp.litU32 64))
      let inIdx4 := Exp.add chunkBaseU (Exp.add tid (Exp.litU32 96))

      let in1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx1
      let in2 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx2
      let in3 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx3
      let in4 ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" inIdx4

      ShaderM.assign "acc" (Exp.add acc
        (Exp.add (Exp.add (Exp.mul w1 in1) (Exp.mul w2 in2))
                 (Exp.add (Exp.mul w3 in3) (Exp.mul w4 in4))))

  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Q6_K Embedding Lookup Kernel -/

/-- Dequant a single Q6_K element at (rowIdx, col) in a [numRows, dim] table.
    Returns the dequantized f32 value.

    Parameters as Exp so they can be computed at runtime.
    Reads directly from the packed table buffer (declared externally).

    @param dim Dimension per row (must be multiple of 256)
    @param tableBufName Name of the buffer declared in the shader
    @param totalU32 Total u32 count in the buffer (for type)
-/
private def dequantQ6KElement (dim : Nat) (tableBufName : String) (totalU32 : Nat)
    (rowIdx : Exp (.scalar .u32)) (col : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .f32)) := do
  -- Byte offset in table: rowIdx * (dim / 256) * 210 + blockOffset + withinBlockOffset
  let blocksPerRow := dim / 256
  let rowByteBase := Exp.mul rowIdx (Exp.litU32 (blocksPerRow * blockSizeBytes))
  let blockIdxInRow := Exp.div col (Exp.litU32 256)
  let elemInBlock := Exp.mod col (Exp.litU32 256)

  let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdxInRow (Exp.litU32 blockSizeBytes))

  -- Read d (FP16) at byte 208 within block
  let dByteOff := Exp.add blockByteBase (Exp.litU32 208)
  let dU32Idx := Exp.div dByteOff (Exp.litU32 4)
  let dByteInU32 := Exp.mul (Exp.mod dByteOff (Exp.litU32 4)) (Exp.litU32 8)
  let dU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalU32) tableBufName dU32Idx
  let dBits := Exp.bitAnd (Exp.shiftRight dU32 dByteInU32) (Exp.litU32 0xFFFF)
  -- FP16 → F32 (arithmetic, supports subnormals)
  --   normal:   (-1)^s * (1 + mant/1024) * 2^(exp-15)
  --   subnormal (exp==0): (-1)^s * (mant/1024) * 2^(-14)
  let sign := Exp.shiftRight dBits (Exp.litU32 15)
  let exp5 := Exp.bitAnd (Exp.shiftRight dBits (Exp.litU32 10)) (Exp.litU32 0x1F)
  let mant := Exp.bitAnd dBits (Exp.litU32 0x3FF)
  let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
  let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
  let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
  let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
  let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
  let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
  let expFSubnormal := Exp.litF32 6.103515625e-5  -- 2^(-14)
  let expF := Exp.select isSubnormal expFSubnormal expFNormal
  let d := Exp.mul signF (Exp.mul mantF expF)

  -- Determine which chunk (0 or 1), group (0..3), and position within group (0..31)
  let chunk := Exp.div elemInBlock (Exp.litU32 128)           -- 0 or 1
  let elemInChunk := Exp.mod elemInBlock (Exp.litU32 128)     -- 0..127
  let group := Exp.div elemInChunk (Exp.litU32 32)            -- 0..3
  let posInGroup := Exp.mod elemInChunk (Exp.litU32 32)       -- 0..31

  -- ql byte: chunk * 64 bytes offset + (group < 2 ? posInGroup : posInGroup) (same ql bytes for groups 0+2, 1+3)
  -- Actually per dequant_row_q6_K: groups 0,1 use ql[0..63], groups 2,3 use ql[0..63] as well (different nibbles)
  -- Wait, looking at llama.cpp: for n=0 chunk, l in 0..31, l+0 uses q1, l+32 uses q2, l+0(high nibble) uses q3, l+32(high nibble) uses q4
  -- So each "l" (posInGroup) indexes ql[l] and ql[l+32]. Group 0 uses low nibble of ql[l], group 1 uses low nibble of ql[l+32],
  --                                                        group 2 uses high nibble of ql[l], group 3 uses high nibble of ql[l+32]
  -- But wait posInGroup goes 0..31 so we can directly compute ql byte position
  let qlBase := Exp.add blockByteBase (Exp.mul chunk (Exp.litU32 64))
  -- For group 0: ql[posInGroup], low nibble
  -- For group 1: ql[posInGroup + 32], low nibble
  -- For group 2: ql[posInGroup], high nibble
  -- For group 3: ql[posInGroup + 32], high nibble
  let groupMod2 := Exp.mod group (Exp.litU32 2)  -- 0 or 1 (selects +0 or +32 offset)
  let qlByteOff := Exp.add (Exp.add qlBase posInGroup) (Exp.mul groupMod2 (Exp.litU32 32))
  let qlU32Idx := Exp.div qlByteOff (Exp.litU32 4)
  let qlByteInU32 := Exp.mul (Exp.mod qlByteOff (Exp.litU32 4)) (Exp.litU32 8)
  let qlU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalU32) tableBufName qlU32Idx
  let qlByte := Exp.bitAnd (Exp.shiftRight qlU32 qlByteInU32) (Exp.litU32 0xFF)
  let useHighNibble := Exp.ge group (Exp.litU32 2)
  let qlNibble := Exp.select useHighNibble
    (Exp.shiftRight qlByte (Exp.litU32 4))
    (Exp.bitAnd qlByte (Exp.litU32 0xF))

  -- qh byte: chunk * 32 offset + posInGroup (same qh byte used for all 4 groups at position posInGroup)
  let qhBase := Exp.add blockByteBase (Exp.litU32 (128 + 0))  -- qh starts at byte 128
  let qhBase2 := Exp.add qhBase (Exp.mul chunk (Exp.litU32 32))
  let qhByteOff := Exp.add qhBase2 posInGroup
  let qhU32Idx := Exp.div qhByteOff (Exp.litU32 4)
  let qhByteInU32 := Exp.mul (Exp.mod qhByteOff (Exp.litU32 4)) (Exp.litU32 8)
  let qhU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalU32) tableBufName qhU32Idx
  let qhByte := Exp.bitAnd (Exp.shiftRight qhU32 qhByteInU32) (Exp.litU32 0xFF)
  -- group 0→bits 0-1, group 1→bits 2-3, group 2→bits 4-5, group 3→bits 6-7
  let qhShift := Exp.mul group (Exp.litU32 2)
  let qhBits := Exp.bitAnd (Exp.shiftRight qhByte qhShift) (Exp.litU32 0x3)

  -- q6 = (qlNibble | (qhBits << 4)) - 32
  let q6Raw := Exp.bitOr qlNibble (Exp.shiftLeft qhBits (Exp.litU32 4))
  let q6 := Exp.sub (Exp.toF32 q6Raw) (Exp.litF32 32.0)

  -- scales[is]: within block, is = (chunk * 8) + (group * 2) + (posInGroup / 16)
  -- Wait: looking at dequant_row_q6_K:
  --   for n in [0, 128]: for l in 0..31: is = l/16; y[l+0] = d*sc[is+0]*q1; y[l+32] = d*sc[is+2]*q2 ...
  -- So is increments by 2 between groups within chunk. Per chunk, we use sc[0..7]
  -- Chunk 0 uses sc[0..7], chunk 1 uses sc[8..15]
  -- Within chunk: group 0 uses sc[is+0] where is = l/16 (0 or 1), so sc[0 or 1]
  --              group 1 uses sc[is+2] = sc[2 or 3]
  --              group 2 uses sc[is+4] = sc[4 or 5]
  --              group 3 uses sc[is+6] = sc[6 or 7]
  -- So per group: scaleIdx = group*2 + (posInGroup/16) [within chunk]
  -- Absolute: chunk*8 + group*2 + (posInGroup/16)
  let scIdx := Exp.add (Exp.mul chunk (Exp.litU32 8))
                       (Exp.add (Exp.mul group (Exp.litU32 2))
                                (Exp.div posInGroup (Exp.litU32 16)))
  let scByteOff := Exp.add blockByteBase (Exp.add (Exp.litU32 192) scIdx)
  let scU32Idx := Exp.div scByteOff (Exp.litU32 4)
  let scByteInU32 := Exp.mul (Exp.mod scByteOff (Exp.litU32 4)) (Exp.litU32 8)
  let scU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalU32) tableBufName scU32Idx
  let scByte := Exp.bitAnd (Exp.shiftRight scU32 scByteInU32) (Exp.litU32 0xFF)
  -- Sign extend 8-bit
  let scSigned := Exp.select (Exp.ge scByte (Exp.litU32 128))
    (Exp.sub (Exp.toF32 scByte) (Exp.litF32 256.0))
    (Exp.toF32 scByte)

  -- Final: y = d * sc * q6
  return Exp.mul d (Exp.mul scSigned q6)

/-- Q6_K embedding lookup kernel.
    Reads the embedding vector for a token ID from a Q6_K-packed table.
    Each thread dequantizes one element of the output row.

    @param vocabSize Table row count
    @param dim Embedding dimension (must be multiple of 256)
-/
def q6kEmbeddingLookupKernel (vocabSize dim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- output element index within the row

  let blocksPerRow := dim / 256
  let totalU32 := (vocabSize * blocksPerRow * blockSizeBytes + 3) / 4

  let _tokenIds ← ShaderM.declareReadOnlyBuffer "token_ids" (.array (.scalar .u32) 1)
  let _table ← ShaderM.declareReadOnlyBuffer "embedding_table" (.array (.scalar .u32) totalU32)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 dim)) (do
    let tokenId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "token_ids" (Exp.litU32 0)
    let val ← dequantQ6KElement dim "embedding_table" totalU32 tokenId idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx val
  ) (pure ())

/-- Batched Q6_K embedding gather: for output element t = p*dim + d,
    output[t] = scale · dequant(table[token_ids[p]], d).  1 thread per element. -/
def q6kEmbedGatherKernel (N vocabSize dim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let blocksPerRow := dim / 256
  let totalU32 := (vocabSize * blocksPerRow * blockSizeBytes + 3) / 4
  let _tokenIds ← ShaderM.declareReadOnlyBuffer "token_ids" (.array (.scalar .u32) N)
  let _table ← ShaderM.declareReadOnlyBuffer "embedding_table" (.array (.scalar .u32) totalU32)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * dim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N * dim))) (do
    let p := Exp.div t (Exp.litU32 dim)
    let d := Exp.sub t (Exp.mul p (Exp.litU32 dim))
    let tokenId ← ShaderM.readBuffer (ty := .scalar .u32) (n := N) "token_ids" p
    let val ← dequantQ6KElement dim "embedding_table" totalU32 tokenId d
    ShaderM.writeBuffer (ty := .scalar .f32) "output" t (Exp.mul val (Exp.litF32 scale))
  ) (pure ())

/-- Single-row Q6_K dequant with a compile-time scale. Used to replace
    the slow `dequantQ6KRowCPU` + map + upload pipeline in
    `forwardSingleToken`'s per-layer-embedding precompute: upload the
    ~33 KB of raw Q6_K bytes for the needed row once, then run this
    kernel to dequant + scale into the target f32 buffer in one pass.
    The row table is just one row (rowIdx = 0 is always used).

    @param dim  Total elements in the row (must be multiple of 256)
    @param scale  Multiplicative scale applied after dequant
-/
def q6kSingleRowDequantScaleKernel (dim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let blocksPerRow := dim / 256
  let totalU32 := (blocksPerRow * blockSizeBytes + 3) / 4

  let _row ← ShaderM.declareReadOnlyBuffer "row" (.array (.scalar .u32) totalU32)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 dim)) (do
    let val ← dequantQ6KElement dim "row" totalU32 (Exp.litU32 0) idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 scale))
  ) (pure ())

/-- Dequant + scale a row from a full Q6_K table on GPU.
    Row is identified by `params[0]` which holds the *row index* (token ID),
    NOT byte offset.  Uses the same `dequantQ6KElement` as `q6kEmbeddingLookupKernel`
    with the full table buffer size — proven to JIT correctly.
    `vocabSize` determines the declared table buffer size. -/
def q6kTableRowDequantScaleKernel (dim : Nat) (scale : Float)
    (vocabSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let blocksPerRow := dim / 256
  let rowU32Size := (blocksPerRow * blockSizeBytes + 3) / 4
  -- Declare a small buffer (2 rows) — CUDA ignores declared sizes for
  -- global memory.  The actual GPU buffer is vocabSize rows; the
  -- runtime tokenId selects the correct row offset.  Keeping the declared
  -- size small prevents PTX array-size explosion in the ShaderM printer.
  let declaredU32 := rowU32Size * 2

  let _table ← ShaderM.declareReadOnlyBuffer "table" (.array (.scalar .u32) declaredU32)
  let _params ← ShaderM.declareReadOnlyBuffer "params" (.array (.scalar .u32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  -- Read runtime token ID from params[0], use as rowStartU32
  let tokenId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 dim)) (do
    let val ← dequantQ6KElement dim "table" declaredU32 tokenId idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 scale))
  ) (pure ())

end Hesper.Quantization.Q6_K
