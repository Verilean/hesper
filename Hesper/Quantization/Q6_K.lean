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
-/
def fusedQ6KLinearKernel (inDim outDim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / blockSize
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4

  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) inDim)
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

  let _tokenIds ← ShaderM.declareInputBuffer "token_ids" (.array (.scalar .u32) 1)
  let _table ← ShaderM.declareInputBuffer "embedding_table" (.array (.scalar .u32) totalU32)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 dim)) (do
    let tokenId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "token_ids" (Exp.litU32 0)
    let val ← dequantQ6KElement dim "embedding_table" totalU32 tokenId idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx val
  ) (pure ())

end Hesper.Quantization.Q6_K
