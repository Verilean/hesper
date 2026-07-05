import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Monad
import Hesper.Quantization.Q4_K_M

/-!
# Q5_K Dequantization

GGML Q5_K super-block layout (256 elements, 176 bytes = 44 u32, 4-byte aligned):
- bytes [0..2):    d    (FP16 super-block scale)
- bytes [2..4):    dmin (FP16 super-block min)
- bytes [4..16):   scales[12] — packed 6-bit scale/min pairs, SAME encoding as Q4_K
- bytes [16..48):  qh[32] — high bits, bit j of qh[l] = 5th bit of sub-block j element l
- bytes [48..176): qs[128] — low 4 bits, nibble-packed like Q4_K

Element value: d·sc[j]·(q4 + 16·hbit) − dmin·m[j]  for sub-block j = local/32.

Discovered need: Gemma 4 **E2B** quantizes `per_layer_token_embd` as Q5_K
(E4B used Q6_K) — the only Q5_K tensor in the file. The scale/min unpacking is
identical to Q4_K, so we reuse `Q4_K_M.getScaleMin` / `fp16ToF32`.
-/

namespace Hesper.Quantization.Q5_K

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.Quantization.Q4_K_M (fp16ToF32 getScaleMin)

/-- Elements per Q5_K super-block -/
def blockSize : Nat := 256

/-- Bytes per Q5_K super-block (4-byte aligned: 44 u32) -/
def blockSizeBytes : Nat := 176

/-- u32 words per Q5_K super-block -/
def blockSizeU32 : Nat := 44

/-- Dequantize ONE Q5_K element at a runtime global element index from a table bound as
    `bufName` (u32-packed Q5_K blocks, declared size `numU32`). Mirrors
    `Q4_K_M.dequantQ4KElementAt` plus the qh high-bit. -/
def dequantQ5KElementAt (bufName : String) (numU32 : Nat)
    (gIdx : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .f32)) := do
  let blockIdx := Exp.div gIdx (Exp.litU32 blockSize)
  let localIdx := Exp.mod gIdx (Exp.litU32 blockSize)
  let blockBaseU32 := Exp.mul blockIdx (Exp.litU32 blockSizeU32)
  -- d / dmin (packed in u32[0])
  let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName blockBaseU32
  let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
  let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
  -- scales[12] = u32[1..3], same packing as Q4_K
  let scalesU32_0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName (Exp.add blockBaseU32 (Exp.litU32 1))
  let scalesU32_1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName (Exp.add blockBaseU32 (Exp.litU32 2))
  let scalesU32_2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName (Exp.add blockBaseU32 (Exp.litU32 3))
  -- Sub-block decomposition (identical to Q4_K): qs chunk = 64 elems (2 sub-blocks
  -- sharing bytes: low nibble = sub-block 2k, high nibble = sub-block 2k+1)
  let chunkIdx := Exp.div localIdx (Exp.litU32 64)
  let posInChunk := Exp.mod localIdx (Exp.litU32 64)
  let isHighNibble := Exp.ge posInChunk (Exp.litU32 32)
  let posInSubBlock := Exp.mod posInChunk (Exp.litU32 32)
  let subBlockIdx := Exp.add (Exp.mul chunkIdx (Exp.litU32 2)) (Exp.select isHighNibble (Exp.litU32 1) (Exp.litU32 0))
  let (sc0, m0) := getScaleMin 0 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc1, m1) := getScaleMin 1 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc2, m2) := getScaleMin 2 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc3, m3) := getScaleMin 3 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc4, m4) := getScaleMin 4 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc5, m5) := getScaleMin 5 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc6, m6) := getScaleMin 6 scalesU32_0 scalesU32_1 scalesU32_2
  let (sc7, m7) := getScaleMin 7 scalesU32_0 scalesU32_1 scalesU32_2
  let scVal := Exp.select (Exp.eq subBlockIdx (Exp.litU32 0)) sc0
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 1)) sc1
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 2)) sc2
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 3)) sc3
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 4)) sc4
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 5)) sc5
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 6)) sc6 sc7))))))
  let mVal := Exp.select (Exp.eq subBlockIdx (Exp.litU32 0)) m0
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 1)) m1
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 2)) m2
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 3)) m3
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 4)) m4
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 5)) m5
    (Exp.select (Exp.eq subBlockIdx (Exp.litU32 6)) m6 m7))))))
  -- qs low nibble: bytes [48..176) = u32[12..44)
  let qsByteIdx := Exp.add (Exp.mul chunkIdx (Exp.litU32 32)) posInSubBlock
  let qsU32Idx := Exp.div qsByteIdx (Exp.litU32 4)
  let qsByteOffset := Exp.mul (Exp.mod qsByteIdx (Exp.litU32 4)) (Exp.litU32 8)
  let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName (Exp.add blockBaseU32 (Exp.add (Exp.litU32 12) qsU32Idx))
  let qsByte := Exp.bitAnd (Exp.shiftRight qsU32 qsByteOffset) (Exp.litU32 0xFF)
  let q4 := Exp.select isHighNibble
    (Exp.shiftRight qsByte (Exp.litU32 4))
    (Exp.bitAnd qsByte (Exp.litU32 0xF))
  -- qh high bit: bytes [16..48) = u32[4..12); bit subBlockIdx of qh[posInSubBlock]
  let qhU32Idx := Exp.div posInSubBlock (Exp.litU32 4)
  let qhByteOffset := Exp.mul (Exp.mod posInSubBlock (Exp.litU32 4)) (Exp.litU32 8)
  let qhU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) bufName (Exp.add blockBaseU32 (Exp.add (Exp.litU32 4) qhU32Idx))
  let qhByte := Exp.bitAnd (Exp.shiftRight qhU32 qhByteOffset) (Exp.litU32 0xFF)
  let hBit := Exp.bitAnd (Exp.shiftRight qhByte subBlockIdx) (Exp.litU32 1)
  let qVal := Exp.bitOr q4 (Exp.shiftLeft hBit (Exp.litU32 4))
  pure (Exp.sub (Exp.mul d (Exp.mul scVal (Exp.toF32 qVal))) (Exp.mul dmin mVal))

/-- Dequant + scale a row from a Q5_K table on GPU. Row identified by `params[0]`
    (row index, not byte offset) or a baked literal. Mirrors
    `Q6_K.q6kTableRowDequantScaleKernel` — see there for the declRows / bakedRow
    rationale (WebGPU robustness clamps to the DECLARED size; batched prefill
    needs per-slot baked rows). Row stride = dim/256 × 44 u32 (176 B blocks are
    4-byte aligned so no rounding is needed). -/
def q5kTableRowDequantScaleKernel (dim : Nat) (scale : Float)
    (_vocabSize : Nat) (declRows : Nat := 2) (bakedRow : Option Nat := none) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let blocksPerRow := dim / 256
  let rowU32Size := blocksPerRow * blockSizeU32
  let declaredU32 := rowU32Size * declRows

  let _table ← ShaderM.declareReadOnlyBuffer "table" (.array (.scalar .u32) declaredU32)
  let _params ← ShaderM.declareReadOnlyBuffer "params" (.array (.scalar .u32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  let tokenId ← match bakedRow with
    | some r => pure (Exp.litU32 r)
    | none => ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 dim)) (do
    let gIdx := Exp.add (Exp.mul tokenId (Exp.litU32 dim)) idx
    let val ← dequantQ5KElementAt "table" declaredU32 gIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 scale))
  ) (pure ())

end Hesper.Quantization.Q5_K
