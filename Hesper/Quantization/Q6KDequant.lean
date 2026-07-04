import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Quantization.Q6_K

/-!
# Q6_K → packed F16 (half2) dequantization kernel.

One-shot kernel run at model load to convert a Q6_K tensor into the
packed `[N, K/2]` u32 layout consumed by `matMulTransposeF16BlockCoopKernel`.
Used for the LM head where on-the-fly Q6_K dequant is the bottleneck
(Q6_K × Q8_1 path = 1140 µs/call vs llama.cpp's f16×f16 = 114 µs/call).

## Layout

Workgroup = 1 Q6_K block of 256 elements (210 input bytes).
64 threads per workgroup; each thread emits 2 consecutive output
elements packed into 1 u32 (half2).  256 / 2 = 128 output u32s per block.
Each thread writes 2 u32s (4 elements).

Grid `numWorkgroups = (blocksPerRow * outDim, 1, 1)` covers every block
of every row.  `dispatchDim` = numBlocks total.

## Output

The output buffer matches `matMulTransposeF16BlockCoopKernel`'s `b` input:
`u32` array of length `outDim * (inDim / 2)`, where row `n` starts at
u32 index `n * (inDim / 2)`.
-/

namespace Hesper.Quantization.Q6_K

open Hesper.WGSL
open Hesper.WGSL.Monad

/-- Q6_K → packed half2 dequantization. -/
def q6kToF16Kernel (inDim outDim : Nat) (gridXWidth : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  -- blockGlobalIdx in [0, outDim * blocksPerRow). 1D when gridXWidth=0; else 2D (x + y*gridXWidth)
  -- so totalBlocks > the 65535 per-dimension limit can be covered by a single 2D dispatch.
  let blockGlobalIdx :=
    if gridXWidth > 0 then Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridXWidth))
    else Exp.vec3X wid
  let tid := Exp.vec3X lid  -- 0..63

  let blocksPerRow := inDim / blockSize
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  -- Output: [N, K/2] packed half2 u32, total = outDim * (inDim / 2) u32
  let totalOutU32 := outDim * (inDim / 2)

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _output  ← ShaderM.declareOutputBuffer    "output"  (.array (.scalar .u32) totalOutU32)

  -- (n, blkInRow) = blockGlobalIdx decomposed
  let nIdx     := Exp.div blockGlobalIdx (Exp.litU32 blocksPerRow)
  let blkInRow := Exp.bitAnd blockGlobalIdx (Exp.litU32 (blocksPerRow - 1))
    -- blocksPerRow is always a power of 2 on Gemma 4 (inDim=2560 → 10 — NOT pow2)
    -- so use mod instead. blocksPerRow may be e.g. 10 → fall back to mod.
  let blkInRowSafe :=
    if (blocksPerRow != 0) && ((blocksPerRow &&& (blocksPerRow - 1)) == 0) then
      blkInRow  -- pow2 fast-path
    else
      Exp.mod blockGlobalIdx (Exp.litU32 blocksPerRow)
  let blockByteBase :=
    Exp.add
      (Exp.mul nIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
      (Exp.mul blkInRowSafe (Exp.litU32 blockSizeBytes))

  -- Each thread covers 2 element pairs in the block.
  -- Following llama.cpp's dequantize_block_q6_K layout:
  --   ip = tid / 32   ∈ {0, 1}   → which 128-element half
  --   il = tid % 32   ∈ [0, 32)  → position within half
  --   thread writes y[il+128*ip + 0/32/64/96]
  -- We pack into half2: each thread writes pair (offset, offset+1)? NO —
  -- llama.cpp's offsets aren't consecutive.  We can't use llama.cpp's exact
  -- thread layout AND emit pack2x16float of consecutive pairs in one step.
  -- Solution: use a thread layout where tid=0..127 is the "linear" element
  -- index (within block / 2), and read the appropriate ql/qh/scale bytes
  -- by deriving them from the *element* index instead.
  --
  -- Alternative used here: 128 threads, tid corresponds to element pair
  -- index `pairIdx = tid` (0..127), pair output = (2*tid, 2*tid+1).
  -- But we have only 64 threads.  So each thread does 2 pairs: pairIdx ∈
  -- {tid, tid + 64}.  4 elements per thread total.
  let _ : Exp (.scalar .u32) := blockByteBase  -- bind so it materialises

  ShaderM.varNamed "blockByteBase_v" (.scalar .u32) blockByteBase
  let bbb : Exp (.scalar .u32) := Exp.var "blockByteBase_v"

  -- Read d (fp16) once per thread (block-uniform). Goes into f32.
  let dBits ← ShaderM.readBufferU16 (n := totalWeightU32) "weights" (Exp.add bbb (Exp.litU32 208))
  let d := Exp.vecX (Exp.unpack2x16float dBits)
  ShaderM.varNamed "d_v" (.scalar .f32) d
  let dF : Exp (.scalar .f32) := Exp.var "d_v"

  -- Helper: read byte at (bbb + offset)
  let readByte (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    ShaderM.readBufferByte (n := totalWeightU32) "weights" (Exp.add bbb offset)

  -- Helper: read sign-extended i8 scale at (bbb + offset) as f32
  let readScaleF (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .f32)) := do
    let b ← readByte offset
    pure (Exp.select (Exp.ge b (Exp.litU32 128))
      (Exp.sub (Exp.toF32 b) (Exp.litF32 256.0))
      (Exp.toF32 b))

  -- Per-thread loop: do 2 pairs (pair0 at tid, pair1 at tid+64)
  for pairOffset in [0:2] do
    -- pairIdx = tid + 64 * pairOffset, which is in [0, 128)
    let pairIdx := Exp.add tid (Exp.litU32 (64 * pairOffset))
    -- Element indices in the block (256 elements per block):
    let e0 := Exp.mul pairIdx (Exp.litU32 2)
    let e1 := Exp.add e0 (Exp.litU32 1)

    -- For each element index e ∈ [0,256), the dequant follows llama.cpp's
    -- 2-chunk × 32-l × 4-output layout.  Map e back to (chunk, l, slot):
    --   chunk = e / 128       ∈ {0, 1}
    --   eInChunk = e % 128
    --   slot = eInChunk / 32  ∈ {0, 1, 2, 3}  (which of y[l+0,32,64,96])
    --   l    = eInChunk % 32  ∈ [0, 32)
    --
    -- Within the block, indices used:
    --   ql byte:    qlBaseOff = chunk * 64 + (slot ∈ {0,2}? l : l+32) + (slot ∈ {1,3}? +0 : ...)
    --     slot 0: ql[chunk*64 + l]            low nibble
    --     slot 1: ql[chunk*64 + l + 32]       low nibble
    --     slot 2: ql[chunk*64 + l]            high nibble
    --     slot 3: ql[chunk*64 + l + 32]       high nibble
    --   qh byte:    qhBaseOff = 128 + chunk * 32 + l, shifted by (slot * 2)
    --   scale idx:  192 + chunk * 8 + (l/16) + slot * 2
    --
    -- We compute deqW for both e0 and e1, then pack into half2.

    let dequantOne (e : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .f32)) := do
      let chunk    := Exp.shiftRight e (Exp.litU32 7)              -- e / 128
      let eInChunk := Exp.bitAnd e (Exp.litU32 0x7F)               -- e % 128
      let slot     := Exp.shiftRight eInChunk (Exp.litU32 5)       -- /32
      let l        := Exp.bitAnd eInChunk (Exp.litU32 0x1F)        -- %32
      let lOdd     := Exp.bitAnd slot (Exp.litU32 1)               -- 0/1: low or high nibble = high
      let lOddIs1  := Exp.eq lOdd (Exp.litU32 1)
      let slotIsHi := Exp.ge slot (Exp.litU32 2)                   -- slot∈{2,3}: high nibble
      -- ql byte index: chunk*64 + l + (slot odd? +32 : 0)
      let qlByteOff :=
        Exp.add (Exp.mul chunk (Exp.litU32 64))
          (Exp.add l (Exp.select lOddIs1 (Exp.litU32 32) (Exp.litU32 0)))
      let qlByte ← readByte qlByteOff
      -- qh byte: 128 + chunk*32 + l, shift by slot*2
      let qhByteOff :=
        Exp.add (Exp.litU32 128)
          (Exp.add (Exp.mul chunk (Exp.litU32 32)) l)
      let qhByte ← readByte qhByteOff
      let qhShift := Exp.mul slot (Exp.litU32 2)
      let qhBits  := Exp.bitAnd (Exp.shiftRight qhByte qhShift) (Exp.litU32 3)
      -- nibble: low 4 bits or high 4 bits depending on slotIsHi
      let qlNib :=
        Exp.select slotIsHi
          (Exp.shiftRight qlByte (Exp.litU32 4))
          (Exp.bitAnd qlByte (Exp.litU32 0xF))
      -- 6-bit raw: nib | (qhBits << 4)
      let q6Raw := Exp.bitOr qlNib (Exp.shiftLeft qhBits (Exp.litU32 4))
      -- signed: q6Raw - 32
      let q6 := Exp.sub (Exp.toF32 q6Raw) (Exp.litF32 32.0)
      -- scale: 192 + chunk*8 + (l/16) + slot*2
      let scOff :=
        Exp.add (Exp.litU32 192)
          (Exp.add (Exp.mul chunk (Exp.litU32 8))
            (Exp.add (Exp.shiftRight l (Exp.litU32 4))
              (Exp.mul slot (Exp.litU32 2))))
      let scF ← readScaleF scOff
      pure (Exp.mul dF (Exp.mul scF q6))

    let w0 ← dequantOne e0
    let w1 ← dequantOne e1
    let packed := Exp.pack2x16float (Exp.vec2 w0 w1)

    -- Output position: row n at u32 index n * (K/2) + pairIdx_within_row
    -- pairIdx_within_row = (blkInRow * 128) + pairIdx (since 256 elem/block = 128 u32 pairs)
    let rowU32Base := Exp.mul nIdx (Exp.litU32 (inDim / 2))
    let pairInRow  := Exp.add (Exp.mul blkInRowSafe (Exp.litU32 (blockSize / 2))) pairIdx
    let outU32Idx  := Exp.add rowU32Base pairInRow
    -- BOUNDS GUARD (critical): a 2D grid rounds workgroups UP (e.g. the lm_head's 65535×45 =
    -- 2,949,075 wgs for 2,883,584 blocks) — the ~65k EXCESS workgroups compute nIdx ≥ outDim and
    -- their writes all robustness-clamp onto the buffer's LAST u32 = the final vocab row's last
    -- f16 pair. 65k racing garbage writes → the last writer wins → when the garbage f16 is huge,
    -- EVERY decode position argmaxes to that tail <unused…> token (the <unused6226>×N failure),
    -- non-deterministically per run. Guard the write so excess workgroups write nothing.
    ShaderM.if_ (Exp.lt blockGlobalIdx (Exp.litU32 (outDim * blocksPerRow)))
      (ShaderM.writeBuffer (ty := .scalar .u32) "output" outU32Idx packed)
      (pure ())

end Hesper.Quantization.Q6_K
