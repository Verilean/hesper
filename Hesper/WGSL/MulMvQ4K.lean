import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Monad

/-!
# Q4_K × f32 matvec — faithful port of llama.cpp Metal `kernel_mul_mv_q4_K_f32`

DEVPLAN 原則 7: the reference structure is ported WHOLE and placed at the CENTER of the
autotune family space (`NR0=2, NSG=2` = llama's shipping config); the sweep explores around it.

Structure (from ggml-metal.metal:7839, N_R0_Q4_K=2, N_SG_Q4_K=2):
- **f32 input** — no Q8_1 quantize dispatch at all (unlike our dp4a path, which also relies
  on `dot4I8Packed` that Apple GPUs emulate — no dp4a hardware).
- Each SIMDGROUP owns `NR0` consecutive rows; a threadgroup has `NSG` simdgroups →
  `NSG*NR0` rows per TG, `32*NSG` threads.
- Thread layout inside a simdgroup: `ix = tiisg/8` (4 blocks in flight), `it = tiisg%8`,
  `iq = it/4` (which 64-element half of the low 128), `ir = it%4` (8-element slice).
- Per block-iteration each thread loads 32 f32 activations (yl/yh) ONCE and reuses them for
  ALL `NR0` rows (register-level input reuse — the key trick our 1-row kernels lack).
- Weights are read as u16 lanes and multiplied MASKED (`y * f32(q & 0x000F)` etc. with
  1/256 and 1/16 corrections) — no nibble unpacking on the critical path.
- Scales: 3 u16 reads + kmask bit tricks → 8 scale/min bytes (llama's sc16/sc8 scheme).
- Reduction: `subgroupAdd` per row, lane 0 writes. No cross-simdgroup traffic.

Q4_K block = 144 B = 36 u32: w0 = d|dmin (f16×2), w1..w3 = scales[12], w4..w35 = qs[128].
-/

namespace Hesper.WGSL.MulMvQ4K

open Hesper.WGSL
open Hesper.WGSL.Monad

private def u32lit (n : Nat) : Exp (.scalar .u32) := Exp.litU32 n
private def f32lit (x : Float) : Exp (.scalar .f32) := Exp.litF32 x

private instance : Inhabited (Exp (.scalar .f32)) := ⟨Exp.litF32 0.0⟩
private instance : Inhabited (Exp (.scalar .u32)) := ⟨Exp.litU32 0⟩

/-- llama.cpp-structure Q4_K×f32 matvec.

    `K` cols (mult of 256), `N` rows. `nr0` rows per simdgroup, `nsg` simdgroups per TG.
    Dispatch: `numWorkgroups = (ceil(N/(nsg*nr0)), 1, 1)` (or 2D via `gridXWidth` when that
    exceeds 65535), `workgroupSize = 32*nsg`, extensions = ["subgroups"].

    Buffers: `weights` (u32-packed Q4_K, row-major), `input` (f32[K]), `output` (f32[N]). -/
def mulMvQ4KF32Kernel (K N : Nat) (nr0 nsg : Nat) (gridXWidth : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid                                   -- 0 .. 32*nsg-1
  let nb := K / 256                                          -- blocks per row
  let totalWeightU32 := N * nb * 36
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input   ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) K)
  let _output  ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) N)

  let wgIdx := if gridXWidth == 0 then Exp.vec3X wid
               else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (u32lit gridXWidth))
  -- simdgroup id + lane id (subgroup size 32 with linear-local alignment — the assumption
  -- every production hesper subgroup kernel already relies on, Metal-validated)
  let sgIdName ← ShaderM.var (.scalar .u32) (Exp.shiftRight tid (u32lit 5))
  let laneName ← ShaderM.var (.scalar .u32) (Exp.bitAnd tid (u32lit 31))
  let sgId : Exp (.scalar .u32) := Exp.var sgIdName
  let lane : Exp (.scalar .u32) := Exp.var laneName
  -- llama thread layout
  let ixName ← ShaderM.var (.scalar .u32) (Exp.shiftRight lane (u32lit 3))          -- 0..3
  let itName ← ShaderM.var (.scalar .u32) (Exp.bitAnd lane (u32lit 7))               -- 0..7
  let ix : Exp (.scalar .u32) := Exp.var ixName
  let it : Exp (.scalar .u32) := Exp.var itName
  let iqName ← ShaderM.var (.scalar .u32) (Exp.shiftRight it (u32lit 2))             -- 0..1
  let irName ← ShaderM.var (.scalar .u32) (Exp.bitAnd it (u32lit 3))                 -- 0..3
  let iq : Exp (.scalar .u32) := Exp.var iqName
  let ir : Exp (.scalar .u32) := Exp.var irName

  let firstRowName ← ShaderM.var (.scalar .u32)
    (Exp.mul (Exp.add (Exp.mul wgIdx (u32lit nsg)) sgId) (u32lit nr0))
  let firstRow : Exp (.scalar .u32) := Exp.var firstRowName

  -- y slice base (element index): 64*iq + 8*ir; the block offset ib*256 is added per iter.
  let ySliceName ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iq (u32lit 64)) (Exp.mul ir (u32lit 8)))
  let ySlice : Exp (.scalar .u32) := Exp.var ySliceName
  -- scale/q byte offsets within a block (see header comment): u16-lane positions
  --   s0 at byte 4+2iq, s2 at 8+2iq, s4 at 12+2iq  (all within w1..w3, offset 0 or 2)
  let scShiftName ← ShaderM.var (.scalar .u32) (Exp.mul (Exp.bitAnd iq (u32lit 1)) (u32lit 16))
  let scShift : Exp (.scalar .u32) := Exp.var scShiftName
  -- q1 u32 index within block: (16 + 32*iq + 8*ir)/4 = 4 + 8*iq + 2*ir
  let qWordName ← ShaderM.var (.scalar .u32)
    (Exp.add (u32lit 4) (Exp.add (Exp.mul iq (u32lit 8)) (Exp.mul ir (u32lit 2))))
  let qWord : Exp (.scalar .u32) := Exp.var qWordName

  -- per-row accumulators
  for r in [0:nr0] do
    ShaderM.varNamed s!"sumf{r}" (.scalar .f32) (f32lit 0.0)

  -- Block loop: ib = ix; ib < nb; ib += 4  (trip count differs per ix — no subgroup ops inside)
  ShaderM.loop ix (u32lit nb) (u32lit 4) fun ib => do
    -- ── load the 32 activations of this thread's slice, once, + partial sums ──
    let yBaseName ← ShaderM.var (.scalar .u32) (Exp.add (Exp.mul ib (u32lit 256)) ySlice)
    let yBase : Exp (.scalar .u32) := Exp.var yBaseName
    let mut yl : Array (Exp (.scalar .f32)) := #[]
    let mut yh : Array (Exp (.scalar .f32)) := #[]
    for i in [0:8] do
      let n ← ShaderM.var (.scalar .f32)
        (← ShaderM.readBuffer (ty := .scalar .f32) (n := K) "input" (Exp.add yBase (u32lit i)))
      yl := yl.push (Exp.var n)
    for i in [0:8] do
      let n ← ShaderM.var (.scalar .f32)
        (← ShaderM.readBuffer (ty := .scalar .f32) (n := K) "input" (Exp.add yBase (u32lit (32+i))))
      yl := yl.push (Exp.var n)
    for i in [0:8] do
      let n ← ShaderM.var (.scalar .f32)
        (← ShaderM.readBuffer (ty := .scalar .f32) (n := K) "input" (Exp.add yBase (u32lit (128+i))))
      yh := yh.push (Exp.var n)
    for i in [0:8] do
      let n ← ShaderM.var (.scalar .f32)
        (← ShaderM.readBuffer (ty := .scalar .f32) (n := K) "input" (Exp.add yBase (u32lit (160+i))))
      yh := yh.push (Exp.var n)
    let sum8 (a : Array (Exp (.scalar .f32))) (o : Nat) : Exp (.scalar .f32) :=
      (List.range 8).foldl (fun acc i => Exp.add acc a[o+i]!) (f32lit 0.0)
    let sumy0N ← ShaderM.var (.scalar .f32) (sum8 yl 0)
    let sumy1N ← ShaderM.var (.scalar .f32) (sum8 yl 8)
    let sumy2N ← ShaderM.var (.scalar .f32) (sum8 yh 0)
    let sumy3N ← ShaderM.var (.scalar .f32) (sum8 yh 8)
    let sumy0 : Exp (.scalar .f32) := Exp.var sumy0N
    let sumy1 : Exp (.scalar .f32) := Exp.var sumy1N
    let sumy2 : Exp (.scalar .f32) := Exp.var sumy2N
    let sumy3 : Exp (.scalar .f32) := Exp.var sumy3N

    -- ── per row: scales + masked-multiply accumulation (yl/yh reused from registers) ──
    for r in [0:nr0] do
      let blockBaseName ← ShaderM.var (.scalar .u32)
        (Exp.mul (Exp.add (Exp.mul (Exp.add firstRow (u32lit r)) (u32lit nb)) ib) (u32lit 36))
      let blockBase : Exp (.scalar .u32) := Exp.var blockBaseName
      let rd := fun (off : Exp (.scalar .u32)) =>
        ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockBase off)
      -- d / dmin
      let w0 ← rd (u32lit 0)
      let dN ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float w0))
      let dminN ← ShaderM.var (.scalar .f32) (Exp.vecY (Exp.unpack2x16float w0))
      let dF : Exp (.scalar .f32) := Exp.var dN
      let dminF : Exp (.scalar .f32) := Exp.var dminN
      -- scales u16 lanes (byte pos 4+2iq / 8+2iq / 12+2iq → w1/w2/w3, shift 16*iq)
      let w1 ← rd (u32lit 1)
      let w2 ← rd (u32lit 2)
      let w3 ← rd (u32lit 3)
      let s0N ← ShaderM.var (.scalar .u32) (Exp.bitAnd (Exp.shiftRight w1 scShift) (u32lit 0xFFFF))
      let s2N ← ShaderM.var (.scalar .u32) (Exp.bitAnd (Exp.shiftRight w2 scShift) (u32lit 0xFFFF))
      let s4N ← ShaderM.var (.scalar .u32) (Exp.bitAnd (Exp.shiftRight w3 scShift) (u32lit 0xFFFF))
      let s0 : Exp (.scalar .u32) := Exp.var s0N
      let s2 : Exp (.scalar .u32) := Exp.var s2N
      let s4 : Exp (.scalar .u32) := Exp.var s4N
      -- kmask trick → sc16[0..3]
      let sc16_0N ← ShaderM.var (.scalar .u32) (Exp.bitAnd s0 (u32lit 0x3f3f))
      let sc16_1N ← ShaderM.var (.scalar .u32) (Exp.bitAnd s2 (u32lit 0x3f3f))
      let sc16_2N ← ShaderM.var (.scalar .u32)
        (Exp.bitOr (Exp.bitAnd s4 (u32lit 0x0f0f))
                   (Exp.shiftRight (Exp.bitAnd s0 (u32lit 0xc0c0)) (u32lit 2)))
      let sc16_3N ← ShaderM.var (.scalar .u32)
        (Exp.bitOr (Exp.bitAnd (Exp.shiftRight s4 (u32lit 4)) (u32lit 0x0f0f))
                   (Exp.shiftRight (Exp.bitAnd s2 (u32lit 0xc0c0)) (u32lit 2)))
      let sc16_0 : Exp (.scalar .u32) := Exp.var sc16_0N
      let sc16_1 : Exp (.scalar .u32) := Exp.var sc16_1N
      let sc16_2 : Exp (.scalar .u32) := Exp.var sc16_2N
      let sc16_3 : Exp (.scalar .u32) := Exp.var sc16_3N
      let byteLo (w : Exp (.scalar .u32)) : Exp (.scalar .f32) := Exp.toF32U (Exp.bitAnd w (u32lit 0xFF))
      let byteHi (w : Exp (.scalar .u32)) : Exp (.scalar .f32) := Exp.toF32U (Exp.shiftRight w (u32lit 8))
      let sc8_0 := byteLo sc16_0; let sc8_1 := byteHi sc16_0
      let sc8_2 := byteLo sc16_1; let sc8_3 := byteHi sc16_1
      let sc8_4 := byteLo sc16_2; let sc8_5 := byteHi sc16_2
      let sc8_6 := byteLo sc16_3; let sc8_7 := byteHi sc16_3
      -- q1 (2 u32 = 4 u16) + q2 (= q1 + 64 B = +16 u32)
      let qA ← rd qWord
      let qB ← rd (Exp.add qWord (u32lit 1))
      let qC ← rd (Exp.add qWord (u32lit 16))
      let qD ← rd (Exp.add qWord (u32lit 17))
      let qAN ← ShaderM.var (.scalar .u32) qA
      let qBN ← ShaderM.var (.scalar .u32) qB
      let qCN ← ShaderM.var (.scalar .u32) qC
      let qDN ← ShaderM.var (.scalar .u32) qD
      let q1 : Array (Exp (.scalar .u32)) := #[
        Exp.bitAnd (Exp.var qAN) (u32lit 0xFFFF), Exp.shiftRight (Exp.var qAN) (u32lit 16),
        Exp.bitAnd (Exp.var qBN) (u32lit 0xFFFF), Exp.shiftRight (Exp.var qBN) (u32lit 16)]
      let q2 : Array (Exp (.scalar .u32)) := #[
        Exp.bitAnd (Exp.var qCN) (u32lit 0xFFFF), Exp.shiftRight (Exp.var qCN) (u32lit 16),
        Exp.bitAnd (Exp.var qDN) (u32lit 0xFFFF), Exp.shiftRight (Exp.var qDN) (u32lit 16)]
      -- masked multiplies (llama FOR_UNROLL i=0..3), built as pure expression trees
      let mask (q : Exp (.scalar .u32)) (m : Nat) : Exp (.scalar .f32) :=
        Exp.toF32U (Exp.bitAnd q (u32lit m))
      let accOf (qs : Array (Exp (.scalar .u32))) (ya : Array (Exp (.scalar .f32))) (m : Nat) (yoff : Nat) : Exp (.scalar .f32) :=
        (List.range 4).foldl (fun acc i =>
          Exp.fma ya[yoff + 2*i]! (mask qs[i]! m) acc) (f32lit 0.0)
      let acc1_0 := accOf q1 yl 0x000F 0
      let acc1_1 := (List.range 4).foldl (fun acc i =>
          Exp.fma yl[2*i+1]! (mask q1[i]! 0x0F00) acc) (f32lit 0.0)
      let acc1_2 := accOf q1 yl 0x00F0 8
      let acc1_3 := (List.range 4).foldl (fun acc i =>
          Exp.fma yl[2*i+9]! (mask q1[i]! 0xF000) acc) (f32lit 0.0)
      let acc2_0 := accOf q2 yh 0x000F 0
      let acc2_1 := (List.range 4).foldl (fun acc i =>
          Exp.fma yh[2*i+1]! (mask q2[i]! 0x0F00) acc) (f32lit 0.0)
      let acc2_2 := accOf q2 yh 0x00F0 8
      let acc2_3 := (List.range 4).foldl (fun acc i =>
          Exp.fma yh[2*i+9]! (mask q2[i]! 0xF000) acc) (f32lit 0.0)
      let inv256 := f32lit (1.0/256.0)
      let inv16  := f32lit (1.0/16.0)
      let dTerm :=
        Exp.add
          (Exp.add (Exp.mul (Exp.fma acc1_1 inv256 acc1_0) sc8_0)
                   (Exp.mul (Exp.mul (Exp.fma acc1_3 inv256 acc1_2) sc8_1) inv16))
          (Exp.add (Exp.mul (Exp.fma acc2_1 inv256 acc2_0) sc8_4)
                   (Exp.mul (Exp.mul (Exp.fma acc2_3 inv256 acc2_2) sc8_5) inv16))
      let mTerm :=
        Exp.add (Exp.add (Exp.mul sumy0 sc8_2) (Exp.mul sumy1 sc8_3))
                (Exp.add (Exp.mul sumy2 sc8_6) (Exp.mul sumy3 sc8_7))
      let prev : Exp (.scalar .f32) := Exp.var s!"sumf{r}"
      ShaderM.assign s!"sumf{r}"
        (Exp.fma dF dTerm (Exp.fma (Exp.neg dminF) mTerm prev))

  -- ── reduction: subgroupAdd per row, lane 0 writes ──
  for r in [0:nr0] do
    ShaderM.varNamed s!"rowTotal{r}" (.scalar .f32) (Exp.subgroupAdd (Exp.var s!"sumf{r}"))
  ShaderM.if_ (Exp.eq lane (u32lit 0)) (do
    for r in [0:nr0] do
      let row := Exp.add firstRow (u32lit r)
      ShaderM.if_ (Exp.lt row (u32lit N)) (do
        ShaderM.writeBuffer (ty := .scalar .f32) "output" row (Exp.var s!"rowTotal{r}")
      ) (pure ())
  ) (pure ())

end Hesper.WGSL.MulMvQ4K
