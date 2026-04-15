import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

set_option maxRecDepth 2048

/-!
# Matmul microbenchmark — isolate per-call bottleneck

Same shape as Q4_K gate+up (inDim=2560, outDim=10240, blocks=10, 1 row/WG,
32 threads/WG, 10240 WGs).  Five kernels, each strictly more work than the
previous.  Timing gaps reveal which specific part is responsible.

Kernels:
  K1: pure weight stream — read all weight u32 and accumulate XOR.
      Represents the DRAM peak BW floor for this access pattern.
  K2: + q4 nibble extract + q4_K scale/min decode.
  K3: + dp4a (int dot with Q8_1 input).
  K4: + subgroup-reduce to lane 0.
  K5: + final f32 output write (reference: full fused kernel minus gate/up pairing).
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM

def blocksPerRow : Nat := 10           -- inDim=2560 / 256
def outDim : Nat := 10240
def totalWeightU32 : Nat := outDim * blocksPerRow * 36
def q8BlocksPerRow : Nat := 2560 / 32  -- 80
def q8InputU32Size : Nat := q8BlocksPerRow * 9  -- 720

/-- K1: pure weight stream.  Each WG reads one row's worth of Q4_K weights
    (10 blocks × 36 u32 = 360 u32 per row) and accumulates XOR into acc.
    Writes to output only at lane 0 to prevent DCE. -/
def k1Stream : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _w ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "acc" (.scalar .u32) (Exp.litU32 0)
  let acc : Exp (.scalar .u32) := Exp.var "acc"
  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 (blocksPerRow * 36 / 32)) (Exp.litU32 1) fun i => do
    -- Each lane reads one u32 per iteration, strided by 32.
    let idx := Exp.add rowBase (Exp.add (Exp.mul i (Exp.litU32 32)) tid)
    let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" idx
    ShaderM.assign "acc" (Exp.bitXor acc w)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx
      (Exp.mul (Exp.toF32 acc) (Exp.litF32 1e-9))
  ) (pure ())

/-- K2: K1 + q4 nibble extract + header decode.  Simulates the per-block
    bookkeeping done by Q4_K matmul, but no dp4a yet. -/
def k2StreamDequant : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _w ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun b => do
    let blockBase := Exp.add rowBase (Exp.mul b (Exp.litU32 36))
    let dm ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockBase
    let d := Exp.toF32 (Exp.bitAnd dm (Exp.litU32 0xFFFF))
    -- Each lane reads its own q4 nibble (32 lanes × 8 u32 per block = 256 u32 stride).
    let q4Idx := Exp.add blockBase (Exp.add (Exp.litU32 4) tid)
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4Idx
    let vLo := Exp.bitAnd v (Exp.litU32 0x0F0F0F0F)
    -- Fake FMA: acc += d * f32(vLo).
    ShaderM.assign "acc" (Exp.add acc (Exp.mul d (Exp.toF32 vLo)))
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx acc
  ) (pure ())

/-- K3: K2 + dp4a against Q8_1 input (no subgroup reduce). -/
def k3StreamDP4A : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _w ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _i ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "acc" (.scalar .i32) (Exp.toI32 (Exp.litU32 0))
  let acc : Exp (.scalar .i32) := Exp.var "acc"
  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun b => do
    let blockBase := Exp.add rowBase (Exp.mul b (Exp.litU32 36))
    let q4Idx := Exp.add blockBase (Exp.add (Exp.litU32 4) tid)
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4Idx
    let vLo := Exp.bitAnd v (Exp.litU32 0x0F0F0F0F)
    -- Read one Q8_1 u32 per lane.
    let q8Idx := Exp.add (Exp.mul b (Exp.litU32 9)) (Exp.add (Exp.litU32 1) (Exp.bitAnd tid (Exp.litU32 7)))
    let u ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Idx
    let dot := Exp.dot4I8Packed vLo u
    ShaderM.assign "acc" (Exp.add acc dot)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.toF32 acc)
  ) (pure ())

/-- fp16 → f32 bit pattern conversion (copy of Linear.lean helper). -/
private def fp16ToF32Bench (bits : Exp (.scalar .u32)) : Exp (.scalar .f32) :=
  let sign := Exp.shiftLeft (Exp.bitAnd (Exp.shiftRight bits (Exp.litU32 15)) (Exp.litU32 1)) (Exp.litU32 31)
  let exp5 := Exp.bitAnd (Exp.shiftRight bits (Exp.litU32 10)) (Exp.litU32 0x1F)
  let mant10 := Exp.bitAnd bits (Exp.litU32 0x3FF)
  let expAdj := Exp.add exp5 (Exp.litU32 (127 - 15))
  let exp8 := Exp.shiftLeft expAdj (Exp.litU32 23)
  let mant23 := Exp.shiftLeft mant10 (Exp.litU32 13)
  let f32bits := Exp.bitOr (Exp.bitOr sign exp8) mant23
  let isZero := Exp.eq exp5 (Exp.litU32 0)
  Exp.bitcast (Exp.select isZero (Exp.litU32 0) f32bits)

/-- K6: full fused gate+up simulation matching `fusedQ4KMGateUpDP4AKernel`
    closely — QR4_K=2 two-pass nibble extraction, full d/dmin decode, scale/min
    bit-unpack, signed dp4a, duplicate-work ×0.5, subgroup reduction, GELU(gate)
    × up.  Should converge to the real kernel's time if the kernel is already
    near-optimal.

    Dispatch unchanged from K5 (outDim WGs, 32 threads) so occupancy matches. -/
def k6FullFusedGateUp : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _wg ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _wu ← ShaderM.declareReadOnlyBuffer "weights_up" (.array (.scalar .u32) totalWeightU32)
  let _i ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"

  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  -- Lane decomposition matching Linear.lean:1052.
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun b => do
    let blockBase := Exp.add rowBase (Exp.mul b (Exp.litU32 36))
    let processWeight (which : String) (accName : String) (acc : Exp (.scalar .f32)) : ShaderM Unit := do
      -- Per-block header: d (fp16), dmin (fp16).
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockBase
      let dF := fp16ToF32Bench (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
      let dminF := fp16ToF32Bench (Exp.shiftRight dmU32 (Exp.litU32 16))
      -- Scales/mins packed in u32[1..3].
      let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockBase (Exp.litU32 1))
      let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockBase (Exp.litU32 2))
      let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockBase (Exp.litU32 3))
      let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
        let isLow := Exp.lt is (Exp.litU32 4)
        let shift4 := Exp.mul is (Exp.litU32 8)
        let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
        let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
        let isHi := Exp.sub is (Exp.litU32 4)
        let shiftHi := Exp.mul isHi (Exp.litU32 8)
        let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
        let scaleHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
        let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
        let minHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let minHigh := Exp.bitOr minHiLo minHiHi
        let scaleU := Exp.select isLow scaleLow scaleHigh
        let minU   := Exp.select isLow minLow   minHigh
        (Exp.toF32 scaleU, Exp.toF32 minU)
      let (scA, mA) := extractScaleMin bq8Off
      let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

      -- q4 nibbles: v0 and v1 = q4[0] and q4[4].
      let q4BaseIdx := Exp.add blockBase
        (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
      let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which q4BaseIdx
      let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add q4BaseIdx (Exp.litU32 4))

      -- Q8_1 block headers + quants.
      let q8Sub0Base := Exp.add (Exp.mul b (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
      let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
      let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
        (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
      let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
        (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
      let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
        (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
      let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
        (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
      let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
      let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
      let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
      let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

      -- i=0: low nibbles.
      let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
      let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
      let dot0 := Exp.add (Exp.dot4I8Packed v0i0 u0) (Exp.dot4I8Packed v1i0 u1)
      let sumU0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                           (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
      let sumfD0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot0) scA)
      let sumfM0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU0) mA)
      -- i=1: high nibbles.
      let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let dot1 := Exp.add (Exp.dot4I8Packed v0i1 u2) (Exp.dot4I8Packed v1i1 u3)
      let sumU1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                           (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
      let sumfD1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1) scB)
      let sumfM1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU1) mB)
      let blockSumfD := Exp.add sumfD0 sumfD1
      let blockSumfM := Exp.add sumfM0 sumfM1
      let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
      ShaderM.assign accName (Exp.add acc blockContrib)

    processWeight "weights_gate" "accG" accG
    processWeight "weights_up" "accU" accU

  -- Subgroup reduce with ×0.5 duplicate-work correction.
  ShaderM.varNamed "totalG" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accG) (Exp.litF32 0.5))
  ShaderM.varNamed "totalU" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accU) (Exp.litF32 0.5))
  let totalG : Exp (.scalar .f32) := Exp.var "totalG"
  let totalU : Exp (.scalar .f32) := Exp.var "totalU"
  -- GELU(tanh) × up, matching the real kernel's epilogue.
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let z := totalG
    let z3 := Exp.mul (Exp.mul z z) z
    let inner := Exp.mul sqrt2OverPi (Exp.add z (Exp.mul (Exp.litF32 0.044715) z3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) z) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul gelu totalU)
  ) (pure ())

/-- K5: simulates fused gate+up — 2 weight buffers, both read per WG.
    Per call reads `2 × totalWeightU32 × 4` bytes.  If the gap between K4
    and the real fused gate+up kernel (124 µs) is just "2× the weight",
    K5 should land near 66 µs on the cold-L2 path. -/
def k5FusedGateUpSim : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _wg ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _wu ← ShaderM.declareReadOnlyBuffer "weights_up" (.array (.scalar .u32) totalWeightU32)
  let _i ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"
  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun b => do
    let blockBase := Exp.add rowBase (Exp.mul b (Exp.litU32 36))
    let q4Idx := Exp.add blockBase (Exp.add (Exp.litU32 4) tid)
    let vg ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights_gate" q4Idx
    let vu ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights_up" q4Idx
    let q8Idx := Exp.add (Exp.mul b (Exp.litU32 9)) (Exp.add (Exp.litU32 1) (Exp.bitAnd tid (Exp.litU32 7)))
    let u ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Idx
    let dotG := Exp.dot4I8Packed (Exp.bitAnd vg (Exp.litU32 0x0F0F0F0F)) u
    let dotU := Exp.dot4I8Packed (Exp.bitAnd vu (Exp.litU32 0x0F0F0F0F)) u
    ShaderM.assign "accG" (Exp.add accG (Exp.toF32 dotG))
    ShaderM.assign "accU" (Exp.add accU (Exp.toF32 dotU))
  ShaderM.varNamed "totalG" (.scalar .f32) (Exp.subgroupAdd accG)
  ShaderM.varNamed "totalU" (.scalar .f32) (Exp.subgroupAdd accU)
  let totalG : Exp (.scalar .f32) := Exp.var "totalG"
  let totalU : Exp (.scalar .f32) := Exp.var "totalU"
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul totalG totalU)
  ) (pure ())

/-- K4: K3 + subgroupAdd reduction. -/
def k4StreamDP4AReduce : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let _w ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _i ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let rowBase := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun b => do
    let blockBase := Exp.add rowBase (Exp.mul b (Exp.litU32 36))
    let q4Idx := Exp.add blockBase (Exp.add (Exp.litU32 4) tid)
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4Idx
    let vLo := Exp.bitAnd v (Exp.litU32 0x0F0F0F0F)
    let q8Idx := Exp.add (Exp.mul b (Exp.litU32 9)) (Exp.add (Exp.litU32 1) (Exp.bitAnd tid (Exp.litU32 7)))
    let u ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Idx
    let dot := Exp.dot4I8Packed vLo u
    ShaderM.assign "acc" (Exp.add acc (Exp.toF32 dot))
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)

def benchKernel (ctx : CUDAContext) (name : String) (shader : ShaderM Unit)
    (buffers : List (String × CUDABuffer)) (syncBuf : CUDABuffer)
    (numWorkgroups : Nat × Nat × Nat) (workgroupX : Nat) : IO Unit := do
  let config : Hesper.ExecConfig := {
    numWorkgroups, workgroupSize := { x := workgroupX, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  -- Warmup (establishes L2 residency, caches PTX).
  for _ in List.range 20 do
    GPUBackend.execute ctx shader buffers config
  -- Force a sync so the warmup cost doesn't leak into the timed region.
  let _ ← Hesper.CUDA.cuMemcpyDtoH syncBuf.ptr 4
  let runs := 200
  let t0 ← IO.monoNanosNow
  for _ in List.range runs do
    GPUBackend.execute ctx shader buffers config
  -- Force GPU sync before reading timer.
  let _ ← Hesper.CUDA.cuMemcpyDtoH syncBuf.ptr 4
  let t1 ← IO.monoNanosNow
  let perCallUs : Float := (t1 - t0).toFloat / (runs.toFloat * 1000.0)
  -- Compute achieved BW: weights = outDim * blocksPerRow * 36 u32 = per kernel read.
  -- K1-K3 each read this amount.  Reduction kernels do the same.
  let weightBytes : Float := (outDim * blocksPerRow * 36 * 4).toFloat
  let bwGBs : Float := weightBytes / (perCallUs * 1e-6) / 1e9
  IO.println s!"  {name}: {perCallUs} µs/call, {bwGBs} GB/s (weight only)"

/-- Same as `benchKernel` but rotates through N distinct weight buffers
    so each call touches previously-uncached memory.  This exposes the
    *real* BW the kernel achieves when L2 is cold (which is what the full
    model inference actually experiences as it walks 42 layers). -/
def benchKernelMultiBuf (ctx : CUDAContext) (name : String) (shader : ShaderM Unit)
    (wBufs : Array CUDABuffer) (q8Buf oBuf : CUDABuffer) (withQ8 : Bool)
    (numWorkgroups : Nat × Nat × Nat) (workgroupX : Nat) : IO Unit := do
  let config : Hesper.ExecConfig := {
    numWorkgroups, workgroupSize := { x := workgroupX, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  let mkBufs (w : CUDABuffer) : List (String × CUDABuffer) :=
    if withQ8 then [("weights", w), ("input_q8", q8Buf), ("output", oBuf)]
    else [("weights", w), ("output", oBuf)]
  -- Warmup.
  for _ in List.range 5 do
    for w in wBufs do
      GPUBackend.execute ctx shader (mkBufs w) config
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  -- Timed: rotate through buffers to evict L2 between calls.
  let iters := 200
  let t0 ← IO.monoNanosNow
  let wBufsList := wBufs.toList
  for i in List.range iters do
    match wBufsList[i % wBufsList.length]? with
    | some w => GPUBackend.execute ctx shader (mkBufs w) config
    | none => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t1 ← IO.monoNanosNow
  let perCallUs : Float := (t1 - t0).toFloat / (iters.toFloat * 1000.0)
  let weightBytes : Float := (outDim * blocksPerRow * 36 * 4).toFloat
  let bwGBs : Float := weightBytes / (perCallUs * 1e-6) / 1e9
  IO.println s!"  {name}: {perCallUs} µs/call, {bwGBs} GB/s (weight only)"

/-- K7: hand-written PTX exercising `fma.rn.f16x2` for the compute
    part of a matmul-style kernel.  Same work shape as K1/K4 (one row per
    WG, 32 threads, 10 blocks × 32 u32-wide inner loop), but replaces
    each scalar `dot4I8Packed + FMA` with a chain of `fma.rn.f16x2`.

    Purpose: directly answer "does the hardware's packed fp16 FMA help?"
    If K7 beats K4 (~33 µs cold L2) on the same BW budget, it's worth
    adding the intrinsic to the DSL.  If not, the kernel is already
    BW-bound and fp16x2 is irrelevant.

    Signature: void k7(u32 *weights, u32 *q8, f32 *output)
    Dispatch: (10240, 1, 1) × 32 threads (same as K1-K6). -/
def k7Ptx : String := "//
.version 8.0
.target sm_89
.address_size 64

.entry k7(
    .param .u64 param_weights,
    .param .u64 param_q8,
    .param .u64 param_output
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<10>;
    .reg .b32 %hx<40>;     // packed f16x2 as b32
    .reg .b16 %h<8>;       // single f16 halves
    .reg .f32 %f<8>;
    .reg .pred %p<2>;

    // Load params.
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_q8];
    ld.param.u64 %rd2, [param_output];

    // rowBase = ctaid.x * 360 (= blocksPerRow * 36 = 10 * 36).
    mov.u32 %r0, %ctaid.x;           // row idx
    mul.lo.u32 %r1, %r0, 360;        // rowBase u32 index
    mov.u32 %r2, %tid.x;             // lane

    // Initial f16x2 accumulator = {0, 0}.
    mov.b32 %hx0, 0;
    mov.b32 %hx1, 0;
    mov.b32 %hx2, 0;
    mov.b32 %hx3, 0;

    // Two initial scale f16x2 (arbitrary, just need non-zero for compute).
    // For bench purposes, use small constants so fp16 doesn't overflow.
    mov.b32 %hx10, 0x3c003c00;      // (1.0, 1.0) in f16x2
    mov.b32 %hx11, 0x3c003c00;

    // Loop over 10 blocks.
    mov.u32 %r3, 0;                  // block counter
    mov.u32 %r4, 10;
LOOP:
    setp.lt.u32 %p0, %r3, %r4;
    @!%p0 bra END;

    // blockBase = rowBase + block * 36.
    mul.lo.u32 %r5, %r3, 36;
    add.u32 %r6, %r1, %r5;
    // Lane's u32 offset inside the block = 4 + lane.
    add.u32 %r7, %r6, 4;
    add.u32 %r8, %r7, %r2;
    // Load weight u32 (we'll reinterpret as two f16x2 = 4 f16).
    mul.wide.u32 %rd3, %r8, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.nc.u32 %r9, [%rd4];
    // Pack into two f16x2 registers (upper/lower halves).
    mov.b32 %hx20, %r9;              // use low 16 bits as f16x2 pair (bogus values,
    mov.b32 %hx21, %r9;              // just to keep data on the critical path)

    // 8 chained fma.rn.f16x2 to simulate the dequant+dp compute density.
    // Each fma: acc = fma(a * b + acc).  Packed f16x2 → 2 FMAs per instruction.
    fma.rn.f16x2 %hx0, %hx20, %hx10, %hx0;
    fma.rn.f16x2 %hx1, %hx21, %hx11, %hx1;
    fma.rn.f16x2 %hx2, %hx20, %hx11, %hx2;
    fma.rn.f16x2 %hx3, %hx21, %hx10, %hx3;
    fma.rn.f16x2 %hx0, %hx20, %hx10, %hx0;
    fma.rn.f16x2 %hx1, %hx21, %hx11, %hx1;
    fma.rn.f16x2 %hx2, %hx20, %hx11, %hx2;
    fma.rn.f16x2 %hx3, %hx21, %hx10, %hx3;

    add.u32 %r3, %r3, 1;
    bra LOOP;
END:
    // Reduce 4 f16x2 accumulators to one f32 so we have something to write
    // (prevents DCE).  mov.b32 unpacks the b32 into two b16 halves.
    mov.b32 {%h0, %h1}, %hx0;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    add.f32 %f2, %f0, %f1;
    mov.b32 {%h2, %h3}, %hx1;
    cvt.f32.f16 %f3, %h2;
    add.f32 %f4, %f2, %f3;
    cvt.f32.f16 %f5, %h3;
    add.f32 %f6, %f4, %f5;
    // Write output[ctaid.x] = f6 only from lane 0.
    setp.eq.u32 %p1, %r2, 0;
    @!%p1 bra SKIP;
    mul.wide.u32 %rd5, %r0, 4;
    add.u64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f6;
SKIP:
    ret;
}
"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════"
  IO.println " Q4_K matmul microbenchmark (inDim=2560, outDim=10240)"
  IO.println "════════════════════════════════════════════════════════"
  let ctx ← CUDAContext.init
  let weightBytes : USize := (totalWeightU32 * 4).toUSize
  let q8Bytes : USize := (q8InputU32Size * 4).toUSize
  let outBytes : USize := (outDim * 4).toUSize
  let wBuf ← GPUBackend.allocBuffer ctx weightBytes
  let q8Buf ← GPUBackend.allocBuffer ctx q8Bytes
  let oBuf ← GPUBackend.allocBuffer ctx outBytes
  IO.println s!"Weight size: {weightBytes / 1024 / 1024} MB"
  IO.println s!"Q8_1 input size: {q8Bytes / 1024} KB"
  IO.println s!"Output size: {outBytes / 1024} KB"
  IO.println ""
  IO.println "── Single-buffer (warm L2) ─────────────────────────────"
  benchKernel ctx "K1 pure weight stream        " k1Stream
    [("weights", wBuf), ("output", oBuf)] oBuf (outDim, 1, 1) 32
  benchKernel ctx "K2 + dequant (d + nibble)    " k2StreamDequant
    [("weights", wBuf), ("output", oBuf)] oBuf (outDim, 1, 1) 32
  benchKernel ctx "K3 + dp4a int dot           " k3StreamDP4A
    [("weights", wBuf), ("input_q8", q8Buf), ("output", oBuf)] oBuf (outDim, 1, 1) 32
  benchKernel ctx "K4 + subgroup reduce        " k4StreamDP4AReduce
    [("weights", wBuf), ("input_q8", q8Buf), ("output", oBuf)] oBuf (outDim, 1, 1) 32

  -- Multi-buffer variant: allocate enough separate weight buffers to fill
  -- well past the 48 MB L2 cache, rotating through them so each call sees
  -- cold weights.  Mirrors what full inference experiences (42 layers).
  IO.println ""
  IO.println "── Multi-buffer (cold L2, 8 × 14 MB = 112 MB rotation) ─"
  let numBufs := 8
  let mut wBufs : Array CUDABuffer := #[]
  for _ in List.range numBufs do
    wBufs := wBufs.push (← GPUBackend.allocBuffer ctx weightBytes)
  benchKernelMultiBuf ctx "K1 pure weight stream        " k1Stream
    wBufs q8Buf oBuf false (outDim, 1, 1) 32
  benchKernelMultiBuf ctx "K2 + dequant (d + nibble)    " k2StreamDequant
    wBufs q8Buf oBuf false (outDim, 1, 1) 32
  benchKernelMultiBuf ctx "K3 + dp4a int dot           " k3StreamDP4A
    wBufs q8Buf oBuf true (outDim, 1, 1) 32
  benchKernelMultiBuf ctx "K4 + subgroup reduce        " k4StreamDP4AReduce
    wBufs q8Buf oBuf true (outDim, 1, 1) 32

  -- K5: 2 weight buffers per call, as the fused gate+up kernel does.
  -- Rotate through pairs so every call is cold for BOTH weights.
  IO.println ""
  IO.println "── K5 fused-gate+up sim (2 weight buffers, cold L2) ────"
  let numPairs : Nat := 4
  let mut wGates : Array CUDABuffer := #[]
  let mut wUps : Array CUDABuffer := #[]
  for _ in List.range numPairs do
    wGates := wGates.push (← GPUBackend.allocBuffer ctx weightBytes)
    wUps := wUps.push (← GPUBackend.allocBuffer ctx weightBytes)
  let config : Hesper.ExecConfig := {
    numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  -- Warmup.
  for _ in List.range 5 do
    for p in List.range numPairs do
      match wGates.toList[p]?, wUps.toList[p]? with
      | some wg, some wu =>
        GPUBackend.execute ctx k5FusedGateUpSim
          [("weights_gate", wg), ("weights_up", wu), ("input_q8", q8Buf), ("output", oBuf)]
          config
      | _, _ => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let iters := 200
  let t0 ← IO.monoNanosNow
  let wgList := wGates.toList
  let wuList := wUps.toList
  for i in List.range iters do
    let idx := i % numPairs
    match wgList[idx]?, wuList[idx]? with
    | some wg, some wu =>
      GPUBackend.execute ctx k5FusedGateUpSim
        [("weights_gate", wg), ("weights_up", wu), ("input_q8", q8Buf), ("output", oBuf)]
        config
    | _, _ => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t1 ← IO.monoNanosNow
  let perCallUs : Float := (t1 - t0).toFloat / (iters.toFloat * 1000.0)
  let totalWB : Float := (2 * totalWeightU32 * 4).toFloat
  let bwGBs : Float := totalWB / (perCallUs * 1e-6) / 1e9
  IO.println s!"  K5 2-buffer fused gate+up    : {perCallUs} µs/call, {bwGBs} GB/s (both weights)"

  -- K6: full fused kernel simulation (Q4_K dequant + QR4_K=2 + GELU).
  -- Compares against the real kernel at ~124 µs — if close, the algorithm
  -- is already near-optimal and gains must come from a different approach.
  let t2 ← IO.monoNanosNow
  for i in List.range iters do
    let idx := i % numPairs
    match wgList[idx]?, wuList[idx]? with
    | some wg, some wu =>
      GPUBackend.execute ctx k6FullFusedGateUp
        [("weights_gate", wg), ("weights_up", wu), ("input_q8", q8Buf), ("output", oBuf)]
        config
    | _, _ => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t3 ← IO.monoNanosNow
  let perCallUs6 : Float := (t3 - t2).toFloat / (iters.toFloat * 1000.0)
  let bwGBs6 : Float := totalWB / (perCallUs6 * 1e-6) / 1e9
  IO.println s!"  K6 full fused (Q4_K+QR4+GELU): {perCallUs6} µs/call, {bwGBs6} GB/s (both weights)"

  -- K6b: same as K6 but via `executeWithConfigCached`.  The gap vs K6
  -- (312 µs) tells us how much of K6's overhead is PTX JIT + buffer
  -- lookup in the non-cached path.
  IO.println ""
  IO.println "── K6b full fused via cached dispatch (cold L2) ────────"
  let k6bRef ← GPUBackend.newCacheRef (β := CUDAContext)
  let cacheKey : UInt64 := 0x6b6b6b6b
  -- Warmup through the cached path (populates the prepared dispatch).
  for _ in List.range 5 do
    for p in List.range numPairs do
      match wgList[p]?, wuList[p]? with
      | some wg, some wu =>
        GPUBackend.executeWithConfigCached ctx k6FullFusedGateUp
          [("weights_gate", wg), ("weights_up", wu), ("input_q8", q8Buf), ("output", oBuf)]
          config cacheKey k6bRef
      | _, _ => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t6 ← IO.monoNanosNow
  for i in List.range iters do
    let idx := i % numPairs
    match wgList[idx]?, wuList[idx]? with
    | some wg, some wu =>
      GPUBackend.executeWithConfigCached ctx k6FullFusedGateUp
        [("weights_gate", wg), ("weights_up", wu), ("input_q8", q8Buf), ("output", oBuf)]
        config cacheKey k6bRef
    | _, _ => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t7 ← IO.monoNanosNow
  let perCallUs6b : Float := (t7 - t6).toFloat / (iters.toFloat * 1000.0)
  let bwGBs6b : Float := totalWB / (perCallUs6b * 1e-6) / 1e9
  IO.println s!"  K6b cached dispatch         : {perCallUs6b} µs/call, {bwGBs6b} GB/s"

  -- K7: raw-PTX fma.rn.f16x2 benchmark.  Compiles the hand-written PTX,
  -- runs it with the same dispatch as K1-K4, and reports the time.
  -- If fp16x2 packed FMA actually helps throughput, this should be faster
  -- than a matching f32-FMA version even without matmul semantics.
  IO.println ""
  IO.println "── K7 raw-PTX fma.rn.f16x2 (cold L2) ───────────────────"
  let mod ← Hesper.CUDA.cuModuleLoadData k7Ptx
  let func ← Hesper.CUDA.cuModuleGetFunction mod "k7"
  let k7Config : Hesper.ExecConfig := {
    numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
  }
  let args : Array USize := #[]
  -- Warmup with multi-buffer rotation to eliminate L2 caching.
  for _ in List.range 5 do
    for wRaw in wBufs do
      Hesper.CUDA.cuLaunchKernel func
        outDim.toUInt32 1 1 32 1 1 0
        #[wRaw.ptr, q8Buf.ptr, oBuf.ptr]
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let wBufList := wBufs.toList
  let iters7 := 200
  let t4 ← IO.monoNanosNow
  for i in List.range iters7 do
    match wBufList[i % wBufList.length]? with
    | some w =>
      Hesper.CUDA.cuLaunchKernel func
        outDim.toUInt32 1 1 32 1 1 0
        #[w.ptr, q8Buf.ptr, oBuf.ptr]
    | none => pure ()
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t5 ← IO.monoNanosNow
  let _ := args
  let perCallUs7 : Float := (t5 - t4).toFloat / (iters7.toFloat * 1000.0)
  let wB : Float := (outDim * blocksPerRow * 36 * 4).toFloat
  let bw7 : Float := wB / (perCallUs7 * 1e-6) / 1e9
  IO.println s!"  K7 fp16x2 FMA (raw PTX)      : {perCallUs7} µs/call, {bw7} GB/s (weight only)"
  IO.println "    (K4 reference for same BW: ~33 µs cold L2, 444 GB/s)"

  IO.println ""
  IO.println "Reference: real fusedQ4KMGateUpDP4AKernel ≈ 124 µs/call (1-row)."
  IO.println "           real fusedQ4KMGateUpDP4A4RowKernel ≈ 124 µs/call (4-row)."
