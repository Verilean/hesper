import Hesper.Backend
import Hesper.Backend.CUDA

/-!
# `if`-guarded `readBuffer` + smem broadcast regression test

The Q4_K Option A rewrite (smem broadcast of scale/dmin/qs) hung the
GPU at runtime even though the PTX dump looked correct.  This test
isolates the suspected pattern: lane 0 of a warp does an
`if (laneId == 0)` guarded `readBuffer` from global memory, writes to
shared memory, then a workgroup barrier, then all 32 lanes read back
from smem.

Expected behaviour:
  - All 32 output entries equal the value at `input[7]` (= 0xCAFEBABE).
  - Kernel returns within ~1 second.

Failure modes to detect:
  - Hang at the barrier (some lanes never reach it under divergence).
  - Garbage in output (smem write happened but barrier didn't sync).
  - Some lanes read stale 0 (lane 0 read with stale predicate).

Run: `lake exe cuda-if-guarded-rb-test`
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)

private def packU32s (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun (acc : ByteArray) (n : UInt32) =>
    acc.push n.toUInt8 |>.push (n>>>8).toUInt8
       |>.push (n>>>16).toUInt8 |>.push (n>>>24).toUInt8

private def unpackU32 (ba : ByteArray) (i : Nat) : UInt32 :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

/-- 1 WG, 32 threads.  Mirrors the structure used in
    `fusedQ4KMLinearDP4A4WarpKernel` (post-Option-A): lane 0 reads
    from global, writes to smem; barrier; all lanes read smem. -/
def ifGuardedKernel : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .u32) 8)
  let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 32)
  ShaderM.sharedNamed "s_value" (.array (.scalar .u32) 1)

  let lid ← ShaderM.localId
  let laneId := Exp.vec3X lid

  -- Lane 0 reads from input[7] and writes to s_value[0].
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 8) "input" (Exp.litU32 7)
    ShaderM.writeWorkgroup (ty := .scalar .u32) "s_value" (Exp.litU32 0) v
  ) (pure ())
  ShaderM.barrier

  -- All 32 lanes read s_value[0] and write to output[laneId].
  let v ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 1) "s_value" (Exp.litU32 0)
  ShaderM.writeBuffer (ty := .scalar .u32) "output" laneId v

/-- Phase 2: 128-thread WG, 8 half-warps each broadcasting their own slot.
    Mirrors fusedQ4KMLinearDP4A4WarpKernel exactly: warpId * 2 +
    (laneId / 16) → halfWarpId in 0..7, each half-warp's lane 0
    reads input[halfWarpId] and broadcasts via smem. -/
def ifGuardedKernel_128 : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .u32) 8)
  let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 128)
  ShaderM.sharedNamed "s_value" (.array (.scalar .u32) 8)

  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let warpId := Exp.shiftRight tid (Exp.litU32 5)        -- 0..3
  let laneId := Exp.bitAnd tid (Exp.litU32 31)           -- 0..31
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)          -- 0..15 (lane within half-warp)
  let halfWarpId := Exp.add (Exp.mul warpId (Exp.litU32 2))
                            (Exp.shiftRight laneId (Exp.litU32 4))  -- 0..7

  -- Lane 0 of each half-warp reads input[halfWarpId] and writes
  -- to s_value[halfWarpId].
  ShaderM.if_ (Exp.eq laneLow (Exp.litU32 0)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 8) "input" halfWarpId
    ShaderM.writeWorkgroup (ty := .scalar .u32) "s_value" halfWarpId v
  ) (pure ())
  ShaderM.barrier

  -- All 128 threads read s_value[halfWarpId] and write to output[tid].
  let v ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 8) "s_value" halfWarpId
  ShaderM.writeBuffer (ty := .scalar .u32) "output" tid v

def runIfGuarded128Test [GPUBackend β] (ctx : β) : IO Bool := do
  -- 8 distinct values, one per half-warp.
  let inputBytes := packU32s #[0xAA000000, 0xBB000001, 0xCC000002, 0xDD000003,
                               0xEE000004, 0xFF000005, 0x11000006, 0x22000007]
  let inputBuf ← GPUBackend.allocBuffer ctx (32 : USize)
  GPUBackend.writeBuffer ctx inputBuf inputBytes

  let outBuf ← GPUBackend.allocBuffer ctx (128 * 4 : USize)
  let zeros := packU32s (Array.replicate 128 (0xDEADBEEF : UInt32))
  GPUBackend.writeBuffer ctx outBuf zeros

  IO.println "  Launching 128-thread kernel (1 WG × 128 threads)..."
  GPUBackend.execute ctx ifGuardedKernel_128
    [("input", inputBuf), ("output", outBuf)]
    ({ workgroupSize := { x := 128, y := 1, z := 1 },
       numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
  IO.println "  Kernel returned, reading back..."

  let bytes ← GPUBackend.readBuffer ctx outBuf (128 * 4)

  let expected : Array UInt32 := #[0xAA000000, 0xBB000001, 0xCC000002, 0xDD000003,
                                   0xEE000004, 0xFF000005, 0x11000006, 0x22000007]
  let mut allOk := true
  let mut errCount := 0
  for tid in [0:128] do
    let warpId := tid / 32
    let laneId := tid % 32
    let halfWarpId := warpId * 2 + laneId / 16
    let exp := expected[halfWarpId]!
    let got := unpackU32 bytes tid
    if got != exp then
      if errCount < 5 then
        IO.println s!"  ✗ output[{tid}] (halfWarp={halfWarpId}) = 0x{Nat.toDigits 16 got.toNat |>.asString}  (expected 0x{Nat.toDigits 16 exp.toNat |>.asString})"
      errCount := errCount + 1
      allOk := false
  if allOk then
    IO.println "  ✓ All 128 threads read correct broadcast values from smem"
  else
    IO.println s!"  ✗ {errCount} mismatches"
  return allOk

def runIfGuardedTest [GPUBackend β] (ctx : β) : IO Bool := do
  -- Input: 8 u32 values, last one is the broadcast target.
  let inputBytes := packU32s #[0x00000000, 0x11111111, 0x22222222, 0x33333333,
                               0x44444444, 0x55555555, 0x66666666, 0xCAFEBABE]
  let inputBuf ← GPUBackend.allocBuffer ctx (32 : USize)
  GPUBackend.writeBuffer ctx inputBuf inputBytes

  let outBuf ← GPUBackend.allocBuffer ctx (32 * 4 : USize)
  -- Zero the output buffer to detect "lane never wrote".
  let zeros := packU32s (Array.replicate 32 (0xDEADBEEF : UInt32))
  GPUBackend.writeBuffer ctx outBuf zeros

  IO.println "  Launching kernel (1 WG × 32 threads)..."
  GPUBackend.execute ctx ifGuardedKernel
    [("input", inputBuf), ("output", outBuf)]
    ({ workgroupSize := { x := 32, y := 1, z := 1 },
       numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
  IO.println "  Kernel returned, reading back..."

  let bytes ← GPUBackend.readBuffer ctx outBuf (32 * 4)

  let expected : UInt32 := 0xCAFEBABE
  let mut allOk := true
  for i in [0:32] do
    let v := unpackU32 bytes i
    if v != expected then
      IO.println s!"  ✗ output[{i}] = 0x{Nat.toDigits 16 v.toNat |>.asString}  (expected 0x{Nat.toDigits 16 expected.toNat |>.asString})"
      allOk := false
  if allOk then
    IO.println s!"  ✓ All 32 lanes read 0x{Nat.toDigits 16 expected.toNat |>.asString} from smem"
  return allOk

/-- Phase 3: like Phase 2, but smem-write is itself nested inside an
    outer `ShaderM.if_` (matching the `if blockInRange` guard in the
    real Q4_K kernel).  This is the configuration that hung. -/
def ifGuardedKernel_nestedIf : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .u32) 8)
  let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 128)
  ShaderM.sharedNamed "s_value" (.array (.scalar .u32) 8)

  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let halfWarpId := Exp.add (Exp.mul warpId (Exp.litU32 2))
                            (Exp.shiftRight laneId (Exp.litU32 4))

  -- Outer if: always-true predicate (so all 128 threads enter).
  -- Models the real kernel's `if blockInRange` which is true for the
  -- in-range case.  Inside that, lane 0 of each half-warp does the
  -- guarded readBuffer + writeWorkgroup, then barrier, then everyone
  -- reads.
  --
  -- The barrier is INSIDE the outer if.  This matches the bug-suspect
  -- configuration: the suspicion was that putting `barrier` inside a
  -- divergent (or even uniformly-true) outer-if breaks workgroup-level
  -- synchronisation.
  let alwaysTrue := Exp.eq (Exp.litU32 1) (Exp.litU32 1)
  ShaderM.if_ alwaysTrue (do
    ShaderM.if_ (Exp.eq laneLow (Exp.litU32 0)) (do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 8) "input" halfWarpId
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_value" halfWarpId v
    ) (pure ())
    ShaderM.barrier
    let v ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 8) "s_value" halfWarpId
    ShaderM.writeBuffer (ty := .scalar .u32) "output" tid v
  ) (pure ())

def runIfGuardedNestedTest [GPUBackend β] (ctx : β) : IO Bool := do
  let inputBytes := packU32s #[0xAA000000, 0xBB000001, 0xCC000002, 0xDD000003,
                               0xEE000004, 0xFF000005, 0x11000006, 0x22000007]
  let inputBuf ← GPUBackend.allocBuffer ctx (32 : USize)
  GPUBackend.writeBuffer ctx inputBuf inputBytes

  let outBuf ← GPUBackend.allocBuffer ctx (128 * 4 : USize)
  let zeros := packU32s (Array.replicate 128 (0xDEADBEEF : UInt32))
  GPUBackend.writeBuffer ctx outBuf zeros

  IO.println "  Launching nested-if kernel (1 WG × 128 threads)..."
  GPUBackend.execute ctx ifGuardedKernel_nestedIf
    [("input", inputBuf), ("output", outBuf)]
    ({ workgroupSize := { x := 128, y := 1, z := 1 },
       numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
  IO.println "  Kernel returned, reading back..."

  let bytes ← GPUBackend.readBuffer ctx outBuf (128 * 4)
  let expected : Array UInt32 := #[0xAA000000, 0xBB000001, 0xCC000002, 0xDD000003,
                                   0xEE000004, 0xFF000005, 0x11000006, 0x22000007]
  let mut allOk := true
  let mut errCount := 0
  for tid in [0:128] do
    let warpId := tid / 32
    let laneId := tid % 32
    let halfWarpId := warpId * 2 + laneId / 16
    let exp := expected[halfWarpId]!
    let got := unpackU32 bytes tid
    if got != exp then
      if errCount < 5 then
        IO.println s!"  ✗ output[{tid}] (halfWarp={halfWarpId}) = 0x{Nat.toDigits 16 got.toNat |>.asString}  (expected 0x{Nat.toDigits 16 exp.toNat |>.asString})"
      errCount := errCount + 1
      allOk := false
  if allOk then
    IO.println "  ✓ All 128 threads OK with barrier inside outer-if"
  return allOk

/-- Phase 4: K-loop with predicate-guarded smem broadcast inside.  This
    matches the real Q4_K kernel: outer `for kbx` loop over 2 iters,
    each iter does `if blockInRange (do { lane-0-write-to-smem; barrier;
    everyone-read })`. -/
def ifGuardedKernel_loop : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .u32) 16)
  let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 128)
  ShaderM.sharedNamed "s_value" (.array (.scalar .u32) 8)

  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let halfWarpId := Exp.add (Exp.mul warpId (Exp.litU32 2))
                            (Exp.shiftRight laneId (Exp.litU32 4))

  -- Accumulator: sum of values read across both iters.
  ShaderM.varNamed "acc" (.scalar .u32) (Exp.litU32 0)
  let acc : Exp (.scalar .u32) := Exp.var "acc"

  -- Outer loop: 2 iterations.
  let blocksPerRow : Nat := 10  -- match Q4_K K=2560 case
  let maxIter : Nat := (blocksPerRow + 7) / 8
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 maxIter) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add halfWarpId (Exp.mul iter (Exp.litU32 8))
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    ShaderM.if_ blockInRange (do
      ShaderM.if_ (Exp.eq laneLow (Exp.litU32 0)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 16) "input" blockIdx
        ShaderM.writeWorkgroup (ty := .scalar .u32) "s_value" halfWarpId v
      ) (pure ())
      ShaderM.barrier
      let v ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 8) "s_value" halfWarpId
      ShaderM.assign "acc" (Exp.add acc v)
    ) (pure ())

  -- Write final acc to output[tid].
  ShaderM.writeBuffer (ty := .scalar .u32) "output" tid acc

def runIfGuardedLoopTest [GPUBackend β] (ctx : β) : IO Bool := do
  -- 16 input values: input[i] = i+1 so the sum over [halfWarpId, halfWarpId+8]
  -- for halfWarpId in 0..7 gives a predictable result.
  let inputBytes := packU32s (Array.range 16 |>.map (fun (i : Nat) => (i + 1).toUInt32))
  let inputBuf ← GPUBackend.allocBuffer ctx (16 * 4 : USize)
  GPUBackend.writeBuffer ctx inputBuf inputBytes

  let outBuf ← GPUBackend.allocBuffer ctx (128 * 4 : USize)
  let zeros := packU32s (Array.replicate 128 (0xDEADBEEF : UInt32))
  GPUBackend.writeBuffer ctx outBuf zeros

  IO.println "  Launching loop kernel (1 WG × 128 threads, 2-iter outer loop)..."
  GPUBackend.execute ctx ifGuardedKernel_loop
    [("input", inputBuf), ("output", outBuf)]
    ({ workgroupSize := { x := 128, y := 1, z := 1 },
       numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
  IO.println "  Kernel returned, reading back..."

  let bytes ← GPUBackend.readBuffer ctx outBuf (128 * 4)

  -- Expected acc per thread: input[halfWarpId] + input[halfWarpId+8]
  -- but only if both blockIdx < 10.  iter 0: blockIdx = halfWarpId (0..7),
  -- always in range.  iter 1: blockIdx = halfWarpId + 8 (8..15), in range
  -- only for halfWarpId < 2 (i.e. blockIdx 8, 9).  So:
  --   halfWarp 0: input[0] + input[8] = 1 + 9 = 10
  --   halfWarp 1: input[1] + input[9] = 2 + 10 = 12
  --   halfWarp 2: input[2]            = 3
  --   ...
  --   halfWarp 7: input[7]            = 8
  let mut allOk := true
  let mut errCount := 0
  for tid in [0:128] do
    let warpId := tid / 32
    let laneId := tid % 32
    let halfWarpId := warpId * 2 + laneId / 16
    let v0 := halfWarpId + 1
    let v1 := if halfWarpId + 8 < 10 then halfWarpId + 8 + 1 else 0
    let exp : UInt32 := (v0 + v1).toUInt32
    let got := unpackU32 bytes tid
    if got != exp then
      if errCount < 5 then
        IO.println s!"  ✗ output[{tid}] (halfWarp={halfWarpId}) = 0x{Nat.toDigits 16 got.toNat |>.asString}  (expected 0x{Nat.toDigits 16 exp.toNat |>.asString})"
      errCount := errCount + 1
      allOk := false
  if allOk then
    IO.println "  ✓ All 128 threads OK with K-loop + barrier inside if-guard"
  return allOk

def main : IO UInt32 := do
  IO.println "═══ if-guarded readBuffer + smem broadcast test ═══"
  let ctx ← CUDAContext.init
  IO.println "─── Phase 1: 1 warp × 32 threads ───"
  let ok1 ← runIfGuardedTest ctx
  IO.println "─── Phase 2: 4 warps × 32 threads, 8 half-warps ───"
  let ok2 ← runIfGuarded128Test ctx
  IO.println "─── Phase 3: nested if (outer always-true) + barrier inside ───"
  let ok3 ← runIfGuardedNestedTest ctx
  IO.println "─── Phase 4: K-loop + if blockInRange + barrier inside ───"
  let ok4 ← runIfGuardedLoopTest ctx
  if ok1 && ok2 && ok3 && ok4 then
    IO.println "✓ PASS — all patterns work"
    return 0
  else
    IO.println s!"✗ FAIL — Phase 1: {ok1}, Phase 2: {ok2}, Phase 3: {ok3}, Phase 4: {ok4}"
    return 1
