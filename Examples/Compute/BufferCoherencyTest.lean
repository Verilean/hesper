import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.WGSL.Monad

open Hesper.WebGPU
abbrev SM := Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.Monad.ShaderM (declareOutputBuffer declareReadOnlyBuffer globalId localId readBuffer writeBuffer if_ sharedNamed writeWorkgroup readWorkgroup barrier)
open Hesper.WGSL (Exp)

namespace Examples.Compute.BufferCoherencyTest

/-- A: data[i] = i+1 -/
def writeK (n : Nat) : SM Unit := do
  let _o ← declareOutputBuffer "data" (.array (.scalar .f32) n)
  let gid ← globalId; let i := Exp.vec3X gid
  if_ (Exp.lt i (Exp.litU32 n)) (do
    writeBuffer (ty := .scalar .f32) "data" i (Exp.toF32 (Exp.add i (Exp.litU32 1)))) (pure ())

/-- B-direct: out[i] = src[i] * 2 -/
def readDirectK (n : Nat) : SM Unit := do
  let _s ← declareReadOnlyBuffer "src" (.array (.scalar .f32) n)
  let _o ← declareOutputBuffer "out" (.array (.scalar .f32) n)
  let gid ← globalId; let i := Exp.vec3X gid
  if_ (Exp.lt i (Exp.litU32 n)) (do
    let v ← readBuffer (ty := .scalar .f32) (n := n) "src" i
    writeBuffer (ty := .scalar .f32) "out" i (Exp.mul v (Exp.litF32 2.0))) (pure ())

/-- B-indirect (gather): out[i] = src[idx[i]] * 2 -/
def readIndirectK (n : Nat) : SM Unit := do
  let _s ← declareReadOnlyBuffer "src" (.array (.scalar .f32) n)
  let _idx ← declareReadOnlyBuffer "idx" (.array (.scalar .u32) n)
  let _o ← declareOutputBuffer "out" (.array (.scalar .f32) n)
  let gid ← globalId; let i := Exp.vec3X gid
  if_ (Exp.lt i (Exp.litU32 n)) (do
    let j ← readBuffer (ty := .scalar .u32) (n := n) "idx" i
    let v ← readBuffer (ty := .scalar .f32) (n := n) "src" j
    writeBuffer (ty := .scalar .f32) "out" i (Exp.mul v (Exp.litF32 2.0))) (pure ())

/-- in-place (like q80): data[i] += 100, reads AND writes the same buffer -/
def inplaceK (n : Nat) : SM Unit := do
  let _d ← declareOutputBuffer "data" (.array (.scalar .f32) n)
  let gid ← globalId; let i := Exp.vec3X gid
  if_ (Exp.lt i (Exp.litU32 n)) (do
    let v ← readBuffer (ty := .scalar .f32) (n := n) "data" i
    writeBuffer (ty := .scalar .f32) "data" i (Exp.add v (Exp.litF32 100.0))) (pure ())

/-- shared-mem + barrier (the wcache pattern): out[i] = (sh[tid]=src[i]; barrier; sh[tid])*2 -/
def sharedK (n : Nat) : SM Unit := do
  let _s ← declareReadOnlyBuffer "src" (.array (.scalar .f32) n)
  let _o ← declareOutputBuffer "out" (.array (.scalar .f32) n)
  sharedNamed "sh" (.array (.scalar .f32) 256)
  let lid ← localId; let tid := Exp.vec3X lid
  let gid ← globalId; let i := Exp.vec3X gid
  let v ← readBuffer (ty := .scalar .f32) (n := n) "src" i
  writeWorkgroup (ty := .scalar .f32) "sh" tid v
  barrier
  let w ← readWorkgroup (ty := .scalar .f32) (n := 256) "sh" tid
  writeBuffer (ty := .scalar .f32) "out" i (Exp.mul w (Exp.litF32 2.0))

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

def disp (device : Device) (k : SM Unit) (bufs : List (String × Buffer)) (n : Nat) (key : UInt64) : IO Unit := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device k bufs
    { numWorkgroups := ((n+255)/256, 1, 1), workgroupSize := {x:=256} } key r

def main : IO Unit := do
  IO.println "=== Buffer coherency repro: kernel A writes buf, kernel B reads it (same batch) ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  let n := 1024
  let bufA ← mkBuf device n
  let idx ← mkBuf device n
  let out ← mkBuf device n
  -- idx[i] = i (identity gather)
  let mut idxBytes : ByteArray := ByteArray.empty
  for i in [0:n] do
    idxBytes := idxBytes.push (i &&& 0xFF).toUInt8
    idxBytes := idxBytes.push ((i >>> 8) &&& 0xFF).toUInt8
    idxBytes := idxBytes.push ((i >>> 16) &&& 0xFF).toUInt8
    idxBytes := idxBytes.push ((i >>> 24) &&& 0xFF).toUInt8
  writeBuffer device idx 0 idxBytes
  let bW : List (String × Buffer) := [("data", bufA)]
  let bRD : List (String × Buffer) := [("src", bufA), ("out", out)]
  let bRI : List (String × Buffer) := [("src", bufA), ("idx", idx), ("out", out)]
  let check (label : String) (arr : Array Float) : IO Unit := do
    let mut ok := 0; let mut nz := 0
    for i in [0:n] do
      let exp := ((i+1)*2 : Nat).toFloat
      if (arr.getD i 0.0 - exp).abs < 0.5 then ok := ok+1
      if arr.getD i 0.0 != 0.0 then nz := nz+1
    let verdict := if ok==n then "✅ coherent" else if nz==0 then "❌ B saw 0 (A→B not flushed)" else "❌ wrong"
    IO.println s!"  [{label}] {ok}/{n} correct, {nz}/{n} non-zero → {verdict}"
  -- Test 1: DIRECT, both kernels in ONE batch (no sync between A and B)
  Hesper.GPUBackend.beginBatch device
  disp device (writeK n) bW n 1
  disp device (readDirectK n) bRD n 2
  Hesper.GPUBackend.endBatch device
  check "direct, 1 batch" (← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize))
  -- Test 2: INDIRECT (gather), both in ONE batch
  Hesper.GPUBackend.beginBatch device
  disp device (writeK n) bW n 3
  disp device (readIndirectK n) bRI n 4
  Hesper.GPUBackend.endBatch device
  check "indirect/gather, 1 batch" (← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize))
  -- Test 3: DIRECT, A and B in SEPARATE batches (sync between)
  Hesper.GPUBackend.beginBatch device
  disp device (writeK n) bW n 5
  Hesper.GPUBackend.endBatch device
  Hesper.GPUBackend.beginBatch device
  disp device (readDirectK n) bRD n 6
  Hesper.GPUBackend.endBatch device
  check "direct, 2 batches" (← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize))
  -- Test 4: INDIRECT, A and B in SEPARATE batches
  Hesper.GPUBackend.beginBatch device
  disp device (writeK n) bW n 7
  Hesper.GPUBackend.endBatch device
  Hesper.GPUBackend.beginBatch device
  disp device (readIndirectK n) bRI n 8
  Hesper.GPUBackend.endBatch device
  check "indirect/gather, 2 batches" (← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize))
  -- Test 5: 3-kernel chain with IN-PLACE middle (A write → B in-place += 100 → C read*2), one batch
  -- expect out[i] = (i+1+100)*2
  let check5 (arr : Array Float) : IO Unit := do
    let mut ok := 0; let mut nz := 0
    for i in [0:n] do
      if (arr.getD i 0.0 - ((i+101)*2 : Nat).toFloat).abs < 0.5 then ok := ok+1
      if arr.getD i 0.0 != 0.0 then nz := nz+1
    let v := if ok==n then "✅ coherent" else if nz==0 then "❌ saw 0" else "❌ wrong"
    IO.println s!"  [3-chain in-place, 1 batch] {ok}/{n} correct, {nz} non-zero → {v}"
  Hesper.GPUBackend.beginBatch device
  disp device (writeK n) bW n 9
  disp device (inplaceK n) bW n 10
  disp device (readDirectK n) bRD n 11
  Hesper.GPUBackend.endBatch device
  check5 (← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize))
  -- Test 6: run the 1-batch indirect repro 20× to catch NON-determinism
  let mut fails := 0
  for t in [0:20] do
    Hesper.GPUBackend.beginBatch device
    disp device (writeK n) bW n (1000+t.toUInt64*2)
    disp device (readIndirectK n) bRI n (1001+t.toUInt64*2)
    Hesper.GPUBackend.endBatch device
    let a ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize)
    let mut ok := 0
    for i in [0:n] do if (a.getD i 0.0 - ((i+1)*2 : Nat).toFloat).abs < 0.5 then ok := ok+1
    if ok != n then fails := fails+1
  let v6 := if fails==0 then "✅ stable" else "❌ non-deterministic"
  IO.println s!"  [indirect ×20 non-determinism] {fails}/20 runs failed → {v6}"
  -- Test 7: shared-mem + barrier kernel (wcache pattern), A→sharedK in one batch, ×20
  let mut sf := 0
  for t in [0:20] do
    Hesper.GPUBackend.beginBatch device
    disp device (writeK n) bW n (2000+t.toUInt64*2)
    disp device (sharedK n) bRD n (2001+t.toUInt64*2)
    Hesper.GPUBackend.endBatch device
    let a ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device out 0 (n*4).toUSize)
    let mut ok := 0
    for i in [0:n] do if (a.getD i 0.0 - ((i+1)*2 : Nat).toFloat).abs < 0.5 then ok := ok+1
    if ok != n then sf := sf+1
  let v7 := if sf==0 then "✅ stable" else "❌ non-deterministic (shared-mem race!)"
  IO.println s!"  [shared-mem+barrier ×20] {sf}/20 runs failed → {v7}"

end Examples.Compute.BufferCoherencyTest

def main : IO Unit := Examples.Compute.BufferCoherencyTest.main
