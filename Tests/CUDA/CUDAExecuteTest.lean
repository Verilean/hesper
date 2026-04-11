import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.Execute
import Hesper.WGSL.Monad

/-!
# CUDA End-to-End Execution Test

Tests the full pipeline: ShaderM → PTX → cuModuleLoadData → cuLaunchKernel.
Runs simple kernels on the GPU and verifies results.
-/

open Hesper.CUDA
open Hesper.CUDA.Execute
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM

/-- Convert Float (f64) to IEEE 754 f32 bits. -/
private def float64ToFloat32Bits (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 := (bits64 >>> 63) &&& 1
  let exp64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 := bits64 &&& 0x000FFFFFFFFFFFFF
  if exp64 == 0 then (0 : UInt32)
  else if exp64 == 0x7FF then
    (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
  else
    let exp32val : Int := exp64.toNat - 1023 + 127
    if exp32val <= 0 then (0 : UInt32)
    else if exp32val >= 255 then (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (sign64.toUInt32 <<< 31) ||| (exp32val.toNat.toUInt32 <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def floatArrayToBytes (arr : Array Float) : ByteArray :=
  arr.foldl (fun (acc : ByteArray) (f : Float) =>
    let bits := float64ToFloat32Bits f
    acc.push bits.toUInt8
       |>.push (bits >>> 8).toUInt8
       |>.push (bits >>> 16).toUInt8
       |>.push (bits >>> 24).toUInt8
  ) ByteArray.empty

private def bytesToFloatArray (bytes : ByteArray) : Array Float :=
  let numFloats := bytes.size / 4
  Array.range numFloats |>.map fun i =>
    let offset := i * 4
    let b0 := bytes.get! offset
    let b1 := bytes.get! (offset + 1)
    let b2 := bytes.get! (offset + 2)
    let b3 := bytes.get! (offset + 3)
    let bits : UInt32 := b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
    -- f32 bits → f64 Float
    let sign32 := (bits >>> 31) &&& 1
    let exp32 := (bits >>> 23) &&& 0xFF
    let mant32 := bits &&& 0x7FFFFF
    if exp32 == 0 then 0.0
    else if exp32 == 0xFF then if mant32 == 0 then (if sign32 == 1 then -1.0/0.0 else 1.0/0.0) else 0.0/0.0
    else
      let exp64 : UInt64 := (exp32.toUInt64 - 127 + 1023)
      let mant64 : UInt64 := mant32.toUInt64 <<< 29
      let sign64 : UInt64 := sign32.toUInt64 <<< 63
      Float.ofBits (sign64 ||| (exp64 <<< 52) ||| mant64)

/-- Simple kernel: output[i] = input[i] * 2.0 -/
def vectorDoubleKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vec3X gid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  writeBuffer (ty := .scalar .f32) "output" idx result

/-- Vector add kernel: output[i] = a[i] + b[i] -/
def vectorAddKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vec3X gid

  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1024)
  let _b ← declareInputBuffer "b" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  let va ← readBuffer (ty := .scalar .f32) (n := 1024) "a" idx
  let vb ← readBuffer (ty := .scalar .f32) (n := 1024) "b" idx
  let result := Exp.add va vb
  writeBuffer (ty := .scalar .f32) "output" idx result

def main : IO Unit := do
  IO.println "═══ CUDA End-to-End Execution Test ═══"
  IO.println ""

  -- Initialize CUDA
  IO.println "[Init] Initializing CUDA..."
  let (dev, _ctx) ← initCUDA

  let cc ← cuComputeCapability dev
  IO.println s!"[Init] Compute capability: sm_{cc}"
  IO.println ""

  -- ═══ Test 1: Vector Double ═══
  IO.println "Test 1: Vector Double (output = input * 2)"
  IO.println "───────────────────────────────────────────"

  let n : Nat := 1024
  let bufSize : USize := (n * 4).toUSize

  -- Create input data: [1.0, 2.0, 3.0, ..., 1024.0]
  let inputArr := Array.range n |>.map (fun i => Float.ofNat (i + 1))
  let inputData := floatArrayToBytes inputArr

  -- Allocate GPU buffers
  let inputBuf ← createCUDABuffer bufSize
  let outputBuf ← createCUDABuffer bufSize

  -- Upload input
  writeCUDABuffer inputBuf inputData

  -- Execute kernel
  let config := CUDAExecutionConfig.dispatch1D n 256
  executeShaderCUDA vectorDoubleKernel
    [("input", inputBuf), ("output", outputBuf)]
    config

  -- Read back results
  let resultBytes ← readCUDABufferFull outputBuf
  let result := bytesToFloatArray resultBytes

  -- Verify
  let mut passed := true
  for i in [0, 1, 2, 3, 1022, 1023] do
    let got := result.getD i 0.0
    let expected := Float.ofNat ((i + 1) * 2)
    if (got - expected).abs > 0.01 then
      IO.println s!"  ✗ output[{i}] = {got}, expected {expected}"
      passed := false
    else
      IO.println s!"  ✓ output[{i}] = {got} (expected {expected})"

  freeCUDABuffer inputBuf
  freeCUDABuffer outputBuf

  if passed then
    IO.println "  → Test 1 PASSED"
  else
    IO.println "  → Test 1 FAILED"
  IO.println ""

  -- ═══ Test 2: Vector Add ═══
  IO.println "Test 2: Vector Add (output = a + b)"
  IO.println "────────────────────────────────────"

  let aBuf ← createCUDABuffer bufSize
  let bBuf ← createCUDABuffer bufSize
  let outBuf ← createCUDABuffer bufSize

  let aArr := Array.range n |>.map (fun i => Float.ofNat (i + 1))
  let bArr := Array.range n |>.map (fun i => Float.ofNat ((i + 1) * 100))

  writeCUDABuffer aBuf (floatArrayToBytes aArr)
  writeCUDABuffer bBuf (floatArrayToBytes bArr)

  executeShaderCUDA vectorAddKernel
    [("a", aBuf), ("b", bBuf), ("output", outBuf)]
    (CUDAExecutionConfig.dispatch1D n)

  let result2Bytes ← readCUDABufferFull outBuf
  let result2 := bytesToFloatArray result2Bytes

  let mut passed2 := true
  for i in [0, 1, 2, 1023] do
    let got := result2.getD i 0.0
    let expected := Float.ofNat ((i + 1) + (i + 1) * 100)
    if (got - expected).abs > 0.5 then
      IO.println s!"  ✗ output[{i}] = {got}, expected {expected}"
      passed2 := false
    else
      IO.println s!"  ✓ output[{i}] = {got} (expected {expected})"

  freeCUDABuffer aBuf
  freeCUDABuffer bBuf
  freeCUDABuffer outBuf

  if passed2 then
    IO.println "  → Test 2 PASSED"
  else
    IO.println "  → Test 2 FAILED"
  IO.println ""

  -- ═══ Test 3: Cache hit ═══
  IO.println "Test 3: Pipeline Cache"
  IO.println "──────────────────────"
  let (hits, misses) ← getCUDACacheStats
  IO.println s!"  Cache hits: {hits}, misses: {misses}"
  if misses == 2 then
    IO.println "  → Test 3 PASSED (2 unique kernels compiled)"
  else
    IO.println s!"  → Test 3: {misses} misses (expected 2)"

  IO.println ""
  if passed && passed2 then
    IO.println "✓ ALL CUDA EXECUTION TESTS PASSED"
  else
    IO.println "✗ SOME TESTS FAILED"
    IO.Process.exit 1
