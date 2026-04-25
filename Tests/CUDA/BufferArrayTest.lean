import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.WGSL.Monad

open Hesper (ExecConfig)

/-!
# bufferArray MVP test

Kernel takes a `bufferArray f32 N` and sums element 0 of every buffer:
```
  out[0] = Σ_{i=0..N-1}  buffers[i][0]
```
Each thread in a N-wide grid reads its own buffer's element 0 and atomically
adds to out[0] (but we avoid atomics by using a single-threaded reduction).

Verifies the full pipeline: ShaderM declare → Exp.indexBuf → PTX ld.global.u64
indirection → CUDA pointer-table materialisation → launch → readback.
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM

/-- Single-thread kernel: out[0] = sum of buffers[i][0] for i in [0, N).
    Grid is (1,1,1), workgroup is (1,1,1).  Keeps the reduction simple
    for a correctness-only test. -/
def sumFirstElemsKernel (numBufs : Nat) : ShaderM Unit := do
  let _bufs ← declareInputBufferArray "bufs" (.scalar .f32) numBufs
  let _out ← declareOutputBuffer "out" (.array (.scalar .f32) 1)

  -- Accumulate in a ShaderM variable bound via Exp.var.
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  for i in [0:numBufs] do
    let iExp : Exp (.scalar .u32) := Exp.litU32 i
    let zero : Exp (.scalar .u32) := Exp.litU32 0
    let v : Exp (.scalar .f32) ←
      readBufferArray (elemTy := .scalar .f32) (n := numBufs) "bufs" iExp zero
    let cur : Exp (.scalar .f32) := Exp.var "acc"
    ShaderM.assign "acc" (Exp.add cur v)

  let acc : Exp (.scalar .f32) := Exp.var "acc"
  writeBuffer (ty := .scalar .f32) "out" (Exp.litU32 0) acc

private def f32Bytes (f : Float) : ByteArray :=
  let b64 : UInt64 := f.toBits
  let s : UInt32 := ((b64 >>> 63) &&& 1).toUInt32
  let e64 : UInt64 := (b64 >>> 52) &&& 0x7FF
  let m64 : UInt64 := b64 &&& 0x000FFFFFFFFFFFFF
  let bits32 : UInt32 :=
    if e64 == 0 then 0
    else if e64 == 0x7FF then
      (s <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let e32 : Int := (e64.toNat : Int) - 1023 + 127
      if e32 ≤ 0 then 0
      else if e32 ≥ 255 then (s <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else
        (s <<< 31) ||| (e32.toNat.toUInt32 <<< 23) ||| ((m64 >>> 29).toUInt32 &&& 0x7FFFFF)
  ByteArray.empty
    |>.push bits32.toUInt8
    |>.push (bits32 >>> 8).toUInt8
    |>.push (bits32 >>> 16).toUInt8
    |>.push (bits32 >>> 24).toUInt8

private def bytesToF32 (bytes : ByteArray) : Float :=
  let b0 := bytes.get! 0
  let b1 := bytes.get! 1
  let b2 := bytes.get! 2
  let b3 := bytes.get! 3
  let bits : UInt32 := b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
  let s := (bits >>> 31) &&& 1
  let e32 := (bits >>> 23) &&& 0xFF
  let m32 := bits &&& 0x7FFFFF
  if e32 == 0 then 0.0
  else if e32 == 0xFF then if m32 == 0 then (if s == 1 then -1.0/0.0 else 1.0/0.0) else 0.0/0.0
  else
    let e64 : UInt64 := (e32.toUInt64 - 127 + 1023)
    let m64 : UInt64 := m32.toUInt64 <<< 29
    let sg64 : UInt64 := s.toUInt64 <<< 63
    Float.ofBits (sg64 ||| (e64 <<< 52) ||| m64)

def main : IO Unit := do
  IO.println "═══ bufferArray MVP test ═══"
  let ctx ← CUDAContext.init
  let n : Nat := 8  -- 8 separate buffers

  -- Allocate N buffers, each holding one f32 value = (i + 1) * 0.5
  -- Expected sum: Σ_{i=1..8} i * 0.5 = 0.5 * 36 = 18.0
  let mut bufs : List CUDABuffer := []
  for i in [0:n] do
    let buf ← createCUDABuffer 4
    let v : Float := (i + 1).toFloat * 0.5
    writeCUDABuffer buf (f32Bytes v)
    bufs := bufs ++ [buf]
  IO.println s!"[Setup] Allocated {n} f32 buffers, each with value (i+1)*0.5"

  -- Output buffer: 1 f32
  let outBuf ← createCUDABuffer 4
  cuMemset outBuf.ptr 4

  -- Dispatch the kernel via executeWithConfigCachedArrays.
  let ref : IO.Ref (Option (GPUBackend.CachedDispatch CUDAContext)) ← IO.mkRef none
  let cfg : ExecConfig := {
    funcName := "sumFirstElems"
    workgroupSize := { x := 1 }
    numWorkgroups := (1, 1, 1)
  }
  GPUBackend.executeWithConfigCachedArrays ctx
    (sumFirstElemsKernel n)
    [("out", outBuf)]
    [("bufs", bufs)]
    cfg
    (hash "bufferArray_sum_first")
    ref

  -- Read result.
  let outBytes ← readCUDABuffer outBuf 4
  let got := bytesToF32 outBytes
  let expected : Float := (List.range n).foldl (fun acc i => acc + (i + 1).toFloat * 0.5) 0.0
  IO.println s!"[Result] got = {got}, expected = {expected}"
  if (got - expected).abs < 1e-5 then
    IO.println "✓ PASS (read)"
  else
    IO.println "✗ FAIL (read)"
    IO.Process.exit 1

  -- Write test: single-WG kernel writes (100 + layerIdx) to buffers[layerIdx][0].
  let writeKernel : ShaderM Unit := do
    let _ ← declareInputBufferArray "outs" (.scalar .f32) n
    for i in [0:n] do
      let iExp : Exp (.scalar .u32) := Exp.litU32 i
      let zero : Exp (.scalar .u32) := Exp.litU32 0
      let v : Exp (.scalar .f32) := Exp.litF32 (100.0 + i.toFloat)
      writeBufferArray (ty := .scalar .f32) "outs" iExp zero v

  let writeRef : IO.Ref (Option (GPUBackend.CachedDispatch CUDAContext)) ← IO.mkRef none
  GPUBackend.executeWithConfigCachedArrays ctx writeKernel []
    [("outs", bufs)]
    { funcName := "writeLayerVals", workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
    (hash "bufferArray_write")
    writeRef

  IO.println ""
  IO.println "[Write test] Expect buffers[i][0] = 100 + i"
  let mut allOk := true
  let mut idx := 0
  for buf in bufs do
    let bytes ← readCUDABuffer buf 4
    let got := bytesToF32 bytes
    let want : Float := 100.0 + idx.toFloat
    if (got - want).abs > 1e-5 then
      IO.println s!"  buf[{idx}]: got {got}, want {want}  ✗"
      allOk := false
    idx := idx + 1
  if allOk then
    IO.println "✓ PASS (write)"
  else
    IO.Process.exit 1
