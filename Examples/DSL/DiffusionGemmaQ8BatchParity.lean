import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Basic

/-! Batched Q8_0 matmul parity: fusedQ8_0BatchKernel (M rows, one dispatch) vs the
    ggml-validated single-row fusedQ8_0LinearKernel per row. -/
open Hesper.WebGPU

def main : IO Unit := do
  let inD := 64; let outD := 8; let M := 4
  IO.println "[dg-q8batch-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let q8 ← IO.FS.readBinFile "/tmp/dg_golden/q8exp/a.bin"   -- expert-0 slab = [outD,inD] Q8_0
  let wBuf ← createBuffer device { size := q8.size.toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device wBuf 0 q8
  let xs := (List.range (M*inD)).toArray.map (fun i => Float.sin (i.toFloat * 0.023) * 0.5)
  let inBuf ← createBuffer device { size := (M*inD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes xs)
  let outBatch ← createBuffer device { size := (M*outD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let kB := Hesper.Layers.Linear.fusedQ8_0BatchKernel { inDim := inD, outDim := outD } M
  let bufsB := ("weights", wBuf) :: ("input", inBuf) :: ("output", outBatch) :: List.nil
  let cfgB : Hesper.ExecConfig := { numWorkgroups := (outD, M, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device kB bufsB cfgB
  let ob ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outBatch 0 (M*outD*4).toUSize)
  unmapBuffer outBatch
  let mut ref := Array.replicate (M*outD) 0.0
  for m in [0:M] do
    let rowBuf ← createBuffer device { size := (inD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
    writeBuffer device rowBuf 0 (← Hesper.Basic.floatArrayToBytes (xs.extract (m*inD) (m*inD+inD)))
    let outR ← createBuffer device { size := (outD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
    let kS := Hesper.Layers.Linear.fusedQ8_0LinearKernel { inDim := inD, outDim := outD }
    let bufsS := ("weights", wBuf) :: ("input", rowBuf) :: ("output", outR) :: List.nil
    let cfgS : Hesper.ExecConfig := { numWorkgroups := (outD, 1, 1), workgroupSize := { x := 256 } }
    Hesper.GPUBackend.execute device kS bufsS cfgS
    let orr ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outR 0 (outD*4).toUSize)
    unmapBuffer outR
    for i in [0:outD] do ref := ref.set! (m*outD+i) orr[i]!
  let mut err := 0.0
  for i in [0:M*outD] do
    let d := (ob[i]! - ref[i]!).abs
    if d > err then err := d
  IO.println s!"  M={M} inD={inD} outD={outD}  batched-vs-single maxAbsErr={err}"
  if err < 1e-4 && ob.size == M*outD then
    IO.println "✓ batched Q8_0 matmul matches per-row single kernel (ggml-validated) — N-row batching correct"
  else
    IO.println s!"✗ FAIL  batch[0..4]={(ob.extract 0 4).toList} single[0..4]={(ref.extract 0 4).toList}"
    throw (IO.userError "q8 batch parity failed")
