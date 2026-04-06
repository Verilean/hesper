import Hesper
import Hesper.Training.AttentionBackward
import Hesper.Training.SafeBuffer

open Hesper.WebGPU
open Hesper.Training.SafeBuffer

def main : IO Unit := do
  IO.println "=== RMSNorm Backward GPU Test ==="

  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  let dim := 2560

  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }

  let xBuf ← mkBuf dim
  let gammaBuf ← mkBuf dim
  let dOutBuf ← mkBuf dim
  let dInBuf ← mkBuf dim

  -- Fill buffers using floatArrayToBytes
  let xArr := Array.ofFn (n := dim) fun i => Float.sin (i.val.toFloat * 0.1) * 2.0
  writeBuffer device xBuf 0 (floatArrayToBytes xArr)

  let gammaArr := Array.replicate dim 1.0
  writeBuffer device gammaBuf 0 (floatArrayToBytes gammaArr)

  let dOutArr := Array.ofFn (n := dim) fun i => Float.cos (i.val.toFloat * 0.05) * 0.1
  writeBuffer device dOutBuf 0 (floatArrayToBytes dOutArr)

  IO.println "Running RMSNorm backward kernel..."
  Hesper.Training.AttentionBackward.executeRmsNormBackward device
    xBuf gammaBuf dOutBuf dInBuf dim

  let result ← safeMapBufferReadF32 device dInBuf 8
  let hasNan := result.any isNaN
  let maxAbs := result.foldl (init := 0.0) fun acc v =>
    let a := if v < 0.0 then 0.0 - v else v
    if a > acc then a else acc

  IO.println s!"Result first 8: {result.toList}"
  IO.println s!"Has NaN: {hasNan}, Max abs: {maxAbs}"

  if hasNan then
    IO.println "✗ FAIL: RMSNorm backward produces NaN"
  else
    IO.println "✓ PASS: RMSNorm backward is valid"
