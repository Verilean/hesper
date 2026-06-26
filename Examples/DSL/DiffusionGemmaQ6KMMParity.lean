import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Quantization.Q6_K
import Hesper.Basic
open Hesper.WebGPU
def readF32 (p : String) : IO (Array Float) := do
  let b ← IO.FS.readBinFile p; let n := b.size/4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let v := (b.get! (i*4)).toUInt32 ||| ((b.get! (i*4+1)).toUInt32<<<8) ||| ((b.get! (i*4+2)).toUInt32<<<16) ||| ((b.get! (i*4+3)).toUInt32<<<24)
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 v)
  return a
def main : IO Unit := do
  let dir := "/tmp/dg_golden/q6kmm"; let inD := 256; let outD := 8
  IO.println "[dg-q6kmm-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init; let device ← getDevice inst
  let w ← IO.FS.readBinFile s!"{dir}/a.bin"
  let x ← readF32 s!"{dir}/x.bin"; let gold ← readF32 s!"{dir}/out.bin"
  let wBuf ← createBuffer device { size := w.size.toUSize, usage := [.storage,.copyDst,.copySrc], mappedAtCreation := false }
  writeBuffer device wBuf 0 w
  let inBuf ← createBuffer device { size := (inD*4).toUSize, usage := [.storage,.copyDst,.copySrc], mappedAtCreation := false }
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes x)
  let outBuf ← createBuffer device { size := (outD*4).toUSize, usage := [.storage,.copyDst,.copySrc], mappedAtCreation := false }
  let bufs := ("weights",wBuf)::("input",inBuf)::("output",outBuf)::List.nil
  let cfg : Hesper.ExecConfig := { numWorkgroups := (outD,1,1), workgroupSize := {x:=256} }
  Hesper.GPUBackend.execute device (Hesper.Quantization.Q6_K.fusedQ6KLinearKernel inD outD) bufs cfg
  let q ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outBuf 0 (outD*4).toUSize)
  unmapBuffer outBuf
  let mut err := 0.0
  for i in [0:outD] do let d := (q[i]! - gold[i]!).abs; if d > err then err := d
  IO.println s!"  Q6_K matmul maxAbsErr={err}"
  IO.println s!"  gpu ={(q.extract 0 outD).toList}"
  IO.println s!"  gold={(gold.extract 0 outD).toList}"
  if err < 1e-3 then IO.println "✓ Metal Q6_K matmul matches ggml golden"
  else IO.println "✗ Q6_K MATMUL BUG — embedding+lm_head use this dequant"; throw (IO.userError "q6k fail")
