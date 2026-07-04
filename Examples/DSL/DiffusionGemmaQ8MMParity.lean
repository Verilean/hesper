import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Basic

/-!
# DiffusionGemma Q8_0 matmul parity on Metal

Validates Hesper's `fusedQ8_0LinearKernel` (inline Q8_0 dequant, packed
weights stay in VRAM) against the ggml golden (dequant-weight × f32-input).
Builds a Q8_0 `LinearLayer` from the dumped packed bytes and runs
`LinearLayer.forward` on the WebGPU→Metal backend.

Run:
  ./scripts/llama_parity/dump_dg_q8mm_golden /tmp/dg_golden/q8mm
  lake exe diffusiongemma-q8mm-parity
-/

open Hesper
open Hesper.WebGPU
open Hesper.Layers.Linear

def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)).toUInt32; let b1 := (bytes.get! (i*4+1)).toUInt32
    let b2 := (bytes.get! (i*4+2)).toUInt32; let b3 := (bytes.get! (i*4+3)).toUInt32
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 (b0 ||| (b1<<<8) ||| (b2<<<16) ||| (b3<<<24)))
  return a

def main : IO Unit := do
  let dir := "/tmp/dg_golden/q8mm"
  let inD := 64; let outD := 32
  IO.println "[dg-q8mm-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let q8 ← IO.FS.readBinFile s!"{dir}/a.bin"
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"

  let mkBuf (nbytes : Nat) : IO Buffer :=
    createBuffer device { size := nbytes.toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let wBuf ← mkBuf q8.size
  writeBuffer device wBuf 0 q8

  let layer : LinearLayer (GPUBackend.Buf Device) (GPUBackend.CachedDispatch Device) := {
    config := { inDim := inD, outDim := outD }
    weightBuf := wBuf
    quantFormat := .Q8_0
    prepared := ← GPUBackend.newCacheRef (β := Device)
    splitKBuf := ← IO.mkRef none
    splitKPartialPrepared := ← GPUBackend.newCacheRef (β := Device)
    splitKReducePrepared := ← GPUBackend.newCacheRef (β := Device)
    dp4aQ8Buf := ← IO.mkRef none
    dp4aQuantizePrepared := ← GPUBackend.newCacheRef (β := Device)
    dp4aMatmulPrepared := ← GPUBackend.newCacheRef (β := Device)
    dp4aBatchQuantizePrepared := ← GPUBackend.newCacheRef (β := Device)
    dp4aBatchMatmulPrepared := ← GPUBackend.newCacheRef (β := Device)
  }

  let inBuf ← mkBuf (inD*4)
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes x)
  let outBuf ← mkBuf (outD*4)
  LinearLayer.forward device layer inBuf outBuf
  let q ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outBuf 0 (outD*4).toUSize)
  unmapBuffer outBuf

  let mut err := 0.0
  for i in [0:outD] do
    let d := (q[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  in={inD} out={outD}  maxAbsErr={err}"
  IO.println s!"  gpu[0..4]  = {(q.extract 0 4).toList}"
  IO.println s!"  gold[0..4] = {(gold.extract 0 4).toList}"
  if q.size == outD && err < 1e-3 then
    IO.println "✓ Q8_0 inline-dequant matmul on Metal matches ggml golden (packed weights, no F32 expansion)"
  else
    IO.println "✗ FAIL"
    throw (IO.userError "q8 parity failed")
