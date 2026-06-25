import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Basic

/-!
# DiffusionGemma expert-indexed Q4_K matmul parity on Metal

Validates `fusedQ4KMExpertKernel` (selects expert `e` along the expert axis
of a `[inDim, outDim, nExpert]` Q4_K tensor, inline dequant) against the
ggml golden (dequant expert-slab × f32 input).

Run:
  ./scripts/llama_parity/dump_dg_q4kexp_golden /tmp/dg_golden/q4kexp
  lake exe diffusiongemma-q4kexp-parity
-/

open Hesper.WebGPU

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
  let dir := "/tmp/dg_golden/q4kexp"
  let inD := 256; let outD := 8; let nExp := 3; let e := 1
  IO.println "[dg-q4kexp-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let q4 ← IO.FS.readBinFile s!"{dir}/a.bin"
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"

  let wBuf ← createBuffer device { size := q4.size.toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device wBuf 0 q4
  let inBuf ← createBuffer device { size := (inD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes x)
  let outBuf ← createBuffer device { size := (outD*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

  let kern := Hesper.Layers.Linear.fusedQ4KMExpertKernel { inDim := inD, outDim := outD } nExp
  let pBuf ← createBuffer device { size := (4 : Nat).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device pBuf 0 (ByteArray.mk #[UInt8.ofNat e, 0, 0, 0])
  let bufs := ("weights", wBuf) :: ("input", inBuf) :: ("params", pBuf) :: ("output", outBuf) :: List.nil
  let cfg : Hesper.ExecConfig := { numWorkgroups := (outD, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device kern bufs cfg

  let q ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outBuf 0 (outD*4).toUSize)
  unmapBuffer outBuf
  let mut err := 0.0
  for i in [0:outD] do
    let d := (q[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  inD={inD} outD={outD} nExp={nExp} expert={e}  maxAbsErr={err}"
  IO.println s!"  gpu[0..4]  = {(q.extract 0 4).toList}"
  IO.println s!"  gold[0..4] = {(gold.extract 0 4).toList}"
  if q.size == outD && err < 1e-3 then
    IO.println "✓ expert-indexed Q4_K matmul on Metal matches ggml golden (expert selected, weights packed)"
  else
    IO.println "✗ FAIL"
    throw (IO.userError "q4kexp parity failed")
