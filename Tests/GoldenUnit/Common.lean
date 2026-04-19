import Hesper.CUDA.FFI
import Hesper.Backend.CUDA
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.Layers.Linear

/-!
# Common helpers for Gemma4 golden-unit tests

Design notes (memory-conscious):
- The GGUF file (2.7 GB) and CUDAContext are loaded **once** in
  `Main.lean` and passed to every test as arguments.  Tests must
  not re-load them.
- Every GPU buffer allocated inside a test must be released via
  `GPUBackend.freeBuffer` before returning.  We provide
  `withTempBuf` / `withTempLinearLayer` helpers that free on exit.

Conventions:
- Inputs in `/tmp/llama_dump/<name>.bin` (populated by
  `llama-eval-callback -p "Hello world how are you"`).
- Tests compare against the **last token's slice** of each dump.
-/

namespace Hesper.Tests.GoldenUnit.Common

def goldenDir : String := "/tmp/llama_dump"
def ggufPath : String := "data/gemma-4-e4b-it-Q4_K_M.gguf"

/-- hiddenSize for Gemma 4 E4B (n_embd). -/
def gemma4HiddenDim : Nat := 2560

/-- RMS eps. -/
def gemma4RmsEps : Float := 1e-6

/-- Load a float32 binary file.  Returns ByteArray (raw bytes). -/
def loadFloat32Bin (path : String) : IO ByteArray := do
  IO.FS.readBinFile path

/-- Interpret 4 bytes at `offset` as little-endian f32. -/
def readF32LE (ba : ByteArray) (offset : Nat) : Float :=
  let b0 : UInt32 := (ba.get! offset).toUInt32
  let b1 : UInt32 := (ba.get! (offset + 1)).toUInt32
  let b2 : UInt32 := (ba.get! (offset + 2)).toUInt32
  let b3 : UInt32 := (ba.get! (offset + 3)).toUInt32
  let bits : UInt32 := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e : UInt32 := (bits >>> 23) &&& 0xFF
  let m : UInt32 := bits &&& 0x7FFFFF
  let s : UInt32 := bits >>> 31
  if e == 0 then 0.0
  else
    let mf : Float := 1.0 + m.toNat.toFloat / 8388608.0
    let v : Float := mf * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

/-- Convert a ByteArray holding n f32 values to Array Float. -/
def byteArrayToF32Array (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut arr : Array Float := Array.mkEmpty n
  for i in [0:n] do
    arr := arr.push (readF32LE ba (i * 4))
  pure arr

/-- Extract the last `dim` f32 values from a ByteArray (returns ByteArray
    of `dim*4` bytes).  Used to slice the last token from a multi-token
    dump. -/
def lastTokenBytes (ba : ByteArray) (dim : Nat) : ByteArray :=
  let totalBytes := ba.size
  let tokBytes := dim * 4
  if totalBytes < tokBytes then ba
  else
    let start := totalBytes - tokBytes
    ba.extract start totalBytes

/-- Relative L2 diff: `||a - b|| / ||b||`. -/
def relDiff (a b : Array Float) : Float := Id.run do
  let mut numSq : Float := 0.0
  let mut denSq : Float := 0.0
  let n := min a.size b.size
  for i in [0:n] do
    let ai := a[i]!
    let bi := b[i]!
    let d := ai - bi
    numSq := numSq + d * d
    denSq := denSq + bi * bi
  let num := numSq.sqrt
  let den := denSq.sqrt
  if den < 1e-30 then num else num / den

/-- Parse GGUF once (call in Main). -/
def loadGGUF : IO Hesper.GGUF.GGUFFile := do
  let bytes ← Hesper.CUDA.readFileFast ggufPath
  match Hesper.GGUF.Parser.parseGGUF bytes with
  | .ok g => pure g
  | .error e => throw (IO.userError s!"GGUF parse error: {e}")

/-- Extract a named float32 tensor from a parsed GGUF. -/
def extractF32 (gguf : Hesper.GGUF.GGUFFile) (name : String) : IO ByteArray :=
  Hesper.GGUF.Loader.extractFloat32Tensor gguf name

/-- Upload a ByteArray to a new GPU buffer.  Caller MUST free. -/
def uploadBuffer [GPUBackend β] (ctx : β) (data : ByteArray) : IO (GPUBackend.Buf β) := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if data.size > 0 then
    GPUBackend.writeBuffer ctx buf data
  return buf

/-- Run `action` with a fresh GPU buffer of `sizeBytes`, free on exit. -/
def withTempBuf [GPUBackend β] (ctx : β) (sizeBytes : Nat)
    (action : GPUBackend.Buf β → IO α) : IO α := do
  let buf ← GPUBackend.allocBuffer ctx sizeBytes.toUSize
  try
    action buf
  finally
    GPUBackend.freeBuffer ctx buf

/-- Run `action` with a fresh GPU buffer initialised from `data`, free on exit. -/
def withTempBufFromBytes [GPUBackend β] (ctx : β) (data : ByteArray)
    (action : GPUBackend.Buf β → IO α) : IO α := do
  let buf ← uploadBuffer ctx data
  try
    action buf
  finally
    GPUBackend.freeBuffer ctx buf

/-- Load a quantized linear layer from a GGUF tensor.  Caller MUST free
    the returned layer's `weightBuf` via `freeLinearLayer`. -/
def loadLinear [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (inDim outDim : Nat)
    : IO (Hesper.Layers.Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let tensorInfo ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok ti => pure ti
    | .error e => throw (IO.userError e)
  let quantFormat : Hesper.Layers.Linear.QuantFormat := match tensorInfo.ggmlType with
    | .Q6_K => .Q6_K
    | _ => .Q4_K
  let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf name with
    | .ok r => pure r
    | .error e => throw (IO.userError e)
  let weightBuf ← uploadBuffer ctx data
  let prepared ← GPUBackend.newCacheRef (β := β)
  let splitKBuf ← IO.mkRef none
  let splitKPartialPrepared ← GPUBackend.newCacheRef (β := β)
  let splitKReducePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aQ8Buf ← IO.mkRef none
  let dp4aQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  return {
    config := { inDim, outDim }
    weightBuf
    quantFormat
    prepared
    splitKBuf
    splitKPartialPrepared
    splitKReducePrepared
    dp4aQ8Buf
    dp4aQuantizePrepared
    dp4aMatmulPrepared
    dp4aBatchQuantizePrepared
    dp4aBatchMatmulPrepared
  }

/-- Free all GPU buffers held by a LinearLayer (weightBuf + lazily-
    allocated splitK / dp4a Q8 scratch). -/
def freeLinearLayer [GPUBackend β] (ctx : β)
    (layer : Hesper.Layers.Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    : IO Unit := do
  GPUBackend.freeBuffer ctx layer.weightBuf
  match ← layer.splitKBuf.get with
  | some b => GPUBackend.freeBuffer ctx b
  | none => pure ()
  match ← layer.dp4aQ8Buf.get with
  | some b => GPUBackend.freeBuffer ctx b
  | none => pure ()

/-- Run `action` with a freshly loaded LinearLayer, free on exit. -/
def withLinearLayer [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (inDim outDim : Nat)
    (action : Hesper.Layers.Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) → IO α)
    : IO α := do
  let layer ← loadLinear ctx gguf name inDim outDim
  try
    action layer
  finally
    freeLinearLayer ctx layer

end Hesper.Tests.GoldenUnit.Common
