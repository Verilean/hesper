import Hesper.CUDA.FFI
import Hesper.Backend.CUDA
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader

/-!
# Common helpers for Gemma4 golden-unit tests

All kernel unit tests share these helpers:
- Load f32 binary dumps (produced by llama.cpp's eval-callback).
- Extract weight tensors from the GGUF file.
- Compute relative L2 diff between actual and expected f32 buffers.

Conventions:
- Inputs in `/tmp/llama_dump/<name>.bin` (populated by
  `llama-eval-callback -p "Hello world how are you"`).
- Tests compare against the **last token's slice** (position 4 in
  our 5-token prompt) because llama.cpp's `inp_out_ids` optimization
  prunes intermediate positions from most dumps.
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

/-- Parse GGUF once and return the file structure. -/
def loadGGUF : IO Hesper.GGUF.GGUFFile := do
  let bytes ← Hesper.CUDA.readFileFast ggufPath
  match Hesper.GGUF.Parser.parseGGUF bytes with
  | .ok g => pure g
  | .error e => throw (IO.userError s!"GGUF parse error: {e}")

/-- Extract a named float32 tensor from a parsed GGUF. -/
def extractF32 (gguf : Hesper.GGUF.GGUFFile) (name : String) : IO ByteArray :=
  Hesper.GGUF.Loader.extractFloat32Tensor gguf name

end Hesper.Tests.GoldenUnit.Common
