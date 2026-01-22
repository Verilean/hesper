import Hesper.Tensor.Types

/-!
# GGUF File Loader (Skeleton)

Support for loading tensors and configuration from GGUF files.
Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
-/

namespace Hesper.IO.GGUF

structure Header where
  magic : UInt32
  version : UInt32
  tensorCount : Nat
  metadataCount : Nat

/-- GGUF Tensor Info -/
structure TensorInfo where
  name : String
  shape : List Nat
  dtype : Hesper.Tensor.DType
  offset : Nat

/-- Main GGUF Container -/
structure GGUFFile where
  header : Header
  metadata : List (String × String) -- Placeholder for KV pairs
  tensors : List TensorInfo

/-- Load GGUF file (Placeholder) -/
def load (path : String) : IO GGUFFile := do
  IO.println s!"Loading GGUF from {path}..."
  return {
    header := { magic := 0x47475546, version := 3, tensorCount := 0, metadataCount := 0 },
    metadata := [],
    tensors := []
  }

end Hesper.IO.GGUF
