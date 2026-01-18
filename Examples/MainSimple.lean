-- Only import the Hesper namespace (not all modules)
namespace Hesper

/-- Initialize the Hesper WebGPU engine.
    Discovers available GPU adapters and sets up the Dawn instance. -/
@[extern "lean_hesper_init"]
opaque init : IO Unit

/-- Run GPU vector addition (Hello World compute example).
    Adds two vectors of the given size element-wise on the GPU. -/
@[extern "lean_hesper_vector_add"]
opaque vectorAdd (size : UInt32) : IO Unit

end Hesper

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verilean Hesper - GPU Inference Engine     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU and enumerate adapters
  Hesper.init
  IO.println ""

  -- Run GPU vector addition example
  IO.println "=== GPU Vector Addition Example ==="
  Hesper.vectorAdd 1024
  IO.println ""

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   GPU Test Complete!                         ║"
  IO.println "╚══════════════════════════════════════════════╝"
