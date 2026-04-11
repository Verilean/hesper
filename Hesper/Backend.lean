/-!
# Backend Selection

Runtime backend selection via `HESPER_BACKEND` environment variable.
Defaults to WebGPU; set `HESPER_BACKEND=cuda` to use the CUDA PTX JIT backend.
-/

namespace Hesper

inductive Backend where
  | WebGPU
  | CUDA
  deriving BEq, Repr

/-- Get the compute backend from `HESPER_BACKEND` env var. -/
def getBackend : IO Backend := do
  let env ← IO.getEnv "HESPER_BACKEND"
  match env with
  | some "cuda" => return .CUDA
  | some "CUDA" => return .CUDA
  | _ => return .WebGPU

/-- Check if CUDA backend is selected. -/
def isCUDABackend : IO Bool := do
  let b ← getBackend
  return b == .CUDA

end Hesper
