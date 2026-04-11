import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-!
# GPU Backend Abstraction

Typeclass-based backend abstraction allowing the same ShaderM kernel
to run on WebGPU or CUDA, selected by `HESPER_BACKEND` env var.

## Usage

```lean
-- Backend-agnostic kernel execution:
def runMyKernel [GPUBackend β] (ctx : β) (bufs : List (String × GPUBackend.Buf β))
    (kernel : ShaderM Unit) (config : ExecConfig) : IO Unit :=
  GPUBackend.execute ctx kernel bufs config

-- At program start:
let backend ← detectBackend   -- reads HESPER_BACKEND
match backend with
| .webgpu device => runMyKernel device [("a", buf)] kernel config
| .cuda cudaCtx  => runMyKernel cudaCtx [("a", cbuf)] kernel config
```
-/

namespace Hesper

open Hesper.WGSL (WorkgroupSize)
open Hesper.WGSL.Monad (ShaderM)

/-- Backend-agnostic execution config.
    Maps to WebGPU's `ExecutionConfig` or CUDA's grid/block dims. -/
structure ExecConfig where
  funcName : String := "main"
  workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1}
  numWorkgroups : Nat × Nat × Nat := (1, 1, 1)

namespace ExecConfig

def dispatch1D (n : Nat) (wgSize : Nat := 256) : ExecConfig :=
  { workgroupSize := {x := wgSize}
    numWorkgroups := ((n + wgSize - 1) / wgSize, 1, 1) }

def dispatch2D (nx ny : Nat) (bx : Nat := 16) (by_ : Nat := 16) : ExecConfig :=
  { workgroupSize := {x := bx, y := by_}
    numWorkgroups := ((nx + bx - 1) / bx, (ny + by_ - 1) / by_, 1) }

end ExecConfig

/-- Typeclass for GPU compute backends.

    `β` = backend context type (e.g., `Device` for WebGPU, `CUDAContext` for CUDA)
    `Buf β` = buffer type associated with backend -/
class GPUBackend (β : Type) where
  /-- Buffer type for this backend -/
  Buf : Type
  /-- Execute a ShaderM kernel with named buffers -/
  execute : β → ShaderM Unit → List (String × Buf) → ExecConfig → IO Unit
  /-- Allocate a GPU buffer of given size (bytes), zero-filled -/
  allocBuffer : β → USize → IO Buf
  /-- Free a GPU buffer -/
  freeBuffer : β → Buf → IO Unit
  /-- Upload host data to GPU buffer -/
  writeBuffer : β → Buf → ByteArray → IO Unit
  /-- Download GPU buffer to host -/
  readBuffer : β → Buf → USize → IO ByteArray

/-- Which backend to use -/
inductive BackendChoice where
  | WebGPU
  | CUDA
  deriving BEq, Repr

/-- Detect backend from `HESPER_BACKEND` env var. Default: WebGPU. -/
def detectBackend : IO BackendChoice := do
  let env ← IO.getEnv "HESPER_BACKEND"
  match env with
  | some "cuda" | some "CUDA" => return .CUDA
  | _ => return .WebGPU

end Hesper
