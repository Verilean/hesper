import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-!
# GPU Backend Abstraction

Typeclass-based backend abstraction. Same ShaderM kernels run on
WebGPU or CUDA — selected by `HESPER_BACKEND` env var at runtime.

## Design

`GPUBackend β` provides all GPU operations parameterized by context type `β`.
- WebGPU instance: `β = Device`, `Buf β = Buffer`
- CUDA instance: `β = CUDAContext`, `Buf β = CUDABuffer`

Layer/model code is written as:
```lean
def myLayer [GPUBackend β] (ctx : β) (input output : GPUBackend.Buf β) : IO Unit := do
  GPUBackend.execute ctx myKernel [("input", input), ("output", output)]
    (GPUBackend.dispatch1D 1024)
```
-/

namespace Hesper

open Hesper.WGSL (WorkgroupSize)
open Hesper.WGSL.Monad (ShaderM)

/-- Typeclass for GPU compute backends.

    `β` is the backend context type.
    All GPU operations go through this interface. -/
class GPUBackend (β : Type) where
  /-- Buffer type for this backend -/
  Buf : Type
  /-- Cached dispatch state for fast-path replay (WebGPU: PreparedDispatch, CUDA: hash key) -/
  CachedDispatch : Type := Unit
  /-- Pre-compiled kernel (WebGPU: CompiledKernel, CUDA: CUfunction + metadata) -/
  CompiledKernel : Type := Unit

  -- ── Kernel execution ──

  /-- Execute a ShaderM kernel with named buffers. -/
  executeKernel : β → ShaderM Unit → List (String × Buf) →
    (funcName : String) → (workgroupSize : WorkgroupSize) →
    (numWorkgroups : Nat × Nat × Nat) → IO Unit

  /-- Execute with optional dispatch cache (PreparedDispatch equivalent).
      If cacheRef contains a cached dispatch, replay it (fast path).
      Otherwise, execute normally and store the result. -/
  executeKernelCached : β → ShaderM Unit → List (String × Buf) →
    (funcName : String) → (workgroupSize : WorkgroupSize) →
    (numWorkgroups : Nat × Nat × Nat) →
    (cacheKey : UInt64) →
    (cacheRef : IO.Ref (Option CachedDispatch)) → IO Unit

  /-- Replay a cached dispatch with different grid dimensions. -/
  replayCached : β → CachedDispatch → Nat × Nat × Nat → IO Unit

  -- ── Buffer management ──

  /-- Allocate a GPU buffer of given size (bytes), zero-filled -/
  allocBuffer : β → USize → IO Buf
  /-- Allocate a buffer with specific usage flags (backend-specific) -/
  allocBufferUsage : β → USize → List String → IO Buf := fun ctx size _ => allocBuffer ctx size
  /-- Free a GPU buffer -/
  freeBuffer : β → Buf → IO Unit
  /-- Upload host ByteArray to GPU buffer (offset 0) -/
  writeBuffer : β → Buf → ByteArray → IO Unit
  /-- Upload host ByteArray to GPU buffer at byte offset -/
  writeBufferOffset : β → Buf → USize → ByteArray → IO Unit := fun ctx buf _ data => writeBuffer ctx buf data
  /-- Download `size` bytes from GPU buffer to host -/
  readBuffer : β → Buf → USize → IO ByteArray

  -- ── Kernel compilation ──

  /-- Pre-compile a kernel (WebGPU: build pipeline, CUDA: JIT PTX) -/
  buildKernel : β → ShaderM Unit →
    (funcName : String) → (workgroupSize : WorkgroupSize) →
    (numWorkgroups : Nat × Nat × Nat) → IO CompiledKernel
  /-- Dispatch a pre-compiled kernel with positional buffers + optional cache -/
  dispatchCompiledKernel : β → CompiledKernel → Array Buf →
    (numWorkgroups : Nat × Nat × Nat) →
    (cacheRef : Option (IO.Ref (Option CachedDispatch))) → IO Unit

  -- ── Dispatch cache management ──

  /-- Create a new empty dispatch cache ref -/
  newCacheRef : IO (IO.Ref (Option CachedDispatch)) := IO.mkRef none

/-- Execution config — mirrors the existing `Execute.ExecutionConfig`
    so that `smartDispatch` etc. work unchanged. -/
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

/-- Convenience: execute with ExecConfig (unpacks into executeKernel args) -/
@[inline]
def GPUBackend.execute [GPUBackend β] (ctx : β) (computation : ShaderM Unit)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit :=
  GPUBackend.executeKernel ctx computation namedBuffers
    config.funcName config.workgroupSize config.numWorkgroups

/-- Smart dispatch: 1D if fits, 2D otherwise.
    Returns `(config, gridDimX)` — same signature as TTT.Kernels.smartDispatch. -/
def smartDispatch (totalThreads : Nat) (wgSize : Nat := 256) : ExecConfig × Nat :=
  let wgCount := (totalThreads + wgSize - 1) / wgSize
  if wgCount <= 65535 then
    (ExecConfig.dispatch1D totalThreads wgSize, 0)
  else
    let gridX : Nat := 4096
    let gridY := (wgCount + gridX - 1) / gridX
    ({ numWorkgroups := (gridX, gridY, 1),
       workgroupSize := { x := wgSize, y := 1, z := 1 } }, gridX * wgSize)

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
