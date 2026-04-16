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

/-- Execution config — backend-agnostic. extensions/diagnostics are used
    by WebGPU (e.g., chromium_experimental_subgroup_matrix) and ignored by CUDA. -/
structure ExecConfig where
  funcName : String := "main"
  workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1}
  numWorkgroups : Nat × Nat × Nat := (1, 1, 1)
  extensions : List String := []
  diagnostics : List (String × String) := []

namespace ExecConfig

def dispatch1D (n : Nat) (wgSize : Nat := 256) : ExecConfig :=
  { workgroupSize := {x := wgSize}
    numWorkgroups := ((n + wgSize - 1) / wgSize, 1, 1) }

def dispatch2D (nx ny : Nat) (bx : Nat := 16) (by_ : Nat := 16) : ExecConfig :=
  { workgroupSize := {x := bx, y := by_}
    numWorkgroups := ((nx + bx - 1) / bx, (ny + by_ - 1) / by_, 1) }

end ExecConfig

/-- Typeclass for GPU compute backends. -/
class GPUBackend (β : Type) where
  Buf : Type
  CachedDispatch : Type := Unit
  CompiledKernel : Type := Unit
  executeWithConfig : β → ShaderM Unit → List (String × Buf) → ExecConfig → IO Unit
  executeWithConfigCached : β → ShaderM Unit → List (String × Buf) → ExecConfig →
    UInt64 → IO.Ref (Option CachedDispatch) → IO Unit
  replayCached : β → CachedDispatch → Nat × Nat × Nat → IO Unit
  allocBuffer : β → USize → IO Buf
  allocBufferUsage : β → USize → List String → IO Buf := fun ctx size _ => allocBuffer ctx size
  freeBuffer : β → Buf → IO Unit
  writeBuffer : β → Buf → ByteArray → IO Unit
  writeBufferOffset : β → Buf → USize → ByteArray → IO Unit := fun ctx buf _ data => writeBuffer ctx buf data
  readBuffer : β → Buf → USize → IO ByteArray
  buildKernel : β → ShaderM Unit → ExecConfig → IO CompiledKernel
  dispatchCompiledKernel : β → CompiledKernel → Array Buf →
    Nat × Nat × Nat → Option (IO.Ref (Option CachedDispatch)) → IO Unit
  hasSubgroupSupport : β → IO Bool := fun _ => pure false
  hasShaderF16Support : β → IO Bool := fun _ => pure false
  newCacheRef : IO (IO.Ref (Option CachedDispatch)) := IO.mkRef none

  /-- Raw device pointer for a buffer, if the backend has a native address
      space exposable as a USize.  CUDA returns `some` (CUdeviceptr), WebGPU
      returns `none`.  Used by Phase-0 hybrid path that calls externally-JIT'd
      PTX directly.  Must not be used by portable code. -/
  rawDevicePtr : β → Buf → IO (Option USize) := fun _ _ => pure none

  -- ── Batching (optional) ──
  /-- Begin recording dispatches for batch submission. CUDA: no-op. -/
  beginBatch : β → IO Unit := fun _ => pure ()
  /-- Submit all recorded dispatches and wait. CUDA: no-op (sync is per-launch). -/
  endBatch : β → IO Unit := fun _ => pure ()

/-- Convenience: execute with ExecConfig -/
@[inline]
def GPUBackend.execute [GPUBackend β] (ctx : β) (computation : ShaderM Unit)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit :=
  GPUBackend.executeWithConfig ctx computation namedBuffers config

/-- A kernel bundled with its dispatch cache. Production code should use
    this instead of bare `GPUBackend.execute` to ensure dispatch caching.
    Create with `CachedKernel.create`, dispatch with `CachedKernel.exec`. -/
structure CachedKernel (β : Type) [GPUBackend β] where
  computation : ShaderM Unit
  cacheKey : UInt64
  cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β))

namespace CachedKernel

/-- Create a cached kernel. Call once at init time (e.g., in createInferenceState). -/
def create [GPUBackend β] (computation : ShaderM Unit) (name : String) : IO (CachedKernel β) := do
  pure { computation, cacheKey := hash name, cacheRef := ← IO.mkRef none }

/-- Dispatch with caching. First call compiles; subsequent calls skip PTX generation. -/
@[inline]
def exec [GPUBackend β] (k : CachedKernel β) (ctx : β)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit :=
  GPUBackend.executeWithConfigCached ctx k.computation namedBuffers config k.cacheKey k.cacheRef

end CachedKernel

/-- For debug/test use only. Creates an ephemeral cache (not reused). -/
def GPUBackend.debugExecuteOnce [GPUBackend β] (ctx : β) (computation : ShaderM Unit)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit := do
  let ref ← IO.mkRef none
  GPUBackend.executeWithConfigCached ctx computation namedBuffers config 0 ref

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
