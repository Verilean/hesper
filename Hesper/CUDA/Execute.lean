import Hesper.CUDA.FFI
import Hesper.CUDA.CodeGen
import Hesper.CUDA.Buffer
import Hesper.WGSL.Monad

/-!
# CUDA Kernel Execution Backend

Compiles ShaderM computations to PTX, JIT-compiles via `cuModuleLoadData`,
and dispatches with `cuLaunchKernel`. Parallel to `Hesper/WGSL/Execute.lean`.

Pipeline caching: PTX source is hashed; compiled CUmodule + CUfunction are
reused for identical kernels (same pattern as WebGPU pipeline cache).
-/

namespace Hesper.CUDA.Execute

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM BufferAccessMode)
open Hesper.CUDA
open Hesper.CUDA.CodeGen

-- ============================================================================
-- Execution config
-- ============================================================================

/-- Configuration for a CUDA kernel dispatch.
    `blockSize` maps to CUDA block dimensions (= WebGPU workgroup size).
    `gridSize` maps to CUDA grid dimensions (= WebGPU num workgroups). -/
structure CUDAExecutionConfig where
  funcName : String := "main"
  blockSize : WorkgroupSize := {x := 256, y := 1, z := 1}
  gridSize : Nat × Nat × Nat
  sharedMem : Nat := 0

namespace CUDAExecutionConfig

/-- 1D dispatch helper: N elements with given block size. -/
def dispatch1D (n : Nat) (blockSize : Nat := 256) : CUDAExecutionConfig :=
  { blockSize := {x := blockSize}
    gridSize := ((n + blockSize - 1) / blockSize, 1, 1) }

/-- 2D dispatch helper. -/
def dispatch2D (nx ny : Nat) (bx : Nat := 16) (by_ : Nat := 16) : CUDAExecutionConfig :=
  { blockSize := {x := bx, y := by_}
    gridSize := ((nx + bx - 1) / bx, (ny + by_ - 1) / by_, 1) }

end CUDAExecutionConfig

-- ============================================================================
-- Module cache (hash → compiled CUfunction)
-- ============================================================================

/-- Cached compiled CUDA module and kernel function. -/
structure CachedCUDAKernel where
  cudaModule : CUmodule
  func : CUfunction
  declaredNames : List String

/-- Global cache of compiled PTX modules. -/
initialize cudaModuleCacheRef : IO.Ref (Array (UInt64 × CachedCUDAKernel)) ←
  IO.mkRef #[]

initialize cudaCacheHitsRef : IO.Ref Nat ← IO.mkRef 0
initialize cudaCacheMissesRef : IO.Ref Nat ← IO.mkRef 0

def findCachedKernel (key : UInt64) (cache : Array (UInt64 × CachedCUDAKernel))
    : Option CachedCUDAKernel :=
  cache.find? (fun entry => entry.1 == key) |>.map (·.2)

/-- Get cache hit/miss statistics. -/
def getCUDACacheStats : IO (Nat × Nat) := do
  let hits ← cudaCacheHitsRef.get
  let misses ← cudaCacheMissesRef.get
  return (hits, misses)

/-- Reset the CUDA module cache (e.g. on context change). -/
def resetCUDACache : IO Unit := do
  cudaModuleCacheRef.set #[]
  cudaCacheHitsRef.set 0
  cudaCacheMissesRef.set 0

-- ============================================================================
-- Compiled kernel API
-- ============================================================================

/-- A pre-compiled CUDA kernel ready for dispatch. -/
structure CompiledCUDAKernel where
  func : CUfunction
  declaredNames : Array String
  sourceHash : UInt64

/-- Compile a ShaderM computation to a CUDA kernel.
    Returns a compiled kernel that can be dispatched multiple times. -/
def buildCUDAKernel
    (computation : ShaderM Unit)
    (config : CUDAExecutionConfig)
    : IO CompiledCUDAKernel := do
  let ptxSource := generatePTX config.funcName
    {x := config.blockSize.x, y := config.blockSize.y, z := config.blockSize.z}
    computation

  let sourceHash := hash ptxSource
  let cache ← cudaModuleCacheRef.get

  match findCachedKernel sourceHash cache with
  | some cached =>
    cudaCacheHitsRef.modify (· + 1)
    return { func := cached.func
             declaredNames := cached.declaredNames.toArray
             sourceHash }
  | none =>
    cudaCacheMissesRef.modify (· + 1)
    let cudaMod ← cuModuleLoadData ptxSource
    let func ← cuModuleGetFunction cudaMod config.funcName
    let state := ShaderM.exec computation
    let declaredNames := state.declaredBuffers.map (·.1)
    cudaModuleCacheRef.modify (·.push (sourceHash, {
      cudaModule := cudaMod
      func
      declaredNames
    }))
    return { func, declaredNames := declaredNames.toArray, sourceHash }

-- ============================================================================
-- Main dispatch functions
-- ============================================================================

/-- Execute a ShaderM computation on CUDA with named buffers.
    Parallel to `executeShaderNamed` in the WebGPU backend.

    - Generates PTX from the ShaderM computation
    - JIT-compiles via `cuModuleLoadData` (cached by hash)
    - Matches named buffers to declared buffer params
    - Launches kernel with `cuLaunchKernel`
-/
def executeShaderCUDA
    (computation : ShaderM Unit)
    (namedBuffers : List (String × CUDABuffer))
    (config : CUDAExecutionConfig)
    : IO Unit := do
  let kernel ← buildCUDAKernel computation config

  -- Match declared buffer names to provided named buffers
  let args ← kernel.declaredNames.foldlM (init := #[]) fun acc name => do
    match namedBuffers.find? (fun p => p.1 == name) with
    | some (_, buf) => return acc.push buf.ptr
    | none => throw (IO.userError s!"CUDA executeShader: missing buffer '{name}'")

  let (gx, gy, gz) := config.gridSize
  cuLaunchKernel kernel.func
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    config.blockSize.x.toUInt32 config.blockSize.y.toUInt32 config.blockSize.z.toUInt32
    config.sharedMem.toUInt32
    args

/-- Execute a ShaderM computation with positional buffers (no name matching).
    Buffers must be in the same order as declared in the shader. -/
def executeShaderCUDADirect
    (computation : ShaderM Unit)
    (buffers : Array CUDABuffer)
    (config : CUDAExecutionConfig)
    : IO Unit := do
  let kernel ← buildCUDAKernel computation config
  let args := buffers.map (·.ptr)

  let (gx, gy, gz) := config.gridSize
  cuLaunchKernel kernel.func
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    config.blockSize.x.toUInt32 config.blockSize.y.toUInt32 config.blockSize.z.toUInt32
    config.sharedMem.toUInt32
    args

/-- Dispatch a pre-compiled kernel with device pointer arguments. -/
def dispatchCUDAKernel
    (kernel : CompiledCUDAKernel)
    (args : Array USize)
    (config : CUDAExecutionConfig)
    : IO Unit := do
  let (gx, gy, gz) := config.gridSize
  cuLaunchKernel kernel.func
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    config.blockSize.x.toUInt32 config.blockSize.y.toUInt32 config.blockSize.z.toUInt32
    config.sharedMem.toUInt32
    args

end Hesper.CUDA.Execute
