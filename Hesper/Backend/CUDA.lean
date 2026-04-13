import Hesper.Backend
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen

/-!
# CUDA Backend Instance
-/

namespace Hesper

open Hesper.CUDA
open Hesper.CUDA.CodeGen
open Hesper.WGSL.Monad (ShaderM)

structure CUDAContext where
  ctx : CUcontext
  deriving Inhabited

def CUDAContext.init : IO CUDAContext := do
  cuDriverInit
  let count ← cuDeviceCount
  if count == 0 then throw (IO.userError "No CUDA devices found")
  let dev ← cuDeviceGet 0
  let name ← cuDeviceName dev
  let cc ← cuComputeCapability dev
  let mem ← cuTotalMem dev
  IO.println s!"[CUDA] Device: {name}, SM {cc / 10}.{cc % 10}, {mem / (1024*1024)} MB"
  let ctx ← cuCtxCreate dev
  return ⟨ctx⟩

/-- CUDA cached dispatch — compiled function + buffer args for replay. -/
structure CUDACachedDispatch where
  func : CUfunction
  sourceHash : UInt64
  args : Array USize
  blockX : UInt32 := 1
  blockY : UInt32 := 1
  blockZ : UInt32 := 1

initialize cudaModuleCache : IO.Ref (Array (USize × CUfunction)) ← IO.mkRef #[]

/-- Auto-cache: PTX hash → (CUfunction, declaredNames). Eliminates
    90-330μs generatePTX overhead on 2nd+ call for same kernel. -/
initialize cudaAutoCache : IO.Ref (Array (USize × CUfunction × Array String)) ← IO.mkRef #[]

/-- Batched launch queue. When batching, executeWithConfig resolves
    func + args but defers cuLaunchKernel. endBatch fires them all. -/
structure PendingLaunch where
  func : CUfunction
  gridX : UInt32
  gridY : UInt32
  gridZ : UInt32
  blockX : UInt32
  blockY : UInt32
  blockZ : UInt32
  args : Array USize

initialize cudaBatchQueue : IO.Ref (Option (Array PendingLaunch)) ← IO.mkRef none

private def cudaExecuteImpl (computation : ShaderM Unit) (namedBuffers : List (String × CUDABuffer))
    (funcName : String) (workgroupSize : Hesper.WGSL.WorkgroupSize)
    (numWorkgroups : Nat × Nat × Nat) : IO CUfunction := do
  -- Fast path: hash ShaderState (stmts repr) to skip 625KB PTX string hashing
  let state := Hesper.WGSL.Monad.ShaderM.exec computation
  let ptx := generatePTX funcName workgroupSize computation
  let sourceHash ← Hesper.CUDA.fastStringHash ptx
  let cache ← cudaModuleCache.get
  match cache.find? (fun e => e.1 == sourceHash) with
  | some (_, f) => return f
  | none =>
    let cudaMod ← cuModuleLoadData ptx
    let f ← cuModuleGetFunction cudaMod funcName
    cudaModuleCache.modify (·.push (sourceHash, f))
    return f

private def cudaLaunchWithBuffers (func : CUfunction) (namedBuffers : List (String × CUDABuffer))
    (computation : ShaderM Unit) (workgroupSize : Hesper.WGSL.WorkgroupSize)
    (numWorkgroups : Nat × Nat × Nat) : IO Unit := do
  let state := Hesper.WGSL.Monad.ShaderM.exec computation
  let declaredNames := state.declaredBuffers.map (·.1)
  let args ← declaredNames.foldlM (init := #[]) fun acc name => do
    match namedBuffers.find? (fun p => p.1 == name) with
    | some (_, buf) => return acc.push buf.ptr
    | none => throw (IO.userError s!"CUDA execute: missing buffer '{name}'")
  let (gx, gy, gz) := numWorkgroups
  cuLaunchKernel func
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    workgroupSize.x.toUInt32 workgroupSize.y.toUInt32 workgroupSize.z.toUInt32
    0 args

/-- CUDA compiled kernel — just the JIT'd function. -/
structure CUDACompiledKernel where
  func : CUfunction
  declaredNames : Array String
  blockX : UInt32 := 1
  blockY : UInt32 := 1
  blockZ : UInt32 := 1

instance : GPUBackend CUDAContext where
  Buf := CUDABuffer
  CachedDispatch := CUDACachedDispatch
  CompiledKernel := CUDACompiledKernel
  executeWithConfig _ctx computation namedBuffers config := do
    let ptx := generatePTX config.funcName config.workgroupSize computation
    let ptxHash ← Hesper.CUDA.fastStringHash ptx
    let autoCache ← cudaAutoCache.get
    let (func, declaredNames) ← match autoCache.find? (fun e => e.1 == ptxHash) with
    | some (_, f, dn) => pure (f, dn)
    | none =>
      let cudaMod ← cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod config.funcName
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let dn := state.declaredBuffers.map (·.1) |>.toArray
      cudaModuleCache.modify (·.push (ptxHash, f))
      cudaAutoCache.modify (·.push (ptxHash, f, dn))
      pure (f, dn)
    let args ← declaredNames.foldlM (init := #[]) fun acc name => do
      match namedBuffers.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"CUDA execute: missing buffer '{name}'")
    let (gx, gy, gz) := config.numWorkgroups
    let pending : PendingLaunch := {
      func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := config.workgroupSize.x.toUInt32, blockY := config.workgroupSize.y.toUInt32,
      blockZ := config.workgroupSize.z.toUInt32, args
    }
    -- If batching, queue the launch; otherwise fire immediately
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => cuLaunchKernel func gx.toUInt32 gy.toUInt32 gz.toUInt32
                config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32
                config.workgroupSize.z.toUInt32 0 args
  executeWithConfigCached _ctx computation namedBuffers config _cacheKey cacheRef := do
    let cached ← cacheRef.get
    let func ← match cached with
    | some c => pure c.func
    | none => do
      let f ← cudaExecuteImpl computation namedBuffers config.funcName config.workgroupSize config.numWorkgroups
      -- Collect buffer args for replay
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let declaredNames := state.declaredBuffers.map (·.1)
      let args ← declaredNames.foldlM (init := #[]) fun acc name => do
        match namedBuffers.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => return acc
      cacheRef.set (some {
        func := f
        sourceHash := hash (generatePTX config.funcName config.workgroupSize computation)
        args
        blockX := config.workgroupSize.x.toUInt32
        blockY := config.workgroupSize.y.toUInt32
        blockZ := config.workgroupSize.z.toUInt32
      })
      pure f
    -- Resolve args and launch (or queue if batching)
    let state := Hesper.WGSL.Monad.ShaderM.exec computation
    let declaredNames := state.declaredBuffers.map (·.1)
    let args ← declaredNames.foldlM (init := #[]) fun acc name => do
      match namedBuffers.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"CUDA execute: missing buffer '{name}'")
    let (gx, gy, gz) := config.numWorkgroups
    let pending : PendingLaunch := {
      func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := config.workgroupSize.x.toUInt32, blockY := config.workgroupSize.y.toUInt32,
      blockZ := config.workgroupSize.z.toUInt32, args
    }
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => cuLaunchKernel func gx.toUInt32 gy.toUInt32 gz.toUInt32
                config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32
                config.workgroupSize.z.toUInt32 0 args
  replayCached _ctx cached dims := do
    let (gx, gy, gz) := dims
    let pending : PendingLaunch := {
      func := cached.func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := cached.blockX, blockY := cached.blockY, blockZ := cached.blockZ, args := cached.args
    }
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => cuLaunchKernel cached.func gx.toUInt32 gy.toUInt32 gz.toUInt32
                cached.blockX cached.blockY cached.blockZ 0 cached.args
  allocBuffer _ctx size := createCUDABuffer size
  freeBuffer _ctx buf := freeCUDABuffer buf
  writeBuffer _ctx buf data := writeCUDABuffer buf data
  readBuffer _ctx buf size := readCUDABuffer buf size
  buildKernel _ctx computation config := do
    let ptx := generatePTX config.funcName config.workgroupSize computation
    let sourceHash ← Hesper.CUDA.fastStringHash ptx
    let cache ← cudaModuleCache.get
    let func ← match cache.find? (fun e => e.1 == sourceHash) with
    | some (_, f) => pure f
    | none => do
      let cudaMod ← cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod config.funcName
      cudaModuleCache.modify (·.push (sourceHash, f))
      pure f
    let state := Hesper.WGSL.Monad.ShaderM.exec computation
    pure { func, declaredNames := state.declaredBuffers.map (·.1) |>.toArray,
           blockX := config.workgroupSize.x.toUInt32,
           blockY := config.workgroupSize.y.toUInt32,
           blockZ := config.workgroupSize.z.toUInt32 }
  dispatchCompiledKernel _ctx kernel buffers numWorkgroups _cacheRef := do
    let args := buffers.map (·.ptr)
    let (gx, gy, gz) := numWorkgroups
    let pending : PendingLaunch := {
      func := kernel.func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := kernel.blockX, blockY := kernel.blockY, blockZ := kernel.blockZ, args
    }
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => cuLaunchKernel kernel.func gx.toUInt32 gy.toUInt32 gz.toUInt32
                kernel.blockX kernel.blockY kernel.blockZ 0 args
  beginBatch _ctx := do
    cudaBatchQueue.set (some #[])
  endBatch _ctx := do
    match ← cudaBatchQueue.get with
    | some queue =>
      for p in queue do
        cuLaunchKernel p.func p.gridX p.gridY p.gridZ p.blockX p.blockY p.blockZ 0 p.args
      cudaBatchQueue.set none
    | none => pure ()
  hasSubgroupSupport _ctx := pure true   -- CUDA warp shuffle
  hasShaderF16Support _ctx := pure true  -- sm_89 has native f16
  newCacheRef := IO.mkRef none

end Hesper
