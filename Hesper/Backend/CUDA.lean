import Hesper.Backend
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Execute

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

/-- Dispatch counter (declared up-front so `CUDAContext.init` can wire
    it into Execute.withSection).  Incremented by `launchKernelMaybeStream`
    below on every kernel launch. -/
initialize dispatchCounter : IO.Ref Nat ← IO.mkRef 0

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
  -- Wire the dispatch counter into Execute.withSection so per-section
  -- profiling (HESPER_DISPATCH_COUNT=1) can attribute kernel launches.
  Hesper.WGSL.Execute.registerDispatchCounter dispatchCounter.get
  return ⟨ctx⟩

/-- CUDA cached dispatch — compiled function + buffer args for replay. -/
structure CUDACachedDispatch where
  func : CUfunction
  sourceHash : UInt64
  declaredNames : Array String  -- buffer names in declaration order
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

/-- Diagnostic: true when a CUDA batch queue is currently open. -/
def Backend.isCudaBatching : IO Bool := do
  return (← cudaBatchQueue.get).isSome

/-! ## CUDA Graph capture stream

When `cudaCaptureStream` is `some s` (and no batch queue is open),
kernel launches go to `s` via `cuLaunchKernelOnStream` instead of the
default stream via `cuLaunchKernel`.  That makes them capturable into a
CUDA Graph.  When the ref is `none`, behaviour is unchanged.

Host→device copies also need to land on the capture stream — hence
`cuMemcpyHtoDMaybeOnStream` below.  All other FFI (cuMemcpyDtoH,
cuMemset) stays on the default stream because they sync + reading
happens only OUTSIDE capture scope. -/
initialize cudaCaptureStream : IO.Ref (Option Hesper.CUDA.CUstream) ← IO.mkRef none

def resetDispatchCounter : IO Unit := dispatchCounter.set 0
def getDispatchCounter : IO Nat := dispatchCounter.get

/-- Helper: route a kernel launch through the capture stream when active.
    Mirrors `cuLaunchKernel` signature.  Does NOT increment the dispatch
    counter — that happens earlier, when the launch is *emitted* (queued
    or immediate).  See `bumpDispatchOnEmit`. -/
private def launchKernelMaybeStream
    (func : CUfunction) (gx gy gz bx byDim bz : UInt32) (smem : UInt32)
    (args : Array USize) : IO Unit := do
  match ← cudaCaptureStream.get with
  | some s => Hesper.CUDA.cuLaunchKernelOnStream func gx gy gz bx byDim bz smem s args
  | none   => cuLaunchKernel func gx gy gz bx byDim bz smem args

/-- Bump the dispatch counter when a launch is emitted.  Called at the
    point where the launch is first decided — whether it gets queued into
    the batch or fired immediately, from the caller's perspective it IS
    a dispatch.  This makes per-section profiling see the launches even
    when batching is active (`beginBatch` is called upstream). -/
private def bumpDispatchOnEmit : IO Unit :=
  dispatchCounter.modify (· + 1)

/-- Cached result of `getenv HESPER_KERNEL_TRACE`.  Env lookup per launch
    is slow enough to show up in traces; check once at first use. -/
initialize kernelTraceEnabled : IO.Ref (Option Bool) ← IO.mkRef none

private def kernelTraceOn : IO Bool := do
  match ← kernelTraceEnabled.get with
  | some b => pure b
  | none =>
    let b := (← IO.getEnv "HESPER_KERNEL_TRACE").isSome
    kernelTraceEnabled.set (some b)
    pure b

/-- Emit a `[hs] funcName grid=(..) block=(..)` line to stderr when
    HESPER_KERNEL_TRACE=1.  Pair with llama.cpp's `[lc] ...` trace and
    diff the two captures to drive fusion work. -/
private def traceLaunch (funcName : String)
    (gx gy gz bx byDim bz : UInt32) : IO Unit := do
  if ← kernelTraceOn then
    IO.eprintln s!"[hs] {funcName} grid=({gx},{gy},{gz}) block=({bx},{byDim},{bz})"

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
    -- Optional: dump PTX for profiling/static analysis (HESPER_PTX_DUMP=dir).
    match ← IO.getEnv "HESPER_PTX_DUMP" with
    | some dir => IO.FS.writeFile s!"{dir}/{funcName}.ptx" ptx
    | none => pure ()
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
  launchKernelMaybeStream func
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
    -- If the caller left funcName at its default "main", derive a
    -- stable unique PTX symbol from the PTX hash.  That lets nsys
    -- break apart the per-kernel time bucket (otherwise every
    -- un-named execute call lands in one giant "main" row).  Same
    -- PTX → same symbol, so the module cache still hits identically.
    --
    -- Two-step: first generate PTX with "main", hash it, then if the
    -- caller didn't provide a name, regenerate with the hash-derived
    -- name.  The regeneration is cheap (same ShaderM state) and only
    -- costs once per unique PTX (cached downstream by ptxHash).
    let ptxInit := generatePTX config.funcName config.workgroupSize computation
    let ptxInitHash ← Hesper.CUDA.fastStringHash ptxInit
    let (effFuncName, ptx) := if config.funcName == "main" then
        let name := s!"k_{(toString ptxInitHash.toNat).take 16}"
        (name, generatePTX name config.workgroupSize computation)
      else (config.funcName, ptxInit)
    let ptxHash ← Hesper.CUDA.fastStringHash ptx
    let autoCache ← cudaAutoCache.get
    let (func, declaredNames) ← match autoCache.find? (fun e => e.1 == ptxHash) with
    | some (_, f, dn) => pure (f, dn)
    | none =>
      let cudaMod ← cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod effFuncName
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
    bumpDispatchOnEmit
    traceLaunch effFuncName gx.toUInt32 gy.toUInt32 gz.toUInt32
      config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32 config.workgroupSize.z.toUInt32
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => launchKernelMaybeStream func gx.toUInt32 gy.toUInt32 gz.toUInt32
                config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32
                config.workgroupSize.z.toUInt32 0 args
  executeWithConfigCached _ctx computation namedBuffers config cacheKey cacheRef := do
    let cached ← cacheRef.get
    -- Fast path: cacheRef hit — skip generatePTX + ShaderM.exec entirely
    let (func, declaredNames) ← match cached with
    | some c => pure (c.func, c.declaredNames.toList)
    | none => do
      -- Derive a unique PTX entry-point name from the caller's cacheKey so
      -- nsys/ncu can distinguish kernels in profiles. When cacheKey=0 (i.e.
      -- user didn't supply one) fall back to config.funcName.
      let funcName :=
        if cacheKey == 0 then config.funcName
        else s!"k_{(toString cacheKey.toNat).take 16}"
      let f ← cudaExecuteImpl computation namedBuffers funcName config.workgroupSize config.numWorkgroups
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let dn := state.declaredBuffers.map (·.1)
      let args ← dn.foldlM (init := #[]) fun acc name => do
        match namedBuffers.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => return acc
      cacheRef.set (some {
        func := f, sourceHash := 0, declaredNames := dn.toArray, args
        blockX := config.workgroupSize.x.toUInt32
        blockY := config.workgroupSize.y.toUInt32
        blockZ := config.workgroupSize.z.toUInt32
      })
      pure (f, dn)
    -- Resolve buffer args fresh (buffers may differ between calls)
    let args ← declaredNames.foldlM (init := #[]) fun acc name => do
      match namedBuffers.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"CUDA executeCached: missing buffer '{name}'")
    let (gx, gy, gz) := config.numWorkgroups
    let pending : PendingLaunch := {
      func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := config.workgroupSize.x.toUInt32, blockY := config.workgroupSize.y.toUInt32,
      blockZ := config.workgroupSize.z.toUInt32, args
    }
    bumpDispatchOnEmit
    let effName :=
      if cacheKey == 0 then config.funcName
      else s!"k_{(toString cacheKey.toNat).take 16}"
    traceLaunch effName gx.toUInt32 gy.toUInt32 gz.toUInt32
      config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32 config.workgroupSize.z.toUInt32
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => launchKernelMaybeStream func gx.toUInt32 gy.toUInt32 gz.toUInt32
                config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32
                config.workgroupSize.z.toUInt32 0 args
  executeWithConfigCachedArrays _ctx computation namedBuffers namedBufferArrays config cacheKey cacheRef := do
    -- Pointer tables live as IO.Ref-owned device allocations.  We allocate
    -- lazily and reuse via cacheRef — the table base pointer is captured
    -- in `cached.args`, so replays pick it up cheaply.  The table contents
    -- (per-layer device ptrs) are updated in-place on each call in case
    -- the layer buffers change.
    let cached ← cacheRef.get
    let (func, declaredNames, tablePtrs) ← match cached with
    | some c => pure (c.func, c.declaredNames.toList,
        -- Re-use captured args' tail which holds pointer-table bases.
        c.args.extract (c.args.size - namedBufferArrays.length.toUSize.toNat) c.args.size)
    | none => do
      let funcName :=
        if cacheKey == 0 then config.funcName
        else s!"k_{(toString cacheKey.toNat).take 16}"
      let f ← cudaExecuteImpl computation namedBuffers funcName config.workgroupSize config.numWorkgroups
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let dn := state.declaredBuffers.map (·.1)
      -- Allocate one device-side pointer table per bufferArray binding.
      let mut tableBases : Array USize := #[]
      for (_, bufs) in namedBufferArrays do
        let n := bufs.length
        let p ← Hesper.CUDA.cuMalloc (n * 8).toUSize
        tableBases := tableBases.push p
      let args := #[]  -- filled fresh below each call
      cacheRef.set (some {
        func := f, sourceHash := 0, declaredNames := dn.toArray, args
        blockX := config.workgroupSize.x.toUInt32
        blockY := config.workgroupSize.y.toUInt32
        blockZ := config.workgroupSize.z.toUInt32
      })
      pure (f, dn, tableBases)
    -- Upload current layer pointers into the tables (H→D copy of N×8 bytes).
    let mut tIdx : Nat := 0
    for (_name, bufs) in namedBufferArrays do
      let tablePtr := tablePtrs[tIdx]!
      let mut bytes : ByteArray := ByteArray.empty
      for buf in bufs do
        -- Little-endian encode CUdeviceptr (USize, 64-bit) as 8 bytes.
        let p := buf.ptr
        for i in [0:8] do
          bytes := bytes.push ((p >>> (i*8).toUSize).toUInt8)
      Hesper.CUDA.cuMemcpyHtoD tablePtr bytes 0 (bufs.length * 8).toUSize
      tIdx := tIdx + 1
    -- Resolve single-buffer args, then append the per-array table base ptrs.
    let mut args : Array USize := #[]
    for name in declaredNames do
      match namedBuffers.find? (fun p => p.1 == name) with
      | some (_, buf) => args := args.push buf.ptr
      | none =>
        match namedBufferArrays.findIdx? (fun p => p.1 == name) with
        | some i => args := args.push tablePtrs[i]!
        | none => throw (IO.userError s!"CUDA executeCachedArrays: missing binding '{name}'")
    let (gx, gy, gz) := config.numWorkgroups
    let effName :=
      if cacheKey == 0 then config.funcName
      else s!"k_{(toString cacheKey.toNat).take 16}"
    traceLaunch effName gx.toUInt32 gy.toUInt32 gz.toUInt32
      config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32 config.workgroupSize.z.toUInt32
    launchKernelMaybeStream func gx.toUInt32 gy.toUInt32 gz.toUInt32
      config.workgroupSize.x.toUInt32 config.workgroupSize.y.toUInt32
      config.workgroupSize.z.toUInt32 0 args
  replayCached _ctx cached dims := do
    let (gx, gy, gz) := dims
    let pending : PendingLaunch := {
      func := cached.func, gridX := gx.toUInt32, gridY := gy.toUInt32, gridZ := gz.toUInt32,
      blockX := cached.blockX, blockY := cached.blockY, blockZ := cached.blockZ, args := cached.args
    }
    bumpDispatchOnEmit
    traceLaunch "<replay>" gx.toUInt32 gy.toUInt32 gz.toUInt32
      cached.blockX cached.blockY cached.blockZ
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => launchKernelMaybeStream cached.func gx.toUInt32 gy.toUInt32 gz.toUInt32
                cached.blockX cached.blockY cached.blockZ 0 cached.args
  allocBuffer _ctx size := createCUDABuffer size
  freeBuffer _ctx buf := freeCUDABuffer buf
  writeBuffer _ctx buf data := do
    -- Route through the capture stream when active so writes become
    -- graph nodes.  Outside capture, use the sync default-stream path.
    match ← cudaCaptureStream.get with
    | some s =>
      Hesper.CUDA.cuMemcpyHtoDAsync buf.ptr data 0 data.size.toUSize s
    | none =>
      writeCUDABuffer buf data
  writeBufferOffset _ctx buf offset data := do
    match ← cudaCaptureStream.get with
    | some s =>
      Hesper.CUDA.cuMemcpyHtoDAsync buf.ptr data offset data.size.toUSize s
    | none =>
      cuMemcpyHtoD buf.ptr data offset data.size.toUSize
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
    bumpDispatchOnEmit
    traceLaunch "<compiled>" gx.toUInt32 gy.toUInt32 gz.toUInt32
      kernel.blockX kernel.blockY kernel.blockZ
    match ← cudaBatchQueue.get with
    | some queue => cudaBatchQueue.set (some (queue.push pending))
    | none => launchKernelMaybeStream kernel.func gx.toUInt32 gy.toUInt32 gz.toUInt32
                kernel.blockX kernel.blockY kernel.blockZ 0 args
  beginBatch _ctx := do
    cudaBatchQueue.set (some #[])
  endBatch _ctx := do
    match ← cudaBatchQueue.get with
    | some queue =>
      for p in queue do
        launchKernelMaybeStream p.func p.gridX p.gridY p.gridZ p.blockX p.blockY p.blockZ 0 p.args
      cudaBatchQueue.set none
    | none => pure ()
  hasSubgroupSupport _ctx := pure true   -- CUDA warp shuffle
  hasShaderF16Support _ctx := pure true  -- sm_89 has native f16
  newCacheRef := IO.mkRef none
  rawDevicePtr _ctx buf := pure (some buf.ptr)

end Hesper
