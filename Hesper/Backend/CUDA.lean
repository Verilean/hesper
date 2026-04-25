import Std.Data.HashMap
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

/-- Default persistent non-null stream for H2D + kernel launches.  See
    the detailed doc at the main declaration site below. -/
initialize cudaDefaultStream : IO.Ref (Option Hesper.CUDA.CUstream) ← IO.mkRef none

/-- Pinned-host ring buffer for async `writeBufferOffset` outside graph
    capture.  Each call grabs a fresh slot (advancing the cursor mod
    `cudaPinnedRingSize`); subsequent calls re-use slots only after the
    ring has wrapped around — long enough that prior async copies have
    drained on the default stream.  Size = 256 slots × 64 B = 16 KB. -/
initialize cudaPinnedRingPtr  : IO.Ref USize ← IO.mkRef 0
initialize cudaPinnedRingCursor : IO.Ref USize ← IO.mkRef 0
def cudaPinnedRingSlotBytes : USize := 64
def cudaPinnedRingSlots     : USize := 256
def cudaPinnedRingTotalBytes : USize := cudaPinnedRingSlotBytes * cudaPinnedRingSlots

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
  -- Allocate the persistent default async stream.  Used by the
  -- GGUF mmap loader path (HESPER_USE_MMAP + HESPER_MMAP_ASYNC) to
  -- issue `cuMemcpyHtoDAsync` on a non-null stream.  Not wired into
  -- writeBuffer / launchKernelMaybeStream by default to avoid the
  -- regressions we hit in earlier sessions (stream ordering races).
  match ← cudaDefaultStream.get with
  | some _ => pure ()
  | none =>
    let s ← Hesper.CUDA.cuStreamCreate
    cudaDefaultStream.set (some s)
  -- Allocate the pinned ring buffer for async per-call scalar/short writes.
  if (← cudaPinnedRingPtr.get) == 0 then
    let p ← Hesper.CUDA.cuMemAllocHost cudaPinnedRingTotalBytes
    cudaPinnedRingPtr.set p
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

/-- Env-flag cache: `HESPER_DESC_PATH=1` → route cacheRef-hit launches
    through the C descriptor pool (metadata-free hot path). -/
initialize descPathEnabled : IO.Ref (Option Bool) ← IO.mkRef none

private def descPathOn : IO Bool := do
  match ← descPathEnabled.get with
  | some b => pure b
  | none =>
    let b := (← IO.getEnv "HESPER_DESC_PATH").isSome
    descPathEnabled.set (some b)
    pure b

/-- Side-table mapping a CUDACachedDispatch's cacheKey (or a surrogate
    identity — we use the CUfunction handle as USize) to a registered
    C descriptor id.  Kept outside `CUDACachedDispatch` to avoid
    inflating that record and triggering Lean elaboration blow-ups in
    downstream typeclass instances. -/
initialize cudaDescIdMap : IO.Ref (Array (USize × USize)) ← IO.mkRef #[]

private def descIdLookup (funcKey : USize) : IO (Option USize) := do
  let m ← cudaDescIdMap.get
  match m.find? (fun p => p.1 == funcKey) with
  | some (_, id) => pure (some id)
  | none => pure none

private def descIdRegister (funcKey : USize) (id : USize) : IO Unit := do
  cudaDescIdMap.modify (·.push (funcKey, id))

/-- Module cache: maps PTX source hash (or the cheaper shape+funcName
    preHash) to the compiled CUfunction.  Was previously an
    `Array (USize × CUfunction)` scanned linearly on every kernel
    dispatch — decode at steady state does ~300 calls/tok and this
    lookup alone was ~8% of CPU time (`perf` 2026-04-24).  Moved to
    `Std.HashMap` for O(1) amortised lookup. -/
initialize cudaModuleCache : IO.Ref (Std.HashMap USize CUfunction) ← IO.mkRef ∅

/-- Per-size allocation counter.  Each entry is `(sizeBytes, count)`.
    Enabled by env `HESPER_ALLOC_TRACE=1`; when true, `allocBuffer`
    bumps the bucket for its size so callers can see which sizes are
    allocated repeatedly (i.e. which call sites aren't cached).
    A cached alloc should appear exactly once; a per-decode-step alloc
    shows up `steps` times. -/
initialize cudaAllocCounter : IO.Ref (Array (USize × Nat)) ← IO.mkRef #[]
initialize cudaAllocTraceEnabled : IO.Ref (Option Bool) ← IO.mkRef none

private def allocTraceOn : IO Bool := do
  match ← cudaAllocTraceEnabled.get with
  | some b => pure b
  | none =>
    let b := (← IO.getEnv "HESPER_ALLOC_TRACE").isSome
    cudaAllocTraceEnabled.set (some b)
    pure b

/-- Reset alloc counter (call at decode-loop start to exclude prefill). -/
def resetAllocCounter : IO Unit := cudaAllocCounter.set #[]

/-- Total wall time spent in `cuModuleLoadData` (PTX JIT) — suspected
    decode-time stall source.  Bump from each load-site when trace on. -/
initialize cudaModuleLoadWallNs : IO.Ref Nat ← IO.mkRef 0
initialize cudaModuleLoadCount : IO.Ref Nat ← IO.mkRef 0

def resetModuleLoadTimer : IO Unit := do
  cudaModuleLoadWallNs.set 0
  cudaModuleLoadCount.set 0

def recordModuleLoad (ns : Nat) : IO Unit := do
  cudaModuleLoadWallNs.modify (· + ns)
  cudaModuleLoadCount.modify (· + 1)

def printModuleLoadStats : IO Unit := do
  let ns ← cudaModuleLoadWallNs.get
  let n ← cudaModuleLoadCount.get
  let ms := ns.toFloat / 1e6
  let avgUs := if n > 0 then ns.toFloat / n.toFloat / 1e3 else 0.0
  IO.println s!"[modload] cuModuleLoadData calls={n}, total={ms} ms, avg={avgUs} µs"

/-- Total wall time spent in cudaExecuteImpl (= generatePTX + hash +
    module cache lookup + possibly cuModuleLoadData). -/
initialize cudaExecuteImplWallNs : IO.Ref Nat ← IO.mkRef 0
initialize cudaExecuteImplCount : IO.Ref Nat ← IO.mkRef 0
/-- Count of `executeWithConfigCached` cacheRef misses, by funcName. -/
initialize cudaCacheMissByName : IO.Ref (Array (String × Nat)) ← IO.mkRef #[]
def resetExecuteImplTimer : IO Unit := do
  cudaExecuteImplWallNs.set 0
  cudaExecuteImplCount.set 0
  cudaCacheMissByName.set #[]
def recordExecuteImpl (ns : Nat) : IO Unit := do
  cudaExecuteImplWallNs.modify (· + ns)
  cudaExecuteImplCount.modify (· + 1)
def recordCacheMiss (name : String) : IO Unit := do
  if ← allocTraceOn then
    cudaCacheMissByName.modify fun arr =>
      match arr.findIdx? (fun p => p.1 == name) with
      | some i => arr.modify i (fun (n, c) => (n, c + 1))
      | none   => arr.push (name, 1)
def printExecuteImplStats : IO Unit := do
  let ns ← cudaExecuteImplWallNs.get
  let n ← cudaExecuteImplCount.get
  IO.println s!"[execImpl] cudaExecuteImpl calls={n}, total={ns.toFloat/1e6} ms"
  let misses ← cudaCacheMissByName.get
  let sorted := misses.qsort (fun a b => a.2 > b.2)
  IO.println "[cacheMiss] executeWithConfigCached cacheRef miss histogram (top 20):"
  let limit := min sorted.size 20
  let mut i := 0
  while i < limit do
    match sorted[i]? with
    | some (name, c) => IO.println s!"  {name} × {c}"
    | none => pure ()
    i := i + 1

/-- Print the alloc histogram sorted by count descending.  Lines where
    `count > 1` are suspect non-cached call sites. -/
def printAllocHistogram : IO Unit := do
  let arr ← cudaAllocCounter.get
  let sorted := arr.qsort (fun a b => a.2 > b.2)
  let unique := arr.size
  let totalNs := arr.foldl (fun s p => s + p.2) 0
  IO.println s!"[alloc] total cuMemAlloc wall={totalNs.toFloat / 1e6} ms, unique sizes={unique}"
  IO.println "[alloc] size (bytes) × total wall (ms)   (now time-of-alloc per size-bucket)"
  let limit := min sorted.size 30
  let mut i := 0
  while i < limit do
    match sorted[i]? with
    | some (sz, ns) =>
      IO.println s!"  {sz} × {ns.toFloat / 1e6} ms"
    | none => pure ()
    i := i + 1

/-- Auto-cache: PTX hash → (CUfunction, declaredNames). Eliminates
    90-330μs generatePTX overhead on 2nd+ call for same kernel. -/
initialize cudaAutoCache : IO.Ref (Array (USize × CUfunction × Array String)) ← IO.mkRef #[]

/-- Cache-overwrite detection.  A `CUDACachedDispatch` stored in a
    caller's `IO.Ref` *should* always represent the same call site (one
    cacheKey per Ref).  When two code paths share the same Ref but
    register with distinct cacheKeys, they overwrite each other every
    call, which guarantees a cacheRef miss.  To surface this bug:
    `executeWithConfigCached` writes `cacheKey` into the dispatch's
    previously-unused `sourceHash` field, and on subsequent writes
    compares the stored key.  Behaviour on mismatch is controlled by
    `HESPER_CACHE_STRICT`:

    - unset (default): emit `[cache-overwrite]` to stderr once per
      (old, new) pair — non-fatal, preserves prod behaviour.
    - `1`: throw `IO.userError` on first mismatch so a test surfaces
      the call-site pair that's thrashing. -/
initialize cacheOverwriteSeen : IO.Ref (Array (UInt64 × UInt64)) ← IO.mkRef #[]
initialize cacheStrictEnabled : IO.Ref (Option Bool) ← IO.mkRef none

private def cacheStrictOn : IO Bool := do
  match ← cacheStrictEnabled.get with
  | some b => pure b
  | none =>
    let b := match ← IO.getEnv "HESPER_CACHE_STRICT" with
             | some "1" => true
             | _        => false
    cacheStrictEnabled.set (some b)
    pure b

/-- Report (or, in strict mode, abort on) a cache overwrite. -/
private def reportCacheOverwrite (oldKey newKey : UInt64) : IO Unit := do
  let seen ← cacheOverwriteSeen.get
  if seen.any (fun p => p.1 == oldKey && p.2 == newKey) then return
  cacheOverwriteSeen.modify (·.push (oldKey, newKey))
  let msg := s!"[cache-overwrite] same Ref used with different cacheKeys: {oldKey} → {newKey}"
  if ← cacheStrictOn then
    throw (IO.userError msg)
  else
    IO.eprintln msg

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

/-- Pick the stream for H2D / kernel launches outside capture:
      1. capture stream (if active — needed for graph recording)
      2. persistent default stream (once initialised)
      3. null stream (legacy / pre-init) -/
private def activeStream : IO (Option Hesper.CUDA.CUstream) := do
  match ← cudaCaptureStream.get with
  | some s => pure (some s)
  | none   => cudaDefaultStream.get

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

/-- Option B+ helper: if a descriptor for this call site is registered
    and `HESPER_DESC_PATH=1`, fire it via `descLaunchWithArgs` which
    rebinds arg pointers in place before launching.  Keyed by
    `cacheKey` alone (the call-site identity) — buffers may change
    between calls, which is exactly what rebind handles. -/
private def tryDescLaunch (cacheKey : UInt64) (args : Array USize) : IO Bool := do
  if ¬ (← descPathOn) then return false
  if cacheKey == 0 then return false
  -- If a batch queue is active, we must NOT fire immediately — the
  -- enclosing code expects to enqueue the launch.  Return false to
  -- fall back to the batch-queue path.
  if (← cudaBatchQueue.get).isSome then return false
  match ← descIdLookup cacheKey.toUSize with
  | some id =>
    bumpDispatchOnEmit
    let stream : USize := match ← cudaCaptureStream.get with
      | some s => s
      | none   => 0
    Hesper.CUDA.descLaunchWithArgs id stream args
    return true
  | none => return false

/-- Option B+ helper: register a descriptor for this call site on
    first observation, then fire it (register captures current args,
    subsequent calls go through `tryDescLaunch` → `descLaunchWithArgs`
    which rebinds). -/
private def tryDescRegisterAndLaunch
    (cacheKey : UInt64) (func : USize)
    (gx gy gz bx by_ bz smem : UInt32)
    (args : Array USize) : IO Bool := do
  if ¬ (← descPathOn) then return false
  if cacheKey == 0 then return false
  match ← descIdLookup cacheKey.toUSize with
  | some _ => return false   -- already registered; fast-path should have hit
  | none   =>
    if (← cudaBatchQueue.get).isNone && (← cudaCaptureStream.get).isNone then
      let id ← Hesper.CUDA.descRegister func gx gy gz bx by_ bz smem args
      descIdRegister cacheKey.toUSize id
      bumpDispatchOnEmit
      Hesper.CUDA.descLaunch id 0
      return true
    else
      return false

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

/-- Returns both the resolved CUfunction and the `ShaderState` that was
    built while probing the cache, so callers can read `declaredBuffers`
    without running `ShaderM.exec` a second time. -/
private def cudaExecuteImpl (computation : ShaderM Unit) (namedBuffers : List (String × CUDABuffer))
    (funcName : String) (workgroupSize : Hesper.WGSL.WorkgroupSize)
    (numWorkgroups : Nat × Nat × Nat) : IO (CUfunction × Hesper.WGSL.Monad.ShaderState) := do
  let traceOn ← allocTraceOn
  let t0 ← if traceOn then IO.monoNanosNow else pure 0
  -- Fast path: hash the ShaderState repr (cheap) to probe the module
  -- cache before running the expensive `generatePTX` + full PTX
  -- string hash.  Only if the preHash isn't already a cache key do
  -- we fall back to generating PTX and hashing it.
  let state := Hesper.WGSL.Monad.ShaderM.exec computation
  let preHash : USize := (hash (funcName, state.stmts.length, state.declaredBuffers.length,
                                workgroupSize.x, workgroupSize.y, workgroupSize.z)).toUSize
  -- Check prehash cache first.
  let preCache ← cudaModuleCache.get
  match preCache[preHash]? with
  | some f =>
    if traceOn then
      let t1 ← IO.monoNanosNow
      recordExecuteImpl (t1 - t0)
    return (f, state)
  | none => pure ()
  -- Cache miss on prehash: generate full PTX and use its hash as the
  -- authoritative key.  Also register the preHash so subsequent calls
  -- hit the fast path.
  let ptx := generatePTX funcName workgroupSize computation
  let sourceHash ← Hesper.CUDA.fastStringHash ptx
  let cache ← cudaModuleCache.get
  match cache[sourceHash]? with
  | some f =>
    -- Register preHash on the sourceHash-hit path too so subsequent
    -- calls skip the generatePTX.
    cudaModuleCache.modify (·.insert preHash f)
    if traceOn then
      let t1 ← IO.monoNanosNow
      recordExecuteImpl (t1 - t0)
    return (f, state)
  | none =>
    -- Optional: dump PTX for profiling/static analysis (HESPER_PTX_DUMP=dir).
    match ← IO.getEnv "HESPER_PTX_DUMP" with
    | some dir => IO.FS.writeFile s!"{dir}/{funcName}.ptx" ptx
    | none => pure ()
    let cudaMod ← do
      if traceOn then
        let tm0 ← IO.monoNanosNow
        let m ← cuModuleLoadData ptx
        let tm1 ← IO.monoNanosNow
        recordModuleLoad (tm1 - tm0)
        if (← IO.getEnv "HESPER_MODLOAD_NAMES").isSome then
          IO.eprintln s!"[modload-miss] {funcName} ({(tm1 - tm0) / 1000} µs)"
        pure m
      else
        cuModuleLoadData ptx
    let f ← cuModuleGetFunction cudaMod funcName
    cudaModuleCache.modify fun c => (c.insert sourceHash f).insert preHash f
    if traceOn then
      let t1 ← IO.monoNanosNow
      recordExecuteImpl (t1 - t0)
    return (f, state)

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
    recordCacheMiss s!"[execNonCached] {config.funcName}"
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
      let cudaMod ← do
        if ← allocTraceOn then
          let t0 ← IO.monoNanosNow
          let m ← cuModuleLoadData ptx
          let t1 ← IO.monoNanosNow
          recordModuleLoad (t1 - t0)
          pure m
        else
          cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod effFuncName
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let dn := state.declaredBuffers.map (·.1) |>.toArray
      cudaModuleCache.modify (·.insert ptxHash f)
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
    -- Overwrite detection: if the Ref already holds a dispatch from a
    -- *different* cacheKey, something is wrong — two call sites share
    -- the same Ref and clobber each other every call.  The dispatch
    -- stored in `sourceHash` is actually the cacheKey (repurposed from
    -- the previously-unused field).
    --
    -- Separately, a cacheRef that arrives `none` every call (throwaway
    -- ref created fresh at the call site) is an even worse anti-
    -- pattern: the caller short-circuits the entire cache.  Detect
    -- that by counting how many times each cacheKey has been seen
    -- with a none-Ref; same key twice in a row means a throwaway.
    let cached ← match cached with
      | some c =>
        if cacheKey != 0 && c.sourceHash != 0 && c.sourceHash != cacheKey then do
          reportCacheOverwrite c.sourceHash cacheKey
          pure none   -- treat as miss; will re-register with the right key
        else
          pure (some c)
      | none =>
        -- none + non-zero cacheKey that we've seen before = throwaway Ref
        if cacheKey != 0 && (← allocTraceOn) then do
          let seen ← cudaCacheMissByName.get
          -- Use cacheKey as a proxy funcName tag to detect repeats.
          let keyTag := s!"[throwaway?] key={cacheKey}"
          let count := (seen.find? (fun p => p.1 == keyTag)).map (·.2) |>.getD 0
          if count == 1 then   -- second sighting, first *throwaway* confirmed
            IO.eprintln s!"[throwaway-ref] cacheKey={cacheKey} seen twice with none-Ref — caller uses IO.mkRef none"
          cudaCacheMissByName.modify fun arr =>
            match arr.findIdx? (fun p => p.1 == keyTag) with
            | some i => arr.modify i (fun (n, c) => (n, c + 1))
            | none   => arr.push (keyTag, 1)
        pure none
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
      -- For human-readable miss histograms, record under the caller's
      -- `config.funcName` when it was tagged with a `ce "..."` name.  The
      -- PTX module still uses `funcName` (the hex key) so nsys kernel
      -- identity is preserved.  For untagged call sites the funcName
      -- is "main" (default); stamp numWG+wgSize to help differentiate.
      let (gx, gy, gz) := config.numWorkgroups
      let wgs := config.workgroupSize
      let shapeTag := s!"nwg=({gx},{gy},{gz}) wg=({wgs.x},{wgs.y},{wgs.z})"
      let missTag := if config.funcName == "" || config.funcName.startsWith "k_"
                     then funcName
                     else if config.funcName == "main"
                     then s!"main-{shapeTag} ({funcName})"
                     else s!"{config.funcName} ({funcName})"
      recordCacheMiss missTag
      -- HESPER_CACHE_MISS_TRACE=1: on the FIRST miss for each missTag, dump
      -- the declared buffer names to stderr so callers can be identified.
      -- Subsequent misses of the same missTag are noise (they mean the
      -- caller is hitting a throwaway ref) but the first-seen names are
      -- enough to grep source for the call site.
      if (← allocTraceOn) then
        let prior ← cudaCacheMissByName.get
        let isFirst := prior.any (fun p => p.1 == missTag && p.2 == 1)
        if isFirst then
          let bufNames := (namedBuffers.map (·.1)).toArray
          IO.eprintln s!"[cacheMissFirst] {missTag} bufs={bufNames}"
      let (f, state) ← cudaExecuteImpl computation namedBuffers funcName config.workgroupSize config.numWorkgroups
      let dn := state.declaredBuffers.map (·.1)
      let args ← dn.foldlM (init := #[]) fun acc name => do
        match namedBuffers.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => return acc
      -- Stamp cacheKey into `sourceHash` so future set-overwrites with a
      -- different key are detectable (see `reportCacheOverwrite`).
      let newDispatch : CUDACachedDispatch := {
        func := f, sourceHash := cacheKey, declaredNames := dn.toArray, args
        blockX := config.workgroupSize.x.toUInt32
        blockY := config.workgroupSize.y.toUInt32
        blockZ := config.workgroupSize.z.toUInt32
      }
      cacheRef.set (some newDispatch)
      pure (f, dn)
    -- Resolve buffer args fresh (buffers may differ between calls)
    let args ← declaredNames.foldlM (init := #[]) fun acc name => do
      match namedBuffers.find? (fun p => p.1 == name) with
      | some (_, buf) => return acc.push buf.ptr
      | none => throw (IO.userError s!"CUDA executeCached: missing buffer '{name}'")
    -- Option B+ fast-fast path: descriptor keyed by (cacheKey, args-hash).
    if ← tryDescLaunch cacheKey args then return
    let (gx, gy, gz) := config.numWorkgroups
    -- Option B+: try registering descriptor for future calls.  Helper
    -- returns true if it both registered and launched, else false.
    if ← tryDescRegisterAndLaunch cacheKey func
         gx.toUInt32 gy.toUInt32 gz.toUInt32
         config.workgroupSize.x.toUInt32
         config.workgroupSize.y.toUInt32
         config.workgroupSize.z.toUInt32
         0 args
    then return
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
      let (f, state) ← cudaExecuteImpl computation namedBuffers funcName config.workgroupSize config.numWorkgroups
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
  allocBuffer _ctx size := do
    if ← allocTraceOn then
      let t0 ← IO.monoNanosNow
      let buf ← createCUDABuffer size
      let t1 ← IO.monoNanosNow
      let ns := (t1 - t0)
      cudaAllocCounter.modify fun arr =>
        match arr.findIdx? (fun p => p.1 == size) with
        | some i => arr.modify i (fun (s, n) => (s, n + ns))
        | none   => arr.push (size, ns)
      return buf
    else
      createCUDABuffer size
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
      -- Outside graph capture: route small writes through the pinned
      -- ring buffer + async H2D on stream 0 (the legacy default
      -- stream).  Stream 0 is implicitly ordered with kernel launches
      -- on stream 0 (also via cuLaunchKernel's null-stream form), so
      -- the kernel still sees the new bytes.  Falls back to the sync
      -- path for anything larger than a ring slot.
      let sz := data.size.toUSize
      let pinPtr ← cudaPinnedRingPtr.get
      if pinPtr ≠ 0 ∧ sz ≤ cudaPinnedRingSlotBytes then
        let cursor ← cudaPinnedRingCursor.get
        let nextCursor := (cursor + 1) % cudaPinnedRingSlots
        cudaPinnedRingCursor.set nextCursor
        let slotOffset := cursor * cudaPinnedRingSlotBytes
        let dstAddr := buf.ptr + offset
        -- stream = 0 → legacy default stream, same lane as plain
        -- cuLaunchKernel.  Kernel-write-kernel ordering is preserved.
        Hesper.CUDA.cuPinnedWriteAndCopy dstAddr pinPtr slotOffset data sz (0 : USize)
      else
        cuMemcpyHtoD buf.ptr data offset data.size.toUSize
  readBuffer _ctx buf size := readCUDABuffer buf size
  buildKernel _ctx computation config := do
    let ptx := generatePTX config.funcName config.workgroupSize computation
    let sourceHash ← Hesper.CUDA.fastStringHash ptx
    let cache ← cudaModuleCache.get
    let func ← match cache[sourceHash]? with
    | some f => pure f
    | none => do
      let cudaMod ← do
        if ← allocTraceOn then
          let t0 ← IO.monoNanosNow
          let m ← cuModuleLoadData ptx
          let t1 ← IO.monoNanosNow
          recordModuleLoad (t1 - t0)
          pure m
        else
          cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod config.funcName
      cudaModuleCache.modify (·.insert sourceHash f)
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
