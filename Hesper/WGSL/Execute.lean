import Std.Data.HashMap
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.Logging

namespace Hesper.WGSL.Execute

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.CodeGen
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-!
# WGSL Shader Execution Layer

Integration between ShaderM monad, code generation, and WebGPU execution.

This module provides high-level functions to:
1. Compile ShaderM computations to WGSL
2. Create GPU pipelines
3. Execute shaders with buffer management
4. Handle synchronization

Usage Pattern:
```lean
def myKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

-- Execute on GPU
executeShader device myKernel
  [("input", inputBuffer), ("output", outputBuffer)]
  {x := 256, y := 1, z := 1}
  (256, 1, 1)
```
-/

/-- GPU Buffer with name for binding -/
structure NamedBuffer where
  name : String
  buffer : Buffer

/-- Execution configuration for compute shaders -/
structure ExecutionConfig where
  funcName : String := "main"
  workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1}
  numWorkgroups : Nat × Nat × Nat
  extensions : List String := []
  diagnostics : List (String × String) := []

instance : Inhabited ExecutionConfig where
  default := {
    funcName := "main"
    workgroupSize := {x := 256, y := 1, z := 1}
    numWorkgroups := (1, 1, 1)
    extensions := []
    diagnostics := []
  }

namespace ExecutionConfig

/-- Create default execution config with specified workgroup count -/
def default (numWorkgroups : Nat × Nat × Nat) : ExecutionConfig :=
  { funcName := "main"
    workgroupSize := {x := 256, y := 1, z := 1}
    numWorkgroups := numWorkgroups }

/-- Create config for 1D dispatch -/
def dispatch1D (totalThreads : Nat) (workgroupSize : Nat := 256) : ExecutionConfig :=
  let numWorkgroups := (totalThreads + workgroupSize - 1) / workgroupSize
  { funcName := "main"
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numWorkgroups, 1, 1) }

end ExecutionConfig

/-! ## Subgroup Feature Detection

Cached runtime check for subgroup support. Queried once per session,
used to select between subgroup-based kernels and shared-memory fallback kernels.
-/

/-- Cached subgroup support flag (queried once, then reused) -/
initialize subgroupSupportRef : IO.Ref (Option Bool) ← IO.mkRef none

/-- Check if the device supports subgroup operations (`subgroupAdd`, etc.).
    Result is cached after the first call. -/
def hasSubgroupSupport (device : Device) : IO Bool := do
  match ← subgroupSupportRef.get with
  | some v => pure v
  | none =>
    let v ← Hesper.WebGPU.deviceHasSubgroups device
    subgroupSupportRef.set (some v)
    pure v

/-- Cached SubgroupMatrix support flag -/
initialize subgroupMatrixSupportRef : IO.Ref (Option Bool) ← IO.mkRef none

/-- Check if the device supports Chromium experimental subgroup matrix
    operations (subgroup_matrix_{left,right,result}, subgroupMatrixLoad,
    subgroupMatrixMultiplyAccumulate, subgroupMatrixStore). Cached. -/
def hasSubgroupMatrixSupport (device : Device) : IO Bool := do
  match ← subgroupMatrixSupportRef.get with
  | some v => pure v
  | none =>
    let v ← Hesper.WebGPU.deviceHasSubgroupMatrix device
    subgroupMatrixSupportRef.set (some v)
    pure v

/-- Cached ShaderF16 support flag -/
initialize shaderF16SupportRef : IO.Ref (Option Bool) ← IO.mkRef none

/-- Check if the device supports ShaderF16 (`f16` values, f16 arithmetic). -/
def hasShaderF16Support (device : Device) : IO Bool := do
  match ← shaderF16SupportRef.get with
  | some v => pure v
  | none =>
    let v ← Hesper.WebGPU.deviceHasShaderF16 device
    shaderF16SupportRef.set (some v)
    pure v

/-! ## Pipeline Cache

Caches compiled GPU pipelines keyed by WGSL source hash.
Pipeline compilation is expensive (~1-5ms per shader). With ~270 dispatches
per forward pass, caching eliminates 270-1350ms of per-token overhead.
-/

/-- Cached GPU pipeline components -/
structure CachedPipeline where
  shaderModule : WebGPU.ShaderModule
  bindGroupLayout : BindGroupLayout
  pipeline : ComputePipeline
  declaredNames : List String
  declaredModes : List Monad.BufferAccessMode

/-- Global pipeline cache: maps WGSL source hash to cached pipeline.
    HashMap — a linear Array scan cost ~ms/token at ~600 dispatches. -/
initialize pipelineCacheRef : IO.Ref (Std.HashMap UInt64 CachedPipeline) ← IO.mkRef {}

/-- Pipeline cache hit counter -/
initialize cacheHitsRef : IO.Ref Nat ← IO.mkRef 0

/-- Pipeline cache miss counter -/
initialize cacheMissesRef : IO.Ref Nat ← IO.mkRef 0

/-- Look up a cached pipeline by key -/
private def findCachedPipeline (key : UInt64) (cache : Std.HashMap UInt64 CachedPipeline) : Option CachedPipeline :=
  cache.get? key

/-- Get pipeline cache statistics: (hits, misses) -/
def getPipelineCacheStats : IO (Nat × Nat) := do
  pure (← cacheHitsRef.get, ← cacheMissesRef.get)

/-- Reset pipeline cache (call when device is destroyed or for benchmarking) -/
def resetPipelineCache : IO Unit := do
  pipelineCacheRef.set {}
  cacheHitsRef.set 0
  cacheMissesRef.set 0

/-! ## Bind Group Cache

Caches WebGPU BindGroups keyed by (pipeline hash, buffer IDs).
BindGroup creation involves internal validation and allocation in the WebGPU runtime,
costing ~20-30µs per call. With 572 dispatches per token, this adds ~12-17ms overhead.

Since inference reuses the same pipelines with the same pre-allocated buffers,
bind groups are almost always identical across tokens. Caching eliminates
572 redundant `createBindGroup` calls per token.
-/

/-- Global bind group cache: maps (pipeline + buffer IDs) hash to cached BindGroup -/
initialize bindGroupCacheRef : IO.Ref (Std.HashMap UInt64 BindGroup) ← IO.mkRef {}

/-- Bind group cache hit/miss counters -/
initialize bgCacheHitsRef : IO.Ref Nat ← IO.mkRef 0
initialize bgCacheMissesRef : IO.Ref Nat ← IO.mkRef 0

/-- Look up a cached bind group -/
private def findCachedBindGroup (key : UInt64) (cache : Std.HashMap UInt64 BindGroup) : Option BindGroup :=
  cache.get? key

/-- Compute a bind group cache key from pipeline hash + buffer IDs (single FFI call) -/
private def computeBindGroupKey (pipelineKey : UInt64) (buffers : List Buffer) : IO UInt64 :=
  hashBufferArray pipelineKey buffers.toArray

/-- Get bind group cache statistics: (hits, misses) -/
def getBindGroupCacheStats : IO (Nat × Nat) := do
  pure (← bgCacheHitsRef.get, ← bgCacheMissesRef.get)

/-! ## PreparedDispatch (Graph Capture)

Pre-computed dispatch state for instant replay. Stores the pipeline and bind group
so that subsequent tokens skip ALL Lean-side processing:
- No WGSL generation/lookup
- No buffer name matching
- No bind group key computation
- No cache lookups
- Just one FFI call: recordDispatch

Usage:
```lean
-- In layer struct:
structure BitLinear where
  ...
  prepared : IO.Ref (Option PreparedDispatch)

-- In forward function:
def forward (device : Device) (layer : BitLinear) ... := do
  let (wx, wy, wz) := computeWorkgroups ...
  -- Fast path: replay if prepared
  if let some p ← layer.prepared.get then
    replayPreparedDispatch device p wx wy wz
    return
  -- Slow path: full execution (first token only)
  executeShaderNamed device shader namedBuffers config cacheKey (some layer.prepared)
```
-/

/-- Pre-computed dispatch: pipeline + bind group, ready for instant replay -/
structure PreparedDispatch where
  pipeline : ComputePipeline
  bindGroup : BindGroup

/-! ## Command Buffer Batching

Global batching mode: when enabled, `executeShaderNamed` records dispatches into
a shared command encoder instead of creating individual command buffers.
This eliminates per-dispatch overhead (encoder creation + submit + wait).

Usage:
```lean
beginBatch device           -- Create shared encoder
-- All executeShaderNamed calls now record instead of submit
layer1.forward device ...
layer2.forward device ...
endBatch device             -- Submit all + wait once
```
-/

/-- Global batch encoder: when `some`, executeShaderNamed records into it -/
initialize batchEncoderRef : IO.Ref (Option CommandEncoder) ← IO.mkRef none

/-- Dispatch counter for current batch (for diagnostics) -/
initialize batchDispatchCountRef : IO.Ref Nat ← IO.mkRef 0

/-- Begin command buffer batching. All subsequent `executeShaderNamed` calls
    will record into a shared encoder instead of submitting individually. -/
def beginBatch (device : Device) : IO Unit := do
  -- DG_NOBATCH: no-op batching so each dispatch submits+waits individually (fully serialized).
  -- Diagnostic for the batch buffer-reuse race + a deterministic (slow) reference mode.
  if (← IO.getEnv "DG_NOBATCH").isSome then return
  let existing ← batchEncoderRef.get
  if existing.isSome then
    throw <| IO.userError "beginBatch: already in batch mode"
  let encoder ← createCommandEncoder device
  batchEncoderRef.set (some encoder)
  batchDispatchCountRef.set 0

/-- End command buffer batching. Submits all recorded dispatches and waits. -/
def endBatch (device : Device) : IO Unit := do
  if (← IO.getEnv "DG_NOBATCH").isSome then return
  match ← batchEncoderRef.get with
  | none => throw <| IO.userError "endBatch: not in batch mode"
  | some encoder => do
    let count ← batchDispatchCountRef.get
    logVerbose s!"[Batch] Submitting {count} recorded dispatches"
    submitAndWait device encoder
    batchEncoderRef.set none
    batchDispatchCountRef.set 0

/-- Split the current batch WITHOUT a CPU sync: submit the recorded encoder (no wait) and start a
    fresh one. Keeps Dawn-on-Metal's per-encoder inter-pass barriers correct (it drops them in a
    very large single encoder) at a fraction of `endBatch`'s cost. Cross-encoder buffer hazards are
    handled by queue ordering + the driver. -/
def flushBatch (device : Device) : IO Unit := do
  if (← IO.getEnv "DG_NOBATCH").isSome then return
  match ← batchEncoderRef.get with
  | none => throw <| IO.userError "flushBatch: not in batch mode"
  | some encoder => do
    submitNoWait device encoder
    let enc ← createCommandEncoder device
    batchEncoderRef.set (some enc)
    batchDispatchCountRef.set 0

/-- Check if currently in batch mode -/
def isBatching : IO Bool := do
  return (← batchEncoderRef.get).isSome

/-! ## Section profiling

Lightweight per-section wall-clock timing for the "everything else" bucket
breakdown. When `sectionProfilingRef` is true, `withSection name act` wraps
`act` with monoNanos timestamps and accumulates total ns + call count into
`sectionTotalsRef` keyed by name. Nested sections are allowed (the outer
section's total includes nested ones). Only meaningful in unbatched mode
where each dispatch auto-syncs.
-/

initialize sectionProfilingRef : IO.Ref Bool ← IO.mkRef false
-- (name, totalNanos, callCount)
initialize sectionTotalsRef : IO.Ref (Array (String × UInt64 × Nat)) ← IO.mkRef #[]

/-- Per-section kernel-dispatch counter.  Populated by `withSection` when
    `sectionProfilingRef` is on.  The caller must wire `preDispatch` /
    `postDispatch` (readers of the global dispatch counter) so withSection
    can compute the dispatches attributed to one section. -/
initialize sectionDispatchesRef : IO.Ref (Array (String × Nat × Nat)) ← IO.mkRef #[]

/-- Callback to read the global dispatch counter.  Set by the CUDA backend
    via `registerDispatchCounter`.  When `none`, per-section dispatch
    counting is disabled. -/
initialize sectionDispatchReader : IO.Ref (Option (IO Nat)) ← IO.mkRef none

def registerDispatchCounter (reader : IO Nat) : IO Unit :=
  sectionDispatchReader.set (some reader)

private def addSectionSample (name : String) (ns : UInt64)
    (dispatchDelta : Nat) : IO Unit := do
  let arr ← sectionTotalsRef.get
  match arr.findIdx? (fun e => e.1 == name) with
  | some i =>
    let (n, t, c) := arr[i]!
    sectionTotalsRef.set (arr.set! i (n, t + ns, c + 1))
  | none =>
    sectionTotalsRef.set (arr.push (name, ns, 1))
  -- also accumulate dispatch count per section
  let darr ← sectionDispatchesRef.get
  match darr.findIdx? (fun e => e.1 == name) with
  | some i =>
    let (n, d, c) := darr[i]!
    sectionDispatchesRef.set (darr.set! i (n, d + dispatchDelta, c + 1))
  | none =>
    sectionDispatchesRef.set (darr.push (name, dispatchDelta, 1))

def resetSectionTotals : IO Unit := do
  sectionTotalsRef.set #[]
  sectionDispatchesRef.set #[]

def getSectionTotals : IO (Array (String × UInt64 × Nat)) := sectionTotalsRef.get

def getSectionDispatches : IO (Array (String × Nat × Nat)) := sectionDispatchesRef.get

/-- Chrome-trace event log for `withSection` calls.  When
    `sectionTraceRef` is on, every withSection invocation appends an
    entry (name, start_ns, dur_ns, depth).  Dumped to a JSON file by
    `dumpSectionTrace` at end-of-run.  Enabled by env
    `HESPER_TRACE_OUT=<path>`. -/
initialize sectionTraceRef : IO.Ref Bool ← IO.mkRef false
-- Reversed-order list (newest event first) for O(1) prepend.  We
-- reverse once at dump time.  An Array.push at every withSection call
-- is O(N) due to immutable copy and turns the run quadratic when
-- there are 50k+ events.
initialize sectionTraceEvents : IO.Ref (List (String × UInt64 × UInt64 × Nat))
  ← IO.mkRef []
initialize sectionTraceDepth : IO.Ref Nat ← IO.mkRef 0

private def appendSectionTrace (name : String) (startNs durNs : UInt64)
    (depth : Nat) : IO Unit :=
  sectionTraceEvents.modify (fun xs => (name, startNs, durNs, depth) :: xs)

/-- Dump accumulated section-trace events to a Chrome-trace-format
    JSON fragment file (just the `traceEvents` array body — the merger
    in `scripts/nsys_to_chrome_trace.py` reads it). -/
def dumpSectionTrace (path : String) : IO Unit := do
  let revList ← sectionTraceEvents.get
  let evs := revList.reverse
  let q : String := "\""
  let bs : String := "\\"
  let escQ : String := bs ++ q
  let mut out : String := "["
  let mut first := true
  for (name, ts, dur, depth) in evs do
    let escName := name.replace q escQ
    if !first then out := out ++ ","
    first := false
    out := out ++
      "{" ++ q ++ "name" ++ q ++ ":" ++ q ++ escName ++ q ++
      "," ++ q ++ "ts_ns" ++ q ++ ":" ++ s!"{ts}" ++
      "," ++ q ++ "dur_ns" ++ q ++ ":" ++ s!"{dur}" ++
      "," ++ q ++ "depth" ++ q ++ ":" ++ s!"{depth}" ++ "}"
  out := out ++ "]"
  IO.FS.writeFile path out

/-- Time an IO action under a section name; no-op when profiling disabled. -/
def withSection (name : String) (act : IO α) : IO α := do
  let traceOn := (← IO.getEnv "HESPER_KERNEL_TRACE").isSome
  let chromeTraceOn ← sectionTraceRef.get
  if (← sectionProfilingRef.get) || chromeTraceOn then
    let reader ← sectionDispatchReader.get
    let dBefore ← match reader with | some r => r | none => pure 0
    let depth ← if chromeTraceOn then do
        let d ← sectionTraceDepth.get
        sectionTraceDepth.set (d + 1)
        pure d
      else pure 0
    let t0 ← IO.monoNanosNow
    if traceOn then IO.eprintln s!"[hs][section begin] {name}"
    let r ← act
    if traceOn then IO.eprintln s!"[hs][section end  ] {name}"
    let t1 ← IO.monoNanosNow
    let dAfter ← match reader with | some r => r | none => pure 0
    if name == "embedLookup" || name == "attnNorm" then
      IO.eprintln s!"[debug-sec {name}] reader={reader.isSome} before={dBefore} after={dAfter}"
    if (← sectionProfilingRef.get) then
      addSectionSample name (t1 - t0).toUInt64 (dAfter - dBefore)
    if chromeTraceOn then do
      appendSectionTrace name t0.toUInt64 (t1 - t0).toUInt64 depth
      sectionTraceDepth.set depth
    pure r
  else if traceOn then do
    IO.eprintln s!"[hs][section begin] {name}"
    let r ← act
    IO.eprintln s!"[hs][section end  ] {name}"
    pure r
  else
    act

/-- Replay a prepared dispatch directly. Skips ALL Lean-side processing.
    Works in both batch mode (record into shared encoder) and standalone mode. -/
def replayPreparedDispatch (device : Device) (prepared : PreparedDispatch)
    (wx wy wz : Nat) : IO Unit := do
  match ← batchEncoderRef.get with
  | some encoder =>
    recordDispatch encoder prepared.pipeline prepared.bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    batchDispatchCountRef.modify (· + 1)
  | none =>
    let future ← dispatchCompute device prepared.pipeline prepared.bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    deviceWait future

/-- Compile a ShaderM computation to WGSL source code -/
def compileToWGSL
    (computation : ShaderM Unit)
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (extensions : List String := [])
    (diagnostics : List (String × String) := [])
    : String :=
  generateWGSL funcName workgroupSize extensions diagnostics computation

/-! ## CompiledKernel (Zero-Overhead Dispatch API)

Separates shader compilation from dispatch. A `CompiledKernel` holds the compiled
pipeline and binding layout, ready for buffer binding and dispatch without any
string matching, WGSL regeneration, or cache lookups.

Usage:
```lean
-- At initialization (once):
let kernel ← buildKernel device myShaderM config
let bg ← bindKernel device kernel [("input", inBuf), ("output", outBuf)]

-- At dispatch time (hot loop, zero overhead):
dispatchKernel device kernel bg (numWorkgroups, 1, 1)

-- Or combine into PreparedDispatch for even fewer indirections:
let prepared := kernel.prepare bg
replayPreparedDispatch device prepared wx wy wz
```
-/

/-- Pre-compiled kernel: pipeline + layout + binding order.
    Created once via `buildKernel`, reused across dispatches. -/
structure CompiledKernel where
  pipeline : ComputePipeline
  bindGroupLayout : BindGroupLayout
  declaredNames : Array String  -- Buffer names in binding order
  sourceHash : UInt64

namespace CompiledKernel

/-- Create a PreparedDispatch from this kernel and a bind group -/
def prepare (kernel : CompiledKernel) (bindGroup : BindGroup) : PreparedDispatch :=
  { pipeline := kernel.pipeline, bindGroup }

end CompiledKernel

/-- Compile a ShaderM computation into a reusable CompiledKernel.
    Uses the global pipeline cache. Thread-safe for repeated calls. -/
def buildKernel (device : Device) (computation : ShaderM Unit)
    (config : ExecutionConfig) : IO CompiledKernel := do
  let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics
  let sourceHash : UInt64 := hash wgslSource
  let cache ← pipelineCacheRef.get
  match findCachedPipeline sourceHash cache with
  | some cp =>
    cacheHitsRef.modify (· + 1)
    pure { pipeline := cp.pipeline, bindGroupLayout := cp.bindGroupLayout,
           declaredNames := cp.declaredNames.toArray, sourceHash }
  | none =>
    cacheMissesRef.modify (· + 1)
    let shaderModule ← createShaderModule device wgslSource
    let state := Monad.ShaderM.exec computation
    let declaredNames := state.declaredBuffers.map (·.1)
    let declaredModes := state.declaredBuffers.map (·.2.2)
    let layoutEntries := declaredModes.mapIdx fun i mode =>
      { binding := i.toUInt32
        visibility := ShaderStage.compute
        bindingType := BindingType.buffer (match mode with | .read => true | .readWrite => false) }
    let bindGroupLayout ← createBindGroupLayout device layoutEntries.toArray
    let pipelineDesc : ComputePipelineDescriptor := {
      shaderModule := shaderModule
      entryPoint := config.funcName
      bindGroupLayout := bindGroupLayout
    }
    let pipeline ← createComputePipeline device pipelineDesc
    pipelineCacheRef.modify (·.insert sourceHash {
      shaderModule := shaderModule
      bindGroupLayout := bindGroupLayout
      pipeline := pipeline
      declaredNames := declaredNames
      declaredModes := declaredModes
    })
    pure { pipeline, bindGroupLayout, declaredNames := declaredNames.toArray, sourceHash }

/-- Create a BindGroup by matching named buffers to a CompiledKernel's bindings.
    Uses the global bind group cache. -/
def bindKernel (device : Device) (kernel : CompiledKernel)
    (namedBuffers : List (String × Buffer)) : IO BindGroup := do
  let bgKey ← computeBindGroupKey kernel.sourceHash (namedBuffers.map (·.snd))
  let bgCache ← bindGroupCacheRef.get
  match findCachedBindGroup bgKey bgCache with
  | some bg =>
    bgCacheHitsRef.modify (· + 1)
    pure bg
  | none =>
    bgCacheMissesRef.modify (· + 1)
    let sortedBuffers := kernel.declaredNames.toList.filterMap fun name =>
      namedBuffers.find? (·.fst == name) |>.map (·.snd)
    if sortedBuffers.length != kernel.declaredNames.size then
      throw <| IO.userError s!"bindKernel: expected {kernel.declaredNames.size} buffers ({kernel.declaredNames.toList}), got {sortedBuffers.length}"
    let bindEntries := sortedBuffers.mapIdx fun i buf =>
      { binding := i.toUInt32, buffer := buf, offset := 0, size := 0 }
    let bg ← createBindGroup device kernel.bindGroupLayout bindEntries.toArray
    bindGroupCacheRef.modify (·.insert bgKey bg)
    pure bg

/-- Create a BindGroup from pre-sorted buffer array (no name matching).
    Buffers must be in binding order (matching kernel.declaredNames). -/
def bindKernelDirect (device : Device) (kernel : CompiledKernel)
    (buffers : Array Buffer) : IO BindGroup := do
  if buffers.size != kernel.declaredNames.size then
    throw <| IO.userError s!"bindKernelDirect: expected {kernel.declaredNames.size} buffers, got {buffers.size}"
  let bindEntries := buffers.mapIdx fun i buf =>
    { binding := i.toUInt32, buffer := buf, offset := 0, size := 0 }
  createBindGroup device kernel.bindGroupLayout bindEntries

/-- Dispatch a compiled kernel with a pre-built BindGroup.
    Zero string matching. Works in both batch mode and standalone mode. -/
def dispatchKernel (device : Device) (kernel : CompiledKernel) (bindGroup : BindGroup)
    (numWorkgroups : Nat × Nat × Nat) : IO Unit := do
  let (wx, wy, wz) := numWorkgroups
  match ← batchEncoderRef.get with
  | some encoder =>
    recordDispatch encoder kernel.pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    batchDispatchCountRef.modify (· + 1)
  | none =>
    let future ← dispatchCompute device kernel.pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    deviceWait future

/-- Create shader module from ShaderM computation -/
def createShaderFromComputation
    (device : Device)
    (computation : ShaderM Unit)
    (config : ExecutionConfig)
    : IO WebGPU.ShaderModule :=
  let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics
  createShaderModule device wgslSource

/-- Execute a ShaderM computation on the GPU with named buffers.

This is the main high-level execution function. It:
1. Compiles the ShaderM computation to WGSL
2. Looks up or creates the GPU pipeline (cached)
3. Binds buffers by name
4. Dispatches the compute shader
5. Waits for completion

Parameters:
- device: GPU device
- computation: ShaderM monad defining the shader
- namedBuffers: List of (name, buffer) pairs for binding
- config: Execution configuration (workgroup size, dispatch size)

Example:
```lean
let kernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

executeShaderNamed device kernel
  [("input", inputBuf), ("output", outputBuf)]
  (ExecutionConfig.dispatch1D 1024)
```
-/
def executeShaderNamed
    (device : Device)
    (computation : ShaderM Unit)
    (namedBuffers : List (String × Buffer))
    (config : ExecutionConfig)
    (cacheKey : Option UInt64 := none)
    (preparedRef : Option (IO.Ref (Option PreparedDispatch)) := none)
    : IO Unit := do
  -- Check cache: use cacheKey if provided (skips WGSL regeneration), otherwise generate and hash
  let cache ← pipelineCacheRef.get
  let (sourceHash, needsCompile) ← match cacheKey with
    | some key =>
      match findCachedPipeline key cache with
      | some _ => pure (key, false)
      | none => pure (key, true)
    | none =>
      let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics
      logVerbose s!"[Execute] Compiled shader ({wgslSource.length} bytes)"
      pure (hash wgslSource, true)

  let (pipeline, bindGroupLayout, declaredNames) ← do
    if !needsCompile then
      -- Cache hit (fast path: no WGSL generation needed when cacheKey was provided)
      match findCachedPipeline sourceHash cache with
      | some cp =>
        cacheHitsRef.modify (· + 1)
        pure (cp.pipeline, cp.bindGroupLayout, cp.declaredNames)
      | none => unreachable!  -- We checked above
    else
      -- Check cache again (for the no-cacheKey path where we just computed the hash)
      match findCachedPipeline sourceHash cache with
      | some cp =>
        cacheHitsRef.modify (· + 1)
        pure (cp.pipeline, cp.bindGroupLayout, cp.declaredNames)
      | none =>
        cacheMissesRef.modify (· + 1)
        -- Generate WGSL (may already have been done for hash)
        let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics
        let shaderModule ← createShaderModule device wgslSource
        let state := ShaderM.exec computation
        let declaredNames := state.declaredBuffers.map (·.1)
        let declaredModes := state.declaredBuffers.map (·.2.2)
        let layoutEntries := declaredModes.mapIdx fun i mode =>
          { binding := i.toUInt32
            visibility := ShaderStage.compute
            bindingType := BindingType.buffer (match mode with | .read => true | .readWrite => false) }
        let bindGroupLayout ← createBindGroupLayout device layoutEntries.toArray
        let pipelineDesc : ComputePipelineDescriptor := {
          shaderModule := shaderModule
          entryPoint := config.funcName
          bindGroupLayout := bindGroupLayout
        }
        let pipeline ← createComputePipeline device pipelineDesc
        pipelineCacheRef.modify (·.insert sourceHash {
          shaderModule := shaderModule
          bindGroupLayout := bindGroupLayout
          pipeline := pipeline
          declaredNames := declaredNames
          declaredModes := declaredModes
        })
        pure (pipeline, bindGroupLayout, declaredNames)

  -- Bind group cache: compute key from namedBuffers BEFORE name matching (skip matching on hit)
  let bgKey ← computeBindGroupKey sourceHash (namedBuffers.map (·.snd))
  let bgCache ← bindGroupCacheRef.get
  let bindGroup ← match findCachedBindGroup bgKey bgCache with
    | some bg =>
      bgCacheHitsRef.modify (· + 1)
      pure bg
    | none =>
      bgCacheMissesRef.modify (· + 1)
      -- Only do buffer name matching on cache miss
      let sortedBuffers := declaredNames.filterMap fun name =>
        namedBuffers.find? (·.fst == name) |>.map (·.snd)
      if sortedBuffers.length != declaredNames.length then
        IO.println s!"[Execute] ERROR: Buffer count mismatch!"
        IO.println s!"  Expected: {declaredNames}"
        IO.println s!"  Provided: {namedBuffers.map (·.fst)}"
        throw <| IO.userError "Buffer binding mismatch"
      let bindEntries := sortedBuffers.mapIdx fun i buf =>
        { binding := i.toUInt32
          buffer := buf
          offset := 0
          size := 0 }  -- 0 means whole buffer
      let bg ← createBindGroup device bindGroupLayout bindEntries.toArray
      bindGroupCacheRef.modify (·.insert bgKey bg)
      pure bg

  -- Save PreparedDispatch for future instant replay (first token only)
  if let some ref := preparedRef then
    ref.set (some { pipeline, bindGroup })

  -- Check if we're in batch mode
  let (wx, wy, wz) := config.numWorkgroups
  match ← batchEncoderRef.get with
  | some encoder =>
    -- Batch mode: record into shared encoder (no submit, no wait)
    recordDispatch encoder pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    batchDispatchCountRef.modify (· + 1)
  | none =>
    -- Normal mode: dispatch + wait (original behavior)
    logVerbose s!"[Execute] Dispatching {wx}×{wy}×{wz} workgroups..."
    let future ← dispatchCompute device pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32
    deviceWait future
    logVerbose "[Execute] Completed successfully"

/-- Record a ShaderM computation into a command encoder (no submit, no wait).

This is the batched variant of `executeShaderNamed`. Instead of creating its own
command encoder and waiting, it records the dispatch into a pre-existing encoder.
The caller must call `submitAndWait` after recording all dispatches.

Pipeline caching is shared with `executeShaderNamed`.
-/
def executeShaderRecorded
    (device : Device)
    (encoder : CommandEncoder)
    (computation : ShaderM Unit)
    (namedBuffers : List (String × Buffer))
    (config : ExecutionConfig)
    : IO Unit := do
  -- Compile to WGSL (pure Lean, fast)
  let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics

  -- Check pipeline cache
  let sourceHash := hash wgslSource
  let cache ← pipelineCacheRef.get

  let (pipeline, bindGroupLayout, declaredNames) ← do
    match findCachedPipeline sourceHash cache with
    | some cp => do
      cacheHitsRef.modify (· + 1)
      pure (cp.pipeline, cp.bindGroupLayout, cp.declaredNames)
    | none => do
      cacheMissesRef.modify (· + 1)
      let shaderModule ← createShaderModule device wgslSource
      let state := ShaderM.exec computation
      let declaredNames := state.declaredBuffers.map (·.1)
      let declaredModes := state.declaredBuffers.map (·.2.2)
      let layoutEntries := declaredModes.mapIdx fun i mode =>
        { binding := i.toUInt32
          visibility := ShaderStage.compute
          bindingType := BindingType.buffer (match mode with | .read => true | .readWrite => false) }
      let bindGroupLayout ← createBindGroupLayout device layoutEntries.toArray
      let pipelineDesc : ComputePipelineDescriptor := {
        shaderModule := shaderModule
        entryPoint := config.funcName
        bindGroupLayout := bindGroupLayout
      }
      let pipeline ← createComputePipeline device pipelineDesc
      pipelineCacheRef.modify (·.insert sourceHash {
        shaderModule := shaderModule
        bindGroupLayout := bindGroupLayout
        pipeline := pipeline
        declaredNames := declaredNames
        declaredModes := declaredModes
      })
      pure (pipeline, bindGroupLayout, declaredNames)

  -- Match buffers to bindings by name
  let sortedBuffers := declaredNames.filterMap fun name =>
    namedBuffers.find? (·.fst == name) |>.map (·.snd)

  if sortedBuffers.length != declaredNames.length then
    throw <| IO.userError "Buffer binding mismatch (batched)"

  let bindEntries := sortedBuffers.mapIdx fun i buf =>
    { binding := i.toUInt32
      buffer := buf
      offset := 0
      size := 0 }

  let bindGroup ← createBindGroup device bindGroupLayout bindEntries.toArray

  -- Record dispatch into encoder (no submit, no wait)
  let (wx, wy, wz) := config.numWorkgroups
  recordDispatch encoder pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32

/-- Execute a simple ShaderM computation with a single input/output buffer.

Convenience wrapper for the common case of one input buffer and one output buffer.

Example:
```lean
let kernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

executeShaderSimple device kernel inputBuf outputBuf 1024
```
-/
def executeShaderSimple
    (device : Device)
    (computation : ShaderM Unit)
    (inputBuffer : Buffer)
    (outputBuffer : Buffer)
    (numThreads : Nat)
    : IO Unit :=
  executeShaderNamed device computation
    [("input", inputBuffer), ("output", outputBuffer)]
    (ExecutionConfig.dispatch1D numThreads)

/-- Print generated WGSL for debugging -/
def debugPrintWGSL
    (computation : ShaderM Unit)
    (config : ExecutionConfig := ExecutionConfig.default (1, 1, 1))
    : IO Unit := do
  let wgsl := compileToWGSL computation config.funcName config.workgroupSize []
  IO.println "═══════════════════════════════════════════════"
  IO.println "Generated WGSL Shader:"
  IO.println "═══════════════════════════════════════════════"
  IO.println wgsl
  IO.println "═══════════════════════════════════════════════"

end Hesper.WGSL.Execute
