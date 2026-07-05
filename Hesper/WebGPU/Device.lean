import Hesper.WebGPU.Types

namespace Hesper.WebGPU

/-- GPU adapter information -/
structure AdapterInfo where
  name : String
  backendType : Nat  -- 0=Null, 1=WebGPU, 2=D3D11, 3=D3D12, 4=Metal, 5=Vulkan, 6=OpenGL, 7=OpenGLES
  deriving Repr

/-- Get the number of available GPU adapters -/
@[extern "lean_hesper_get_adapter_count"]
opaque getAdapterCount (inst : @& Instance) : IO Nat

/-- Get information about a specific GPU adapter by index -/
@[extern "lean_hesper_get_adapter_info"]
opaque getAdapterInfo (inst : @& Instance) (gpuIdx : @& UInt32) : IO AdapterInfo

/-- Get the default GPU device.
    This is a simplified wrapper that gets the first available adapter
    and creates a device from it. -/
@[extern "lean_hesper_get_device"]
opaque getDevice (inst : @& Instance) : IO Device

/-- Get a GPU device with advanced features enabled (Subgroups, Float16).
    This is required for high-performance compute tasks like subgroup matrix multiplication. -/
@[extern "lean_hesper_get_device_with_features"]
opaque getDeviceWithFeatures (inst : @& Instance) : IO Device

/-- Get a GPU device from a specific adapter index.
    Use this to select which GPU to use in multi-GPU systems.
    Example:
    ```lean
    let device ← getDeviceByIndex inst 0  -- Use first GPU
    let device ← getDeviceByIndex inst 1  -- Use second GPU
    ```
-/
@[extern "lean_hesper_get_device_by_index"]
opaque getDeviceByIndex (inst : @& Instance) (gpuIdx : @& UInt32) : IO Device

/-- Check if the device was created with subgroup support.
    Returns `true` if `subgroupAdd` and related operations are available. -/
@[extern "lean_hesper_device_has_subgroups"]
opaque deviceHasSubgroups (device : @& Device) : IO Bool

/-- metal_replacer STEP 1 PoC: the name/props of the MTLDevice behind this WGPUDevice (via Dawn's
    GetMTLDevice). Proves the Metal interop is live — the foundation for swapping in llama.cpp's Metal
    kernels. See METAL_REPLACER_INTEGRATION.md. -/
@[extern "lean_hesper_mtl_device_name"]
opaque mtlDeviceName (device : @& Device) : IO String

/-- DG_GPUBUSY: read+reset the compute-pass count + host (record/finish) vs GPU (submit+wait)
    time split accumulated since the last read. Quantifies encoder-per-dispatch host overhead. -/
@[extern "lean_hesper_gpubusy_read"]
opaque gpuBusyRead : IO String

/-- DEVPLAN M1: the most recent Tint-MSL dump captured in-process by the HESPER_DUMP_MSL logging
    callback (empty if nothing dumped yet / env not set). Feed to `mslOccupancyProbe`. Set
    HESPER_DUMP_MSL_QUIET=1 to capture without the stderr spew (sweep mode). -/
@[extern "lean_hesper_last_dumped_msl"]
opaque lastDumpedMsl : IO String

/-- DEVPLAN M1: compile a Tint-dumped MSL source with the real Metal compiler and report the
    pipeline's resource stats: "maxThreads=N execWidth=M tgMem=K". maxThreads is the
    register-pressure inverse signal (drops on spill); this is the MANDATORY resource column of
    the autotune sweep. macOS only (IO error elsewhere). -/
@[extern "lean_hesper_msl_occupancy_probe"]
opaque mslOccupancyProbe (device : @& Device) (msl : @& String) : IO String

/-- Native serialized GPU timing of a dumped Tint-MSL kernel: `nDispatches` back-to-back in
    one SERIAL Metal encoder, GPU ms from GPUStart/EndTime. Buffers in BINDING ORDER.
    The honest per-kernel bench (Dawn adds ~35 µs/dispatch that drowns small matvecs). -/
@[extern "lean_hesper_msl_bench_serial"]
opaque mslBenchSerial (device : @& Device) (msl : @& String) (bufs : @& Array Buffer)
    (nDispatches gridX gridY gridZ wgX : UInt32) : IO String

/-- M3 de-risk probe: run `nDispatches` of a dumped Tint-MSL kernel back-to-back in ONE
    native Metal encoder with Serial or Concurrent dispatch type (no barriers — timing
    only, racing writes tolerated). Returns GPU wall ms as a string. -/
@[extern "lean_hesper_msl_concurrent_probe"]
opaque mslConcurrentProbe (device : @& Device) (msl : @& String)
    (b0 b1 b2 : @& Buffer) (nDispatches gridX gridY wgX : UInt32)
    (concurrent : UInt8) : IO String

/-- DG_GPUBUSY: read+reset the MSL-path split (count / WaitForCommandsToBeScheduled / encode+commit
    / kernel GPU time). Complements gpuBusyRead (which covers the Dawn WGSL path only). -/
@[extern "lean_hesper_msl_busy_read"]
opaque mslBusyRead : IO String

/-- Exp 2 Phase A (native replay): clear the recorded dispatch sequence. -/
@[extern "lean_hesper_replay_reset"]
opaque replayReset : IO Unit

/-- Exp 2 Phase A: push a barrier marker (layer boundary) into the replay sequence.
    Only honored by `replayRun mode=2` (concurrent + barriers). -/
@[extern "lean_hesper_replay_barrier"]
opaque replayBarrier : IO Unit

/-- Exp 2 Phase A: record one dispatch for native replay. `bufs` must be in MSL
    `[[buffer(i)]]` order (parse the Tint-CLI entry signature); `entry` = the MSL kernel
    function name; `tgBytes` = threadgroup memory upper bound (0 if none; set via
    setThreadgroupMemoryLength at index 0). PSO compiled once per unique MSL. -/
@[extern "lean_hesper_replay_record"]
opaque replayRecord (device : @& Device) (msl : @& String) (entry : @& String)
    (bufs : @& Array Buffer) (gx gy gz tx ty tz tgBytes : UInt32) : IO Unit

/-- Exp 2 Phase A: replay the recorded token in ONE native command buffer.
    mode: 0 = Serial (sanity vs Dawn GPU time), 1 = Concurrent no-barrier (upper bound),
    2 = Concurrent + barriers at recorded markers (realistic). Returns
    "count=<ops> min=<ms> avg=<ms>". TIMING ONLY — buffer contents end up garbage.
    Device comes from the record-time stash, so this is callable from
    backend-generic code. -/
@[extern "lean_hesper_replay_run"]
opaque replayRun (mode iters : UInt32) : IO String

/-- metal_replacer STEP 4: Apple's tuned MPS f16 matmul (C=A·Bᵀ, our reg-matmul shape) as the CEILING —
    returns "ms/call | GFLOPS | %peak". Diff vs the WGSL reg (harness) at the same shape to quantify the
    WGSL→Tint→Metal gap. macOS DEBUG/REFERENCE only. See METAL_REPLACER_INTEGRATION.md. -/
@[extern "lean_hesper_mps_matmul_bench"]
opaque mpsMatmulBench (device : @& Device) (M N K iters : UInt32) : IO String

/-- metal_replacer MSL PoC: bench the hand-written native-Metal port of
    q4kMatmulGroupedRegIndexedKernel on the given Dawn buffers (same algorithm as the WGSL kernel).
    Returns ms/iter from MTLCommandBuffer GPU timestamps. Caller syncs input writes first and
    writes `c` once (Dawn lazy-clear). macOS DEBUG/REFERENCE only. -/
@[extern "lean_hesper_msl_q4k_bench"]
opaque mslQ4kBench (device : @& Device) (src idx b c te tr : @& Buffer)
    (M N K nExpert srcRows iters : UInt32) : IO String

/-- HOT-PATH MSL q4k gate/up dispatch (DG_MSL, ~1.61× vs WGSL/Tint). PSO cached after the first
    call; encode+commit with NO CPU wait. ORDERING CONTRACT: caller must flushBatch (commit the
    Dawn producer encoder) immediately BEFORE; Dawn buffers are hazard-tracked so Metal orders the
    command buffers by commit order on the shared buffers. macOS only. -/
@[extern "lean_hesper_msl_q4k_dispatch"]
opaque mslQ4kDispatch (device : @& Device) (src idx b c te tr : @& Buffer)
    (M N K nExpert srcRows : UInt32) : IO Unit

/-- HOT-PATH MSL Q8_0 MoE-down dispatch (DG_MSLDOWN): the native port of
    q8MatmulGroupedRegIndexedScatterKernel (A read direct from the grouped geglu output, ragged
    sub-tile skip, C scatter-on-store through pos/slot into dst[slot,NTOK,N]). Same ordering
    contract as mslQ4kDispatch (flushBatch before; hazard-tracked commit order). macOS only. -/
@[extern "lean_hesper_msl_q8down_dispatch"]
opaque mslQ8DownDispatch (device : @& Device) (a b te tr pos slot dst : @& Buffer)
    (M N K nExpert nUsed nTok : UInt32) : IO Unit

/-- HOT-PATH MSL Q5_0 MoE-down dispatch: the Q5_0 (22B/block) analogue of mslQ8DownDispatch —
    covers the 16/30 layers whose down_exps are Q5_0 (previously the WGSL warp fallback + staged
    scatter). Same ordering contract. macOS only. -/
@[extern "lean_hesper_msl_q5down_dispatch"]
opaque mslQ5DownDispatch (device : @& Device) (a b te tr pos slot dst : @& Buffer)
    (M N K nExpert nUsed nTok : UInt32) : IO Unit

/-- SINGLE-STREAM (DG_MSLONESTREAM): encode the MSL gate/up (q4k) AND the FUSED MSL down (q8/q5,
    reads sGatheredGU + inline geglu) into ONE MTLCommandBuffer with two compute encoders + ONE commit,
    so the gate/up→down MSL chain runs back-to-back with no inter-cb handoff bubble. gate/up writes
    guC=sGatheredGU which the down reads as its A; down reuses guIdx as pos. isQ5 selects the down
    kernel. Implies the fused down. macOS only. -/
@[extern "lean_hesper_msl_gateup_down_onecb"]
opaque mslGateupDownOnecb (device : @& Device)
    (guSrc guIdx guB guC te tr dnB dnSlot dnDst : @& Buffer)
    (maxPadded guN guK nExpert srcRows dnN dnK nUsed nTok isQ5 : UInt32) : IO Unit

/-- Check if the device was created with the Chromium experimental
    subgroup matrix feature. `subgroup_matrix_left/right/result` types
    and `subgroupMatrixLoad/Store/MultiplyAccumulate` are available iff
    this returns `true`. -/
@[extern "lean_hesper_device_has_subgroup_matrix"]
opaque deviceHasSubgroupMatrix (device : @& Device) : IO Bool

/-- Check if the device was created with ShaderF16 support. `f16` values
    and related arithmetic are available iff this returns `true`. -/
@[extern "lean_hesper_device_has_shader_f16"]
opaque deviceHasShaderF16 (device : @& Device) : IO Bool

/-- Tick the device (process callbacks and events).
    Should be called regularly when doing async operations. -/
@[extern "lean_hesper_device_tick"]
opaque deviceTick (device : @& Device) : IO Unit

/-- Wait for GPU work to complete (takes Future from dispatchCompute) -/
@[extern "lean_hesper_device_wait"]
opaque deviceWait (future : @& Future) : IO Unit

/-- List all available GPU adapters with their information -/
def listAdapters (inst : Instance) : IO Unit := do
  let count ← getAdapterCount inst
  IO.println s!"Found {count} GPU adapter(s):"
  for i in [0:count] do
    let info ← getAdapterInfo inst i.toUInt32
    let backend := match info.backendType with
      | 0 => "Null"
      | 1 => "WebGPU"
      | 2 => "D3D11"
      | 3 => "D3D12"
      | 4 => "Metal"
      | 5 => "Vulkan"
      | 6 => "OpenGL"
      | 7 => "OpenGLES"
      | _ => "Unknown"
    IO.println s!"  [{i}] {info.name} (Backend: {backend})"

end Hesper.WebGPU
