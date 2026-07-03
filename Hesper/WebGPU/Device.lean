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
