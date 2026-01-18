import Hesper.WebGPU.Types

namespace Hesper.WebGPU

/-- GPU adapter information -/
structure AdapterInfo where
  name : String
  backendType : Nat  -- 0=Null, 1=WebGPU, 2=D3D11, 3=D3D12, 4=Metal, 5=Vulkan, 6=OpenGL, 7=OpenGLES
  deriving Repr

/-- Get the number of available GPU adapters -/
@[extern "lean_hesper_get_adapter_count"]
opaque getAdapterCount : IO Nat

/-- Get information about a specific GPU adapter by index -/
@[extern "lean_hesper_get_adapter_info"]
opaque getAdapterInfo (gpuIdx : @& UInt32) : IO AdapterInfo

/-- Get the default GPU device.
    This is a simplified wrapper that gets the first available adapter
    and creates a device from it. -/
@[extern "lean_hesper_get_device"]
opaque getDevice : IO Device

/-- Get a GPU device from a specific adapter index.
    Use this to select which GPU to use in multi-GPU systems.
    Example:
    ```lean
    let device ← getDeviceByIndex 0  -- Use first GPU
    let device ← getDeviceByIndex 1  -- Use second GPU
    ```
-/
@[extern "lean_hesper_get_device_by_index"]
opaque getDeviceByIndex (gpuIdx : @& UInt32) : IO Device

/-- Release a device (cleanup resources) -/
@[extern "lean_hesper_release_device"]
opaque releaseDevice (device : @& Device) : IO Unit

/-- Tick the device (process callbacks and events).
    Should be called regularly when doing async operations. -/
@[extern "lean_hesper_device_tick"]
opaque deviceTick (device : @& Device) : IO Unit

/-- Wait for all device operations to complete -/
@[extern "lean_hesper_device_wait"]
opaque deviceWait (device : @& Device) : IO Unit

/-- List all available GPU adapters with their information -/
def listAdapters : IO Unit := do
  let count ← getAdapterCount
  IO.println s!"Found {count} GPU adapter(s):"
  for i in [0:count] do
    let info ← getAdapterInfo i.toUInt32
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
