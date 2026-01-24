/-!
# WebGPU Type Definitions

Opaque types for WebGPU resources. Resources are managed via Lean External objects
with automatic cleanup by finalizers on the C++ side.
-/

namespace Hesper.WebGPU

/-- Opaque handle to a WebGPU Instance (Dawn native instance) with automatic cleanup via External finalizer -/
opaque Instance : Type

/-- Internal opaque handle to a GPU work completion Future -/
opaque FuturePtr : Type

/-- GPU work completion Future that maintains a reference to its parent Instance.
    This ensures the Instance stays alive while waiting for GPU work to complete. -/
structure Future where
  ptr : FuturePtr
  parentInstance : Instance

/-- Internal opaque handle to the raw WebGPU Device pointer -/
opaque DevicePtr : Type

/-- WebGPU Device that maintains a reference to its parent Instance.
    This ensures the Instance stays alive as long as the Device is in use,
    preventing premature garbage collection that would cause segfaults. -/
structure Device where
  ptr : DevicePtr
  parentInstance : Instance

/-- Internal opaque handles for raw WebGPU pointers -/
opaque BufferPtr : Type
opaque ShaderModulePtr : Type
opaque ComputePipelinePtr : Type
opaque BindGroupPtr : Type
opaque BindGroupLayoutPtr : Type
opaque CommandEncoderPtr : Type

/-- WebGPU Buffer that maintains a reference to its parent Device.
    This ensures Device (and Instance) stay alive while Buffer is in use. -/
structure Buffer where
  ptr : BufferPtr
  parentDevice : Device

/-- WebGPU Shader Module that maintains a reference to its parent Device. -/
structure ShaderModule where
  ptr : ShaderModulePtr
  parentDevice : Device

/-- WebGPU Compute Pipeline that maintains a reference to its parent Device. -/
structure ComputePipeline where
  ptr : ComputePipelinePtr
  parentDevice : Device

/-- WebGPU Bind Group that maintains a reference to its parent Device. -/
structure BindGroup where
  ptr : BindGroupPtr
  parentDevice : Device

/-- WebGPU Bind Group Layout that maintains a reference to its parent Device. -/
structure BindGroupLayout where
  ptr : BindGroupLayoutPtr
  parentDevice : Device

/-- WebGPU Command Encoder that maintains a reference to its parent Device. -/
structure CommandEncoder where
  ptr : CommandEncoderPtr
  parentDevice : Device

/-- Buffer usage flags -/
inductive BufferUsage where
  | storage    -- Storage buffer (read/write in shaders)
  | uniform    -- Uniform buffer (read-only in shaders)
  | copyDst    -- Can be used as copy destination
  | copySrc    -- Can be used as copy source
  | mapRead    -- Can be mapped for reading
  | mapWrite   -- Can be mapped for writing
  deriving Repr, BEq

/-- Shader stage visibility -/
inductive ShaderStage where
  | vertex
  | fragment
  | compute
  deriving Repr, BEq

/-- Binding type for bind group layout -/
inductive BindingType where
  | buffer (readOnly : Bool)   -- Storage buffer
  | uniformBuffer              -- Uniform buffer
  | sampler                    -- Texture sampler
  | texture                    -- Texture
  deriving Repr, BEq

end Hesper.WebGPU
