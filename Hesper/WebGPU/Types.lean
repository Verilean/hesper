/-!
# WebGPU Type Definitions

Opaque types for WebGPU resources. Resources are managed via Lean External objects
with automatic cleanup by finalizers on the C++ side.
-/

namespace Hesper.WebGPU

/-- Opaque handle to a WebGPU Instance (Dawn native instance) with automatic cleanup via External finalizer -/
opaque Instance : Type

/-- Opaque handle to a WebGPU Device with automatic cleanup via External finalizer -/
opaque Device : Type

/-- Opaque handle to a WebGPU Buffer with automatic cleanup via External finalizer -/
opaque Buffer : Type

/-- Opaque handle to a WebGPU Shader Module with automatic cleanup via External finalizer -/
opaque ShaderModule : Type

/-- Opaque handle to a WebGPU Compute Pipeline with automatic cleanup via External finalizer -/
opaque ComputePipeline : Type

/-- Opaque handle to a WebGPU Bind Group with automatic cleanup via External finalizer -/
opaque BindGroup : Type

/-- Opaque handle to a WebGPU Bind Group Layout with automatic cleanup via External finalizer -/
opaque BindGroupLayout : Type

/-- Opaque handle to a WebGPU Command Encoder with automatic cleanup via External finalizer -/
opaque CommandEncoder : Type

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
