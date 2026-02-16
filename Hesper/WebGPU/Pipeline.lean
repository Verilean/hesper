import Hesper.WebGPU.Types

namespace Hesper.WebGPU

/-- Bind group layout entry descriptor -/
structure BindGroupLayoutEntry where
  binding : UInt32          -- Binding number
  visibility : ShaderStage  -- Shader stage visibility
  bindingType : BindingType -- Type of binding

instance : Inhabited BindGroupLayoutEntry where
  default := { binding := 0, visibility := .compute, bindingType := .buffer false }

/-- Create a bind group layout.
    Defines the interface between shader and bound resources.
    Resources are automatically cleaned up by Lean's GC via External finalizers.
    @param device The GPU device
    @param entries List of binding entries
-/
@[extern "lean_hesper_create_bind_group_layout"]
opaque createBindGroupLayout (device : @& Device) (entries : @& Array BindGroupLayoutEntry) : IO BindGroupLayout

/-- Bind group entry - associates a buffer with a binding point -/
structure BindGroupEntry where
  binding : UInt32  -- Binding number (matches shader)
  buffer : Buffer   -- The buffer to bind
  offset : USize    -- Offset in bytes
  size : USize      -- Size in bytes

/-- Create a bind group.
    Binds actual resources (buffers) to the layout.
    Resources are automatically cleaned up by Lean's GC via External finalizers.
    @param device The GPU device
    @param layout The bind group layout
    @param entries The buffer bindings
-/
@[extern "lean_hesper_create_bind_group"]
opaque createBindGroup (device : @& Device) (layout : @& BindGroupLayout) (entries : @& Array BindGroupEntry) : IO BindGroup

/-- Compute pipeline descriptor -/
structure ComputePipelineDescriptor where
  shaderModule : ShaderModule    -- Compiled shader
  entryPoint : String            -- Entry function name (e.g., "main")
  bindGroupLayout : BindGroupLayout  -- Resource layout

/-- Create a compute pipeline.
    Resources are automatically cleaned up by Lean's GC via External finalizers.
    @param device The GPU device
    @param desc Pipeline configuration
-/
@[extern "lean_hesper_create_compute_pipeline"]
opaque createComputePipeline (device : @& Device) (desc : @& ComputePipelineDescriptor) : IO ComputePipeline

/-- Dispatch compute work (async - returns Future).
    @param device The GPU device
    @param pipeline The compute pipeline to execute
    @param bindGroup The bound resources
    @param workgroupsX Number of workgroups in X dimension
    @param workgroupsY Number of workgroups in Y dimension (default 1)
    @param workgroupsZ Number of workgroups in Z dimension (default 1)
    @return Future for GPU work completion (pass to deviceWait to wait)
-/
@[extern "lean_hesper_dispatch_compute"]
opaque dispatchCompute
  (device : @& Device)
  (pipeline : @& ComputePipeline)
  (bindGroup : @& BindGroup)
  (workgroupsX : UInt32)
  (workgroupsY : UInt32 := 1)
  (workgroupsZ : UInt32 := 1) : IO Future

/-! ## Command Buffer Batching

Record multiple dispatches into a single command encoder, then submit once.
Eliminates per-dispatch overhead (command encoder creation + queue submit + wait).
-/

/-- Create a new command encoder for recording multiple dispatches. -/
@[extern "lean_hesper_create_command_encoder"]
opaque createCommandEncoder (device : @& Device) : IO CommandEncoder

/-- Record a compute dispatch into an existing command encoder (no submit, no wait). -/
@[extern "lean_hesper_record_dispatch"]
opaque recordDispatch
  (encoder : @& CommandEncoder)
  (pipeline : @& ComputePipeline)
  (bindGroup : @& BindGroup)
  (workgroupsX : UInt32)
  (workgroupsY : UInt32 := 1)
  (workgroupsZ : UInt32 := 1) : IO Unit

/-- Finish the command encoder, submit to queue, and wait for all recorded work to complete. -/
@[extern "lean_hesper_submit_and_wait"]
opaque submitAndWait (device : @& Device) (encoder : @& CommandEncoder) : IO Unit

end Hesper.WebGPU
