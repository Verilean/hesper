import Hesper.WebGPU.Types

namespace Hesper.WebGPU

/-- Compile a WGSL shader module from source code.
    Resources are automatically cleaned up by Lean's GC via External finalizers.
    @param device The GPU device
    @param source WGSL shader source code
    @return Compiled shader module
-/
@[extern "lean_hesper_create_shader_module"]
opaque createShaderModule (device : @& Device) (source : @& String) : IO ShaderModule

end Hesper.WebGPU
