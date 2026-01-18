import Hesper.GLFW.Types
import Hesper.WebGPU.Types

/-!
# GLFW Internal FFI Bindings

Low-level FFI bindings to GLFW and WebGPU rendering functions.

Resources are automatically managed via Lean.External with C++ finalizers.
No manual cleanup is required.
-/

namespace Hesper.GLFW.Internal

open Hesper.GLFW
open Hesper.WebGPU

-- GLFW Initialization

/-- Initialize GLFW. Must be called before creating windows. Throws error on failure. -/
@[extern "lean_glfw_init"]
opaque glfwInit : IO Unit

/-- Terminate GLFW. Should be called when done with all windows. -/
@[extern "lean_glfw_terminate"]
opaque glfwTerminate : IO Unit

-- Window Management

/-- Create a window with the given dimensions and title. Automatically cleaned up by GC. -/
@[extern "lean_glfw_create_window"]
opaque createWindow (width : UInt32) (height : UInt32) (title : String) : IO Window

/-- Check if a window should close -/
@[extern "lean_glfw_window_should_close"]
opaque windowShouldClose (window : Window) : IO Bool

/-- Poll for window events (must be called regularly in the render loop) -/
@[extern "lean_glfw_poll_events"]
opaque pollEvents : IO Unit

/-- Get key state for a specific key -/
@[extern "lean_glfw_window_get_key"]
opaque windowGetKey (window : Window) (key : UInt32) : IO UInt32

-- Surface Management

/-- Create a WebGPU surface for the given window. Automatically cleaned up by GC. -/
@[extern "lean_glfw_create_surface"]
opaque createSurface (device : Device) (window : Window) : IO Surface

/-- Configure the surface with the given dimensions and format -/
@[extern "lean_glfw_configure_surface"]
opaque configureSurface (surface : Surface) (width : UInt32) (height : UInt32) (format : UInt32) : IO Unit

/-- Get the preferred texture format for the surface -/
@[extern "lean_glfw_surface_get_preferred_format"]
opaque getSurfacePreferredFormat (surface : Surface) : IO UInt32

/-- Get the current texture from the surface for rendering. Automatically cleaned up by GC. -/
@[extern "lean_glfw_surface_get_current_texture"]
opaque getCurrentTexture (surface : Surface) : IO Texture

/-- Present the surface (swap buffers) -/
@[extern "lean_glfw_surface_present"]
opaque surfacePresent (surface : Surface) : IO Unit

-- Texture Management

/-- Create a view of the texture for rendering. Automatically cleaned up by GC. -/
@[extern "lean_glfw_texture_create_view"]
opaque createTextureView (texture : Texture) : IO TextureView

-- Command Encoding

/-- Create a command encoder for recording commands. Automatically cleaned up by GC. -/
@[extern "lean_glfw_create_command_encoder"]
opaque createCommandEncoder (device : Device) : IO CommandEncoder

/-- Begin a render pass with the given texture view. Automatically cleaned up by GC. -/
@[extern "lean_glfw_begin_render_pass"]
opaque beginRenderPass (encoder : CommandEncoder) (view : TextureView) : IO RenderPassEncoder

/-- Set the render pipeline for the render pass -/
@[extern "lean_glfw_render_pass_set_pipeline"]
opaque setRenderPipeline (pass : RenderPassEncoder) (pipeline : RenderPipeline) : IO Unit

/-- Draw vertices -/
@[extern "lean_glfw_render_pass_draw"]
opaque draw (pass : RenderPassEncoder) (vertexCount : UInt32) : IO Unit

/-- End the render pass -/
@[extern "lean_glfw_render_pass_end"]
opaque endRenderPass (pass : RenderPassEncoder) : IO Unit

/-- Finish recording commands and create a command buffer. Automatically cleaned up by GC. -/
@[extern "lean_glfw_encoder_finish"]
opaque finishEncoder (encoder : CommandEncoder) : IO CommandBuffer

/-- Submit a command buffer to the GPU queue -/
@[extern "lean_glfw_queue_submit"]
opaque submitCommand (device : Device) (cmd : CommandBuffer) : IO Unit

-- Pipeline Management

/-- Create a shader module from WGSL code. Automatically cleaned up by GC. -/
@[extern "lean_glfw_create_shader_module"]
opaque createShaderModule (device : Device) (code : String) : IO ShaderModule

/-- Create a render pipeline from a shader module. Automatically cleaned up by GC. -/
@[extern "lean_glfw_create_render_pipeline"]
opaque createRenderPipeline (device : Device) (shader : ShaderModule) (format : UInt32) : IO RenderPipeline

end Hesper.GLFW.Internal
