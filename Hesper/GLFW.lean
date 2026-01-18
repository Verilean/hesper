import Hesper.GLFW.Types
import Hesper.GLFW.Internal
import Hesper.WebGPU.Types

/-!
# GLFW Safe API

Safe GLFW bindings with automatic resource management using Lean.External.

This module provides a clean API for GLFW resources. All resources are
automatically cleaned up by Lean's GC when they go out of scope.

## Usage Example

```lean
import Hesper.GLFW
import Hesper.WebGPU.Device

def main : IO Unit := do
  -- Initialize GLFW
  withGLFW do
    -- Create window (automatically cleaned up when out of scope)
    let window ← createWindow 800 600 "Hesper Window"

    -- Get device
    let device ← getDevice

    -- Create surface (automatically cleaned up)
    let surface ← createSurface device window

    -- Get preferred format and configure
    let format ← getSurfacePreferredFormat surface
    configureSurface surface 800 600 format

    -- Create shader and pipeline (automatically cleaned up)
    let shader ← createShaderModule device shaderCode
    let pipeline ← createRenderPipeline device shader format

    -- Main render loop
    while !(← windowShouldClose window) do
      renderFrame device surface pipeline
      pollEvents
```
-/

namespace Hesper.GLFW

open Hesper.GLFW.Internal
open Hesper.WebGPU

-- GLFW Initialization

/-- Initialize GLFW with automatic cleanup -/
def withGLFW (action : IO α) : IO α := do
  glfwInit
  try
    action
  finally
    glfwTerminate

-- Window Management

/-- Create a window. Automatically cleaned up by GC. -/
def createWindow (width height : Nat) (title : String) : IO Window :=
  Internal.createWindow width.toUInt32 height.toUInt32 title

/-- Check if window should close -/
def windowShouldClose (window : Window) : IO Bool :=
  Internal.windowShouldClose window

/-- Poll for events -/
def pollEvents : IO Unit :=
  Internal.pollEvents

/-- Get key state -/
def getKey (window : Window) (key : Key) : IO KeyAction := do
  let state ← Internal.windowGetKey window key.toNat.toUInt32
  return KeyAction.fromNat state.toNat

-- Surface Management

/-- Create a surface. Automatically cleaned up by GC. -/
def createSurface (device : Device) (window : Window) : IO Surface :=
  Internal.createSurface device window

/-- Configure surface dimensions and format -/
def configureSurface (surface : Surface) (width height : Nat) (format : Nat) : IO Unit :=
  Internal.configureSurface surface width.toUInt32 height.toUInt32 format.toUInt32

/-- Get preferred surface format -/
def getSurfacePreferredFormat (surface : Surface) : IO Nat := do
  let format ← Internal.getSurfacePreferredFormat surface
  return format.toNat

/-- Present surface -/
def present (surface : Surface) : IO Unit :=
  Internal.surfacePresent surface

-- Texture Management

/-- Get current texture. Automatically cleaned up by GC. -/
def getCurrentTexture (surface : Surface) : IO Texture :=
  Internal.getCurrentTexture surface

/-- Create texture view. Automatically cleaned up by GC. -/
def createTextureView (texture : Texture) : IO TextureView :=
  Internal.createTextureView texture

-- Command Encoding

/-- Create command encoder. Automatically cleaned up by GC. -/
def createCommandEncoder (device : Device) : IO CommandEncoder :=
  Internal.createCommandEncoder device

/-- Begin render pass. Automatically cleaned up by GC. -/
def beginRenderPass (encoder : CommandEncoder) (view : TextureView) : IO RenderPassEncoder :=
  Internal.beginRenderPass encoder view

/-- Set pipeline in render pass -/
def setPipeline (pass : RenderPassEncoder) (pipeline : RenderPipeline) : IO Unit :=
  setRenderPipeline pass pipeline

/-- Draw vertices in render pass -/
def drawVertices (pass : RenderPassEncoder) (count : Nat) : IO Unit :=
  draw pass count.toUInt32

/-- End render pass -/
def endRenderPass (pass : RenderPassEncoder) : IO Unit :=
  Internal.endRenderPass pass

/-- Finish encoder and get command buffer. Automatically cleaned up by GC. -/
def finishEncoder (encoder : CommandEncoder) : IO CommandBuffer :=
  Internal.finishEncoder encoder

/-- Submit command buffer to GPU -/
def submit (device : Device) (cmd : CommandBuffer) : IO Unit :=
  submitCommand device cmd

-- Pipeline Management

/-- Create shader module. Automatically cleaned up by GC. -/
def createShaderModule (device : Device) (code : String) : IO ShaderModule :=
  Internal.createShaderModule device code

/-- Create render pipeline. Automatically cleaned up by GC. -/
def createRenderPipeline (device : Device) (shader : ShaderModule) (format : Nat) : IO RenderPipeline :=
  Internal.createRenderPipeline device shader format.toUInt32

-- High-level render helpers

/-- Render a single frame -/
def renderFrame (device : Device) (surface : Surface) (pipeline : RenderPipeline) : IO Unit := do
  let texture ← getCurrentTexture surface
  let view ← createTextureView texture
  let encoder ← createCommandEncoder device
  let pass ← beginRenderPass encoder view
  setPipeline pass pipeline
  drawVertices pass 3  -- Draw a triangle
  endRenderPass pass
  let cmd ← finishEncoder encoder
  submit device cmd
  present surface
  -- All resources automatically cleaned up by GC

end Hesper.GLFW
