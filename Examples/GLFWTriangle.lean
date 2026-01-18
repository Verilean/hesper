import Hesper
import Hesper.GLFW

/-!
# GLFW Triangle Rendering Demo

Renders a colored triangle in a GLFW window using WebGPU.

All resources are automatically managed by Lean's GC!
-/

namespace Examples.GLFWTriangle

open Hesper.GLFW

def triangleShader : String := "
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5)
    );
    let pos = positions[in_vertex_index];
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.2, 1.0);  // Orange color
}
"

def main : IO Unit := do
  IO.println "Initializing Hesper..."
  Hesper.init

  IO.println "Initializing GLFW..."
  withGLFW do
    IO.println "✓ GLFW initialized"

    IO.println "Creating window..."
    let window ← createWindow 800 600 "Hesper - Triangle Demo"
    IO.println "✓ Window created!"

    -- Get the WebGPU device
    let device ← Hesper.WebGPU.getDevice
    IO.println "✓ Device obtained"

    -- Create surface (automatically cleaned up by GC)
    let surface ← createSurface device window
    IO.println "✓ Surface created"

    -- Get and configure surface
    let format ← getSurfacePreferredFormat surface
    configureSurface surface 800 600 format
    IO.println s!"✓ Surface configured (format: {format})"

    -- Create shader and pipeline (automatically cleaned up by GC)
    let shader ← createShaderModule device triangleShader
    IO.println "✓ Shader module created"

    let pipeline ← createRenderPipeline device shader format
    IO.println "✓ Render pipeline created"
    IO.println ""
    IO.println "Rendering triangle... Press ESC to close or close the window."
    IO.println ""

    -- Main render loop
    let mut frameCount := 0
    while !(← windowShouldClose window) do
      pollEvents

      -- Check for ESC key
      let escKey ← getKey window Key.escape
      if escKey == KeyAction.press then
        break

      -- Render the triangle
      renderFrame device surface pipeline

      frameCount := frameCount + 1
      if frameCount % 60 == 0 then
        IO.println s!"Frame {frameCount}"

    IO.println ""
    IO.println s!"✓ Rendered {frameCount} frames"

  -- All resources (window, surface, shader, pipeline, etc.) are automatically
  -- cleaned up by Lean's GC when they go out of scope!

  IO.println "✓ Demo complete!"

end Examples.GLFWTriangle

def main : IO Unit := Examples.GLFWTriangle.main
