import Hesper
import Hesper.GLFW
import Hesper.WebGPU.Device

/-!
# GLFW Window and Rendering Demo

Demonstrates creating a window with GLFW and rendering a simple triangle.
-/

namespace Examples.GLFWDemo

open Hesper.GLFW
open Hesper.WebGPU

/-- Simple triangle shader in WGSL -/
def triangleShader : String := "
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Define triangle vertices directly in shader
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),   // Top
        vec2<f32>(-0.5, -0.5), // Bottom left
        vec2<f32>(0.5, -0.5)   // Bottom right
    );

    let pos = positions[in_vertex_index];
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Red triangle
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"

/-- Main rendering demo -/
def demo : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  GLFW Window and Rendering Demo"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Initialize Hesper first (creates g_instance and g_device)
  Hesper.init
  IO.println "✓ Hesper initialized"

  withGLFW do
    IO.println "✓ GLFW initialized"

    withWindow 800 600 "Hesper GLFW Demo" fun window => do
      IO.println "✓ Window created (800x600)"

      -- Get WebGPU device (already initialized by Hesper.init)
      let device ← getDevice
      IO.println "✓ WebGPU device acquired"

      withSurface device window fun surface => do
        IO.println "✓ Surface created"

        -- Configure surface
        let format ← getSurfacePreferredFormat surface
        configureSurface surface 800 600 format
        IO.println s!"✓ Surface configured (format: {format})"

        -- Create shader and pipeline
        withShaderModule device triangleShader fun shader => do
          IO.println "✓ Shader module created"

          withRenderPipeline device shader format fun pipeline => do
            IO.println "✓ Render pipeline created"
            IO.println ""
            IO.println "Entering render loop (press ESC to exit)..."
            IO.println ""

            -- Main render loop
            let mut frameCount := 0
            let mut shouldQuit := false

            while !shouldQuit do
              -- Check for window close or ESC key
              let windowClosing ← windowShouldClose window
              let escKey ← getKey window Key.escape
              shouldQuit := windowClosing || (escKey == KeyAction.press)

              if !shouldQuit then
                -- Render frame
                renderFrame device surface pipeline

                -- Poll events
                pollEvents

                -- Print progress every 60 frames
                frameCount := frameCount + 1
                if frameCount % 60 == 0 then
                  IO.println s!"  Rendered {frameCount} frames..."

            IO.println ""
            IO.println s!"✓ Render loop exited after {frameCount} frames"

      IO.println "✓ Cleanup complete"

  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Demo complete!"
  IO.println "═══════════════════════════════════════════════"

/-- Demo with detailed explanations -/
def interactiveDemo : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   GLFW Interactive Demo                       ║"
  IO.println "║   Demonstrates window management & rendering  ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "This demo will:"
  IO.println "  1. Initialize GLFW"
  IO.println "  2. Create an 800x600 window"
  IO.println "  3. Set up WebGPU rendering"
  IO.println "  4. Render a red triangle"
  IO.println "  5. Run until you press ESC or close the window"
  IO.println ""
  IO.println "Press ENTER to start..."
  _ ← (← IO.getStdin).getLine

  demo

  IO.println ""
  IO.println "Key features demonstrated:"
  IO.println "  ✓ Automatic resource management (with* functions)"
  IO.println "  ✓ Window creation and event handling"
  IO.println "  ✓ WebGPU surface configuration"
  IO.println "  ✓ Shader compilation and pipeline setup"
  IO.println "  ✓ Render loop with frame counting"
  IO.println "  ✓ Keyboard input detection"
  IO.println ""

/-- Simple non-interactive demo (for automated testing) -/
def quickDemo : IO Unit := do
  IO.println "Running quick GLFW demo (10 frames)..."

  withGLFW do
    withWindow 800 600 "Hesper Quick Demo" fun window => do
      let device ← getDevice

      withSurface device window fun surface => do
        let format ← getSurfacePreferredFormat surface
        configureSurface surface 800 600 format

        withShaderModule device triangleShader fun shader => do
          withRenderPipeline device shader format fun pipeline => do
            -- Render just 10 frames
            for i in [:10] do
              renderFrame device surface pipeline
              pollEvents
              IO.println s!"  Frame {i+1}/10"

  IO.println "✓ Quick demo complete!"

def main : IO Unit := do
  -- Run the actual demo (FFI is now implemented!)
  quickDemo

end Examples.GLFWDemo

def main : IO Unit := Examples.GLFWDemo.main
