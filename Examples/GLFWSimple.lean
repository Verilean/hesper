import Hesper
import Hesper.GLFW

/-!
# Simple GLFW Window Test

Just creates a window and polls events - no rendering.
-/

namespace Examples.GLFWSimple

open Hesper.GLFW

def main : IO Unit := do
  IO.println "Initializing Hesper..."
  Hesper.init

  IO.println "Initializing GLFW..."
  withGLFW do
    IO.println "✓ GLFW initialized"

    IO.println "Creating window..."
    withWindow 800 600 "Hesper - Simple Window Test" fun window => do
      IO.println "✓ Window created!"
      IO.println ""
      IO.println "Window is open. Press ESC to close or just close the window."
      IO.println ""

      -- Simple event loop
      let mut frameCount := 0
      while !(← windowShouldClose window) do
        pollEvents

        -- Check for ESC key
        let escKey ← getKey window Key.escape
        if escKey == KeyAction.press then
          break

        frameCount := frameCount + 1
        if frameCount % 60 == 0 then
          IO.println s!"Still running... (frame {frameCount})"

      IO.println ""
      IO.println s!"✓ Window closed after {frameCount} frames"

  IO.println "✓ GLFW terminated"
  IO.println "✓ Demo complete!"

end Examples.GLFWSimple

def main : IO Unit := Examples.GLFWSimple.main
