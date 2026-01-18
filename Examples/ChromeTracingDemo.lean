import Hesper.Profile
import Hesper.Profile.Trace

/-!
# Chrome Tracing Profiling Demo

This example demonstrates:
1. Creating mock ProfileEvent data
2. Converting to Chrome Tracing format
3. Writing JSON trace files for visualization

Usage:
  lake build chrome-tracing-demo && ./.lake/build/bin/chrome-tracing-demo
  # Then open chrome://tracing and load gpu_profile.json
-/

open Hesper

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Chrome Tracing Profiling Demo               ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  -- Create mock profiling events to demonstrate trace export
  -- In a real application, these would come from actual GPU profiling
  let events : List ProfileEvent := [
    ProfileEvent.fromTimestamps "kernel_matmul" 1000000 3500000,  -- 1ms to 3.5ms
    ProfileEvent.fromTimestamps "kernel_reduce" 4000000 5200000,  -- 4ms to 5.2ms
    ProfileEvent.fromTimestamps "kernel_copy" 5500000 6000000,    -- 5.5ms to 6ms
    ProfileEvent.fromTimestamps "kernel_matmul" 7000000 9500000,  -- 7ms to 9.5ms
    ProfileEvent.fromTimestamps "kernel_reduce" 10000000 11200000 -- 10ms to 11.2ms
  ]

  IO.println "Created mock profiling events:"
  for event in events do
    IO.println s!"  {event}"
  IO.println ""

  -- Print profiling report
  printReport events

  -- Convert to Chrome Tracing format
  IO.println "Converting to Chrome Tracing format..."
  let trace := Profile.Trace.profileEventsToTrace events

  IO.println s!"  Total events: {trace.traceEvents.length}"
  IO.println s!"  Display unit: {trace.displayTimeUnit}"
  IO.println ""

  -- Export to JSON file
  IO.println "Writing to gpu_profile.json..."
  Profile.Trace.writeChromeTrace "gpu_profile.json" events

  IO.println ""
  IO.println "✅ Done!"
  IO.println ""
  IO.println "Next steps:"
  IO.println "1. Open Chrome browser"
  IO.println "2. Navigate to chrome://tracing"
  IO.println "3. Click 'Load' button"
  IO.println "4. Select gpu_profile.json"
  IO.println "5. You'll see a timeline visualization of GPU kernels"
  IO.println ""
  IO.println "Features to explore in chrome://tracing:"
  IO.println "  • WASD keys to navigate the timeline"
  IO.println "  • Click events to see details (duration, name)"
  IO.println "  • Identify overlapping operations and bottlenecks"
  IO.println "  • Measure time between events"
  IO.println ""
