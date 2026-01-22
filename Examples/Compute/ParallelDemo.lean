import Hesper
import Hesper.Compute

open Hesper.WebGPU
open Hesper.Compute

/--
# High-Level Parallel API Demo

Demonstrates the simplified `parallelFor` API, providing a similar
experience to `webgpu-dawn` but with Lean's type safety.
-/

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper High-Level Parallel API Demo        ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- 1. Initialize
  let inst ← Hesper.init
  let device ← getDevice inst

  -- 2. Define data
  let data := (Array.range 10).map (·.toFloat)
  IO.println s!"Input Data: {data}"

  -- 3. Run parallel_for (mirroring webgpu-dawn functionality)
  IO.println "🚀 Running parallelFor (x = x * 1000.0)..."

  let shader := "
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i < arrayLength(&data)) {
        data[i] = data[i] * 1000.0;
      }
    }
  "

  let result ← parallelFor device shader data

  IO.println s!"Result Data: {result}"
  IO.println ""

  -- Verify
  let expected := data.map (· * 1000.0)
  if result == expected then
    IO.println "✅ Success: GPU results match expected values!"
  else
    IO.println "❌ Error: GPU results do not match!"

  IO.println ""
  IO.println "✅ High-level API verification complete!"
