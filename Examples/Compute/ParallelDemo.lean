import Hesper
import Hesper.Compute

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL

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

  -- 3. Run parallel_for using type-safe DSL
  IO.println "🚀 Running parallelForDSL (x = x * 1000.0)..."

  let result ← parallelForDSL device (fun x => x * Exp.litF32 1000.0) data

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
