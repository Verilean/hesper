import Hesper.Float16

open Hesper.Float16

def main : IO Unit := do
  IO.println "Testing Float16 conversion..."

  -- Create a simple Float64 array
  let f64 := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
  IO.println s!"Input Float64 array: {f64.data}"

  -- Convert to Float16
  let f16 ← fromFloatArray f64
  IO.println s!"Float16Array.data.size (bytes): {f16.data.size}"
  IO.println s!"Float16Array.numElements: {f16.numElements}"

  -- Print raw bytes
  IO.println "Raw bytes:"
  for i in [0:8] do
    IO.println s!"  byte[{i}] = {f16.data.get! i}"

  -- Try to read elements back
  IO.println "\nElements (via get):"
  for i in [0:4] do
    let val ← get f16 i.toUSize
    IO.println s!"  [{i}] = {val}"

  -- Convert back to Float64
  let f64_back ← toFloatArray f16
  IO.println s!"\nConverted back to Float64: {f64_back.data}"
