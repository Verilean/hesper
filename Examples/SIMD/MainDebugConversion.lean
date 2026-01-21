import Hesper.Float32

open Hesper.Float32

def main : IO Unit := do
  IO.println "Testing Float32 conversion..."

  -- Create a Float64 array
  let f64 := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
  IO.println s!"Input Float64 array size: {f64.size}"

  -- Convert to Float32
  let f32 := fromFloatArray f64
  IO.println s!"Float32Array.data.size (bytes): {f32.data.size}"
  IO.println s!"Float32Array size() function: {size f32}"

  -- Try to read elements
  IO.println "Elements:"
  for i in [0:4] do
    let val := get f32 i.toUSize
    IO.println s!"  [{i}] = {val}"
