import Hesper.Simd

open Hesper.Simd

def main : IO Unit := do
  IO.println "Starting SIMD test..."

  -- Create small test arrays
  let a := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
  let b := FloatArray.mk #[5.0, 6.0, 7.0, 8.0]

  IO.println s!"A size: {a.size}"
  IO.println s!"B size: {b.size}"

  -- Test naive addition
  IO.println "Testing naive addition..."
  let c := naiveAdd a b
  IO.println s!"Naive result size: {c.size}"
  IO.println s!"Naive result: {c.data}"

  -- Test SIMD addition
  IO.println "Testing SIMD addition..."
  try
    let d := simdAdd a b
    IO.println s!"SIMD result size: {d.size}"
    IO.println s!"SIMD result: {d.data}"

    -- Verify
    if verifyEqual c d then
      IO.println "✓ Results match!"
    else
      IO.println "✗ Results differ!"
  catch e =>
    IO.println s!"SIMD failed: {e}"

  IO.println "Done!"
