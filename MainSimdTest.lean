-- Minimal test without any SIMD FFI calls

def main : IO Unit := do
  IO.println "Test 1: Basic IO"

  let a := FloatArray.mk #[1.0, 2.0, 3.0]
  IO.println s!"Array size: {a.size}"
  IO.println s!"Array data: {a.data}"

  IO.println "Done!"
