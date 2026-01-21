import Hesper.Simd

open Hesper.Simd

def testSize (generateTestArray : Nat → FloatArray) (size : Nat) : IO Unit := do
  IO.println s!"Size: {size}"

  IO.print "  Generating arrays... "
  let a := generateTestArray size
  let b := generateTestArray size
  IO.println "done"

  IO.print "  Naive add... "
  let c := naiveAdd a b
  IO.println s!"done ({c.size})"

  IO.print "  SIMD add... "
  let d := simdAdd a b
  IO.println s!"done ({d.size})"

  -- Skip verification for large arrays (too expensive)
  if size <= 10000 then
    IO.print "  Verifying... "
    if verifyEqual c d then
      IO.println "✓"
    else
      IO.println "✗"
  else
    IO.println "  (verification skipped)"

def main : IO Unit := do
  IO.println "Testing with various sizes..."

  let backend ← backendInfo
  IO.println s!"Backend: {backend}\n"

  testSize generateTestArray 1000
  testSize generateTestArray 10000

  IO.println "\nDone!"

where
  generateTestArray (size : Nat) : FloatArray :=
    let data := Array.range size |>.map (·.toFloat)
    FloatArray.mk data
