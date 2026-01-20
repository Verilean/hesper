import Hesper.Simd

open Hesper.Simd

def main : IO Unit := do
  IO.println "Testing SIMD with various sizes..."

  let backend ← backendInfo
  IO.println s!"Backend: {backend}\n"

  let sizes := [100, 1000, 10000, 100000]

  for size in sizes do
    IO.println s!"Testing size: {size}"

    IO.print "  Generating arrays... "
    let a := generateTestArray size
    let b := generateTestArray size
    IO.println "done"

    IO.print "  Naive add... "
    let c := naiveAdd a b
    IO.println s!"done (size: {c.size})"

    IO.print "  SIMD add... "
    let d := simdAdd a b
    IO.println s!"done (size: {d.size})"

    IO.print "  Verifying... "
    if verifyEqual c d then
      IO.println "✓ match"
    else
      IO.println "✗ differ"

  IO.println "\nAll tests completed!"

where
  generateTestArray (size : Nat) : FloatArray :=
    let data := Array.range size |>.map (·.toFloat)
    FloatArray.mk data
