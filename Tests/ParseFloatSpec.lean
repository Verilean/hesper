import LSpec
import Hesper.Training.ParseFloat
import Hesper.WGSL.Exp

open LSpec
open Hesper.Training.ParseFloat
open Hesper.WGSL
open Std (HashMap)

/-- Check float equality within relative tolerance -/
def floatApproxEq (a b : Float) (tol : Float := 1e-5) : Bool :=
  if a == 0.0 && b == 0.0 then true
  else
    let diff := if a - b < 0.0 then b - a else a - b
    let denom := (if a < 0.0 then -a else a) + (if b < 0.0 then -b else b)
    diff / (if denom < 1e-15 then 1e-15 else denom) < tol

-- parseFloat tests
def parseFloatTests : List TestSeq := [
  group "integers" (
    test "42" (parseFloat "42" == 42.0) ++
    test "0" (parseFloat "0" == 0.0) ++
    test "100" (parseFloat "100" == 100.0)
  ),
  group "decimals" (
    test "3.14" (floatApproxEq (parseFloat "3.14") 3.14) ++
    test "0.001" (floatApproxEq (parseFloat "0.001") 0.001) ++
    test "-0.5" (floatApproxEq (parseFloat "-0.5") (-0.5)) ++
    test "0.0" (parseFloat "0.0" == 0.0) ++
    test "1.0" (parseFloat "1.0" == 1.0)
  ),
  group "scientific" (
    test "1e-4" (floatApproxEq (parseFloat "1e-4") 1e-4) ++
    test "2e-4" (floatApproxEq (parseFloat "2e-4") 2e-4) ++
    test "1e-7" (floatApproxEq (parseFloat "1e-7") 1e-7) ++
    test "5e-5" (floatApproxEq (parseFloat "5e-5") 5e-5) ++
    test "1e3" (floatApproxEq (parseFloat "1e3") 1000.0) ++
    test "2.5e3" (floatApproxEq (parseFloat "2.5e3") 2500.0) ++
    test "1.0E-7" (floatApproxEq (parseFloat "1.0E-7") 1e-7) ++
    test "-1e-4" (floatApproxEq (parseFloat "-1e-4") (-1e-4)) ++
    test "1e+2" (floatApproxEq (parseFloat "1e+2") 100.0)
  ),
  group "edge" (
    test "empty" (parseFloat "" == 0.0)
  )
]

-- floatToWGSL tests
def wgslTests : List TestSeq := [
  group "precision" (
    -- THE critical test: 1e-7 must not become "0.0" (caused AdamW NaN)
    test "1e-7 ≠ 0.0" (floatToWGSL 1e-7 != "0.0") ++
    test "1e-7 ≠ 0.000000" (floatToWGSL 1e-7 != "0.000000") ++
    test "1e-4 ≠ 0.0" (floatToWGSL 1e-4 != "0.0") ++
    test "0.001 ≠ 0.0" (floatToWGSL 0.001 != "0.0") ++
    test "1e-10 ≠ 0.0" (floatToWGSL 1e-10 != "0.0")
  ),
  group "format" (
    test "0.0 is 0.0" (floatToWGSL 0.0 == "0.0") ++
    test "negative has -" ((floatToWGSL (-0.5)).startsWith "-") ++
    test "1.0 has dot" ((floatToWGSL 1.0).any (· == '.')) ++
    test "pi has dot" ((floatToWGSL 3.14159).any (· == '.'))
  ),
  group "roundtrip" (
    test "rt 1e-7" (floatApproxEq (parseFloat (floatToWGSL 1e-7)) 1e-7 1e-3) ++
    test "rt 1e-4" (floatApproxEq (parseFloat (floatToWGSL 1e-4)) 1e-4 1e-3) ++
    test "rt 0.9" (floatApproxEq (parseFloat (floatToWGSL 0.9)) 0.9 1e-3) ++
    test "rt 0.999" (floatApproxEq (parseFloat (floatToWGSL 0.999)) 0.999 1e-3) ++
    test "rt 500000" (floatApproxEq (parseFloat (floatToWGSL 500000.0)) 500000.0 1e-3) ++
    test "rt -0.5" (floatApproxEq (parseFloat (floatToWGSL (-0.5))) (-0.5) 1e-3)
  )
]

def main (args : List String) : IO UInt32 := do
  let map : HashMap String (List TestSeq) :=
    HashMap.ofList [("parseFloat", parseFloatTests), ("floatToWGSL", wgslTests)]
  lspecIO map args
