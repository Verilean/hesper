-- No WebGPU dependencies - pure Lean test

/-!
# i2_s Dequantization Validation Tests

Test i2_s unpacking against known expected values.

## Test Strategy
1. Create test cases with known i2_s packed data
2. Verify dequantization produces correct ternary values
3. Compare with bitnet.cpp reference implementation
-/

namespace Tests.I2S_Validation

/-! ## Test Cases -/

/-- Test case: Simple ternary values
    Input: 4 weights packed in 1 byte + scale
    Bit layout: idx0=bits[1:0], idx1=bits[3:2], idx2=bits[5:4], idx3=bits[7:6]
    Packed: 0x8A = 0b10001010
      idx0: bits[1:0] = 10 (binary 2) → +1
      idx1: bits[3:2] = 10 (binary 2) → +1
      idx2: bits[5:4] = 00 (binary 0) → -1
      idx3: bits[7:6] = 10 (binary 2) → +1
    Scale: 1.0
    Expected output: [1.0, 1.0, -1.0, 1.0]
-/
def testCase1 : ByteArray × Float × Nat :=
  let packed := ByteArray.mk #[0x8A]  -- 0b10_00_10_10 = [+1, +1, -1, +1]
  let scale := 1.0
  let numElements := 4
  (packed, scale, numElements)

/-- Expected output for test case 1 -/
def expected1 : Array Float :=
  #[1.0, 1.0, -1.0, 1.0]

/-- Test case: All zeros
    Packed: 0b01_01_01_01 = 0x55
    Scale: 1.0
    Expected: [0.0, 0.0, 0.0, 0.0]
-/
def testCase2 : ByteArray × Float × Nat :=
  let packed := ByteArray.mk #[0x55]  -- All 01 (zero)
  let scale := 1.0
  let numElements := 4
  (packed, scale, numElements)

def expected2 : Array Float :=
  #[0.0, 0.0, 0.0, 0.0]

/-- Test case: All positive
    Packed: 0b10_10_10_10 = 0xAA
    Scale: 2.0
    Expected: [2.0, 2.0, 2.0, 2.0]
-/
def testCase3 : ByteArray × Float × Nat :=
  let packed := ByteArray.mk #[0xAA]  -- All 10 (+1)
  let scale := 2.0
  let numElements := 4
  (packed, scale, numElements)

def expected3 : Array Float :=
  #[2.0, 2.0, 2.0, 2.0]

/-- Test case: All negative
    Packed: 0b00_00_00_00 = 0x00
    Scale: 0.5
    Expected: [-0.5, -0.5, -0.5, -0.5]
-/
def testCase4 : ByteArray × Float × Nat :=
  let packed := ByteArray.mk #[0x00]  -- All 00 (-1)
  let scale := 0.5
  let numElements := 4
  (packed, scale, numElements)

def expected4 : Array Float :=
  #[-0.5, -0.5, -0.5, -0.5]

/-! ## CPU Dequantization Test -/

/-- Dequantize i2_s on CPU (reference implementation)

    Based on bitnet.cpp logic:
    - 00 (0) → -1
    - 01 (1) → 0
    - 10 (2) → +1
    - 11 (3) → unused
-/
def dequantizeI2S_CPU (packed : ByteArray) (scale : Float) (numElements : Nat) : Array Float := Id.run do
  let mut result := Array.mkEmpty numElements
  for i in [:numElements] do
    let byteIdx := i / 4
    let bitPos := (i % 4) * 2

    if byteIdx < packed.size then
      let byte := packed.get! byteIdx
      let val2bit := (byte.toNat >>> bitPos) &&& 0x03

      let ternary : Float :=
        if val2bit == 0 then -1.0
        else if val2bit == 1 then 0.0
        else 1.0  -- val2bit == 2

      result := result.push (ternary * scale)
    else
      result := result.push 0.0

  return result

/-! ## Validation Functions -/

def floatApproxEq (a b : Float) (epsilon : Float := 0.0001) : Bool :=
  let diff := if a > b then a - b else b - a
  diff < epsilon

def arrayApproxEq (a b : Array Float) (epsilon : Float := 0.0001) : Bool :=
  if a.size != b.size then false
  else
    a.zip b |>.all fun (x, y) => floatApproxEq x y epsilon

/-! ## Test Execution -/

def runTest (name : String) (testCase : ByteArray × Float × Nat) (expected : Array Float) : IO Bool := do
  let (packed, scale, numElements) := testCase

  IO.println s!"[TEST] {name}"
  IO.println s!"  Packed data: {packed.data}"
  IO.println s!"  Scale: {scale}"
  IO.println s!"  Num elements: {numElements}"

  let result := dequantizeI2S_CPU packed scale numElements

  IO.println s!"  Result:   {result}"
  IO.println s!"  Expected: {expected}"

  let passed := arrayApproxEq result expected

  if passed then
    IO.println "  ✓ PASSED"
  else
    IO.println "  ✗ FAILED"
    -- Print detailed diff
    for i in [:min result.size expected.size] do
      if !floatApproxEq result[i]! expected[i]! then
        IO.println s!"    Mismatch at index {i}: got {result[i]!}, expected {expected[i]!}"

  IO.println ""
  return passed

def runAll : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  i2_s Dequantization Validation Tests"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let test1 ← runTest "Simple ternary values" testCase1 expected1
  let test2 ← runTest "All zeros" testCase2 expected2
  let test3 ← runTest "All positive (scaled)" testCase3 expected3
  let test4 ← runTest "All negative (scaled)" testCase4 expected4

  let allPassed := test1 && test2 && test3 && test4

  IO.println "═══════════════════════════════════════════════"
  if allPassed then
    IO.println "  ✓ ALL TESTS PASSED"
  else
    IO.println "  ✗ SOME TESTS FAILED"
  IO.println "═══════════════════════════════════════════════"

end Tests.I2S_Validation

def main : IO Unit := Tests.I2S_Validation.runAll
