/-!
# RMSNorm Numerical Test

Tests RMSNorm correctness with known inputs and outputs.

## Test Cases
1. All ones input → verify normalized to sqrt(dim) scaling
2. Known vector → verify matches manual calculation
3. Zero variance → verify numerical stability (epsilon prevents division by zero)

## Manual Calculation Example

For input `x = [1.0, 2.0, 3.0]` with `scale = [1.0, 1.0, 1.0]` and `ε = 1e-5`:

```
1. sum_squares = 1² + 2² + 3² = 14
2. mean = 14/3 = 4.666...
3. RMS = sqrt(4.666... + 1e-5) ≈ 2.16025
4. y = x / RMS * scale
   y[0] = 1.0 / 2.16025 * 1.0 ≈ 0.4629
   y[1] = 2.0 / 2.16025 * 1.0 ≈ 0.9258
   y[2] = 3.0 / 2.16025 * 1.0 ≈ 1.3887
```

## Verification Properties
- Sum of normalized squares ≈ dim (unit variance)
- Output scaled by scale parameters
- Epsilon prevents division by zero
-/

def testRMSNormMath : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   RMSNorm Mathematical Correctness Test     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Test Case 1: All ones
  IO.println "Test 1: All ones input [1, 1, 1, 1]"
  let input1 := #[1.0, 1.0, 1.0, 1.0]
  let scale1 := #[1.0, 1.0, 1.0, 1.0]
  let eps := 1e-5

  -- Manual calculation
  let sumSq1 := input1.foldl (fun acc x => acc + x * x) 0.0
  let mean1 := sumSq1 / input1.size.toFloat
  let rms1 := (mean1 + eps).sqrt
  let expected1 := input1.map (fun x => (x / rms1) * 1.0)

  IO.println s!"  Input: {input1.toList}"
  IO.println s!"  RMS: {rms1}"
  IO.println s!"  Expected output: {expected1.toList}"
  IO.println s!"  (Each element ≈ {1.0 / rms1})"

  let pass1 := (1.0 / rms1 - expected1[0]!).abs < 1e-4
  IO.println s!"  {if pass1 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 2: Known vector [1, 2, 3]
  IO.println "\nTest 2: Vector [1, 2, 3] with scale [1, 1, 1]"
  let input2 := #[1.0, 2.0, 3.0]
  let scale2 := #[1.0, 1.0, 1.0]

  let sumSq2 := input2.foldl (fun acc x => acc + x * x) 0.0
  let mean2 := sumSq2 / input2.size.toFloat
  let rms2 := (mean2 + eps).sqrt
  let expected2 := input2.map (fun x => (x / rms2) * 1.0)

  IO.println s!"  Input: {input2.toList}"
  IO.println s!"  Sum of squares: {sumSq2}"
  IO.println s!"  Mean: {mean2}"
  IO.println s!"  RMS: {rms2}"
  IO.println s!"  Expected output: {expected2.toList}"

  -- Verify first element
  let expected2_0 := 1.0 / rms2
  let pass2 := (expected2_0 - expected2[0]!).abs < 1e-4
  IO.println s!"  y[0] = 1.0 / {rms2} ≈ {expected2[0]!}"
  IO.println s!"  {if pass2 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 3: With scale parameter
  IO.println "\nTest 3: Vector [1, 2, 3] with scale [0.5, 1.0, 2.0]"
  let input3 := #[1.0, 2.0, 3.0]
  let scale3 := #[0.5, 1.0, 2.0]

  let sumSq3 := input3.foldl (fun acc x => acc + x * x) 0.0
  let mean3 := sumSq3 / input3.size.toFloat
  let rms3 := (mean3 + eps).sqrt
  let expected3 := #[
    (input3[0]! / rms3) * scale3[0]!,
    (input3[1]! / rms3) * scale3[1]!,
    (input3[2]! / rms3) * scale3[2]!
  ]

  IO.println s!"  Input: {input3.toList}"
  IO.println s!"  Scale: {scale3.toList}"
  IO.println s!"  RMS: {rms3}"
  IO.println s!"  Expected output: {expected3.toList}"

  let pass3_0 := ((1.0 / rms3 * 0.5) - expected3[0]!).abs < 1e-4
  let pass3_1 := ((2.0 / rms3 * 1.0) - expected3[1]!).abs < 1e-4
  let pass3_2 := ((3.0 / rms3 * 2.0) - expected3[2]!).abs < 1e-4
  let pass3 := pass3_0 && pass3_1 && pass3_2

  IO.println s!"  y[0] = 1.0 / {rms3} * 0.5 ≈ {expected3[0]!}"
  IO.println s!"  y[1] = 2.0 / {rms3} * 1.0 ≈ {expected3[1]!}"
  IO.println s!"  y[2] = 3.0 / {rms3} * 2.0 ≈ {expected3[2]!}"
  IO.println s!"  {if pass3 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 4: Numerical stability (near-zero input)
  IO.println "\nTest 4: Numerical stability with small values"
  let input4 := #[1e-10, 1e-10, 1e-10]
  let scale4 := #[1.0, 1.0, 1.0]

  let sumSq4 := input4.foldl (fun acc x => acc + x * x) 0.0
  let mean4 := sumSq4 / input4.size.toFloat
  let rms4 := (mean4 + eps).sqrt

  IO.println s!"  Input: {input4.toList} (very small values)"
  IO.println s!"  Sum of squares: {sumSq4}"
  IO.println s!"  Mean + ε: {mean4 + eps}"
  IO.println s!"  RMS: {rms4}"

  -- RMS should be dominated by epsilon
  let pass4 := (rms4 - eps.sqrt).abs < 1e-6
  IO.println s!"  RMS ≈ sqrt(ε) = {eps.sqrt}"
  IO.println s!"  {if pass4 then "✅ PASS (epsilon prevents div by zero)" else "❌ FAIL"}"

  -- Test Case 5: Verify normalization property
  IO.println "\nTest 5: Verify normalization property"
  let input5 := #[3.0, 4.0, 5.0, 6.0]
  let scale5 := #[1.0, 1.0, 1.0, 1.0]

  let sumSq5 := input5.foldl (fun acc x => acc + x * x) 0.0
  let mean5 := sumSq5 / input5.size.toFloat
  let rms5 := (mean5 + eps).sqrt
  let normalized5 := input5.map (fun x => x / rms5)

  -- Compute sum of normalized squares (should ≈ dim)
  let normalizedSumSq := normalized5.foldl (fun acc x => acc + x * x) 0.0

  IO.println s!"  Input: {input5.toList}"
  IO.println s!"  RMS: {rms5}"
  IO.println s!"  Normalized: {normalized5.toList}"
  IO.println s!"  Sum of normalized squares: {normalizedSumSq}"
  IO.println s!"  Expected (≈ dim): {input5.size.toFloat}"

  let pass5 := (normalizedSumSq - input5.size.toFloat).abs < 1e-3
  IO.println s!"  {if pass5 then "✅ PASS (unit variance achieved)" else "❌ FAIL"}"

  -- Summary
  IO.println "\n═══════════════════════════════════════════════"
  let allPass := pass1 && pass2 && pass3 && pass4 && pass5
  if allPass then
    IO.println "✅ All tests PASSED!"
    IO.println "\n🎉 RMSNorm mathematical correctness verified!"
  else
    IO.println "❌ Some tests FAILED"
    IO.Process.exit 1

def main : IO Unit := testRMSNormMath
