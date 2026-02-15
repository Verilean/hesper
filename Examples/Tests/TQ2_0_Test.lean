import Hesper.GGUF.Quantization

/-!
# TQ2_0 Pack/Unpack Roundtrip Test

Tests that ternary packing and unpacking are correct.

## Test Cases
1. Pack 4 ternary values → single byte
2. Unpack byte → verify all 4 values match
3. Test edge cases: all zeros, all ones, all negative ones, mixed
-/

open Hesper.GGUF.Quantization

def testPackUnpack : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   TQ2_0 Pack/Unpack Roundtrip Test          ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Test Case 1: All zeros
  IO.println "Test 1: All zeros [0, 0, 0, 0]"
  let packed1 := packTernary2bit 0 0 0 0
  let v0 := unpackTernary2bit packed1 0
  let v1 := unpackTernary2bit packed1 1
  let v2 := unpackTernary2bit packed1 2
  let v3 := unpackTernary2bit packed1 3
  IO.println s!"  Packed byte: 0x{packed1.toNat.toDigits 16}"
  IO.println s!"  Unpacked: [{v0}, {v1}, {v2}, {v3}]"
  let pass1 := v0 == 0 && v1 == 0 && v2 == 0 && v3 == 0
  IO.println s!"  {if pass1 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 2: All ones
  IO.println "\nTest 2: All ones [1, 1, 1, 1]"
  let packed2 := packTernary2bit 1 1 1 1
  let v0 := unpackTernary2bit packed2 0
  let v1 := unpackTernary2bit packed2 1
  let v2 := unpackTernary2bit packed2 2
  let v3 := unpackTernary2bit packed2 3
  IO.println s!"  Packed byte: 0x{packed2.toNat.toDigits 16}"
  IO.println s!"  Unpacked: [{v0}, {v1}, {v2}, {v3}]"
  let pass2 := v0 == 1 && v1 == 1 && v2 == 1 && v3 == 1
  IO.println s!"  {if pass2 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 3: All negative ones
  IO.println "\nTest 3: All -1 [-1, -1, -1, -1]"
  let packed3 := packTernary2bit (-1) (-1) (-1) (-1)
  let v0 := unpackTernary2bit packed3 0
  let v1 := unpackTernary2bit packed3 1
  let v2 := unpackTernary2bit packed3 2
  let v3 := unpackTernary2bit packed3 3
  IO.println s!"  Packed byte: 0x{packed3.toNat.toDigits 16}"
  IO.println s!"  Unpacked: [{v0}, {v1}, {v2}, {v3}]"
  let pass3 := v0 == -1 && v1 == -1 && v2 == -1 && v3 == -1
  IO.println s!"  {if pass3 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 4: Mixed values
  IO.println "\nTest 4: Mixed [-1, 0, 1, 0]"
  let packed4 := packTernary2bit (-1) 0 1 0
  let v0 := unpackTernary2bit packed4 0
  let v1 := unpackTernary2bit packed4 1
  let v2 := unpackTernary2bit packed4 2
  let v3 := unpackTernary2bit packed4 3
  IO.println s!"  Packed byte: 0x{packed4.toNat.toDigits 16}"
  IO.println s!"  Unpacked: [{v0}, {v1}, {v2}, {v3}]"
  let pass4 := v0 == -1 && v1 == 0 && v2 == 1 && v3 == 0
  IO.println s!"  {if pass4 then "✅ PASS" else "❌ FAIL"}"

  -- Test Case 5: Pattern [1, -1, 1, -1]
  IO.println "\nTest 5: Alternating [1, -1, 1, -1]"
  let packed5 := packTernary2bit 1 (-1) 1 (-1)
  let v0 := unpackTernary2bit packed5 0
  let v1 := unpackTernary2bit packed5 1
  let v2 := unpackTernary2bit packed5 2
  let v3 := unpackTernary2bit packed5 3
  IO.println s!"  Packed byte: 0x{packed5.toNat.toDigits 16}"
  IO.println s!"  Unpacked: [{v0}, {v1}, {v2}, {v3}]"
  let pass5 := v0 == 1 && v1 == -1 && v2 == 1 && v3 == -1
  IO.println s!"  {if pass5 then "✅ PASS" else "❌ FAIL"}"

  -- Summary
  IO.println "\n═══════════════════════════════════════════════"
  let allPass := pass1 && pass2 && pass3 && pass4 && pass5
  if allPass then
    IO.println "✅ All tests PASSED!"
    IO.println "\n🎉 TQ2_0 pack/unpack is working correctly!"
  else
    IO.println "❌ Some tests FAILED"
    IO.Process.exit 1

def main : IO Unit := testPackUnpack
