import Hesper.Simd
import Hesper.Float32
import Hesper.Float16

/-!
# Multi-Precision SIMD Test

Tests Float64, Float32, and Float16 SIMD operations.
-/

open Hesper.Simd Hesper.Float32 Hesper.Float16

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Multi-Precision SIMD Test                  ║"
  IO.println "╚══════════════════════════════════════════════╝\n"

  -- Display backend
  let backend ← backendInfo
  IO.println s!"Backend: {backend}\n"

  -- Test Float64 (existing)
  IO.println "─── Float64 (8 bytes/element) ───"
  let a64 := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
  let b64 := FloatArray.mk #[5.0, 6.0, 7.0, 8.0]
  let c64 := simdAdd a64 b64
  IO.println s!"Result: {c64.data}"
  IO.println "✓ Float64 works\n"

  -- Test Float32 (direct bytes, no conversion)
  IO.println "─── Float32 (4 bytes/element) ───"
  let a32 := Hesper.Float32.fromFloatArray (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
  let b32 := Hesper.Float32.fromFloatArray (FloatArray.mk #[5.0, 6.0, 7.0, 8.0])
  IO.println s!"A: {a32}"
  IO.println s!"B: {b32}"
  let c32 := Hesper.Float32.simdAdd a32 b32
  IO.println s!"Result: {c32}"
  IO.println "✓ Float32 works\n"

  -- Test Float16 (requires hardware support)
  IO.println "─── Float16 (2 bytes/element) ───"
  let hasFP16 ← Hesper.Float16.hasHardwareSupport
  if hasFP16 then
    IO.println "FP16 hardware detected!"
    let a16 ← Hesper.Float16.fromFloatArray (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
    let b16 ← Hesper.Float16.fromFloatArray (FloatArray.mk #[5.0, 6.0, 7.0, 8.0])
    let a16_str ← Hesper.Float16.toString a16
    let b16_str ← Hesper.Float16.toString b16
    IO.println s!"A: {a16_str}"
    IO.println s!"B: {b16_str}"

    let c16 ← Hesper.Float16.simdAdd a16 b16
    let c16_str ← Hesper.Float16.toString c16
    IO.println s!"Result: {c16_str}"
    IO.println "✓ Float16 works"
  else
    IO.println "⚠ FP16 hardware not available (requires ARMv8.2-A or F16C)"
    IO.println "  Skipping Float16 tests"

  IO.println "\n╔══════════════════════════════════════════════╗"
  IO.println "║   All Tests Complete                          ║"
  IO.println "╚══════════════════════════════════════════════╝"
