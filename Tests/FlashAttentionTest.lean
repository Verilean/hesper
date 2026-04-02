import Hesper.WGSL.FlashAttention

open Hesper.WGSL.FlashAttention

def main : IO Unit := do
  IO.println "=== Flash Attention Equivalence Test ==="
  IO.println ""

  -- CPU equivalence: flash spec == standard spec
  let cpuOk := verifyFlashEquivalence
  IO.println s!"CPU equivalence (flash spec == standard): {if cpuOk then "PASS" else "FAIL"}"

  if cpuOk then
    IO.println "✓ Flash attention produces identical results to standard attention"
  else
    IO.println "✗ Flash attention DIFFERS from standard attention"
