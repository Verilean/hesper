import Hesper.AD.Chain

open Hesper.AD.Chain

def main : IO Unit := do
  IO.println "=== Backward Chain Completeness Test ==="
  IO.println ""

  -- Build the current state of attention backward
  let builder : TransformerBackwardBuilder := {
    preNorm := some { name := "preNorm (skipped: residual bypass)", inDim := 2560, outDim := 2560, verified := false }
    qProjection := none  -- BitLinear Q backward (not needed for LoRA — LoRA does its own)
    vProjection := none  -- BitLinear V backward (same)
    ropeQ := some { name := "ropeQ", inDim := 2560, outDim := 2560, verified := true }
    attentionScores := some { name := "attentionScores", inDim := 2560, outDim := 40960, verified := true }
    softmax := some { name := "softmax", inDim := 40960, outDim := 40960, verified := true }
    attentionApply := some { name := "attentionApply", inDim := 40960, outDim := 2560, verified := true }
    subNorm := some { name := "subNorm", inDim := 2560, outDim := 2560, verified := true }
    oProjection := some { name := "oProjection", inDim := 2560, outDim := 2560, verified := true }
    -- FFN: all implemented
    ffnNorm := some { name := "ffnNorm", inDim := 2560, outDim := 2560, verified := true }
    ffnGate := some { name := "ffnGate (BitLinear transpose)", inDim := 2560, outDim := 6912, verified := false }
    ffnUp := some { name := "ffnUp (BitLinear transpose)", inDim := 2560, outDim := 6912, verified := false }
    ffnActivation := some { name := "ReLU²×Mul", inDim := 6912, outDim := 6912, verified := true }
    ffnSubNorm := some { name := "ffnSubNorm", inDim := 6912, outDim := 6912, verified := true }
    ffnDown := some { name := "ffnDown (BitLinear transpose)", inDim := 6912, outDim := 2560, verified := false }
  }

  -- Check attention completeness
  IO.println "Attention backward:"
  let missingAttn := builder.missingAttentionOps
  if missingAttn.isEmpty then
    IO.println "  ✓ All attention backward ops implemented"
  else
    IO.println s!"  ✗ Missing {missingAttn.size} ops: {missingAttn.toList}"

  -- Check FFN completeness
  IO.println ""
  IO.println "FFN backward:"
  let missingFFN := builder.missingFFNOps
  if missingFFN.isEmpty then
    IO.println "  ✓ All FFN backward ops implemented"
  else
    IO.println s!"  ✗ Missing {missingFFN.size} ops: {missingFFN.toList}"

  -- Build attention chain (should succeed)
  IO.println ""
  match builder.buildAttentionChain with
  | some chain =>
    IO.println "Attention DiffChain built successfully:"
    chain.printChain
    let dimOk := chain.checkDimensions
    IO.println s!"  Dimension check: {if dimOk then "PASS" else "FAIL"}"
  | none =>
    IO.println "  ✗ Cannot build attention chain — missing ops"

  -- Overall status
  IO.println ""
  let totalMissing := missingAttn.size + missingFFN.size
  IO.println s!"Total: {totalMissing} missing backward ops"
  if totalMissing == 0 then
    IO.println "✓ Backward chain is COMPLETE"
  else
    IO.println s!"  Attention: {if missingAttn.isEmpty then "complete" else s!"{missingAttn.size} missing"}"
    IO.println s!"  FFN: {if missingFFN.isEmpty then "complete" else s!"{missingFFN.size} missing"}"
