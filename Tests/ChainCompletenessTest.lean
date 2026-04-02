import Hesper.AD.Chain
import Hesper.AD.BackwardOps

open Hesper.AD.Chain
open Hesper.AD.BackwardOps

def main : IO Unit := do
  IO.println "=== Backward Chain Completeness Test ==="
  IO.println ""

  -- Construct a LayerBackwardOps with dummy kernels.
  -- If any field is missing, this WON'T COMPILE.
  -- This is the compile-time completeness guarantee.
  let dummyKernel : BackwardKernel := fun _ => pure ()

  let layerOps : LayerBackwardOps := {
    attention := {
      finalNormBwd := dummyKernel      -- executeRmsNormBackward (final)
      oProjectionBwd := dummyKernel    -- executeBitLinearTranspose (W_O)
      subNormBwd := dummyKernel        -- executeRmsNormBackward (sub-norm)
      applyBwd := dummyKernel          -- executeApplyBackward
      softmaxBwd := dummyKernel        -- executeSoftmaxBackward
      scoreBwd := dummyKernel          -- executeScoreBackwardQ
      ropeBwd := dummyKernel           -- executeRopeBackward
    }
    ffn := {
      ffnDownBwd := dummyKernel        -- executeBitLinearTranspose (W_down)
      ffnSubNormBwd := dummyKernel     -- executeRmsNormBackward (ffn sub-norm)
      ffnActivationBwd := dummyKernel  -- executeReluSqrMulBackward
      ffnGateBwd := dummyKernel        -- executeBitLinearTranspose (W_gate)
      ffnUpBwd := dummyKernel          -- executeBitLinearTranspose (W_up)
      ffnNormBwd := dummyKernel        -- executeRmsNormBackward (pre-FFN)
    }
  }

  -- This line proves completeness at compile time
  let complete := verifyComplete layerOps
  IO.println s!"LayerBackwardOps constructed: {if complete then "COMPLETE" else "INCOMPLETE"}"

  -- Print the structure
  IO.println ""
  IO.println "Attention backward ops (7 ops):"
  IO.println "  ✓ finalNormBwd      — RMSNorm backward (final norm)"
  IO.println "  ✓ oProjectionBwd    — BitLinear transpose (W_O^T)"
  IO.println "  ✓ subNormBwd        — RMSNorm backward (sub-norm)"
  IO.println "  ✓ applyBwd          — Attention apply backward"
  IO.println "  ✓ softmaxBwd        — Softmax backward"
  IO.println "  ✓ scoreBwd          — Score backward (dQ)"
  IO.println "  ✓ ropeBwd           — RoPE backward"
  IO.println ""
  IO.println "FFN backward ops (6 ops):"
  IO.println "  ✓ ffnDownBwd        — BitLinear transpose (W_down^T)"
  IO.println "  ✓ ffnSubNormBwd     — RMSNorm backward (ffn sub-norm)"
  IO.println "  ✓ ffnActivationBwd  — ReLU²×Mul backward"
  IO.println "  ✓ ffnGateBwd        — BitLinear transpose (W_gate^T)"
  IO.println "  ✓ ffnUpBwd          — BitLinear transpose (W_up^T)"
  IO.println "  ✓ ffnNormBwd        — RMSNorm backward (pre-FFN)"
  IO.println ""
  IO.println "✓ Backward chain is COMPLETE (13/13 ops)"
  IO.println ""
  IO.println "Compile-time guarantee: adding a new forward op to"
  IO.println "AttentionBackwardOps or FFNBackwardOps without providing"
  IO.println "a backward implementation will cause a compilation error."
