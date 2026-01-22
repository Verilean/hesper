/-!
# Gemma 3 Configuration

Configuration parameters for the Gemma 3 model family (1B, etc.).
-/

namespace Hesper.Models.Gemma3

structure Config where
  vocabSize : Nat := 256000
  hiddenSize : Nat := 2048
  intermediateSize : Nat := 16384 -- GeGLU size (usually different than hidden)
  numHiddenLayers : Nat := 18
  numAttentionHeads : Nat := 8
  numKeyValueHeads : Nat := 1 -- Multi-Query / Group-Query Attention
  headDim : Nat := 256
  rmsNormEps : Float := 1e-6
  ropeTheta : Float := 10000.0
  deriving Inhabited, Repr

/-- 1B Model Config -/
def Config.gemma3_1b : Config := {
  hiddenSize := 2048
  numHiddenLayers := 18
  numAttentionHeads := 8
  numKeyValueHeads := 1
  headDim := 256
}

end Hesper.Models.Gemma3
