import Hesper
import Hesper.AD.Reverse
import Hesper.Optimizer.SGD
import Examples.MachineLearning.MNISTData

/-!
# MNIST MLP Training with Automatic Differentiation

Full training example with:
- 2-layer MLP neural network
- Automatic differentiation for gradients
- SGD optimizer
- Training loop with real backpropagation

Network: 784 → 128 → 10 with ReLU and Softmax
-/

namespace Examples.MachineLearning.MNISTTrain

open Hesper.WebGPU
open Hesper.AD.Reverse
open Hesper.Optimizer.SGD
open Examples.MachineLearning.MNISTData

/-- MLP configuration -/
structure MLPConfig where
  inputSize : Nat := 784
  hiddenSize : Nat := 128
  outputSize : Nat := 10
  batchSize : Nat := 32
  learningRate : Float := 0.01
  numEpochs : Nat := 3
  deriving Repr

/-- MLP parameters -/
structure MLPParams where
  w1 : Array Float  -- [inputSize, hiddenSize]
  b1 : Array Float  -- [hiddenSize]
  w2 : Array Float  -- [hiddenSize, outputSize]
  b2 : Array Float  -- [outputSize]
  deriving Repr

/-- Initialize random weights (Xavier initialization) -/
def initWeights (inputSize outputSize seed : Nat) : Array Float :=
  let scale := Float.sqrt (2.0 / inputSize.toFloat)
  Array.range (inputSize * outputSize) |>.map fun i =>
    let x := ((i + seed) % 1000).toFloat / 1000.0
    (x - 0.5) * scale

/-- Initialize bias to zeros -/
def initBias (size : Nat) : Array Float :=
  Array.mk (List.replicate size 0.0)

/-- Initialize MLP parameters -/
def initParams (config : MLPConfig) : MLPParams :=
  {
    w1 := initWeights config.inputSize config.hiddenSize 42
    b1 := initBias config.hiddenSize
    w2 := initWeights config.hiddenSize config.outputSize 123
    b2 := initBias config.outputSize
  }

/-- Flatten parameters into array for optimizer -/
def paramsToArray (params : MLPParams) : Array Float :=
  params.w1 ++ params.b1 ++ params.w2 ++ params.b2

/-- Reconstruct parameters from flattened array -/
def arrayToParams (arr : Array Float) (config : MLPConfig) : MLPParams :=
  let w1Size := config.inputSize * config.hiddenSize
  let b1Size := config.hiddenSize
  let w2Size := config.hiddenSize * config.outputSize
  let b2Size := config.outputSize

  let w1 := arr.extract 0 w1Size
  let b1 := arr.extract w1Size (w1Size + b1Size)
  let w2 := arr.extract (w1Size + b1Size) (w1Size + b1Size + w2Size)
  let b2 := arr.extract (w1Size + b1Size + w2Size) (w1Size + b1Size + w2Size + b2Size)

  { w1, b1, w2, b2 }

/-- Matrix-vector multiplication with AD -/
def matVecMul (ctx : ADContext) (input : Array Dual) (weights : Array Dual) (bias : Array Dual)
    (inputSize outputSize : Nat) : ADContext × Array Dual :=
  Array.range outputSize |>.foldl (init := (ctx, #[])) fun (ctx, acc) i =>
    -- Compute dot product for output i
    let (ctx, sum) := Array.range inputSize |>.foldl (init := (ctx, bias[i]!)) fun (ctx, acc) j =>
      let (ctx, prod) := ctx.mul input[j]! weights[j * outputSize + i]!
      ctx.add acc prod
    (ctx, acc.push sum)

/-- ReLU activation -/
def applyReLU (ctx : ADContext) (xs : Array Dual) : ADContext × Array Dual :=
  xs.foldl (init := (ctx, #[])) fun (ctx, acc) x =>
    let (ctx, y) := ctx.relu x
    (ctx, acc.push y)

/-- Softmax activation (numerically stable) with AD -/
def applySoftmax (ctx : ADContext) (xs : Array Dual) : ADContext × Array Dual :=
  -- Find max for numerical stability (this is just for numerical stability, not part of gradient)
  let maxVal := xs.foldl (init := xs[0]!.primal) fun acc x => max acc x.primal
  let maxDual := Dual.const maxVal

  -- Compute exp(x - max) for each element using AD
  let (ctx, exps) := xs.foldl (init := (ctx, #[])) fun (ctx, acc) x =>
    let (ctx, diff) := ctx.sub x maxDual
    let (ctx, expVal) := ctx.exp diff
    (ctx, acc.push expVal)

  -- Sum of exponentials using AD
  let (ctx, sumExp) := exps.foldl (init := (ctx, Dual.const 0.0)) fun (ctx, acc) expVal =>
    ctx.add acc expVal

  -- Normalize: divide each exp by sum using AD
  let (ctx, normalized) := exps.foldl (init := (ctx, #[])) fun (ctx, acc) expVal =>
    let (ctx, prob) := ctx.div expVal sumExp
    (ctx, acc.push prob)

  (ctx, normalized)

/-- Forward pass through MLP with AD -/
def forwardPassAD (ctx : ADContext) (input : Array Dual) (params : Array Dual) (config : MLPConfig)
    : ADContext × Array Dual :=
  let w1Size := config.inputSize * config.hiddenSize
  let b1Size := config.hiddenSize
  let w2Size := config.hiddenSize * config.outputSize

  -- Extract parameters
  let w1 := params.extract 0 w1Size
  let b1 := params.extract w1Size (w1Size + b1Size)
  let w2 := params.extract (w1Size + b1Size) (w1Size + b1Size + w2Size)
  let b2 := params.extract (w1Size + b1Size + w2Size) (params.size)

  -- Layer 1: Input -> Hidden + ReLU
  let (ctx, h1) := matVecMul ctx input w1 b1 config.inputSize config.hiddenSize
  let (ctx, h1_relu) := applyReLU ctx h1

  -- Layer 2: Hidden -> Output (pre-softmax)
  let (ctx, logits) := matVecMul ctx h1_relu w2 b2 config.hiddenSize config.outputSize

  (ctx, logits)

/-- Cross-entropy loss with AD -/
def crossEntropyLossAD (ctx : ADContext) (predictions : Array Dual) (labels : Array Float)
    : ADContext × Dual :=
  let epsilon := 1e-7

  -- Apply softmax to predictions with AD
  let (ctx, probs) := applySoftmax ctx predictions

  -- Compute -sum(y_true * log(y_pred)) using AD operations
  (Array.range labels.size).foldl (init := (ctx, Dual.const 0.0)) fun (ctx, acc) i =>
    let label := labels[i]!
    if label > 0.0 then
      let prob := probs[i]!
      -- Add epsilon for numerical stability: log(max(prob, epsilon))
      let (ctx, probClamped) := ctx.add prob (Dual.const epsilon)
      let (ctx, logProb) := ctx.log probClamped
      -- Multiply by -label
      let labelDual := Dual.const (-label)
      let (ctx, term) := ctx.mul labelDual logProb
      ctx.add acc term
    else
      (ctx, acc)

/-- Compute loss and gradients for a single sample -/
def computeGradientsForSample (input : Array Float) (labels : Array Float) (params : Array Float)
    (config : MLPConfig) : Float × Array Float :=
  let ctx := ADContext.new

  -- Create variables for parameters
  let (ctx, paramDuals) := params.foldl (init := (ctx, #[])) fun (ctx, acc) p =>
    let (ctx, pDual) := ctx.var p
    (ctx, acc.push pDual)

  -- Create constants for input
  let inputDuals := input.map Dual.const

  -- Forward pass
  let (ctx, predictions) := forwardPassAD ctx inputDuals paramDuals config

  -- Compute loss
  let (ctx, loss) := crossEntropyLossAD ctx predictions labels

  -- Backpropagate
  let grads := backprop ctx.tape loss.tapeIdx

  -- Extract gradients for parameters
  let paramGrads := paramDuals.map fun p => grads[p.tapeIdx]!

  (loss.primal, paramGrads)

/-- Train for one epoch -/
def trainEpoch (params : MLPParams) (config : MLPConfig) (epochNum : Nat)
    : IO (MLPParams × Float) := do
  IO.println s!"📊 Epoch {epochNum}/{config.numEpochs}"

  -- Generate training batch
  let batch := generateSyntheticBatch config.batchSize (epochNum * 1000)
  let oneHotLabels := labelsToOneHot batch.labels

  let mut totalLoss := 0.0
  let mut allGrads := Array.mk (List.replicate (paramsToArray params).size 0.0)

  -- Compute gradients for each sample in batch
  for sampleIdx in [0:batch.batchSize] do
    let inputStart := sampleIdx * imageSize
    let inputEnd := inputStart + imageSize
    let input := batch.images.extract inputStart inputEnd

    let labelStart := sampleIdx * numClasses
    let labelEnd := labelStart + numClasses
    let sampleLabels := oneHotLabels.extract labelStart labelEnd

    -- Compute loss and gradients
    let paramArray := paramsToArray params
    let (loss, grads) := computeGradientsForSample input sampleLabels paramArray config

    totalLoss := totalLoss + loss

    -- Accumulate gradients
    allGrads := (Array.range allGrads.size).map fun i =>
      allGrads[i]! + grads[i]!

  -- Average gradients
  let avgGrads := allGrads.map fun g => g / batch.batchSize.toFloat
  let avgLoss := totalLoss / batch.batchSize.toFloat

  -- Update parameters with SGD
  let paramArray := paramsToArray params
  let updatedParams := (Array.range paramArray.size).map fun i =>
    paramArray[i]! - config.learningRate * avgGrads[i]!

  let newParams := arrayToParams updatedParams config

  IO.println s!"  Loss: {avgLoss}"

  return (newParams, avgLoss)

/-- Evaluate accuracy on test data -/
def evaluate (params : MLPParams) (config : MLPConfig) : IO Float := do
  let testBatch := generateSyntheticBatch 100 999
  let mut correct := 0

  for sampleIdx in [0:testBatch.batchSize] do
    let inputStart := sampleIdx * imageSize
    let inputEnd := inputStart + imageSize
    let input := testBatch.images.extract inputStart inputEnd

    -- Forward pass (without AD for evaluation)
    let ctx := ADContext.new
    let inputDuals := input.map Dual.const
    let paramArray := paramsToArray params
    let paramDuals := paramArray.map Dual.const
    let (ctx, predictions) := forwardPassAD ctx inputDuals paramDuals config
    let (_, probs) := applySoftmax ctx predictions

    -- Find predicted class
    let predClass := (Array.range numClasses).foldl (init := (0, 0.0)) fun (maxIdx, maxVal) j =>
      let val := probs[j]!.primal
      if val > maxVal then (j, val) else (maxIdx, maxVal)

    if predClass.1 == testBatch.labels[sampleIdx]! then
      correct := correct + 1

  let accuracy := correct.toFloat / testBatch.batchSize.toFloat * 100.0
  return accuracy

/-- Main training loop -/
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   MNIST MLP Training with AD                 ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize
  IO.println "🚀 Initializing..."
  let _inst ← Hesper.init
  let config : MLPConfig := {}
  IO.println s!"  Architecture: {config.inputSize} → {config.hiddenSize} → {config.outputSize}"
  IO.println s!"  Learning rate: {config.learningRate}"
  IO.println s!"  Epochs: {config.numEpochs}"
  IO.println s!"  Batch size: {config.batchSize}"
  IO.println ""

  -- Initialize parameters
  let mut params := initParams config
  let totalParams := (paramsToArray params).size
  IO.println s!"✓ Initialized {totalParams} parameters"
  IO.println ""

  -- Initial evaluation
  IO.println "🧪 Initial evaluation (random weights)..."
  let initialAcc ← evaluate params config
  IO.println s!"  Accuracy: {initialAcc}%"
  IO.println ""

  -- Training loop
  IO.println "🏋️  Training..."
  IO.println ""

  for epoch in [1:config.numEpochs + 1] do
    let (newParams, _loss) ← trainEpoch params config epoch
    params := newParams

    -- Evaluate after each epoch
    let acc ← evaluate params config
    IO.println s!"  Test Accuracy: {acc}%"
    IO.println ""

  IO.println "✅ Training complete!"
  IO.println ""
  IO.println "📝 Note: Using synthetic data for demonstration"
  IO.println "   For real MNIST: http://yann.lecun.com/exdb/mnist/"

end Examples.MachineLearning.MNISTTrain

def main : IO Unit := Examples.MachineLearning.MNISTTrain.main
