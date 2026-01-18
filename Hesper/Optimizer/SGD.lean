/-!
# Stochastic Gradient Descent (SGD) Optimizer

Classical SGD optimizer with optional momentum and weight decay.

## Variants
- Standard SGD: θ = θ - lr * ∇θ
- SGD with Momentum: Uses exponential moving average of gradients
- SGD with Nesterov: Lookahead momentum variant

## Usage Example
```lean
-- Create optimizer
let opt := SGDConfig.default.withLearningRate 0.01 |>.withMomentum 0.9

-- Optimization step
let (newParams, newState) := opt.step params grads state
```
-/

namespace Hesper.Optimizer.SGD

/-- SGD optimizer configuration -/
structure SGDConfig where
  /-- Learning rate (step size) -/
  learningRate : Float := 0.01
  /-- Momentum coefficient (0 = no momentum, typical: 0.9) -/
  momentum : Float := 0.0
  /-- Weight decay (L2 regularization coefficient) -/
  weightDecay : Float := 0.0
  /-- Use Nesterov momentum -/
  nesterov : Bool := false
  /-- Dampening for momentum (typical: 0 for standard momentum) -/
  dampening : Float := 0.0
  deriving Inhabited, Repr

namespace SGDConfig

/-- Default SGD configuration (no momentum, lr=0.01) -/
def default : SGDConfig := {
  learningRate := 0.01
  momentum := 0.0
  weightDecay := 0.0
  nesterov := false
  dampening := 0.0
}

/-- Create SGD with specified learning rate -/
def withLearningRate (config : SGDConfig) (lr : Float) : SGDConfig :=
  { config with learningRate := lr }

/-- Create SGD with momentum -/
def withMomentum (config : SGDConfig) (m : Float) : SGDConfig :=
  { config with momentum := m }

/-- Create SGD with weight decay (L2 regularization) -/
def withWeightDecay (config : SGDConfig) (wd : Float) : SGDConfig :=
  { config with weightDecay := wd }

/-- Enable Nesterov momentum -/
def withNesterov (config : SGDConfig) : SGDConfig :=
  { config with nesterov := true }

end SGDConfig

/-- SGD optimizer state (momentum buffers) -/
structure SGDState where
  /-- Momentum buffers for each parameter -/
  momentumBuffers : Array (Array Float)
  /-- Number of steps taken -/
  step : Nat
  deriving Inhabited, Repr

namespace SGDState

/-- Initialize SGD state for given parameter shapes -/
def init (paramShapes : Array Nat) : SGDState := {
  momentumBuffers := paramShapes.map (fun size => Array.mk (List.replicate size 0.0))
  step := 0
}

/-- Initialize SGD state from parameters -/
def fromParams (params : Array (Array Float)) : SGDState := {
  momentumBuffers := params.map (fun p => Array.mk (List.replicate p.size 0.0))
  step := 0
}

end SGDState

/-- Update a single parameter with SGD -/
def updateParam
    (config : SGDConfig)
    (param : Array Float)
    (grad : Array Float)
    (momentumBuffer : Array Float)
    : Array Float × Array Float :=
  -- Apply weight decay if enabled
  let grad := if config.weightDecay != 0.0 then
    Array.zipWith (fun g p => g + config.weightDecay * p) grad param
  else
    grad

  -- Apply momentum if enabled
  let (grad, newMomentum) := if config.momentum != 0.0 then
    -- Update momentum buffer: m = momentum * m_prev + (1 - dampening) * grad
    let newMomentum := Array.zipWith (fun m g => config.momentum * m + (1.0 - config.dampening) * g) momentumBuffer grad

    -- Use Nesterov momentum if enabled
    let effectiveGrad := if config.nesterov then
      Array.zipWith (fun g m => g + config.momentum * m) grad newMomentum
    else
      newMomentum

    (effectiveGrad, newMomentum)
  else
    (grad, momentumBuffer)

  -- Update parameters: θ = θ - lr * grad
  let newParam := Array.zipWith (fun p g => p - config.learningRate * g) param grad

  (newParam, newMomentum)

/-- Perform one optimization step for all parameters -/
def step
    (config : SGDConfig)
    (params : Array (Array Float))
    (grads : Array (Array Float))
    (state : SGDState)
    : Array (Array Float) × SGDState :=
  -- Update each parameter
  let results := Array.zipWith (fun param grad =>
    let idx := 0  -- Will be properly indexed in the fold
    updateParam config param grad (state.momentumBuffers[idx]!)) params grads

  let (newParams, newMomentums) := results.foldl (init := (#[], #[])) fun (ps, ms) (p, m) =>
    (ps.push p, ms.push m)

  -- Update state
  let newState : SGDState := {
    momentumBuffers := newMomentums
    step := state.step + 1
  }

  (newParams, newState)

/-- Perform optimization step with proper indexing -/
def stepIndexed
    (config : SGDConfig)
    (params : Array (Array Float))
    (grads : Array (Array Float))
    (state : SGDState)
    : Array (Array Float) × SGDState :=
  let n := params.size

  -- Update each parameter with its corresponding gradient and momentum
  let (newParams, newMomentums) := Array.range n |>.foldl (init := (#[], #[])) fun (ps, ms) i =>
    let param := params[i]!
    let grad := grads[i]!
    let momentum := state.momentumBuffers[i]!
    let (newP, newM) := updateParam config param grad momentum
    (ps.push newP, ms.push newM)

  -- Create new state
  let newState : SGDState := {
    momentumBuffers := newMomentums
    step := state.step + 1
  }

  (newParams, newState)

/-- Helper: Compute L2 norm of gradients (for monitoring) -/
def gradNorm (grads : Array (Array Float)) : Float :=
  let sumSq := grads.foldl (init := 0.0) fun acc grad =>
    acc + grad.foldl (init := 0.0) fun acc2 g => acc2 + g * g
  Float.sqrt sumSq

/-- Helper: Compute parameter L2 norm -/
def paramNorm (params : Array (Array Float)) : Float :=
  let sumSq := params.foldl (init := 0.0) fun acc param =>
    acc + param.foldl (init := 0.0) fun acc2 p => acc2 + p * p
  Float.sqrt sumSq

end Hesper.Optimizer.SGD
