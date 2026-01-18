/-!
# Adam Optimizer (Adaptive Moment Estimation)

Adam optimizer with adaptive learning rates per parameter.
Combines ideas from RMSprop and momentum.

## Algorithm
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          # First moment (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         # Second moment (uncentered variance)
m̂_t = m_t / (1 - β₁^t)                        # Bias correction
v̂_t = v_t / (1 - β₂^t)                        # Bias correction
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)       # Parameter update
```

## Typical Hyperparameters
- Learning rate (α): 0.001
- β₁ (momentum): 0.9
- β₂ (RMS decay): 0.999
- ε (numerical stability): 1e-8

## Usage Example
```lean
-- Create optimizer
let opt := AdamConfig.default.withLearningRate 0.001

-- Optimization step
let (newParams, newState) := opt.step params grads state
```

## References
- Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
- https://arxiv.org/abs/1412.6980
-/

namespace Hesper.Optimizer.Adam

/-- Adam optimizer configuration -/
structure AdamConfig where
  /-- Learning rate (typical: 0.001) -/
  learningRate : Float := 0.001
  /-- Exponential decay rate for first moment estimates (typical: 0.9) -/
  beta1 : Float := 0.9
  /-- Exponential decay rate for second moment estimates (typical: 0.999) -/
  beta2 : Float := 0.999
  /-- Small constant for numerical stability (typical: 1e-8) -/
  epsilon : Float := 1e-8
  /-- Weight decay (L2 regularization) -/
  weightDecay : Float := 0.0
  /-- Use AMSGrad variant (maintains max of v_t) -/
  amsgrad : Bool := false
  deriving Inhabited, Repr

namespace AdamConfig

/-- Default Adam configuration (recommended hyperparameters) -/
def default : AdamConfig := {
  learningRate := 0.001
  beta1 := 0.9
  beta2 := 0.999
  epsilon := 1e-8
  weightDecay := 0.0
  amsgrad := false
}

/-- Create Adam with specified learning rate -/
def withLearningRate (config : AdamConfig) (lr : Float) : AdamConfig :=
  { config with learningRate := lr }

/-- Create Adam with weight decay (AdamW variant) -/
def withWeightDecay (config : AdamConfig) (wd : Float) : AdamConfig :=
  { config with weightDecay := wd }

/-- Create Adam with custom beta parameters -/
def withBetas (config : AdamConfig) (beta1 beta2 : Float) : AdamConfig :=
  { config with beta1 := beta1, beta2 := beta2 }

/-- Enable AMSGrad variant -/
def withAMSGrad (config : AdamConfig) : AdamConfig :=
  { config with amsgrad := true }

end AdamConfig

/-- Adam optimizer state (moment estimates) -/
structure AdamState where
  /-- First moment estimates (momentum) for each parameter -/
  m : Array (Array Float)
  /-- Second moment estimates (uncentered variance) for each parameter -/
  v : Array (Array Float)
  /-- Maximum of v_t (for AMSGrad) -/
  vMax : Array (Array Float)
  /-- Number of steps taken -/
  step : Nat
  deriving Inhabited, Repr

namespace AdamState

/-- Initialize Adam state for given parameter shapes -/
def init (paramShapes : Array Nat) : AdamState := {
  m := paramShapes.map (fun size => Array.mk (List.replicate size 0.0))
  v := paramShapes.map (fun size => Array.mk (List.replicate size 0.0))
  vMax := paramShapes.map (fun size => Array.mk (List.replicate size 0.0))
  step := 0
}

/-- Initialize Adam state from parameters -/
def fromParams (params : Array (Array Float)) : AdamState := {
  m := params.map (fun p => Array.mk (List.replicate p.size 0.0))
  v := params.map (fun p => Array.mk (List.replicate p.size 0.0))
  vMax := params.map (fun p => Array.mk (List.replicate p.size 0.0))
  step := 0
}

end AdamState

/-- Update a single parameter with Adam -/
def updateParam
    (config : AdamConfig)
    (param : Array Float)
    (grad : Array Float)
    (m : Array Float)
    (v : Array Float)
    (vMax : Array Float)
    (step : Nat)
    : Array Float × Array Float × Array Float × Array Float :=
  let t := step + 1  -- Step count starts at 1

  -- Apply weight decay if enabled (AdamW style - decoupled weight decay)
  let grad := if config.weightDecay != 0.0 then
    Array.zipWith (fun g p => g + config.weightDecay * p) grad param
  else
    grad

  -- Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
  let newM := Array.zipWith (fun m_prev g => config.beta1 * m_prev + (1.0 - config.beta1) * g) m grad

  -- Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
  let newV := Array.zipWith (fun v_prev g => config.beta2 * v_prev + (1.0 - config.beta2) * (g * g)) v grad

  -- Compute bias-corrected moment estimates
  let beta1Power := Float.pow config.beta1 (Float.ofNat t)
  let beta2Power := Float.pow config.beta2 (Float.ofNat t)
  let biasCorrection1 := 1.0 - beta1Power
  let biasCorrection2 := 1.0 - beta2Power

  -- m̂_t = m_t / (1 - β₁^t)
  let mHat := newM.map (· / biasCorrection1)

  -- v̂_t = v_t / (1 - β₂^t)
  let vHat := newV.map (· / biasCorrection2)

  -- AMSGrad: maintain max of v_t
  let (effectiveV, newVMax) := if config.amsgrad then
    let newVMax := Array.zipWith (fun v v_max => max v v_max) vHat vMax
    (newVMax, newVMax)
  else
    (vHat, vMax)

  -- Update parameters: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
  let newParam := Array.range param.size |>.map fun i =>
    let p := param[i]!
    let m := mHat[i]!
    let v := effectiveV[i]!
    p - config.learningRate * m / (Float.sqrt v + config.epsilon)

  (newParam, newM, newV, newVMax)

/-- Perform one optimization step for all parameters -/
def step
    (config : AdamConfig)
    (params : Array (Array Float))
    (grads : Array (Array Float))
    (state : AdamState)
    : Array (Array Float) × AdamState :=
  let n := params.size

  -- Update each parameter with its corresponding gradient and moments
  let (newParams, newMs, newVs, newVMaxs) :=
    Array.range n |>.foldl (init := (#[], #[], #[], #[])) fun (ps, ms, vs, vms) i =>
      let param := params[i]!
      let grad := grads[i]!
      let m := state.m[i]!
      let v := state.v[i]!
      let vMax := state.vMax[i]!
      let (newP, newM, newV, newVMax) := updateParam config param grad m v vMax state.step
      (ps.push newP, ms.push newM, vs.push newV, vms.push newVMax)

  -- Create new state
  let newState : AdamState := {
    m := newMs
    v := newVs
    vMax := newVMaxs
    step := state.step + 1
  }

  (newParams, newState)

/-- Helper: Compute effective learning rate at step t -/
def effectiveLearningRate (config : AdamConfig) (step : Nat) : Float :=
  let t := step + 1
  let beta1Power := Float.pow config.beta1 (Float.ofNat t)
  let beta2Power := Float.pow config.beta2 (Float.ofNat t)
  let biasCorrection1 := 1.0 - beta1Power
  let biasCorrection2 := 1.0 - beta2Power
  config.learningRate * Float.sqrt biasCorrection2 / biasCorrection1

/-- Helper: Compute L2 norm of first moments (momentum magnitude) -/
def momentumNorm (state : AdamState) : Float :=
  let sumSq := state.m.foldl (init := 0.0) fun acc m =>
    acc + m.foldl (init := 0.0) fun acc2 mi => acc2 + mi * mi
  Float.sqrt sumSq

/-- Helper: Compute L2 norm of second moments (variance magnitude) -/
def varianceNorm (state : AdamState) : Float :=
  let sumSq := state.v.foldl (init := 0.0) fun acc v =>
    acc + v.foldl (init := 0.0) fun acc2 vi => acc2 + vi * vi
  Float.sqrt sumSq

/-- Helper: Get statistics for monitoring -/
structure AdamStats where
  step : Nat
  effectiveLR : Float
  momentumNorm : Float
  varianceNorm : Float
  deriving Repr

def getStats (config : AdamConfig) (state : AdamState) : AdamStats := {
  step := state.step
  effectiveLR := effectiveLearningRate config state.step
  momentumNorm := momentumNorm state
  varianceNorm := varianceNorm state
}

end Hesper.Optimizer.Adam
