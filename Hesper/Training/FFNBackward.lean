import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.BitLinear
import Hesper.Training.AttentionBackward
import Hesper.Training.BitLinearBackward
import Hesper.LoRA.Forward

/-!
# FFN Backward GPU Kernels

Backward pass for the FFN (Feed-Forward Network) sub-layer:

Forward:
  gate = W_gate @ normed2
  up = W_up @ normed2
  hidden = ReLU²(gate) × up
  ffnNormed = RMSNorm(hidden)
  output = residual + W_down @ ffnNormed

Backward (reverse):
  1. dFFNNormed = W_down^T @ dOutput
  2. dHidden = RMSNorm_bwd(hidden, gamma, dFFNNormed)
  3. dGate, dUp = ReLU²Mul_bwd(gate, up, dHidden)
  4. dNormed2 = W_gate^T @ dGate + W_up^T @ dUp
  5. dResidual += RMSNorm_bwd(residual, gamma, dNormed2)

## ReLU²×Mul Backward

Forward: h = max(0, gate)² × up
Backward:
  dGate = dH × up × 2 × ReLU(gate)
  dUp = dH × max(0, gate)²
-/

namespace Hesper.Training.FFNBackward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-- ReLU²×Mul backward kernel.
    Forward: hidden[i] = max(0, gate[i])² × up[i]
    Backward:
      dGate[i] = dHidden[i] × up[i] × 2 × max(0, gate[i])
      dUp[i] = dHidden[i] × max(0, gate[i])²

    Reads: gate, up, dHidden
    Writes: dGate, dUp -/
def reluSqrMulBackwardKernel (numElements : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) numElements)
  let _up ← ShaderM.declareInputBuffer "up" (.array (.scalar .f32) numElements)
  let _dHidden ← ShaderM.declareInputBuffer "dHidden" (.array (.scalar .f32) numElements)
  let _dGate ← ShaderM.declareOutputBuffer "dGate" (.array (.scalar .f32) numElements)
  let _dUp ← ShaderM.declareOutputBuffer "dUp" (.array (.scalar .f32) numElements)

  ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
    let gateVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "gate" i
    let upVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "up" i
    let dH ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "dHidden" i

    -- ReLU(gate) = max(0, gate)
    let relu := Exp.max gateVal (Exp.litF32 0.0)
    -- ReLU²(gate) = relu²
    let reluSq := Exp.mul relu relu

    -- dGate = dH × up × 2 × relu
    let dGateVal := Exp.mul (Exp.mul dH upVal) (Exp.mul (Exp.litF32 2.0) relu)
    -- dUp = dH × relu²
    let dUpVal := Exp.mul dH reluSq

    ShaderM.writeBuffer (ty := .scalar .f32) "dGate" i dGateVal
    ShaderM.writeBuffer (ty := .scalar .f32) "dUp" i dUpVal
  ) (pure ())

def executeReluSqrMulBackward (device : Device) (gateBuf upBuf dHiddenBuf dGateBuf dUpBuf : Buffer)
    (numElements : Nat) : IO Unit := do
  let shader := reluSqrMulBackwardKernel numElements
  let namedBuffers := [("gate", gateBuf), ("up", upBuf), ("dHidden", dHiddenBuf),
                       ("dGate", dGateBuf), ("dUp", dUpBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute full FFN backward for one layer.
    Requires saved forward activations: gate, up, hidden, residual1.

    @param device GPU device
    @param block Transformer block (for weight access)
    @param dOutputBuf Gradient from next layer [dim]
    @param dHiddenBuf Scratch buffer [dim] — will contain dResidual contribution
    @param savedGate Saved gate buffer [ffnDim] from forward
    @param savedUp Saved up buffer [ffnDim] from forward
    @param savedHidden Saved hidden buffer [ffnDim] from forward (pre sub-norm)
    @param savedResidual1 Saved residual1 buffer [dim] from forward (pre ffn-norm)
    @param dFFNNormed Scratch [ffnDim]
    @param dFFNHidden Scratch [ffnDim]
    @param dGate Scratch [ffnDim]
    @param dUp Scratch [ffnDim]
    @param dNormed2 Scratch [dim] -/
def executeFFNBackward (device : Device)
    (wDown wGate wUp : Hesper.Layers.BitLinear.BitLinear)
    (ffnSubNormScale ffnNormScale : Buffer)
    (dOutputBuf : Buffer)
    (savedHidden savedResidual1 savedGate savedUp : Buffer)
    (dFFNNormed dFFNHidden dGate dUp dNormed2 dHiddenBuf : Buffer)
    (dim ffnDim : Nat) : IO Unit := do
  -- Step 1: dFFNNormed = W_down^T @ dOutput
  Hesper.Training.BitLinearBackward.executeBitLinearTranspose device wDown dOutputBuf dFFNNormed

  -- Step 2: dFFNHidden = RMSNorm_bwd(savedHidden, ffnSubNormScale, dFFNNormed)
  Hesper.Training.AttentionBackward.executeRmsNormBackward device
    savedHidden ffnSubNormScale dFFNNormed dFFNHidden ffnDim

  -- Step 3: dGate, dUp = ReLU²Mul_bwd(savedGate, savedUp, dFFNHidden)
  executeReluSqrMulBackward device savedGate savedUp dFFNHidden dGate dUp ffnDim

  -- Step 4: dNormed2 = W_gate^T @ dGate + W_up^T @ dUp
  Hesper.Training.BitLinearBackward.executeBitLinearTranspose device wGate dGate dNormed2
  -- Add W_up^T @ dUp to dNormed2
  Hesper.Training.BitLinearBackward.executeBitLinearTranspose device wUp dUp dHiddenBuf
  -- dNormed2 += dHiddenBuf (using addScaled with scale=1.0)
  Hesper.LoRA.Forward.executeAddScaled device dHiddenBuf dNormed2 dim 1.0

  -- Step 5: dResidual1_contribution = RMSNorm_bwd(savedResidual1, ffnNormScale, dNormed2)
  -- Write result to dHiddenBuf (which represents the FFN's contribution to dResidual)
  Hesper.Training.AttentionBackward.executeRmsNormBackward device
    savedResidual1 ffnNormScale dNormed2 dHiddenBuf dim

end Hesper.Training.FFNBackward
