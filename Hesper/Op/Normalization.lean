import Hesper.WGSL.Kernel
import Hesper.WGSL.Exp
import Hesper.WGSL.Types
import Hesper.Tensor.Types

/-!
# Normalization Operators with Fusion Support

Implements normalization operations critical for modern neural networks:
- **Softmax**: Converts logits to probability distribution
- **LayerNorm**: Normalizes activations across features
- **BatchNorm**: Normalizes activations across batch (TODO)
- **RMSNorm**: Root mean square normalization (used in Llama)

## Mathematical Definitions

### Softmax
```
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```
- Numerically stable via max subtraction
- Sum of outputs = 1
- Common in classification and attention

### LayerNorm
```
y = (x - μ) / √(σ² + ε) * γ + β
where μ = mean(x), σ² = variance(x)
```
- Normalizes across feature dimension
- Learnable scale (γ) and shift (β)
- Critical in Transformers

### RMSNorm (Simplified LayerNorm)
```
y = x / √(mean(x²) + ε) * γ
```
- No mean subtraction or bias
- Faster than LayerNorm
- Used in Llama, GPT-NeoX

## Fusion Patterns

### Softmax Fusion
```lean
-- Attention: Q @ K^T |> Softmax |> @ V
attention_scores |> softmaxKernel |> matmul_with_V
```

### LayerNorm Fusion
```lean
-- Transformer: Attention + Residual |> LayerNorm
residual_add |> layerNormKernel
```
-/

namespace Hesper.Op.Normalization

open Hesper.WGSL
open Hesper.Tensor

/-! ## Softmax -/

/-- Softmax kernel (simplified for demonstration).

    **Challenge**: Softmax requires two passes:
    1. Find max and compute sum of exp(x - max)
    2. Normalize by dividing by sum

    For fusion, we can express this as a composable transformation,
    but true implementation requires shared memory and reduction. -/
def softmaxExp (x : Exp (.scalar .f32)) (max_val sum_exp : Exp (.scalar .f32))
    : Exp (.scalar .f32) :=
  -- Numerically stable softmax: exp(x - max) / sum
  let shifted := Exp.sub x max_val
  let exp_val := Exp.exp shifted
  Exp.div exp_val sum_exp

/-- Softmax kernel (conceptual).

    In a full implementation, this would:
    1. Use shared memory for reduction
    2. Compute max in first pass
    3. Compute sum in second pass
    4. Apply normalization

    For now, this is a placeholder showing the structure. -/
def softmaxKernel {N : Nat} : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  -- Placeholder: assumes max and sum are pre-computed
  -- Real implementation would use workgroup reduction
  let max_placeholder := Exp.litF32 0.0
  let sum_placeholder := Exp.litF32 1.0
  mapK (fun x => softmaxExp x max_placeholder sum_placeholder)

/-! ## LayerNorm -/

/-- Layer normalization expression.

    LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

    For fusion, we express this as a transformation that can be
    applied after computing mean and variance. -/
def layerNormExp (x mean variance : Exp (.scalar .f32))
    (gamma beta epsilon : Float) : Exp (.scalar .f32) :=
  -- Normalize: (x - mean) / sqrt(var + eps)
  let centered := Exp.sub x mean
  let var_plus_eps := Exp.add variance (Exp.litF32 epsilon)
  let std_dev := Exp.sqrt var_plus_eps
  let normalized := Exp.div centered std_dev

  -- Scale and shift: normalized * gamma + beta
  let scaled := Exp.mul normalized (Exp.litF32 gamma)
  let shifted := Exp.add scaled (Exp.litF32 beta)
  shifted

/-- LayerNorm kernel (conceptual).

    In a full implementation:
    1. Compute mean via parallel reduction
    2. Compute variance via parallel reduction
    3. Apply normalization, scale, and shift

    This can be fused with subsequent operations. -/
def layerNormKernel {N : Nat} (gamma beta epsilon : Float)
    : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  -- Placeholder: assumes mean and variance are pre-computed
  let mean_placeholder := Exp.litF32 0.0
  let var_placeholder := Exp.litF32 1.0
  mapK (fun x => layerNormExp x mean_placeholder var_placeholder gamma beta epsilon)

/-! ## RMSNorm (Root Mean Square Normalization) -/

/-- RMS normalization expression.

    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

    Simpler than LayerNorm (no mean subtraction, no bias).
    Used in Llama and other modern LLMs. -/
def rmsNormExp (x rms : Exp (.scalar .f32)) (gamma epsilon : Float)
    : Exp (.scalar .f32) :=
  -- Normalize by RMS: x / sqrt(mean(x²) + eps)
  let rms_plus_eps := Exp.add rms (Exp.litF32 epsilon)
  let rms_sqrt := Exp.sqrt rms_plus_eps
  let normalized := Exp.div x rms_sqrt

  -- Scale: normalized * gamma
  Exp.mul normalized (Exp.litF32 gamma)

/-- RMSNorm kernel (conceptual). -/
def rmsNormKernel {N : Nat} (gamma epsilon : Float)
    : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  -- Placeholder: assumes RMS is pre-computed
  let rms_placeholder := Exp.litF32 1.0
  mapK (fun x => rmsNormExp x rms_placeholder gamma epsilon)

/-! ## Fusion Examples -/

/-- Fused pattern: Dense + LayerNorm

    Common in Transformers:
    ```
    x → Linear → LayerNorm → next layer
    ```

    Without fusion: 3 kernels (matmul, layernorm_mean_var, layernorm_apply)
    With fusion: 2 kernels (matmul, layernorm_fused) -/
def linearLayerNorm {D : Nat} (gamma beta epsilon : Float)
    : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  -- First apply linear (placeholder), then LayerNorm
  let norm := layerNormKernel (N := D) gamma beta epsilon
  norm

/-- Fused pattern: Attention + Softmax

    Attention mechanism:
    ```
    scores = Q @ K^T
    attention_weights = Softmax(scores / sqrt(d_k))
    output = attention_weights @ V
    ```

    The Softmax can be fused with the preceding score computation. -/
def attentionSoftmax {D : Nat} (scale : Float)
    : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  -- Scale then softmax
  let scaled := mapK (Exp.mul · (Exp.litF32 scale))
  let softmax := softmaxKernel (N := D)
  Kernel.comp softmax scaled

/-! ## Performance Notes -/

/-- Documentation: Why normalization benefits from fusion

    **Without Fusion**:
    1. Kernel 1: Compute running statistics (mean, var)
    2. Kernel 2: Read x, apply normalization, write back
    3. Kernel 3: Read normalized, apply scale/shift, write back

    **With Fusion**:
    1. Kernel 1: Compute statistics in shared memory
    2. Kernel 2: Apply normalization + scale + shift in one pass

    **Benefit**:
    - 3 kernels → 2 kernels
    - 2 memory roundtrips → 1 memory roundtrip
    - Better cache utilization -/
def fusionBenefits : Unit := ()

/-! ## CPU Specifications -/

/-- CPU Softmax for verification -/
def cpuSoftmax (input : Array Float) : Array Float :=
  -- Find max for numerical stability
  let max_val := input.foldl max (-1e38)  -- Large negative number

  -- Compute exp(x - max) and sum
  let exps := input.map (fun x => Float.exp (x - max_val))
  let sum_exp := exps.foldl (· + ·) 0.0

  -- Normalize
  if sum_exp > 0.0 then
    exps.map (· / sum_exp)
  else
    input

/-- CPU LayerNorm for verification -/
def cpuLayerNorm (input : Array Float) (gamma beta epsilon : Float) : Array Float :=
  let n := input.size.toFloat

  -- Compute mean
  let sum := input.foldl (· + ·) 0.0
  let mean := sum / n

  -- Compute variance
  let var_sum := input.foldl (fun acc x => acc + (x - mean) * (x - mean)) 0.0
  let variance := var_sum / n

  -- Apply normalization
  let std_dev := Float.sqrt (variance + epsilon)
  input.map fun x =>
    let normalized := (x - mean) / std_dev
    normalized * gamma + beta

/-- CPU RMSNorm for verification -/
def cpuRMSNorm (input : Array Float) (gamma epsilon : Float) : Array Float :=
  let n := input.size.toFloat

  -- Compute RMS: sqrt(mean(x²))
  let sq_sum := input.foldl (fun acc x => acc + x * x) 0.0
  let rms := Float.sqrt (sq_sum / n + epsilon)

  -- Apply normalization
  input.map fun x =>
    (x / rms) * gamma

end Hesper.Op.Normalization
