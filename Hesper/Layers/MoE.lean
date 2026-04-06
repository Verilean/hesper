import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# Mixture of Experts (MoE) Layer

Implements the Gemma 4 hybrid MoE architecture:
1. **Shared expert** (always active): standard GeGLU FFN
2. **Router**: softmax gating to select top-K experts
3. **Routed experts**: parallel FFN computation with merged weight tensors
4. **Combine**: shared_expert + weighted sum of routed experts

## Gemma 4 MoE Flow (from gemma4-iswa.cpp:116-168)

```
attn_out
├─ Shared Expert:
│   ffn_norm(attn_out) → GeGLU FFN (gate/up/down) → post_norm_1
│
├─ Router (operates on attn_out):
│   rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale → ffn_gate_inp → softmax → top-K
│
├─ Routed Experts:
│   ffn_pre_norm_2(attn_out) → expert FFN (merged gate_up_exps, down_exps) → post_norm_2
│
└─ Combine: shared_expert + routed_experts
```

## Router

The router is custom for Gemma 4:
- Input: `rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale`
- Weights: `ffn_gate_inp` [n_expert, n_embd]
- Output: softmax logits → select top-K experts with highest probability

## Expert Weight Format

Experts use merged 3D tensors:
- `ffn_gate_up_exps`: [n_expert, 2*expert_ff_size, n_embd] — merged gate+up weights
- `ffn_down_exps`: [n_expert, n_embd, expert_ff_size] — down projection

## References
- llama.cpp/src/models/gemma4-iswa.cpp lines 116-168
- llama.cpp/src/llama-graph.cpp (build_moe_ffn)
-/

namespace Hesper.Layers.MoE

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- MoE layer configuration -/
structure Config where
  hiddenSize : Nat         -- Model hidden dimension (n_embd)
  expertFFSize : Nat       -- Expert FFN intermediate size
  numExperts : Nat          -- Total number of experts (e.g., 16)
  numExpertsUsed : Nat      -- Top-K experts per token (e.g., 2)
  rmsNormEps : Float        -- RMSNorm epsilon
  deriving Repr, Inhabited

/-! ## Router Kernels -/

/-- Router logits kernel: compute expert selection scores.

    Gemma 4 router:
    1. rms_norm(attn_out)
    2. scale by 1/sqrt(n_embd)
    3. elementwise multiply by router_scale (ffn_gate_inp_s)
    4. matmul with ffn_gate_inp weights → [n_expert] logits

    This kernel handles steps 1-3 (normalization + scaling).
    Step 4 (matmul) uses the existing Linear layer.

    @param hiddenSize Model dimension
    @param eps RMSNorm epsilon
-/
def routerPreprocessKernel (hiddenSize : Nat) (eps : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) hiddenSize)
  let _routerScale ← ShaderM.declareInputBuffer "router_scale" (.array (.scalar .f32) hiddenSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) hiddenSize)

  -- First pass: compute sum of squares for RMSNorm
  -- Using shared memory reduction
  let wgSize := if hiddenSize < 256 then hiddenSize else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)

  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop idx (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "input" i
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" idx localSum
  ShaderM.barrier

  -- Tree reduction
  let mut stride := wgSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt idx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" idx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.add idx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" idx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Compute RMS and apply: output = rms_norm(input) * (1/sqrt(n_embd)) * router_scale
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 hiddenSize.toFloat)) (Exp.litF32 eps))
  let invSqrtEmbd := Exp.litF32 (1.0 / Float.sqrt hiddenSize.toFloat)

  ShaderM.loop idx (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "input" i
    let rs ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "router_scale" i
    let normed := Exp.mul val rms
    let scaled := Exp.mul (Exp.mul normed invSqrtEmbd) rs
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i scaled

/-! ## Softmax + Top-K Selection -/

/-- Softmax over expert logits + top-K selection kernel.

    Input: [numExperts] logits
    Output: [numExpertsUsed] expert indices and weights

    For single-token inference (M=1), this is small enough for a single workgroup.

    @param numExperts Total experts
    @param numExpertsUsed Top-K to select
-/
def softmaxTopKKernel (numExperts numExpertsUsed : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let tid := Exp.vec3X gid

  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) numExperts)
  let _indices ← ShaderM.declareOutputBuffer "indices" (.array (.scalar .u32) numExpertsUsed)
  let _weights ← ShaderM.declareOutputBuffer "weights" (.array (.scalar .f32) numExpertsUsed)

  -- Only thread 0 does the work (numExperts is small, typically 16)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    -- Step 1: Find max for numerical stability
    ShaderM.varNamed "max_val" (.scalar .f32) (Exp.litF32 (-1e38))
    let maxVal : Exp (.scalar .f32) := Exp.var "max_val"
    for i in [0:numExperts] do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExperts) "logits" (Exp.litU32 i)
      ShaderM.assign "max_val" (Exp.max maxVal v)

    -- Step 2: Compute exp and sum
    ShaderM.varNamed "exp_sum" (.scalar .f32) (Exp.litF32 0.0)
    let expSum : Exp (.scalar .f32) := Exp.var "exp_sum"
    for i in [0:numExperts] do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExperts) "logits" (Exp.litU32 i)
      let ev := Exp.exp (Exp.sub v maxVal)
      ShaderM.assign "exp_sum" (Exp.add expSum ev)

    -- Step 3: Top-K selection (simple iterative: find max K times, mask used)
    -- For small numExperts (16), this is efficient enough
    -- We use a "used" mask via setting selected logits to -inf
    -- Store softmax probs temporarily by overwriting logits conceptually
    -- Actually, just find top-K from raw logits, then compute softmax weights

    for k in [0:numExpertsUsed] do
      ShaderM.varNamed s!"best_idx_{k}" (.scalar .u32) (Exp.litU32 0)
      ShaderM.varNamed s!"best_val_{k}" (.scalar .f32) (Exp.litF32 (-1e38))
      let bestIdx : Exp (.scalar .u32) := Exp.var s!"best_idx_{k}"
      let bestVal : Exp (.scalar .f32) := Exp.var s!"best_val_{k}"

      for i in [0:numExperts] do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExperts) "logits" (Exp.litU32 i)
        -- Check if this expert was already selected (compare with previously found indices)
        let mut isUsed := Exp.litBool false
        for prevK in [0:k] do
          let prevIdx : Exp (.scalar .u32) := Exp.var s!"best_idx_{prevK}"
          isUsed := Exp.or isUsed (Exp.eq (Exp.litU32 i) prevIdx)
        let isBetter := Exp.and (Exp.not isUsed) (Exp.gt v bestVal)
        ShaderM.assign s!"best_idx_{k}" (Exp.select isBetter (Exp.litU32 i) bestIdx)
        ShaderM.assign s!"best_val_{k}" (Exp.select isBetter v bestVal)

      -- Write selected expert index
      ShaderM.writeBuffer (ty := .scalar .u32) "indices" (Exp.litU32 k) bestIdx

      -- Compute softmax weight for this expert
      let expVal := Exp.exp (Exp.sub bestVal maxVal)
      let weight := Exp.div expVal expSum
      ShaderM.writeBuffer (ty := .scalar .f32) "weights" (Exp.litU32 k) weight
  ) (pure ())

/-! ## Expert FFN Kernel -/

/-- Expert FFN kernel for a single selected expert.

    Reads from merged 3D weight tensors using expert index.
    Computes GeGLU: GELU(x @ gate_up[:ffSize]) * (x @ gate_up[ffSize:]) then @ down

    For single-token inference, this processes one expert at a time.
    The caller dispatches this once per selected expert.

    @param config MoE configuration
    @param expertIdx Which expert (0..numExpertsUsed-1) to read from indices buffer
-/
def expertFFNGateUpKernel (config : Config) (expertIdx : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Merged gate_up weights: [numExperts, 2*expertFFSize, hiddenSize] stored as u32 (Q4_K_M)
  -- For now, assume F32 expert weights (TODO: Q4_K_M for experts)
  let gateUpSize := config.numExperts * 2 * config.expertFFSize * config.hiddenSize

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.hiddenSize)
  let _gateUpWeights ← ShaderM.declareInputBuffer "gate_up_weights" (.array (.scalar .f32) gateUpSize)
  let _expertIndices ← ShaderM.declareInputBuffer "expert_indices" (.array (.scalar .u32) config.numExpertsUsed)
  let _gateOutput ← ShaderM.declareOutputBuffer "gate_output" (.array (.scalar .f32) config.expertFFSize)
  let _upOutput ← ShaderM.declareOutputBuffer "up_output" (.array (.scalar .f32) config.expertFFSize)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 config.expertFFSize)) (do
    -- Read which expert this is
    let expertId ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.numExpertsUsed) "expert_indices" (Exp.litU32 expertIdx)

    -- Weight offset for this expert's gate row: expert * (2*ffSize*hiddenSize) + idx * hiddenSize
    let expertBase := Exp.mul expertId (Exp.litU32 (2 * config.expertFFSize * config.hiddenSize))
    let gateRowBase := Exp.add expertBase (Exp.mul idx (Exp.litU32 config.hiddenSize))
    let upRowBase := Exp.add expertBase (Exp.mul (Exp.add idx (Exp.litU32 config.expertFFSize)) (Exp.litU32 config.hiddenSize))

    -- Dot product: gate[idx] = sum(gate_up_weights[expert, idx, :] * input[:])
    ShaderM.varNamed "gate_acc" (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.varNamed "up_acc" (.scalar .f32) (Exp.litF32 0.0)
    let gateAcc : Exp (.scalar .f32) := Exp.var "gate_acc"
    let upAcc : Exp (.scalar .f32) := Exp.var "up_acc"

    for chunk in [0: (config.hiddenSize + 3) / 4] do
      let baseI := chunk * 4
      for off in [0:4] do
        let i := baseI + off
        if i < config.hiddenSize then
          let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.hiddenSize) "input" (Exp.litU32 i)
          let gw ← ShaderM.readBuffer (ty := .scalar .f32) (n := gateUpSize) "gate_up_weights" (Exp.add gateRowBase (Exp.litU32 i))
          let uw ← ShaderM.readBuffer (ty := .scalar .f32) (n := gateUpSize) "gate_up_weights" (Exp.add upRowBase (Exp.litU32 i))
          ShaderM.assign "gate_acc" (Exp.add gateAcc (Exp.mul inVal gw))
          ShaderM.assign "up_acc" (Exp.add upAcc (Exp.mul inVal uw))

    ShaderM.writeBuffer (ty := .scalar .f32) "gate_output" idx gateAcc
    ShaderM.writeBuffer (ty := .scalar .f32) "up_output" idx upAcc
  ) (pure ())

/-! ## Expert Output Combination -/

/-- Combine expert outputs with router weights.

    output = sum_k(weight_k * expert_output_k)

    @param hiddenSize Output dimension
    @param numExpertsUsed Number of selected experts
-/
def combineExpertOutputsKernel (hiddenSize numExpertsUsed : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Expert outputs stacked: [numExpertsUsed, hiddenSize]
  let _expertOutputs ← ShaderM.declareInputBuffer "expert_outputs" (.array (.scalar .f32) (numExpertsUsed * hiddenSize))
  let _expertWeights ← ShaderM.declareInputBuffer "expert_weights" (.array (.scalar .f32) numExpertsUsed)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) hiddenSize)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 hiddenSize)) (do
    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    for k in [0:numExpertsUsed] do
      let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExpertsUsed) "expert_weights" (Exp.litU32 k)
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExpertsUsed * hiddenSize) "expert_outputs"
        (Exp.add (Exp.litU32 (k * hiddenSize)) idx)
      ShaderM.assign "acc" (Exp.add acc (Exp.mul w v))

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx acc
  ) (pure ())

end Hesper.Layers.MoE
