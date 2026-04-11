import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.Backend
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
open Hesper
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

/-! ## Expert FFN Kernels -/

/-- Expert gate+up matmul kernel: computes gate and up projections for one expert.

    Reads expert index from indices buffer at `expertIdx` position.
    Indexes into merged 3D gate_up_weights: [numExperts, 2*expertFFSize, hiddenSize]

    One workgroup per output element. Each workgroup cooperatively computes
    one dot product using shared memory + tree reduction.

    @param config MoE configuration
    @param expertIdx Which selected expert (0..numExpertsUsed-1)
    @param isGate true = compute gate projection, false = compute up projection
    @param workgroupSize Threads per workgroup
-/
def expertGateUpKernel (config : Config) (expertIdx : Nat) (isGate : Bool) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid  -- output element index
  let tid := Exp.vec3X lid

  let gateUpTotalSize := config.numExperts * 2 * config.expertFFSize * config.hiddenSize

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.hiddenSize)
  let _gateUpWeights ← ShaderM.declareInputBuffer "gate_up_weights" (.array (.scalar .f32) gateUpTotalSize)
  let _expertIndices ← ShaderM.declareInputBuffer "expert_indices" (.array (.scalar .u32) config.numExpertsUsed)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.expertFFSize)

  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt outIdx (Exp.litU32 config.expertFFSize)) (do
    -- Read expert ID
    let expertId ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.numExpertsUsed) "expert_indices" (Exp.litU32 expertIdx)

    -- Row offset in gate_up_weights: expert * (2*ffSize*hiddenSize) + rowIdx * hiddenSize
    let rowIdx := if isGate then outIdx else Exp.add outIdx (Exp.litU32 config.expertFFSize)
    let rowBase := Exp.add
      (Exp.mul expertId (Exp.litU32 (2 * config.expertFFSize * config.hiddenSize)))
      (Exp.mul rowIdx (Exp.litU32 config.hiddenSize))

    -- Cooperative dot product
    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    ShaderM.loop tid (Exp.litU32 config.hiddenSize) (Exp.litU32 workgroupSize) fun i => do
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.hiddenSize) "input" i
      let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := gateUpTotalSize) "gate_up_weights" (Exp.add rowBase i)
      ShaderM.assign "acc" (Exp.add acc (Exp.mul inVal wVal))

    -- Tree reduction
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
    ShaderM.barrier
    let mut stride := workgroupSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2

    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
    ) (pure ())
  ) (pure ())

/-- Expert down projection kernel: computes output = hidden @ down_weights[expert].

    down_weights: [numExperts, hiddenSize, expertFFSize]
    One workgroup per output element (hiddenSize).

    @param config MoE configuration
    @param expertIdx Which selected expert (0..numExpertsUsed-1)
    @param workgroupSize Threads per workgroup
-/
def expertDownKernel (config : Config) (expertIdx : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let downTotalSize := config.numExperts * config.hiddenSize * config.expertFFSize

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.expertFFSize)
  let _downWeights ← ShaderM.declareInputBuffer "down_weights" (.array (.scalar .f32) downTotalSize)
  let _expertIndices ← ShaderM.declareInputBuffer "expert_indices" (.array (.scalar .u32) config.numExpertsUsed)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.hiddenSize)

  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt outIdx (Exp.litU32 config.hiddenSize)) (do
    let expertId ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.numExpertsUsed) "expert_indices" (Exp.litU32 expertIdx)

    -- Row: expert * (hiddenSize * expertFFSize) + outIdx * expertFFSize
    let rowBase := Exp.add
      (Exp.mul expertId (Exp.litU32 (config.hiddenSize * config.expertFFSize)))
      (Exp.mul outIdx (Exp.litU32 config.expertFFSize))

    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
    let acc : Exp (.scalar .f32) := Exp.var "acc"

    ShaderM.loop tid (Exp.litU32 config.expertFFSize) (Exp.litU32 workgroupSize) fun i => do
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.expertFFSize) "input" i
      let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := downTotalSize) "down_weights" (Exp.add rowBase i)
      ShaderM.assign "acc" (Exp.add acc (Exp.mul inVal wVal))

    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
    ShaderM.barrier
    let mut stride := workgroupSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2

    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
    ) (pure ())
  ) (pure ())

/-- GELU + multiply kernel for expert FFN:
    output[i] = GELU(gate[i]) * up[i]
    @param size Expert FFN intermediate size
-/
def expertGeluMulKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) size)
  let _up ← ShaderM.declareInputBuffer "up" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "gate" idx
    let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "up" idx
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let x3 := Exp.mul (Exp.mul g g) g
    let inner := Exp.mul sqrt2OverPi (Exp.add g (Exp.mul (Exp.litF32 0.044715) x3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) g) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul gelu u)
  ) (pure ())

/-- Weighted accumulate kernel: output += weight * expert_output
    Used to accumulate weighted expert outputs.
    @param size Hidden dimension
    @param expertIdx Which expert's weight to read
-/
def weightedAccumulateKernel (size numExpertsUsed expertIdx : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _accumulator ← ShaderM.declareOutputBuffer "accumulator" (.array (.scalar .f32) size)
  let _expertOutput ← ShaderM.declareInputBuffer "expert_output" (.array (.scalar .f32) size)
  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .f32) numExpertsUsed)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := numExpertsUsed) "weights" (Exp.litU32 expertIdx)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "expert_output" idx
    let acc ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "accumulator" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "accumulator" idx (Exp.add acc (Exp.mul w v))
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
