import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.Backend
import Hesper.Logging

/-!
# Per-Layer Embedding

Implements Gemma 4's per-layer input embedding mechanism.

## Overview

Gemma 4 has an optional per-layer embedding that provides layer-specific
input modifications via gated projections:

```
Per-layer embedding flow (per layer il):
  gate = GELU(per_layer_inp_gate @ cur)        -- [embdPerLayer]
  per_layer_input = gate * inp_per_layer[il]    -- [embdPerLayer]
  projected = per_layer_proj @ per_layer_input  -- [hiddenSize]
  normed = RMSNorm(projected)
  output = cur + normed                         -- residual
```

## Pre-processing (once per forward pass)

Before the layer loop, per-layer inputs are pre-computed:
```
1. tok_embd_per_layer[token_ids] * sqrt(embdPerLayer)  -- [embdPerLayer * numLayers]
2. per_layer_model_proj @ input_embeds / sqrt(hiddenSize)  -- [embdPerLayer * numLayers]
3. RMSNorm(projected)
4. inp_per_layer = projected + token_embeds
5. inp_per_layer *= 1/sqrt(2)
```

## KV Cache Sharing

Last `numKVSharedLayers` layers skip K/V computation:
- Only compute Q projection
- Reuse KV cache from an earlier layer (the last layer that has its own KV)
- Saves memory and compute for deep models

## References
- llama.cpp/src/models/gemma4-iswa.cpp lines 192-213 (per-layer embedding)
- llama.cpp/src/models/gemma4-iswa.cpp lines 69-100 (KV sharing: has_kv check)
- llama.cpp/src/models/gemma4-iswa.cpp lines 258-311 (pre-projection)
-/

namespace Hesper.Layers.PerLayerEmbedding

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

structure Config where
  hiddenSize : Nat       -- Model hidden dimension
  embdPerLayer : Nat     -- Per-layer embedding dimension
  numLayers : Nat        -- Total number of layers
  rmsNormEps : Float     -- RMSNorm epsilon
  deriving Repr, Inhabited

/-! ## GELU Gate + Multiply Kernel -/

/-- GELU gated multiply: output = GELU(gate) * perLayerInput

    This is the per-layer step applied within each transformer block.
    gate: output of per_layer_inp_gate @ cur  [embdPerLayer]
    perLayerInput: pre-computed per-layer embedding for this layer [embdPerLayer]

    @param size embdPerLayer dimension
-/
def geluGateMulKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) size)
  let _perLayerInput ← ShaderM.declareInputBuffer "per_layer_input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let gateVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "gate" idx
    let plVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "per_layer_input" idx

    -- GELU approximation
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let x3 := Exp.mul (Exp.mul gateVal gateVal) gateVal
    let inner := Exp.mul sqrt2OverPi (Exp.add gateVal (Exp.mul (Exp.litF32 0.044715) x3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) gateVal) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))

    -- output = GELU(gate) * per_layer_input
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul gelu plVal)
  ) (pure ())

/-- Same as geluGateMulKernel but reads per_layer_input from offset within a larger buffer.
    Used to slice plInputAll[layerIdx * embdPerLayer .. (layerIdx+1) * embdPerLayer]
    without copying.

    @param size embdPerLayer
    @param plTotalSize Total size of the per_layer_input buffer (embdPerLayer * numLayers)
    @param plOffset Starting offset (layerIdx * embdPerLayer)
-/
def geluGateMulSliceKernel (size plTotalSize plOffset : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) size)
  let _perLayerInput ← ShaderM.declareInputBuffer "per_layer_input" (.array (.scalar .f32) plTotalSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let gateVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "gate" idx
    let plIdx := Exp.add idx (Exp.litU32 plOffset)
    let plVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := plTotalSize) "per_layer_input" plIdx

    -- GELU approximation
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let x3 := Exp.mul (Exp.mul gateVal gateVal) gateVal
    let inner := Exp.mul sqrt2OverPi (Exp.add gateVal (Exp.mul (Exp.litF32 0.044715) x3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) gateVal) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))

    -- output = GELU(gate) * per_layer_input[plOffset + idx]
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul gelu plVal)
  ) (pure ())

/-! ## Per-Layer Input Pre-Processing Kernels -/

/-- Scale kernel: y = x * scaleFactor
    Used for: tok_embd_per_layer * sqrt(embdPerLayer)
              and: inp_per_layer * (1/sqrt(2))
-/
def scaleKernel (size : Nat) (scaleFactor : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 scaleFactor))
  ) (pure ())

/-- Layer output scale kernel: y = x * layer_scale
    Applied when model.layers[il].out_scale is present.
    out_scale is a scalar stored as a 1-element tensor.
-/
def layerScaleKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val s)
  ) (pure ())

/-! ## Slice Extraction Kernel -/

/-- Extract a 2D slice from a 3D tensor along the last dimension.
    Used to get per-layer embeddings: inp_per_layer[:, :, layerIdx]
    Input shape: [embdPerLayer, numTokens, numLayers] (after permutation)
    Output shape: [embdPerLayer, numTokens]

    For single-token inference: numTokens=1, so output is [embdPerLayer]

    @param embdPerLayer First dimension
    @param numTokens Second dimension (typically 1 for decode)
    @param layerIdx Which layer to extract
-/
def sliceLayerKernel (embdPerLayer numTokens layerIdx : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalOutput := embdPerLayer * numTokens
  let numLayers := layerIdx + 1  -- only need to know stride, use layerIdx as offset

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (embdPerLayer * numTokens * (layerIdx + 1)))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutput)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalOutput)) (do
    -- For permuted shape [embdPerLayer, numTokens, numLayers]:
    -- slice at layerIdx = input[idx + layerIdx * embdPerLayer * numTokens]
    let srcIdx := Exp.add idx (Exp.litU32 (layerIdx * embdPerLayer * numTokens))
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := embdPerLayer * numTokens * numLayers) "input" srcIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx val
  ) (pure ())

end Hesper.Layers.PerLayerEmbedding
