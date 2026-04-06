import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Kernel Fusion Framework

Compose multiple ShaderM operations into a single GPU dispatch.

## Key Insight

ShaderM is a monad that generates WGSL code. When two ShaderM
computations write to / read from the same buffer, fusing them
eliminates the intermediate buffer and reduces dispatch count.

## Fusion Types

1. **Element-wise chain**: op1 writes out[i], op2 reads out[i] → inline
2. **Multi-copy**: N independent copies → 1 kernel with N read/writes
3. **Sequential with shared memory**: reduction → element-wise
-/

namespace Hesper.WGSL.Fusion

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Multi-Buffer Copy (fused save activations) -/

/-- Fused copy of up to 4 buffers in a single dispatch.
    Each (src, dst) pair is copied element-wise.
    All copies must have the same element count. -/
def fusedCopy4Kernel (numElements : Nat) (numPairs : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
    if numPairs >= 1 then do
      let _s0 ← ShaderM.declareInputBuffer "src0" (.array (.scalar .f32) numElements)
      let _d0 ← ShaderM.declareOutputBuffer "dst0" (.array (.scalar .f32) numElements)
      let v0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "src0" i
      ShaderM.writeBuffer (ty := .scalar .f32) "dst0" i v0
    if numPairs >= 2 then do
      let _s1 ← ShaderM.declareInputBuffer "src1" (.array (.scalar .f32) numElements)
      let _d1 ← ShaderM.declareOutputBuffer "dst1" (.array (.scalar .f32) numElements)
      let v1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "src1" i
      ShaderM.writeBuffer (ty := .scalar .f32) "dst1" i v1
    if numPairs >= 3 then do
      let _s2 ← ShaderM.declareInputBuffer "src2" (.array (.scalar .f32) numElements)
      let _d2 ← ShaderM.declareOutputBuffer "dst2" (.array (.scalar .f32) numElements)
      let v2 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "src2" i
      ShaderM.writeBuffer (ty := .scalar .f32) "dst2" i v2
    if numPairs >= 4 then do
      let _s3 ← ShaderM.declareInputBuffer "src3" (.array (.scalar .f32) numElements)
      let _d3 ← ShaderM.declareOutputBuffer "dst3" (.array (.scalar .f32) numElements)
      let v3 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "src3" i
      ShaderM.writeBuffer (ty := .scalar .f32) "dst3" i v3
  ) (pure ())

/-- Execute fused copy of up to 4 buffer pairs of the same size -/
def executeFusedCopy (device : Device) (pairs : Array (Buffer × Buffer))
    (numElements : Nat) : IO Unit := do
  if pairs.isEmpty then return
  let numPairs := min pairs.size 4
  let shader := fusedCopy4Kernel numElements numPairs
  let mut namedBuffers : List (String × Buffer) := []
  for i in [:numPairs] do
    if h : i < pairs.size then
    let (src, dst) := pairs[i]
    namedBuffers := namedBuffers ++ [(s!"src{i}", src), (s!"dst{i}", dst)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Fused save of attention activations (normed + attnOut = 2 pairs of dim size)
    + attention weights (1 pair of attnSize) in 2 dispatches instead of 3 -/
def fusedSaveAttentionActivations (device : Device)
    (normedBuf savedNormed attnOutBuf savedAttnOut : Buffer) (dim : Nat)
    (attnBuf savedAttn : Buffer) (attnSize : Nat) : IO Unit := do
  -- Pair 1+2: dim-sized buffers (normed + attnOut)
  executeFusedCopy device #[(normedBuf, savedNormed), (attnOutBuf, savedAttnOut)] dim
  -- Pair 3: attn-sized buffer (different size, separate dispatch)
  executeFusedCopy device #[(attnBuf, savedAttn)] attnSize

/-- Fused save of FFN activations (gate + up + hidden = 3 pairs of ffnDim)
    + residual1 (1 pair of dim) in 2 dispatches instead of 4 -/
def fusedSaveFFNActivations (device : Device)
    (gateBuf savedGate upBuf savedUp hiddenBuf savedHidden : Buffer) (ffnDim : Nat)
    (residual1Buf savedResidual1 : Buffer) (dim : Nat) : IO Unit := do
  -- 3 pairs of ffnDim-sized buffers
  executeFusedCopy device #[(gateBuf, savedGate), (upBuf, savedUp), (hiddenBuf, savedHidden)] ffnDim
  -- 1 pair of dim-sized buffer
  executeFusedCopy device #[(residual1Buf, savedResidual1)] dim

end Hesper.WGSL.Fusion
