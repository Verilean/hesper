import Hesper.Tensor.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

/-!
# Convolutional Neural Network Operations

GPU kernels for convolution and pooling operations using ShaderM monad.
Supports 2D convolution, depthwise convolution, and pooling layers.
-/

namespace Hesper.NN.Conv

open Hesper.Tensor
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

/-- Configuration for 2D convolution -/
structure Conv2DConfig where
  /-- Input dimensions: [batch, height, width, channels] -/
  batch : Nat
  inputHeight : Nat
  inputWidth : Nat
  inputChannels : Nat
  /-- Kernel dimensions -/
  kernelHeight : Nat
  kernelWidth : Nat
  /-- Output channels -/
  outputChannels : Nat
  /-- Stride -/
  stride : Nat := 1
  /-- Padding -/
  padding : Nat := 0
  /-- Workgroup size -/
  workgroupSize : Nat := 16
  deriving Inhabited, Repr

namespace Conv2DConfig

  /-- Output height after convolution -/
  def outputHeight (c : Conv2DConfig) : Nat :=
    (c.inputHeight + 2 * c.padding - c.kernelHeight) / c.stride + 1

  /-- Output width after convolution -/
  def outputWidth (c : Conv2DConfig) : Nat :=
    (c.inputWidth + 2 * c.padding - c.kernelWidth) / c.stride + 1

  /-- Number of workgroups -/
  def numWorkgroups (c : Conv2DConfig) : Nat × Nat × Nat :=
    let oh := c.outputHeight
    let ow := c.outputWidth
    let wgX := (ow + c.workgroupSize - 1) / c.workgroupSize
    let wgY := (oh + c.workgroupSize - 1) / c.workgroupSize
    (wgX, wgY, c.outputChannels)

end Conv2DConfig


/-- Configuration for pooling operations -/
structure PoolingConfig where
  /-- Input dimensions: [batch, height, width, channels] -/
  batch : Nat
  inputHeight : Nat
  inputWidth : Nat
  channels : Nat
  /-- Pool dimensions -/
  poolHeight : Nat
  poolWidth : Nat
  /-- Stride -/
  stride : Nat := 2
  /-- Workgroup size -/
  workgroupSize : Nat := 16
  deriving Inhabited, Repr

namespace PoolingConfig

  /-- Output height after pooling -/
  def outputHeight (c : PoolingConfig) : Nat :=
    (c.inputHeight - c.poolHeight) / c.stride + 1

  /-- Output width after pooling -/
  def outputWidth (c : PoolingConfig) : Nat :=
    (c.inputWidth - c.poolWidth) / c.stride + 1

  /-- Number of workgroups -/
  def numWorkgroups (c : PoolingConfig) : Nat × Nat × Nat :=
    let oh := c.outputHeight
    let ow := c.outputWidth
    let wgX := (ow + c.workgroupSize - 1) / c.workgroupSize
    let wgY := (oh + c.workgroupSize - 1) / c.workgroupSize
    (wgX, wgY, c.channels)

end PoolingConfig

/-- Max pooling kernel using ShaderM monad -/
def maxPoolingKernel (config : PoolingConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let outY := Exp.vec3Y gid
  let outX := Exp.vec3X gid
  let ch := Exp.vecZ gid

  let outputHeight := litU config.outputHeight
  let outputWidth := litU config.outputWidth
  let channels := litU config.channels

  -- Nested bounds check
  if_ (Exp.lt outY outputHeight) (do
    if_ (Exp.lt outX outputWidth) (do
      if_ (Exp.lt ch channels) (do
        -- Initialize max value
        let maxVal ← var (.scalar .f32) (litF (-1e38))

        -- Loop over pool window
        loop (litU 0) (litU config.poolHeight) (litU 1) fun py => do
          loop (litU 0) (litU config.poolWidth) (litU 1) fun px => do
            -- Calculate input position: output_pos * stride + pool_offset
            let inY := Exp.add (Exp.mul outY (litU config.stride)) py
            let inX := Exp.add (Exp.mul outX (litU config.stride)) px

            if_ (Exp.lt inY (litU config.inputHeight)) (do
              if_ (Exp.lt inX (litU config.inputWidth)) (do
                -- NHWC indexing: (y * width + x) * channels + c
                let inIdx := Exp.add
                  (Exp.mul (Exp.add (Exp.mul inY (litU config.inputWidth)) inX) (litU config.channels))
                  ch
                let val ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf inIdx
                assign maxVal (Exp.max (Exp.var maxVal) val)
              ) (pure ())
            ) (pure ())

        -- Output index in NHWC format
        let outIdx := Exp.add
          (Exp.mul (Exp.add (Exp.mul outY (litU config.outputWidth)) outX) (litU config.channels))
          ch
        writeBuffer (ty := .scalar .f32) outputBuf outIdx (Exp.var maxVal)
      ) (pure ())
    ) (pure ())
  ) (pure ())

/-- Average pooling kernel using ShaderM monad -/
def avgPoolingKernel (config : PoolingConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let outY := Exp.vec3Y gid
  let outX := Exp.vec3X gid
  let ch := Exp.vecZ gid

  let outputHeight := litU config.outputHeight
  let outputWidth := litU config.outputWidth
  let channels := litU config.channels

  -- Nested bounds check
  if_ (Exp.lt outY outputHeight) (do
    if_ (Exp.lt outX outputWidth) (do
      if_ (Exp.lt ch channels) (do
        let sum ← var (.scalar .f32) (litF 0.0)

        loop (litU 0) (litU config.poolHeight) (litU 1) fun py => do
          loop (litU 0) (litU config.poolWidth) (litU 1) fun px => do
            -- Calculate input position for average pooling
            let inY := Exp.add (Exp.mul outY (litU config.stride)) py
            let inX := Exp.add (Exp.mul outX (litU config.stride)) px

            if_ (Exp.lt inY (litU config.inputHeight)) (do
              if_ (Exp.lt inX (litU config.inputWidth)) (do
                let inIdx := Exp.add
                  (Exp.mul (Exp.add (Exp.mul inY (litU config.inputWidth)) inX) (litU config.channels))
                  ch
                let val ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf inIdx
                assign sum (Exp.add (Exp.var sum) val)
              ) (pure ())
            ) (pure ())

        -- Average pooling: divide sum by pool window size
        let poolSize := litF (config.poolHeight * config.poolWidth).toFloat
        let outIdx := Exp.add
          (Exp.mul (Exp.add (Exp.mul outY (litU config.outputWidth)) outX) (litU config.channels))
          ch
        writeBuffer (ty := .scalar .f32) outputBuf outIdx (Exp.div (Exp.var sum) poolSize)
      ) (pure ())
    ) (pure ())
  ) (pure ())

/-- 2D convolution kernel using ShaderM monad (NHWC format) -/
def conv2DKernel (config : Conv2DConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let kernelBuf ← declareInputBuffer "kernel" (.scalar .f32)
  let biasBuf ← declareInputBuffer "bias" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let outY := Exp.vec3Y gid
  let outX := Exp.vec3X gid
  let outC := Exp.vecZ gid

  let outputHeight := litU config.outputHeight
  let outputWidth := litU config.outputWidth
  let outputChannels := litU config.outputChannels

  -- Nested bounds check
  if_ (Exp.lt outY outputHeight) (do
    if_ (Exp.lt outX outputWidth) (do
      if_ (Exp.lt outC outputChannels) (do
        -- Initialize sum with bias
        let biasVal ← readBuffer (ty := .scalar .f32) (n := 1024) biasBuf outC
        let sum ← var (.scalar .f32) biasVal

        -- Triple nested loop: kernel height, kernel width, input channels
        loop (litU 0) (litU config.kernelHeight) (litU 1) fun ky => do
          loop (litU 0) (litU config.kernelWidth) (litU 1) fun kx => do
            -- Calculate input position: output_pos * stride + kernel_offset
            let inY := Exp.add (Exp.mul outY (litU config.stride)) ky
            let inX := Exp.add (Exp.mul outX (litU config.stride)) kx

            -- Simple bounds check (TODO: add padding support)
            if_ (Exp.lt inY (litU config.inputHeight)) (do
              if_ (Exp.lt inX (litU config.inputWidth)) (do
                -- Loop over input channels
                loop (litU 0) (litU config.inputChannels) (litU 1) fun inC => do
                  -- Input index: [batch=0, y, x, c] in NHWC format
                  let inIdx := Exp.add
                    (Exp.mul (Exp.add (Exp.mul inY (litU config.inputWidth)) inX) (litU config.inputChannels))
                    inC

                  -- Kernel index: [ky, kx, in_c, out_c]
                  let kIdx := Exp.add
                    (Exp.mul
                      (Exp.add
                        (Exp.mul (Exp.add (Exp.mul ky (litU config.kernelWidth)) kx) (litU config.inputChannels))
                        inC)
                      (litU config.outputChannels))
                    outC

                  let inputVal ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf inIdx
                  let kernelVal ← readBuffer (ty := .scalar .f32) (n := 1024) kernelBuf kIdx
                  -- Accumulate: sum += input * kernel
                  assign sum (Exp.add (Exp.var sum) (Exp.mul inputVal kernelVal))
              ) (pure ())
            ) (pure ())

        -- Output index: [batch=0, out_y, out_x, out_c] in NHWC format
        let outIdx := Exp.add
          (Exp.mul (Exp.add (Exp.mul outY outputWidth) outX) outputChannels)
          outC
        writeBuffer (ty := .scalar .f32) outputBuf outIdx (Exp.var sum)
      ) (pure ())
    ) (pure ())
  ) (pure ())

/-- Depthwise convolution kernel using ShaderM monad (for MobileNets) -/
def depthwiseConv2DKernel (config : Conv2DConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let kernelBuf ← declareInputBuffer "kernel" (.scalar .f32)
  let biasBuf ← declareInputBuffer "bias" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let outY := Exp.vec3Y gid
  let outX := Exp.vec3X gid
  let ch := Exp.vecZ gid

  let outputHeight := litU config.outputHeight
  let outputWidth := litU config.outputWidth
  let channels := litU config.inputChannels  -- Same as output for depthwise

  -- Nested bounds check (avoiding .&&. which may not be defined)
  if_ (Exp.lt outY outputHeight) (do
    if_ (Exp.lt outX outputWidth) (do
      if_ (Exp.lt ch channels) (do
        -- Initialize sum with bias
        let biasVal ← readBuffer (ty := .scalar .f32) (n := 1024) biasBuf ch
        let sum ← var (.scalar .f32) biasVal

        -- Depthwise convolution: one kernel per input channel (MobileNets)
        loop (litU 0) (litU config.kernelHeight) (litU 1) fun ky => do
          loop (litU 0) (litU config.kernelWidth) (litU 1) fun kx => do
            let inY := Exp.add (Exp.mul outY (litU config.stride)) ky
            let inX := Exp.add (Exp.mul outX (litU config.stride)) kx

            -- Simple bounds check (TODO: add padding support)
            if_ (Exp.lt inY (litU config.inputHeight)) (do
              if_ (Exp.lt inX (litU config.inputWidth)) (do
                -- Input index: [batch=0, y, x, ch] in NHWC format
                let inIdx := Exp.add
                  (Exp.mul (Exp.add (Exp.mul inY (litU config.inputWidth)) inX) channels)
                  ch

                -- Kernel index: [ky, kx, ch] (one kernel per channel)
                let kIdx := Exp.add
                  (Exp.mul (Exp.add (Exp.mul ky (litU config.kernelWidth)) kx) channels)
                  ch

                let inputVal ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf inIdx
                let kernelVal ← readBuffer (ty := .scalar .f32) (n := 1024) kernelBuf kIdx
                assign sum (Exp.add (Exp.var sum) (Exp.mul inputVal kernelVal))
              ) (pure ())
            ) (pure ())

        -- Output index: [batch=0, out_y, out_x, ch] in NHWC format
        let outIdx := Exp.add
          (Exp.mul (Exp.add (Exp.mul outY outputWidth) outX) channels)
          ch
        writeBuffer (ty := .scalar .f32) outputBuf outIdx (Exp.var sum)
      ) (pure ())
    ) (pure ())
  ) (pure ())

/-- Generate WGSL shader for 2D convolution -/
def generateConv2DShaderFromMonad (config : Conv2DConfig) : String :=
  generateWGSL "main"
    {x := config.workgroupSize, y := config.workgroupSize, z := 1}
    []
    []
    (conv2DKernel config)

/-- Generate WGSL shader for depthwise convolution -/
def generateDepthwiseConv2DShaderFromMonad (config : Conv2DConfig) : String :=
  generateWGSL "main"
    {x := config.workgroupSize, y := config.workgroupSize, z := 1}
    []
    []
    (depthwiseConv2DKernel config)

/-- Generate WGSL shader for max pooling -/
def generateMaxPoolingShaderFromMonad (config : PoolingConfig) : String :=
  generateWGSL "main"
    {x := config.workgroupSize, y := config.workgroupSize, z := 1}
    []
    []
    (maxPoolingKernel config)

/-- Generate WGSL shader for average pooling -/
def generateAvgPoolingShaderFromMonad (config : PoolingConfig) : String :=
  generateWGSL "main"
    {x := config.workgroupSize, y := config.workgroupSize, z := 1}
    []
    []
    (avgPoolingKernel config)


end Hesper.NN.Conv
