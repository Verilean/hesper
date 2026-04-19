import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# FFN (dense gate+up+down) golden-unit tests

Reference: `llama.cpp/src/llama-graph.cpp::build_ffn`
    gate_out = lora_mm(gate, ffn_norm)
    up_out   = lora_mm(up,   ffn_norm)
    geglu    = GELU(gate_out) * up_out         ← ffn_geglu
    ffn_out  = lora_mm(down, geglu)            ← ffn_out

GELU formula (llama.cpp CUDA, llama.cpp/ggml/src/ggml-cuda/unary.cuh:102):
    0.5*x*(1 + tanhf(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A*x*x)))
Hesper uses the algebraically identical form
    0.5*x*(1 + tanh(SQRT_2_OVER_PI * (x + 0.044715*x^3)))
in `geluMulKernel` (Hesper/Models/Gemma4.lean:160).

## Dumps
- ffn_norm-<li> : input to FFN   [hiddenDim=2560, seqLen]
- ffn_gate-<li> : gate matmul out [intermediateSize, seqLen]
- ffn_geglu-<li>: GELU(gate) * up [intermediateSize, seqLen]
- ffn_out-<li>  : down output     [hiddenDim, seqLen] (pre-post-ffn-norm,
                                                      pre-residual)

## Tests
- Test 1 (ffn_gate):   forwardBatchDP4A(wGate) vs ffn_gate    — pure matmul
- Test 2 (ffn_geglu):  gate + up + geluMul     vs ffn_geglu   — adds GELU+mul
- Test 3 (ffn_out):    full FFN                vs ffn_out     — adds down matmul

All tested at L0 (dense layer in Gemma 4 — L0..L5 are dense, L6+ are MoE).
-/

namespace Hesper.Tests.GoldenUnit.FFN

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

/-- Gemma 4 E4B intermediate size (ffn_gate/up outDim). -/
def gemma4IntermediateSize : Nat := 10240

/-- Run `Linear.forwardBatchDP4A` over seqLen tokens, return full batched
    f32 output bytes (no last-token slicing — caller compares full tensor). -/
unsafe def runBatchMatmulFull
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (inputBytes : ByteArray) (outDim seqLen : Nat) : IO ByteArray := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (outDim * seqLen * 4) fun outBuf => do
      Linear.forwardBatchDP4A ctx layer inBuf outBuf seqLen
      GPUBackend.readBuffer ctx outBuf (outDim * seqLen * 4).toUSize

/-- Test 1: ffn_gate matmul alone. Input is `ffn_norm-<li>`, target is
    `ffn_gate-<li>`.  Compares last-token. -/
unsafe def testFFNGateAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li seqLen : Nat) (threshold : Float) : IO TestSeq := do
  let inDim := gemma4HiddenDim
  let outDim := gemma4IntermediateSize
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/ffn_norm-{li}.bin"
  if inputBytes.size ≠ inDim * seqLen * 4 then
    throw (IO.userError s!"ffn_norm-{li}.bin size={inputBytes.size}, expected {inDim * seqLen * 4}")
  let expFull ← loadFloat32Bin s!"{goldenDir}/ffn_gate-{li}.bin"
  if expFull.size ≠ outDim * seqLen * 4 then
    throw (IO.userError s!"ffn_gate-{li}.bin size={expFull.size}, expected {outDim * seqLen * 4}")
  let expected := byteArrayToF32Array (lastTokenBytes expFull outDim) outDim
  let actualBytes ← withLinearLayer ctx gguf s!"blk.{li}.ffn_gate.weight" inDim outDim fun wGate =>
    runBatchMatmulFull ctx wGate inputBytes outDim seqLen
  let actual := byteArrayToF32Array (lastTokenBytes actualBytes outDim) outDim
  let rel := relDiff actual expected
  IO.println s!"[FFN L{li} gate matmul, outDim={outDim}] rel = {rel}"
  pure (test s!"hesper FFN.gate L{li} matches llama.cpp ffn_gate-{li} (rel={rel} < {threshold})" (rel < threshold))

/-- Test 2: ffn_geglu (= GELU(gate) * up). Compares last-token. -/
unsafe def testFFNGegluAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li seqLen : Nat) (threshold : Float) : IO TestSeq := do
  let inDim := gemma4HiddenDim
  let outDim := gemma4IntermediateSize
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/ffn_norm-{li}.bin"
  if inputBytes.size ≠ inDim * seqLen * 4 then
    throw (IO.userError s!"ffn_norm-{li}.bin size={inputBytes.size}, expected {inDim * seqLen * 4}")
  let expFull ← loadFloat32Bin s!"{goldenDir}/ffn_geglu-{li}.bin"
  if expFull.size ≠ outDim * seqLen * 4 then
    throw (IO.userError s!"ffn_geglu-{li}.bin size={expFull.size}, expected {outDim * seqLen * 4}")
  let expected := byteArrayToF32Array (lastTokenBytes expFull outDim) outDim
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  let actual ← withLinearLayer ctx gguf s!"blk.{li}.ffn_gate.weight" inDim outDim fun wGate => do
    withLinearLayer ctx gguf s!"blk.{li}.ffn_up.weight" inDim outDim fun wUp => do
      withTempBufFromBytes ctx inputBytes fun inBuf => do
        withTempBuf ctx (outDim * seqLen * 4) fun gateBuf => do
          withTempBuf ctx (outDim * seqLen * 4) fun upBuf => do
            withTempBuf ctx (outDim * seqLen * 4) fun gegluBuf => do
              Linear.forwardBatchDP4A ctx wGate inBuf gateBuf seqLen
              Linear.forwardBatchDP4A ctx wUp inBuf upBuf seqLen
              -- Apply geluMulKernel element-wise across all seqLen * outDim
              let totalSize := outDim * seqLen
              GPUBackend.execute ctx
                (Hesper.Models.Gemma4.geluMulKernel totalSize)
                [("gate", gateBuf), ("up", upBuf), ("output", gegluBuf)]
                (.dispatch1D totalSize)
              let outBytes ← GPUBackend.readBuffer ctx gegluBuf (totalSize * 4).toUSize
              pure (byteArrayToF32Array (lastTokenBytes outBytes outDim) outDim)
  let rel := relDiff actual expected
  IO.println s!"[FFN L{li} geglu (GELU(gate)*up), outDim={outDim}] rel = {rel}"
  pure (test s!"hesper FFN.geglu L{li} matches llama.cpp ffn_geglu-{li} (rel={rel} < {threshold})" (rel < threshold))

/-- Test 3: full FFN (gate + up + geluMul + down). Compares against
    `ffn_out-<li>` (pre-post-ffn-norm, pre-residual). -/
unsafe def testFFNOutAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li seqLen : Nat) (threshold : Float) : IO TestSeq := do
  let hiddenDim := gemma4HiddenDim
  let interSize := gemma4IntermediateSize
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/ffn_norm-{li}.bin"
  if inputBytes.size ≠ hiddenDim * seqLen * 4 then
    throw (IO.userError s!"ffn_norm-{li}.bin size={inputBytes.size}, expected {hiddenDim * seqLen * 4}")
  let expFull ← loadFloat32Bin s!"{goldenDir}/ffn_out-{li}.bin"
  if expFull.size ≠ hiddenDim * seqLen * 4 then
    throw (IO.userError s!"ffn_out-{li}.bin size={expFull.size}, expected {hiddenDim * seqLen * 4}")
  let expected := byteArrayToF32Array (lastTokenBytes expFull hiddenDim) hiddenDim
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  let actual ← withLinearLayer ctx gguf s!"blk.{li}.ffn_gate.weight" hiddenDim interSize fun wGate => do
    withLinearLayer ctx gguf s!"blk.{li}.ffn_up.weight" hiddenDim interSize fun wUp => do
      withLinearLayer ctx gguf s!"blk.{li}.ffn_down.weight" interSize hiddenDim fun wDown => do
        withTempBufFromBytes ctx inputBytes fun inBuf => do
          withTempBuf ctx (interSize * seqLen * 4) fun gateBuf => do
            withTempBuf ctx (interSize * seqLen * 4) fun upBuf => do
              withTempBuf ctx (interSize * seqLen * 4) fun gegluBuf => do
                withTempBuf ctx (hiddenDim * seqLen * 4) fun outBuf => do
                  Linear.forwardBatchDP4A ctx wGate inBuf gateBuf seqLen
                  Linear.forwardBatchDP4A ctx wUp inBuf upBuf seqLen
                  let totalSize := interSize * seqLen
                  GPUBackend.execute ctx
                    (Hesper.Models.Gemma4.geluMulKernel totalSize)
                    [("gate", gateBuf), ("up", upBuf), ("output", gegluBuf)]
                    (.dispatch1D totalSize)
                  Linear.forwardBatchDP4A ctx wDown gegluBuf outBuf seqLen
                  let outBytes ← GPUBackend.readBuffer ctx outBuf (hiddenDim * seqLen * 4).toUSize
                  pure (byteArrayToF32Array (lastTokenBytes outBytes hiddenDim) hiddenDim)
  let rel := relDiff actual expected
  IO.println s!"[FFN L{li} full FFN out, hiddenDim={hiddenDim}] rel = {rel}"
  pure (test s!"hesper FFN.full L{li} matches llama.cpp ffn_out-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0 is a dense FFN layer in Gemma 4 (MoE starts at some later layer).
  -- seqLen=5 for the "Hello world how are you" prompt.
  let gate0 ← testFFNGateAtLayer ctx gguf 0 5 1e-2
  let geglu0 ← testFFNGegluAtLayer ctx gguf 0 5 1e-2
  -- Threshold relaxed to 2e-2: ffn_out adds a second Q4_K matmul (down)
  -- on top of gate+up, roughly doubling the ~7e-3 noise floor.
  let out0 ← testFFNOutAtLayer ctx gguf 0 5 2e-2
  pure [
    ("FFN gate L0 matmul last-token", [gate0]),
    ("FFN geglu L0 last-token", [geglu0]),
    ("FFN full out L0 last-token", [out0])
  ]

end Hesper.Tests.GoldenUnit.FFN
