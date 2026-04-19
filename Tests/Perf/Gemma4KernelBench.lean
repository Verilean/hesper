import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# Gemma 4 per-kernel microbench

Measures μs/call of each Linear kernel used in hesper Gemma 4 decode.
Target: identify the biggest delta vs llama.cpp Vulkan kernel times
(documented in docs/#92), then decide which kernel to rewrite first.

Pattern (borrowed from Tests/CUDA/CUDAMatmulMicrobench.lean):
  1. Load the actual Gemma 4 weight tensor from the GGUF.
  2. Warmup 20 iters (prime PTX cache, input Q8_1 prep, L2).
  3. Timed loop of `iters=500` `Linear.forwardDP4A` calls.
  4. Final 4-byte DtoH read as a sync fence.
  5. `(t1 - t0) / iters` → μs/call.

No correctness check here (that's in Tests/GoldenUnit).  Output/input
buffers are uninitialised f32; matmul correctness is irrelevant.

Usage:
    lake exe gemma4-kernel-bench
-/

open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

/-- Run `Linear.forwardDP4A` iters times on the same (layer, input, output),
    timed via CPU clock + GPU sync fence.  Returns μs/call. -/
unsafe def benchForwardDP4A
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (inBuf outBuf : GPUBackend.Buf CUDAContext) (outDim : Nat) (iters : Nat) : IO Float := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  -- Warmup: prime the PTX cache, Q8_1 scratch, L2.
  for _ in [0:20] do
    Linear.forwardDP4A ctx layer inBuf outBuf
  -- Sync fence (small DtoH read on the output buffer).
  let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let _ := outDim
  let t0 ← IO.monoNanosNow
  for _ in [0:iters] do
    Linear.forwardDP4A ctx layer inBuf outBuf
  -- Final fence.
  let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let t1 ← IO.monoNanosNow
  return (t1 - t0).toFloat / (iters.toFloat * 1000.0)

/-- Same as `benchForwardDP4A` but for `forwardBatchDP4A` with seqLen=5. -/
unsafe def benchForwardBatchDP4A
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (inBuf outBuf : GPUBackend.Buf CUDAContext) (outDim seqLen iters : Nat) : IO Float := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  let _ := outDim
  for _ in [0:20] do
    Linear.forwardBatchDP4A ctx layer inBuf outBuf seqLen
  let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let t0 ← IO.monoNanosNow
  for _ in [0:iters] do
    Linear.forwardBatchDP4A ctx layer inBuf outBuf seqLen
  let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let t1 ← IO.monoNanosNow
  return (t1 - t0).toFloat / (iters.toFloat * 1000.0)

/-- Bench one (name, tensor, inDim, outDim) row.  Prints μs/call, weight
    bytes touched, and achieved GB/s (useful for BW-bound kernels). -/
unsafe def benchLinear (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (name tensorName : String) (inDim outDim : Nat) (iters : Nat := 500) : IO Unit := do
  withLinearLayer ctx gguf tensorName inDim outDim fun layer => do
    -- Input & output scratch buffers (uninit is fine — we're not checking values).
    withTempBuf ctx (inDim * 4) fun inBuf => do
      withTempBuf ctx (outDim * 4) fun outBuf => do
        let us ← benchForwardDP4A ctx layer inBuf outBuf outDim iters
        -- Weight bytes touched per call.
        -- Q4_K: 0.5 bytes/element + 4-byte headers per 256 block → ≈ 0.578 B/el
        -- Q6_K: 0.75 bytes/element + headers
        let quant := match layer.quantFormat with
          | .Q4_K => "Q4_K"
          | .Q6_K => "Q6_K"
        let bytesPerEl : Float := match layer.quantFormat with
          | .Q6_K => 0.8125
          | _     => 0.578125
        let wBytes : Float := (inDim * outDim).toFloat * bytesPerEl
        let gbs : Float := wBytes / (us * 1e-6) / 1e9
        IO.println s!"  {name} [{quant} {inDim}×{outDim}]: {us} μs/call, {gbs} GB/s (weight)"

unsafe def main : IO Unit := do
  IO.println "[Init] CUDA + GGUF..."
  let ctx ← CUDAContext.init
  let gguf ← loadGGUF
  IO.println "[Init] done.\n"
  -- Enable dp4a.
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true

  IO.println "=== Gemma 4 E4B per-kernel microbench (decode single-token) ==="
  IO.println s!"GPU: RTX 4070 Ti; model: Q4_K_M weights + Q6_K lm_head"
  IO.println s!"Pattern: 20 warmup iters + 500 timed calls, cold L2 not enforced (warm).\n"

  -- SWA attention layer (L0): qDim=2048, kvDim=512
  IO.println "-- L0 (SWA, headDim=256, numHeads=8, numKVHeads=2) --"
  benchLinear ctx gguf "wQ_L0"        "blk.0.attn_q.weight"       2560 2048
  benchLinear ctx gguf "wK_L0"        "blk.0.attn_k.weight"       2560 512
  benchLinear ctx gguf "wV_L0"        "blk.0.attn_v.weight"       2560 512
  benchLinear ctx gguf "wO_L0"        "blk.0.attn_output.weight"  2048 2560
  benchLinear ctx gguf "ffn_gate_L0"  "blk.0.ffn_gate.weight"     2560 10240
  benchLinear ctx gguf "ffn_up_L0"    "blk.0.ffn_up.weight"       2560 10240
  benchLinear ctx gguf "ffn_down_L0"  "blk.0.ffn_down.weight"     10240 2560

  -- Full-attention layer (L17): qDim=4096, kvDim=1024
  IO.println "\n-- L17 (full-attn, headDim=512, numHeads=8, numKVHeads=2) --"
  benchLinear ctx gguf "wQ_L17"       "blk.17.attn_q.weight"      2560 4096
  benchLinear ctx gguf "wK_L17"       "blk.17.attn_k.weight"      2560 1024
  benchLinear ctx gguf "wV_L17"       "blk.17.attn_v.weight"      2560 1024
  benchLinear ctx gguf "wO_L17"       "blk.17.attn_output.weight" 4096 2560
  benchLinear ctx gguf "ffn_gate_L17" "blk.17.ffn_gate.weight"    2560 10240
  benchLinear ctx gguf "ffn_up_L17"   "blk.17.ffn_up.weight"      2560 10240
  benchLinear ctx gguf "ffn_down_L17" "blk.17.ffn_down.weight"    10240 2560

  -- LM head (Q6_K, largest single kernel in decode)
  IO.println "\n-- LM head --"
  benchLinear ctx gguf "lm_head"      "token_embd.weight"         2560 262144 (iters := 200)

  -- Fused gate+up — the actual decode path for FFN.
  IO.println "\n-- Fused gate+up (the real decode path for FFN) --"
  withLinearLayer ctx gguf "blk.0.ffn_gate.weight" 2560 10240 fun gateL => do
    withLinearLayer ctx gguf "blk.0.ffn_up.weight" 2560 10240 fun upL => do
      withTempBuf ctx (2560 * 4) fun inBuf => do
        withTempBuf ctx (10240 * 4) fun outBuf => do
          let preparedRef ← GPUBackend.newCacheRef (β := CUDAContext)
          Linear.dp4aEnabled.set true
          -- Warmup.
          for _ in [0:20] do
            Linear.forwardFusedGateUp ctx gateL upL inBuf outBuf preparedRef
          let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
          let iters := 500
          let t0 ← IO.monoNanosNow
          for _ in [0:iters] do
            Linear.forwardFusedGateUp ctx gateL upL inBuf outBuf preparedRef
          let _ ← GPUBackend.readBuffer ctx outBuf (4 : USize)
          let t1 ← IO.monoNanosNow
          let us : Float := (t1 - t0).toFloat / (iters.toFloat * 1000.0)
          -- Each call reads gate + up weight buffers (2 × Q4_K 2560×10240).
          let wBytes : Float := 2.0 * (2560 * 10240).toFloat * 0.578125
          let gbs : Float := wBytes / (us * 1e-6) / 1e9
          IO.println s!"  forwardFusedGateUp L0 [2560→2×10240]: {us} μs/call, {gbs} GB/s (both weights)"

  IO.println "\n[Done]"
