import Hesper.Circuit.IRv2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4_v2
import Hesper.Models.Gemma4Bridge
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
# Phase F3: Monolith TPS benchmark (scoped)

Measures the host-side cost of three paths running the **same single
transformer-block body** N times:

  A) **Production forwardBlock**  — 1× `Gemma4.forwardBlock(L5)` × N
  B) **Monolith eager**            — 1× 6-block BlockGraph × N via
                                     `runMonolithicGraph`
  C) **Monolith + CUDA capture**   — capture once, `cuGraphLaunch` × N

The Monolith currently only supports the all-Q4_K + freqFactors-bearing
variant of `GemmaAttentionMonolith`, which is layer 5's configuration.
So this benchmark fixes layer=5 and measures *how much host overhead the
IRv2 dispatcher adds vs production* and *how much CUDA Graph capture
removes* — both apples-to-apples.

Acceptance:
- **B/A ≈ 1.0**: IRv2 expansion is zero-cost (same kernels, same work)
- **C/A < 1.0**: Capture wins — one launch replaces ~14 launches per layer
-/

open Hesper
open Hesper.Models.Gemma4
open Hesper.Circuit.IRv2

def msSince (startNs : Nat) : IO Float := do
  let endNs ← IO.monoNanosNow
  return (endNs - startNs).toFloat / 1000000.0

def main : IO Unit := do
  IO.println "=== Phase F3: Monolith TPS benchmark (scoped to L5) ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  Hesper.Layers.Linear.dp4aEnabled.set true
  let ctx ← Hesper.CUDAContext.init
  IO.println s!"[Load] {modelPath}"
  let model ← Gemma4Model.fromGGUF ctx modelPath
  let cfg := model.config
  let state ← createInferenceState ctx cfg

  let hiddenSize := cfg.hiddenSize
  let li : Nat := 5       -- First all-Q4_K, freqFactors-bearing layer.
  let pos : Nat := 0
  let iters : Nat :=
    (match ← IO.getEnv "HESPER_F3_ITERS" with
     | some s => s.toNat?.getD 100
     | none   => 100)
  IO.println s!"[Config] layer={li} hiddenSize={hiddenSize} iters={iters}"

  let block ← match model.blocks[li]? with
    | some b => pure b
    | none   => throw (IO.userError s!"no block {li}")

  -- Seed input + paramsBuf.
  let inputArr : Array Float :=
    (List.range hiddenSize).toArray.map
      (fun i => Float.sin (i.toFloat * 0.011) * 0.3)
  let inputBytes ← Hesper.Basic.floatArrayToBytes inputArr
  GPUBackend.writeBuffer ctx state.buf2 inputBytes
  let packU32 (v : UInt32) : ByteArray := Id.run do
    let mut b : ByteArray := ByteArray.empty
    b := b.push (UInt8.ofNat (v.toNat % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 256) % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 65536) % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 16777216) % 256))
    return b
  let paramsBytes : ByteArray := (packU32 pos.toUInt32) ++ (packU32 (pos + 1).toUInt32)
  GPUBackend.writeBuffer ctx state.paramsBuf paramsBytes

  -- Build Monolith 1-layer graph for L5 via `forwardTokenLazyMonolith`.
  let firstInputId : Nat := 10_000
  let lastOutputId : Nat := 10_999
  let baseTensorId : Nat := 20_000
  let bundleBase : UInt64 := 0xCAFE0000
  let firstLayerKey : UInt64 := bundleBase + li.toUInt64
  let (attnBundles, ffnBundles) ←
    Hesper.Models.Gemma4.extractMonolithBundles (β := CUDAContext) model state bundleBase
  let (_, graph) := Hesper.Circuit.IRv2.runBuilder
    (Hesper.Models.Gemma4_v2.forwardTokenLazyMonolith
       1 firstInputId lastOutputId baseTensorId firstLayerKey pos)
  IO.println s!"[IRv2] graph: {graph.blocks.size} blocks"

  let v2Out ← GPUBackend.allocBuffer ctx (hiddenSize * 4).toUSize
  let externalBufs : List (Nat × GPUBackend.Buf CUDAContext) :=
    [(firstInputId, state.buf2),
     (baseTensorId + 0, state.qBuf),
     (baseTensorId + 1, state.kBuf),
     (baseTensorId + 2, state.vBuf),
     (baseTensorId + 3, state.attnOutBuf),
     (baseTensorId + 4, state.normedBuf),
     (baseTensorId + 5, state.attnResidualBuf),
     (baseTensorId + 6, state.ffnOutBuf),
     (lastOutputId,   v2Out)]

  -- --------------------------------------------------------------
  -- Warmup: each path once → trigger PTX JIT + cache population.
  -- --------------------------------------------------------------
  IO.println "[Warmup] Monolith eager ..."
  Hesper.Circuit.IRv2.runMonolithicGraph ctx graph externalBufs attnBundles ffnBundles

  IO.println "[Warmup] Production forwardBlock ..."
  GPUBackend.writeBuffer ctx state.buf2 inputBytes
  GPUBackend.writeBuffer ctx state.paramsBuf paramsBytes
  let prodOut ← GPUBackend.allocBuffer ctx (hiddenSize * 4).toUSize
  forwardBlock ctx block cfg state.buf2 prodOut state pos
    (kcr := none) (perLayerEmbd := none) (perLayerInput := none)
  Hesper.CUDA.cuStreamSynchronize (0 : USize)

  -- --------------------------------------------------------------
  -- Path A: Production forwardBlock × iters
  -- --------------------------------------------------------------
  IO.println ""
  IO.println "─── Path A: production forwardBlock ───"
  let startA ← IO.monoNanosNow
  for _ in [0:iters] do
    forwardBlock ctx block cfg state.buf2 prodOut state pos
      (kcr := none) (perLayerEmbd := none) (perLayerInput := none)
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let msA ← msSince startA
  let perIterA := msA / iters.toFloat
  IO.println s!"  total {msA} ms / {iters} iters = {perIterA} ms/layer"

  -- --------------------------------------------------------------
  -- Path B: Monolith eager × iters
  -- --------------------------------------------------------------
  IO.println ""
  IO.println "─── Path B: Monolith eager (runMonolithicGraph) ───"
  let startB ← IO.monoNanosNow
  for _ in [0:iters] do
    Hesper.Circuit.IRv2.runMonolithicGraph ctx graph externalBufs attnBundles ffnBundles
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let msB ← msSince startB
  let perIterB := msB / iters.toFloat
  IO.println s!"  total {msB} ms / {iters} iters = {perIterB} ms/layer"

  -- --------------------------------------------------------------
  -- Path C: Monolith + CUDA Graph capture × iters
  -- --------------------------------------------------------------
  IO.println ""
  IO.println "─── Path C: Monolith + CUDA Graph capture ───"
  let stream ← Hesper.CUDA.cuStreamCreate
  IO.println "[Capture] building graph ..."
  let startCap ← IO.monoNanosNow
  let exec ← Hesper.Circuit.IRv2.captureMonolithicGraph ctx stream graph externalBufs
               attnBundles ffnBundles
  let msCap ← msSince startCap
  IO.println s!"[Capture] instantiated in {msCap} ms (one-time cost)"

  let startC ← IO.monoNanosNow
  for _ in [0:iters] do
    Hesper.CUDA.cuGraphLaunch exec stream
  Hesper.CUDA.cuStreamSynchronize stream
  let msC ← msSince startC
  let perIterC := msC / iters.toFloat
  IO.println s!"  total {msC} ms / {iters} iters = {perIterC} ms/layer"

  Hesper.CUDA.cuStreamDestroy stream

  -- --------------------------------------------------------------
  -- Summary.  Extrapolate per-token by ×42 (simulating full model body).
  -- --------------------------------------------------------------
  IO.println ""
  IO.println "══════════════════ Summary ══════════════════"
  IO.println s!"  Path A (production forwardBlock)    : {perIterA} ms/layer"
  IO.println s!"  Path B (Monolith eager)             : {perIterB} ms/layer"
  IO.println s!"  Path C (Monolith + CUDA capture)    : {perIterC} ms/layer"
  IO.println ""
  let tokA := perIterA * 42.0
  let tokB := perIterB * 42.0
  let tokC := perIterC * 42.0
  IO.println s!"  Extrapolated to 42 layers (per token):"
  IO.println s!"    A: {tokA} ms/tok ≈ {1000.0 / tokA} TPS"
  IO.println s!"    B: {tokB} ms/tok ≈ {1000.0 / tokB} TPS"
  IO.println s!"    C: {tokC} ms/tok ≈ {1000.0 / tokC} TPS"
  IO.println ""
  let ratioBA := perIterB / perIterA
  let ratioCA := perIterC / perIterA
  IO.println s!"  B/A = {ratioBA}x  (want ≈ 1.0 — IRv2 expansion is free)"
  IO.println s!"  C/A = {ratioCA}x  (want < 1.0 — capture removes launch overhead)"
  IO.println ""
  if ratioBA < 1.1 ∧ ratioBA > 0.9 then
    IO.println "✓ Path B matches Path A within ±10% — Monolith dispatcher is zero-overhead"
  else if ratioBA > 1.1 then
    IO.println s!"⚠ Path B is {100.0 * (ratioBA - 1.0)}% slower than A — investigate dispatcher overhead"
  else
    IO.println s!"?  Path B is {100.0 * (1.0 - ratioBA)}% FASTER than A — unexpected, double-check kernel selection"
  if ratioCA < 0.9 then
    IO.println s!"✓ Path C beats Path A by {100.0 * (1.0 - ratioCA)}% — CUDA Graph capture wins"
  else if ratioCA > 1.1 then
    IO.println s!"⚠ Path C is {100.0 * (ratioCA - 1.0)}% slower than A — replay path has sync points"
  else
    IO.println "= Path C within ±10% of A — capture is neutral; launch overhead dominated by something else"
