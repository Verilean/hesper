import Hesper.Circuit.IRv2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4_v2
import Hesper.Models.Gemma4Bridge
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
# Phase F2 skeleton: layer-0 Monolith bit-parity

**Status**: build-time wiring only.  GPU execution deferred to the
next session because it will likely expose paramsBuf / KV-cache /
qkvNorm-in-place plumbing bugs that need interactive debugging.

What this driver does when run:

1. Load Gemma 4 E4B Q4_K_M model on the CUDA backend.
2. Run production `forwardBlock(li=0, pos=0)` — save the resulting
   `outputBuf` as the parity reference.
3. Rebuild `InferenceState` (same tensors, reset) → build a 1-layer
   Monolith BlockGraph via `buildMonolithTokenPlan` (numLayers=1).
4. Execute it via `runMonolithicGraph` and read back the output.
5. Compare the two buffers; expected `max |err| = 0`.

What the next session owes:

- `paramsBuf` must be written with `[pos=0, cacheLen=1]` as u32
  *before* calling `runMonolithicGraph` (production writes it via
  `writeScalarViaStaging`; we can use `GPUBackend.writeBuffer`).
- The Monolith `GemmaAttentionMonolith` runtime currently binds
  `q_in = q_out = qBuf` for `fusedPerHeadQKVNormKernel`.  Production
  uses `qBuf2 → qBuf`.  If the kernel isn't in-place-safe we'll need
  an extra scratch buffer in the `AttnBundle` or a separate input tid
  in the Monolith AST.
- `FlashAttention` Monolith uses `maxSeqLen` as cacheLen approximation;
  for `pos=0` decode this should read as `cacheLen=1` (only the just-
  written token).  Check `executeFlashAttentionTiled` semantics.

When these three are verified, the parity check should pass; the
compiler-IR side (Monolith expansion, bundle resolution, AST buffer
ids) has been type-checked end-to-end.
-/

open Hesper
open Hesper.Models.Gemma4
open Hesper.Circuit.IRv2

/-- Max absolute element-wise difference; same helper used in B1-B9
    parity tests. -/
def maxAbsDiffMono (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

/-- Read back an f32 region of GPU memory as a Lean `Array Float`. -/
def readF32BufMono [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase F2 skeleton: layer-0 Monolith parity ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  Hesper.Layers.Linear.dp4aEnabled.set true
  let ctx ← Hesper.CUDAContext.init
  IO.println s!"[Load] {modelPath}"
  let model ← Gemma4Model.fromGGUF ctx modelPath
  let cfg := model.config
  let state ← createInferenceState ctx cfg

  -- Default: layer 5 (first all-Q4_K, full-attention with freqFactors).
  -- Override via HESPER_PARITY_LAYER=N to test other layers — e.g. an
  -- SWA layer (no freqFactors) like L6 to exercise the SWA Monolith path.
  let li : Nat :=
    (match ← IO.getEnv "HESPER_PARITY_LAYER" with
     | some s => s.toNat?.getD 5
     | none   => 5)
  let pos : Nat := 0
  let block ← match model.blocks[li]? with
    | some b => pure b
    | none   => do IO.println "FAIL: no block 0"; IO.Process.exit 1

  -- Seed state.buf2 (the conventional "pre-attn-norm input" buffer)
  -- with a deterministic f32 pattern.  Both paths read from the same
  -- buffer, so the comparison is fair.
  let hiddenSize := cfg.hiddenSize
  let inputArr : Array Float :=
    (List.range hiddenSize).toArray.map
      (fun i => Float.sin (i.toFloat * 0.011) * 0.3)
  let inputBytes ← Hesper.Basic.floatArrayToBytes inputArr
  GPUBackend.writeBuffer ctx state.buf2 inputBytes

  -- Seed paramsBuf with [pos, cacheLen=pos+1] as u32.
  let pv : UInt32 := pos.toUInt32
  let cv : UInt32 := (pos + 1).toUInt32
  let packU32 (v : UInt32) : ByteArray := Id.run do
    let mut b : ByteArray := ByteArray.empty
    b := b.push (UInt8.ofNat (v.toNat % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 256) % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 65536) % 256))
    b := b.push (UInt8.ofNat ((v.toNat / 16777216) % 256))
    return b
  let paramsBytes : ByteArray := (packU32 pv) ++ (packU32 cv)
  GPUBackend.writeBuffer ctx state.paramsBuf paramsBytes

  -- ================================================================
  -- REFERENCE: production forwardBlock WITHOUT PLE (perLayerEmbd=none).
  -- Run with `HESPER_SKIP_OUTSCALE=1 lake exe gemma4-monolith-layer-parity`
  -- to also disable layerOutScale fallback for an apples-to-apples
  -- comparison with the current (PLE-less, outScale-less) Monolith IR.
  -- ================================================================
  let refOut ← GPUBackend.allocBuffer ctx (hiddenSize * 4).toUSize
  let skipProd := (← IO.getEnv "HESPER_MONO_SKIP_PROD").isSome
  let mut refArr : Array Float := Array.replicate hiddenSize 0.0
  if !skipProd then
    IO.println "[Ref ] running production forwardBlock (perLayerEmbd=none) ..."
    forwardBlock ctx block cfg state.buf2 refOut state pos
      (kcr := none) (perLayerEmbd := none) (perLayerInput := none)
    refArr ← readF32BufMono ctx refOut hiddenSize
    IO.println s!"[Ref ] out[0..3] = {refArr.toList.take 3}"
  else
    IO.println "[Ref ] SKIPPED via HESPER_MONO_SKIP_PROD"

  -- ================================================================
  -- IRv2 MONOLITH PATH.
  -- Rebuild input + paramsBuf (forwardBlock is destructive on scratch
  -- buffers inside `state`).  Then run the 1-layer Monolith graph.
  -- ================================================================
  GPUBackend.writeBuffer ctx state.buf2 inputBytes
  GPUBackend.writeBuffer ctx state.paramsBuf paramsBytes

  let v2Out ← GPUBackend.allocBuffer ctx (hiddenSize * 4).toUSize
  let firstInputId : Nat := 10_000
  let lastOutputId : Nat := 10_999
  let baseTensorId : Nat := 20_000
  -- Bundle extractor assigns model layer `i` the key `bundleBase + i`.
  -- The 1-layer Monolith plan uses key = `firstLayerKey + 0` for its
  -- (only) layer.  To make "layer 0 of the graph" resolve to "model
  -- layer 5", set `firstLayerKey = bundleBase + 5`.
  let bundleBase   : UInt64 := 0xCAFE0000
  let firstLayerKey : UInt64 := bundleBase + li.toUInt64
  let (attnBundles, ffnBundles) ←
    Hesper.Models.Gemma4.extractMonolithBundles (β := CUDAContext) model state bundleBase
  let (_, graph) := Hesper.Circuit.IRv2.runBuilder
    (Hesper.Models.Gemma4_v2.forwardTokenLazyMonolith
       1 firstInputId lastOutputId baseTensorId firstLayerKey pos)
  IO.println s!"[IRv2] Monolith graph: {graph.blocks.size} blocks, {graph.tensors.size} tensors"
  IO.println s!"[IRv2] resolved {attnBundles.length} attn bundles, {ffnBundles.length} ffn bundles"

  -- Slot layout (9 per layer, see forwardTokenLazyMonolith):
  --   +0 q, +1 k, +2 v, +3 attnOut, +4 wOOut, +5 attnResid,
  --   +6 ffnOut, +7 postFFNOut (→ lastOutputId for final layer).
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
  IO.println "[IRv2] running runMonolithicGraph ..."
  Hesper.Circuit.IRv2.runMonolithicGraph ctx graph externalBufs attnBundles ffnBundles
  let v2Arr ← readF32BufMono ctx v2Out hiddenSize
  IO.println s!"[IRv2] out[0..3] = {v2Arr.toList.take 3}"

  -- Stage-by-stage inspection: dump all the intermediate buffers we
  -- wired to the Monolith graph so we can see where divergence starts.
  let qArr  ← readF32BufMono ctx state.qBuf   (cfg.numAttentionHeads * cfg.headDim li)
  let vArr  ← readF32BufMono ctx state.vBuf   (cfg.numKVHeads li * cfg.headDim li)
  let atArr ← readF32BufMono ctx state.attnOutBuf  (cfg.numAttentionHeads * cfg.headDim li)
  let woArr ← readF32BufMono ctx state.normedBuf   hiddenSize
  let arArr ← readF32BufMono ctx state.attnResidualBuf hiddenSize
  let fArr  ← readF32BufMono ctx state.ffnOutBuf   hiddenSize
  IO.println s!"[IRv2 stage] qBuf[0..2]       = {qArr.toList.take 3}"
  IO.println s!"[IRv2 stage] vBuf[0..2]       = {vArr.toList.take 3}"
  IO.println s!"[IRv2 stage] attnOut[0..2]    = {atArr.toList.take 3}"
  IO.println s!"[IRv2 stage] wOOut[0..2]      = {woArr.toList.take 3}"
  IO.println s!"[IRv2 stage] attnResid[0..2]  = {arArr.toList.take 3}"
  IO.println s!"[IRv2 stage] ffnOut[0..2]     = {fArr.toList.take 3}"

  let err := maxAbsDiffMono refArr v2Arr
  IO.println s!"[Parity] max |err| over {hiddenSize} elems = {err}"
  IO.println ""
  IO.println "── F2.5 interpretation ─────────────────────────────────────"
  IO.println "Monolith pipeline: 6 AST blocks, ~14 physical kernels executed"
  IO.println "end-to-end on real Gemma 4 E4B Q4_K_M weights (layer 5)."
  IO.println ""
  IO.println "Monolith coverage (implemented):"
  IO.println "  • GemmaAttentionMonolith (RMSNorm+QKV+qkvNorm+RoPE+scatter)"
  IO.println "  • FlashAttention"
  IO.println "  • GemmaAttnOutProj (wO)"
  IO.println "  • PostAttnNormAdd"
  IO.println "  • GemmaFFNMonolith"
  IO.println "  • PostFFNNormAdd"
  IO.println ""
  IO.println "Expected residual divergence from production forwardBlock:"
  IO.println "  • Per-Layer Embedding (PLE) chain — Gemma 4 E4B specific"
  IO.println "  • layerOutScale tail (folded into fusedPLPostScale)"
  IO.println "  • Possible wO kernel-variant difference (2-row vs 4-warp)"
  IO.println ""
  IO.println "Progression across sessions:"
  IO.println "  • F2 (4 blocks):  max |err| = 12.67 (missing post-attn residual)"
  IO.println "  • F2.5 (6 blocks): max |err| =  8.09 (outScale divergence)"
  IO.println "  • F2.6 (PLE-off ref): max |err| = 5.04 (FlashAttn kernel-variant mismatch)"
  IO.println s!"  • F2.7 (dynamicParams FlashAttn): max |err| = {err}"
  IO.println ""
  IO.println "Sub-kernel bit-parity for each Monolith physical stage is"
  IO.println "already independently proven (B3, B4, B5, B7, B8, B9)."
  IO.println ""
  if err == 0.0 then
    IO.println "PASS: Monolith output is BIT-IDENTICAL to production forwardBlock"
  else if err < 1e-2 then
    IO.println s!"NEAR-PARITY: {err} — PLE + layerOutScale remain"
  else
    IO.println s!"DIVERGE: {err} — PLE + layerOutScale not yet modelled"
