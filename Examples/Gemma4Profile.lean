import Hesper
import Hesper.Models.Gemma4
import Hesper.Layers.Linear
import Hesper.WebGPU.Device

/-!
# Gemma 4 Per-Token Profiling Exe

Runs `forwardSingleToken` without the batched command encoder so each
dispatch auto-syncs via `deviceWait`. Measures:
  * total wall time per decode token
  * total time spent inside `LinearLayer.forward` (Q4_K / Q6_K linears)
  * implicit "non-linear" time (everything else) = total − linear

Warmup step: first a few prefill passes so shader compile costs are
paid and the pipeline cache is warm. Then N decode steps are timed and
averaged.

Usage:
  lake exe gemma4-profile
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Gemma 4 Per-Token Profile"
  IO.println "═══════════════════════════════════════════════"

  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println s!"[Load] {modelPath}"
  let model ← Gemma4Model.fromGGUF device modelPath
  let state ← createInferenceState device model.config
  let tokenId := 9259  -- "Hello"
  IO.println ""

  -- Warmup: one full forward pass to compile all shaders and populate the
  -- pipeline/bind-group/prepared-dispatch caches. This is the slow
  -- first-token cost; subsequent calls hit the fast path.
  IO.println "[Warmup] First decode step (compiling shaders)..."
  Hesper.Layers.Linear.profilingRef.set false  -- normal batched mode
  let warmStart ← IO.monoNanosNow
  forwardSingleToken device model tokenId 0 state
  let warmEnd ← IO.monoNanosNow
  IO.println s!"[Warmup] {(warmEnd - warmStart).toFloat / 1_000_000.0} ms"

  -- A couple more passes to warm the pipeline cache fully (e.g. the fallback
  -- paths for layers that don't hit the same buffers on every call).
  for i in [1:5] do
    forwardSingleToken device model tokenId i state

  -- Profiling run: disable batching (so each dispatch deviceWait-s) and
  -- enable the Linear profiling counters. Measure N tokens.
  IO.println ""
  IO.println "[Profile] Timing N decode steps in unbatched mode..."
  Hesper.Layers.Linear.profilingRef.set true
  Hesper.Layers.Linear.totalNanosRef.set 0
  Hesper.Layers.Linear.callCountRef.set 0
  Hesper.Layers.Linear.perShapeRef.set #[]

  let n := 10
  let start ← IO.monoNanosNow
  for i in [5:(5 + n)] do
    forwardSingleToken device model tokenId i state
  let stop ← IO.monoNanosNow

  Hesper.Layers.Linear.profilingRef.set false

  let totalNs := stop - start
  let linearNs := (← Hesper.Layers.Linear.totalNanosRef.get).toNat
  let linearCalls := (← Hesper.Layers.Linear.callCountRef.get)

  let nf := n.toFloat
  let totalMsPerToken := (totalNs.toFloat / 1_000_000.0) / nf
  let linearMsPerToken := (linearNs.toFloat / 1_000_000.0) / nf
  let otherMsPerToken := totalMsPerToken - linearMsPerToken
  let linearCallsPerToken := linearCalls / n

  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Per-token breakdown (averaged over 10 decodes)"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"  Total wall time     : {totalMsPerToken} ms/tok   ({1000.0 / totalMsPerToken} TPS)"
  IO.println s!"  LinearLayer.forward : {linearMsPerToken} ms/tok   ({linearCallsPerToken} calls/tok)"
  IO.println s!"    avg per call      : {linearMsPerToken / linearCallsPerToken.toFloat} ms"
  IO.println s!"  Everything else     : {otherMsPerToken} ms/tok"
  IO.println ""
  IO.println s!"  Linear share        : {(linearMsPerToken / totalMsPerToken) * 100.0}%"
  IO.println s!"  Other  share        : {(otherMsPerToken  / totalMsPerToken) * 100.0}%"
  IO.println "═══════════════════════════════════════════════"

  -- Per-shape linear breakdown
  let perShape ← Hesper.Layers.Linear.perShapeRef.get
  let rows := perShape.toList.map (fun (i, o, ns, cnt) =>
    let total := ns.toNat.toFloat / 1_000_000.0
    let avg := total / cnt.toFloat
    let perTok := total / nf
    (i, o, cnt, total, avg, perTok))
  -- Sort by per-token contribution descending.
  let rows := rows.toArray.qsort (fun a b =>
    let (_, _, _, _, _, pt1) := a
    let (_, _, _, _, _, pt2) := b
    pt1 > pt2)
  IO.println ""
  IO.println "  Linear shape breakdown (sorted by ms/tok)"
  IO.println "  ─────────────────────────────────────────────────────────────"
  IO.println "    inDim   outDim   calls   total_ms   avg_ms   ms/tok"
  for row in rows do
    let (i, o, cnt, total, avg, perTok) := row
    IO.println s!"  {i}×{o}  calls={cnt}  total={total} ms  avg={avg} ms  ms/tok={perTok}"
