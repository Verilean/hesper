import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaForwardPrefill
import Hesper.Models.Gemma4.ScratchPool
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser
import Hesper.WebGPU.BufferOps

/-!
# Phase 0 v3 LlamaPath prefill dispatch-count driver + parity driver

Two modes:
* **No prompt** → dispatch-count check only.  Runs the stub with no token
  input; reports total dispatches vs llama.cpp's ~2016 reference.
* **With prompt** → parity mode.  Tokenises the prompt, runs the stub
  with real embeddings, and (if `HESPER_GOLDEN_DUMP_DIR` is set) writes
  matching llama.cpp `cb()`-named tensors to that directory for diffing
  against `llama-eval-callback` output.

Usage:
  # dispatch count only
  HESPER_DP4A=1 lake exe gemma4-llama-prefill-skeleton \
    data/gemma-4-e4b-it-Q4_K_M.gguf 50

  # parity mode
  HESPER_DP4A=1 HESPER_GOLDEN_DUMP_DIR=/tmp/hesper_dump \
    lake exe gemma4-llama-prefill-skeleton \
      data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you"
-/

open Hesper
open Hesper.Models.Gemma4

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  -- Second arg: either an integer seqLen (dispatch-count mode) or a prompt string.
  let arg1 := args.getD 1 "50"
  let promptMode := arg1.toNat?.isNone
  let prompt := arg1
  -- Third arg: number of tokens to generate in parity mode (default 1).
  -- Each decode step re-runs the full prefill on `(prompt ++ generated)`,
  -- so this is O(N²) — fine for flow-correctness checks and small N.
  let maxTokens := (args.getD 2 "1").toNat!

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 LlamaPath v3 Prefill Skeleton       ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println s!"  Model: {ggufPath}"
  if promptMode then
    IO.println s!"  Mode: parity (prompt=\"{prompt}\")"
  else
    IO.println s!"  Mode: dispatch-count (seqLen={prompt})"

  let ctx ← CUDAContext.init
  -- Install llama.cpp PTX override if HESPER_USE_LLAMACPP_PTX=1.  With
  -- HESPER_LLAMACPP_Q6K=1 (and Q4K=0), Q6_K matmul dispatches route to
  -- llama.cpp's kernel instead of hesper's, letting us measure the ms
  -- gap that's attributable to the kernel body alone (vs周辺overhead).
  let _ ← Hesper.LlamaCppPTX.autoInstall
  IO.println "[Load] Reading GGUF..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  let model ← Gemma4Model.fromGGUFData ctx ggufData
  let cfg := model.config
  IO.println s!"[Config] {cfg.numHiddenLayers} layers, hidden={cfg.hiddenSize}"

  let state ← createInferenceState ctx cfg

  if !promptMode then
    -- Dispatch-count-only: single forward, no prompt tokens, no sampling.
    let seqLen := prompt.toNat!
    Hesper.resetDispatchCounter
    let startNs ← IO.monoNanosNow
    let _ ← forwardPrefillLlamaCpp ctx model seqLen state
    let endNs ← IO.monoNanosNow
    let totalDisp ← Hesper.getDispatchCounter
    let wallMs := (endNs - startNs).toFloat / 1000000.0
    IO.println ""
    IO.println "───────────── Result ─────────────"
    IO.println s!"  total dispatches : {totalDisp}"
    IO.println s!"  wall clock (ms)  : {wallMs}"
    if totalDisp >= 1900 ∧ totalDisp <= 2250 then
      IO.println "✓ PASS: dispatch count near llama.cpp prefill reference (±10%)"
    else
      IO.println s!"✗ FAIL: expected 1900..2250 total dispatches, got {totalDisp}"
    return

  -- ─────────────────── Parity / greedy generation mode ───────────────────
  IO.println "[Tokenize] Loading tokenizer..."
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf
  let initialToks := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  IO.println s!"[Tokenize] Prompt: {initialToks.size} tokens: {initialToks.toList}"

  let vocab := cfg.vocabSize
  -- Running generated tokens.
  let mut generatedIds : Array Nat := #[]

  -- Allocate persistent KV caches (one K/V pair per own-KV layer).
  let maxKVHeads := max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let cacheSize := maxKVHeads * cfg.maxSeqLen * maxHeadDim
  let ownKVLayers := cfg.numHiddenLayers - cfg.numKVSharedLayers
  let mut kvPairs : Array (Hesper.CUDA.CUDABuffer × Hesper.CUDA.CUDABuffer) := Array.empty
  for _ in [0:ownKVLayers] do
    let k ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
    let v ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
    kvPairs := kvPairs.push (k, v)

  -- Token-id GPU buffer (persists, re-used for prefill and each decode step).
  let maxSeq := initialToks.size + maxTokens
  let tokenIdsBuf ← GPUBackend.allocBuffer ctx (maxSeq * 4).toUSize

  -- ScratchPool: all per-forward transient buffers (batch scratch, PLE
  -- staging, etc.) go through this pool.  On the first forward the pool
  -- grows to N slots; every subsequent forward reuses the same slots
  -- with zero cuMemAlloc calls.
  let scratchPool ← Hesper.Models.Gemma4.ScratchPool.new ctx

  let genStart ← IO.monoNanosNow
  let mut lastDisp : Nat := 0

  -- 1. Prefill: run full prompt to populate KV caches.
  --    With HESPER_PREFILL_WARMUP=1, run the prefill twice and only
  --    report the second (warm) timing — useful for kernel-speed
  --    comparisons where PTX JIT cost otherwise dominates the wall.
  let warmupPrefill ← match ← IO.getEnv "HESPER_PREFILL_WARMUP" with
    | some "1" => pure true
    | _        => pure false
  do
    let mut bytes : ByteArray := ByteArray.empty
    for i in [0:initialToks.size] do
      bytes := bytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes initialToks[i]!.toUInt32
    GPUBackend.writeBuffer ctx tokenIdsBuf bytes

    if warmupPrefill then
      scratchPool.reset
      Hesper.resetDispatchCounter
      let warmStart ← IO.monoNanosNow
      let _ ← forwardPrefillLlamaCpp ctx model initialToks.size state
        (tokenIdsBuf := some tokenIdsBuf) (startPos := 0)
        (persistentCaches := some kvPairs)
        (scratchPool := some scratchPool)
      let warmEnd ← IO.monoNanosNow
      let warmMs := (warmEnd - warmStart).toFloat / 1000000.0
      IO.println s!"[prefill warmup] seqLen={initialToks.size} ({warmMs}ms, cold)"
      -- Reset KV caches and scratch so the measured prefill starts clean.
      -- We don't actually need to re-init the buffers — the prefill writes
      -- into them from startPos=0 which overwrites everything.

    let prefillStart ← IO.monoNanosNow
    scratchPool.reset
    Hesper.resetDispatchCounter
    let logitsOpt ← forwardPrefillLlamaCpp ctx model initialToks.size state
      (tokenIdsBuf := some tokenIdsBuf) (startPos := 0)
      (persistentCaches := some kvPairs)
      (scratchPool := some scratchPool)
    let prefillEnd ← IO.monoNanosNow
    lastDisp ← Hesper.getDispatchCounter

    match logitsOpt with
    | some logitsBuf =>
      let lbytes ← GPUBackend.readBuffer ctx logitsBuf (vocab * 4).toUSize
      let mut bestIdx : Nat := 0
      let mut bestVal : Float := Float.ofNat 0
      let mut first := true
      for i in [0:vocab] do
        let fb ← Hesper.Basic.bytesToFloat32 lbytes (i * 4)
        if first ∨ fb > bestVal then
          bestVal := fb; bestIdx := i; first := false
      let decoded := Hesper.Tokenizer.SentencePiece.decode tokenizer #[bestIdx]
      let prefillMs := (prefillEnd - prefillStart).toFloat / 1000000.0
      let label := if warmupPrefill then "prefill warm" else "prefill"
      IO.println s!"[{label}] seqLen={initialToks.size} next={bestIdx} \"{decoded}\" ({prefillMs}ms, {lastDisp} dispatches)"
      generatedIds := generatedIds.push bestIdx
    | none => IO.println "[prefill] no logits"

  -- 2. Decode loop: each step feeds the most recently generated token T_{k-1}
  --    as seqLen=1 with startPos = `prevSeqLen`, i.e. the position where
  --    T_{k-1}'s own KV will be written.  Attention then reads cache[0..startPos].
  --
  -- CUDA Graphs opt-in: HESPER_LLAMA_GRAPHS=1.
  -- First decode step runs eagerly so ScratchPool grows to its final slot
  -- count (CUDA forbids cuMemAlloc during stream capture).  Second step is
  -- captured; subsequent steps replay with one `cuGraphLaunch` instead of
  -- ~1500 `cuLaunchKernel` calls.  To let the graph see the new startPos
  -- on each replay, we pass a persistent `paramsBufOverride` that lives
  -- outside ScratchPool and is updated by the caller on the capture stream.
  let graphsEnabled ← match ← IO.getEnv "HESPER_LLAMA_GRAPHS" with
    | some "1" => pure true
    | _        => pure false
  let persistentParamsBuf ← GPUBackend.allocBuffer ctx (4 : USize)
  -- Pinned host slots: graph replay needs a source pointer that stays
  -- valid across the capture→replay boundary.  Lean ByteArrays are GC'd
  -- and their addresses aren't stable, which is exactly the
  -- CUDA_ERROR_ILLEGAL_ADDRESS hazard that Hesper/CUDA/FFI.lean warns
  -- about near cuMemAllocHost.  Allocate two 4-byte pinned slots (one
  -- for startPos, one for tokenIdsBuf[0]) and issue captured memcpys
  -- against them instead of ByteArrays.
  let pinnedStartPos ← Hesper.CUDA.cuMemAllocHost (4 : USize)
  let pinnedTokenId  ← Hesper.CUDA.cuMemAllocHost (4 : USize)
  let mut graphExec : Option Hesper.CUDA.CUgraphExec := none
  let mut graphStream : Option Hesper.CUDA.CUstream := none
  let mut capturedLogits : Option Hesper.CUDA.CUDABuffer := none

  for step in [1:maxTokens] do
    -- Last-generated token = the token we feed through this step.
    let lastGen := generatedIds[generatedIds.size - 1]!
    -- startPos = number of tokens whose KV is already in the cache =
    -- initialToks.size (written during prefill).  After this decode
    -- step we'll have initialToks.size + 1 cached tokens (slot 0..size).
    let startPos := initialToks.size + (generatedIds.size - 1)

    -- Stage the two varying values into pinned host memory.  These
    -- memcpys do NOT touch the GPU — they are plain memcpys — so they
    -- never fall inside any capture scope.
    Hesper.CUDA.cuWritePinned pinnedTokenId 0
      (Hesper.WebGPU.BufferOps.uint32ToBytes lastGen.toUInt32) (4 : USize)
    Hesper.CUDA.cuWritePinned pinnedStartPos 0
      (Hesper.WebGPU.BufferOps.uint32ToBytes startPos.toUInt32) (4 : USize)

    scratchPool.reset
    Hesper.resetDispatchCounter
    let stepStart ← IO.monoNanosNow

    let logitsOpt ← match graphsEnabled, graphExec with
    | true, some exec =>
      match graphStream with
      | none => pure none
      | some s =>
        -- Replay: push the fresh pinned values to device on the capture
        -- stream, then launch the graph.  Because the host-side pointers
        -- (pinnedStartPos / pinnedTokenId) were also the sources used
        -- during capture, the graph's memcpy nodes stay valid.
        Hesper.CUDA.cuMemcpyHtoDFromPinned persistentParamsBuf.ptr
          pinnedStartPos 0 (4 : USize) s
        Hesper.CUDA.cuMemcpyHtoDFromPinned tokenIdsBuf.ptr
          pinnedTokenId 0 (4 : USize) s
        Hesper.CUDA.cuGraphLaunch exec s
        Hesper.CUDA.cuStreamSynchronize s
        pure capturedLogits
    | true, none =>
      if step == 1 then
        -- Eager warm-up: grow ScratchPool to final size.  Write the
        -- current values via the usual host writeBuffer path — this is
        -- outside any capture so ByteArray stability doesn't matter.
        GPUBackend.writeBuffer ctx persistentParamsBuf
          (Hesper.WebGPU.BufferOps.uint32ToBytes startPos.toUInt32)
        GPUBackend.writeBuffer ctx tokenIdsBuf
          (Hesper.WebGPU.BufferOps.uint32ToBytes lastGen.toUInt32)
        forwardPrefillLlamaCpp ctx model 1 state
          (tokenIdsBuf := some tokenIdsBuf) (startPos := startPos)
          (persistentCaches := some kvPairs)
          (scratchPool := some scratchPool)
          (paramsBufOverride := some persistentParamsBuf)
      else
        -- Capture: update the two varying devices from pinned sources
        -- on the capture stream so the memcpys become graph nodes whose
        -- source pointers outlive GC.  Then run the forward to record
        -- the kernel sequence.
        let stream ← Hesper.CUDA.cuStreamCreate
        Hesper.cudaCaptureStream.set (some stream)
        Hesper.CUDA.cuStreamBeginCapture stream
        Hesper.CUDA.cuMemcpyHtoDFromPinned persistentParamsBuf.ptr
          pinnedStartPos 0 (4 : USize) stream
        Hesper.CUDA.cuMemcpyHtoDFromPinned tokenIdsBuf.ptr
          pinnedTokenId 0 (4 : USize) stream
        let logits ← forwardPrefillLlamaCpp ctx model 1 state
          (tokenIdsBuf := some tokenIdsBuf) (startPos := startPos)
          (persistentCaches := some kvPairs)
          (scratchPool := some scratchPool)
          (paramsBufOverride := some persistentParamsBuf)
          (skipConstantWrites := true)
        let graph ← Hesper.CUDA.cuStreamEndCapture stream
        Hesper.cudaCaptureStream.set none
        let exec ← Hesper.CUDA.cuGraphInstantiate graph
        Hesper.CUDA.cuGraphDestroy graph
        -- Capture only RECORDS; it doesn't run.  Launch once now so this
        -- step's logits are correct (kernels actually execute with the
        -- current pinnedStartPos / pinnedTokenId), and the caller sees
        -- coherent output instead of stale step-1 contents.
        Hesper.CUDA.cuGraphLaunch exec stream
        Hesper.CUDA.cuStreamSynchronize stream
        graphExec := some exec
        graphStream := some stream
        capturedLogits := logits
        IO.println s!"[Graph] captured decode step (step={step})"
        pure logits
    | false, _ =>
      -- No graphs: write tokenIdsBuf the usual way (pinned is
      -- graph-capture specific) and run eager.
      GPUBackend.writeBuffer ctx tokenIdsBuf
        (Hesper.WebGPU.BufferOps.uint32ToBytes lastGen.toUInt32)
      forwardPrefillLlamaCpp ctx model 1 state
        (tokenIdsBuf := some tokenIdsBuf) (startPos := startPos)
        (persistentCaches := some kvPairs)
        (scratchPool := some scratchPool)
        (skipConstantWrites := step > 1)

    let stepEnd ← IO.monoNanosNow
    lastDisp ← Hesper.getDispatchCounter

    match logitsOpt with
    | some logitsBuf =>
      let lbytes ← GPUBackend.readBuffer ctx logitsBuf (vocab * 4).toUSize
      let mut bestIdx : Nat := 0
      let mut bestVal : Float := Float.ofNat 0
      let mut first := true
      for i in [0:vocab] do
        let fb ← Hesper.Basic.bytesToFloat32 lbytes (i * 4)
        if first ∨ fb > bestVal then
          bestVal := fb; bestIdx := i; first := false
      let decoded := Hesper.Tokenizer.SentencePiece.decode tokenizer #[bestIdx]
      let stepMs := (stepEnd - stepStart).toFloat / 1000000.0
      IO.println s!"[decode {step}] startPos={startPos} next={bestIdx} \"{decoded}\" ({stepMs}ms, {lastDisp} dispatches)"
      generatedIds := generatedIds.push bestIdx
    | none =>
      IO.println s!"[decode {step}] no logits; stopping"
      break
  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1000000.0
  let tps := if genMs > 0 then generatedIds.size.toFloat * 1000.0 / genMs else 0.0

  IO.println ""
  IO.println "───────────── Result ─────────────"
  IO.println s!"  prompt tokens    : {initialToks.size}"
  IO.println s!"  generated tokens : {generatedIds.size}"
  IO.println s!"  total time (ms)  : {genMs}"
  IO.println s!"  tokens / sec     : {tps}"
  IO.println s!"  last forward disp: {lastDisp}"
  let fullDecoded := Hesper.Tokenizer.SentencePiece.decode tokenizer generatedIds
  IO.println s!"  prompt  : \"{prompt}\""
  IO.println s!"  completion: \"{fullDecoded}\""
  match ← IO.getEnv "HESPER_GOLDEN_DUMP_DIR" with
  | some dir => IO.println s!"[Golden] Dumps from the LAST forward written to {dir}/"
  | none => pure ()
