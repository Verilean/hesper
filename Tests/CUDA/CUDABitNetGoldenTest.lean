import Hesper.Backend.CUDA
import Hesper.Backend.WebGPU
import Hesper.Models.BitNet
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Reader
import Hesper

/-!
# BitNet Golden Value Test: WebGPU ↔ CUDA logits comparison

Loads the same BitNet model on both backends, runs forward pass for the
same prompt tokens, and compares logits after each prefill step and the
first generated token.
-/

open Hesper
open Hesper.Models.BitNet

private def uf (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -fv else fv

def compareLogits (name : String) (wa ca : ByteArray) (n : Nat) (tol : Float := 0.5) : IO Bool := do
  let mut maxDiff : Float := 0.0
  let mut maxIdx := 0
  let mut sumDiff : Float := 0.0
  -- Find top-5 by WebGPU logits
  let mut top5 : Array (Nat × Float) := #[]
  for i in [:n] do
    let w := uf wa i; let c := uf ca i
    let d := (w - c).abs
    sumDiff := sumDiff + d
    if d > maxDiff then maxDiff := d; maxIdx := i
    -- Track top-5
    if top5.size < 5 then
      top5 := top5.push (i, w)
    else
      let mut minVal := top5[0]!.2; let mut minJ := 0
      for j in [1:5] do
        if top5[j]!.2 < minVal then minVal := top5[j]!.2; minJ := j
      if w > minVal then top5 := top5.set! minJ (i, w)
  let avgDiff := sumDiff / n.toFloat
  -- Check top-5 ranking agreement
  let mut wTop := top5.qsort (fun a b => a.2 > b.2)
  let mut cTop5 : Array (Nat × Float) := #[]
  for (idx, _) in wTop do
    cTop5 := cTop5.push (idx, uf ca idx)
  IO.println s!"  {name}: maxDiff={maxDiff} avgDiff={avgDiff}"
  IO.println s!"    WebGPU top-5: {wTop.map (fun (i,v) => s!"[{i}]={v}")}"
  IO.println s!"    CUDA   top-5: {cTop5.map (fun (i,v) => s!"[{i}]={v}")}"
  if maxDiff < tol then
    IO.println s!"  ✓ PASS (maxDiff < {tol})"
    return true
  else
    IO.println s!"  ✗ FAIL (maxDiff={maxDiff} ≥ {tol})"
    return false

def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gguf/ggml-model-i2_s.gguf"
  IO.println "═══ BitNet WebGPU↔CUDA Golden Value Test ═══\n"

  -- Load GGUF
  IO.println "[1] Loading GGUF..."
  let gguf ← Hesper.GGUF.loadGGUF ggufPath

  -- Init backends
  IO.println "[2] Initializing backends..."
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let cuda ← CUDAContext.init

  -- Load model on both backends
  IO.println "[3] Loading model on WebGPU..."
  let wModel ← fromGGUFObject (β := Hesper.WebGPU.Device) device gguf none
  IO.println "[4] Loading model on CUDA..."
  let cModel ← fromGGUFObject (β := CUDAContext) cuda gguf none

  -- Tokenize
  IO.println "[5] Tokenizing..."
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf true false
  let prompt := "The meaning of life is"
  let promptTokens := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  IO.println s!"  Prompt: \"{prompt}\" → {promptTokens.size} tokens: {promptTokens}"

  -- Create KV cache states
  IO.println "[6] Creating KV caches..."
  let wCache ← createKVCacheState device wModel
  let cCache ← createKVCacheState cuda cModel

  let vocabSize := wModel.config.vocabSize
  let logitsBytes := (vocabSize * 4).toUSize

  -- Prefill and compare logits at each position
  IO.println "\n── Prefill logits comparison ──"
  let mut passed := 0
  let mut failed := 0

  for i in [:promptTokens.size] do
    let tokenId := promptTokens[i]!
    forwardSingleToken device wModel tokenId i wCache
    forwardSingleToken cuda cModel tokenId i cCache

    let wLogits ← GPUBackend.readBuffer device wCache.logitsBuf logitsBytes
    let cLogits ← GPUBackend.readBuffer cuda cCache.logitsBuf logitsBytes

    if ← compareLogits s!"pos={i} token={tokenId}" wLogits cLogits vocabSize then
      passed := passed + 1
    else failed := failed + 1

  -- Generate 1 token and compare
  IO.println "\n── First generated token ──"

  let wArgmax ← gpuArgmax device wCache.logitsBuf wCache.argmaxBuf vocabSize
  let cArgmax ← gpuArgmax cuda cCache.logitsBuf cCache.argmaxBuf vocabSize
  IO.println s!"  WebGPU token: {wArgmax}"
  IO.println s!"  CUDA   token: {cArgmax}"
  if wArgmax == cArgmax then
    IO.println s!"  ✓ First generated token matches: {wArgmax}"
    passed := passed + 1
  else
    IO.println s!"  ✗ Token mismatch: WebGPU={wArgmax} CUDA={cArgmax}"
    failed := failed + 1

  -- Summary
  IO.println s!"\n═══ {passed} passed, {failed} failed ═══"
  if failed > 0 then IO.println "✗ SOME TESTS FAILED"
  else IO.println "✓ ALL TESTS PASSED"
