import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Embedding
import Hesper.Layers.RMSNorm
import Hesper.Layers.Softmax
import Hesper.TTT.Kernels

/-!
# CUDA BitNet-style Forward Pass Test

Tests the core inference pipeline on CUDA via GPUBackend typeclass:
  Embedding → RMSNorm → matVec → result

Same code runs on WebGPU or CUDA — only the context type differs.
-/

open Hesper
open Hesper.CUDA
open Hesper.TTT.Kernels

-- Float helpers
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def unpackFloat (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

/-- Backend-agnostic mini forward pass: embed → normalize → matVec → output -/
def miniForward [GPUBackend β] (ctx : β)
    (embLayer : Hesper.Layers.Embedding.Embedding (GPUBackend.Buf β))
    (normLayer : Hesper.Layers.RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (weightBuf : GPUBackend.Buf β)
    (tokenBuf outputBuf : GPUBackend.Buf β)
    (vocabSize hiddenDim : Nat) : IO Unit := do
  -- Step 1: Embedding lookup → hidden
  let hiddenBuf ← GPUBackend.allocBuffer ctx (hiddenDim * 4).toUSize
  Hesper.Layers.Embedding.forward ctx embLayer tokenBuf hiddenBuf 1 1

  -- Step 2: RMSNorm → normalized hidden
  let normBuf ← GPUBackend.allocBuffer ctx (hiddenDim * 4).toUSize
  Hesper.Layers.RMSNorm.forward ctx normLayer hiddenBuf normBuf

  -- Step 3: matVec (weight @ normHidden → logits)
  executeMatVec ctx weightBuf normBuf outputBuf vocabSize hiddenDim

  GPUBackend.freeBuffer ctx hiddenBuf
  GPUBackend.freeBuffer ctx normBuf

def main : IO Unit := do
  IO.println "═══ CUDA BitNet-style Forward Pass Test ═══"

  let ctx ← CUDAContext.init

  let vocabSize := 8; let hiddenDim := 4

  -- Create embedding table: [vocabSize × hiddenDim]
  -- Token 0 → [0.1, 0.2, 0.3, 0.4], Token 1 → [1.1, ...], Token 2 → [2.1, 2.2, 2.3, 2.4], ...
  let mut embData : Array Float := #[]
  for i in List.range vocabSize do
    for j in List.range hiddenDim do
      embData := embData.push (i.toFloat + (j + 1).toFloat / 10.0)

  let embLayer ← Hesper.Layers.Embedding.createFromFloat32 ctx
    { vocabSize, dim := hiddenDim } (packFloats embData)
  IO.println s!"  Embedding: {vocabSize}×{hiddenDim}"

  -- Create RMSNorm with scale = all 1.0 (identity normalization)
  let scaleData := packFloats (Array.replicate hiddenDim 1.0)
  let normLayer ← Hesper.Layers.RMSNorm.create ctx
    { dim := hiddenDim, eps := 1.0e-5 } scaleData
  IO.println "  RMSNorm: identity scale"

  -- W = [vocabSize × hiddenDim], row i selects hidden[i] (identity for first 4, zero rest)
  let mut wData : Array Float := #[]
  for i in List.range vocabSize do
    for j in List.range hiddenDim do
      wData := wData.push (if i < hiddenDim && i == j then 1.0 else 0.0)
  let weightBuf ← GPUBackend.allocBuffer ctx (vocabSize * hiddenDim * 4).toUSize
  GPUBackend.writeBuffer ctx weightBuf (packFloats wData)

  -- Token input: token ID = 2
  let tokenBuf ← GPUBackend.allocBuffer ctx 4
  GPUBackend.writeBuffer ctx tokenBuf (ByteArray.mk #[2, 0, 0, 0])

  let outputBuf ← GPUBackend.allocBuffer ctx (vocabSize * 4).toUSize

  -- Full pipeline: embed(2) → RMSNorm → W @ hidden → logits
  IO.println "  Running: embed(2) → RMSNorm → W @ hidden → logits"
  miniForward ctx embLayer normLayer weightBuf tokenBuf outputBuf vocabSize hiddenDim

  let result ← GPUBackend.readBuffer ctx outputBuf (vocabSize * 4).toUSize
  IO.println "  Logits:"
  let mut ok := true
  for i in List.range vocabSize do
    let v := unpackFloat result i
    -- Token 2 → embed = [2.1, 2.2, 2.3, 2.4]
    -- RMSNorm: rms = sqrt((2.1²+2.2²+2.3²+2.4²)/4) = sqrt(5.075) ≈ 2.2528
    -- normed[j] = embed[j] / rms, W = identity(4)×8 → logit[i] = normed[i] for i<4
    let rms := Float.sqrt 5.075
    let expected := if i < hiddenDim then (2.0 + (i + 1).toFloat / 10.0) / rms else 0.0
    IO.println s!"    logit[{i}] = {v} (expect {expected})"
    if (v - expected).abs > 0.1 then ok := false

  GPUBackend.freeBuffer ctx weightBuf
  GPUBackend.freeBuffer ctx tokenBuf
  GPUBackend.freeBuffer ctx outputBuf

  if ok then
    IO.println "✓ CUDA BITNET-STYLE FORWARD PASS PASSED"
  else
    IO.println "✗ FAILED"
    IO.Process.exit 1
