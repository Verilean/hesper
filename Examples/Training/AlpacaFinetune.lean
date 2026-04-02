import Hesper
import Hesper.LoRA.Types
import Hesper.LoRA.Init
import Hesper.LoRA.Forward
import Hesper.LoRA.Backward
import Hesper.LoRA.IO
import Hesper.LoRA.Inference
import Hesper.Training.Loss
import Hesper.Training.AlpacaDataset
import Hesper.Training.TrainLoop
import Hesper.Optimizer.AdamGPU
import Hesper.Optimizer.GradientClip
import Hesper.Training.LRScheduler
import Hesper.Models.BitNet
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Reader
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute
import Hesper.WGSL.MatMul
import Hesper.WGSL.Elementwise
import Hesper.Training.ParseFloat
import Hesper.Training.SafeBuffer

/-!
# Alpaca-Style LoRA Finetuning for BitNet

End-to-end instruction finetuning of BitNet b1.58 2B using LoRA adapters.

## GPU Optimization

The training loop uses batched GPU execution:
- Forward + loss + backward are recorded into a SINGLE GPU command buffer per token
- Loss is accumulated on GPU, read once per example (not per token)
- SGD parameter updates are batched into a single GPU submit
- This eliminates ~20 GPU sync points per token vs naive implementation
-/

open Hesper.WebGPU
open Hesper.LoRA
open Hesper.Training
open Hesper.Training.ParseFloat
open Hesper.Models.BitNet
open Hesper.Tokenizer.SentencePiece
open Hesper.GGUF

def printUsage : IO Unit := do
  IO.println "Usage: alpaca-finetune [OPTIONS]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --model PATH     Path to BitNet GGUF model file (required)"
  IO.println "  --data PATH      Path to Alpaca JSON dataset (required)"
  IO.println "  --output PATH    Path to save LoRA weights (default: lora_weights.bin)"
  IO.println "  --rank N         LoRA rank (default: 8)"
  IO.println "  --alpha F        LoRA alpha scaling (default: 8.0)"
  IO.println "  --lr F           Learning rate (default: 1e-4)"
  IO.println "  --epochs N       Number of training epochs (default: 3)"
  IO.println "  --max-seq-len N  Maximum sequence length (default: 512)"
  IO.println "  --log-every N    Log every N steps (default: 10)"
  IO.println "  --max-grad-norm F Max gradient norm for clipping (0=disabled, default: 0)"

structure Args where
  modelPath : String
  dataPath : String
  outputPath : String := "lora_weights.bin"
  rank : Nat := 8
  alpha : Float := 8.0
  lr : Float := 2e-4  -- PyTorch/HuggingFace LoRA default
  epochs : Nat := 3
  maxSeqLen : Nat := 512
  logEvery : Nat := 10
  maxGradNorm : Float := 1.0  -- PyTorch default (0 = disabled)

def parseArgs (args : List String) : IO Args := do
  let mut modelPath := ""
  let mut dataPath := ""
  let mut outputPath := "lora_weights.bin"
  let mut rank : Nat := 8
  let mut alpha : Float := 8.0
  let mut lr : Float := 1e-4
  let mut epochs : Nat := 3
  let mut maxSeqLen : Nat := 512
  let mut logEvery : Nat := 10
  let mut maxGradNorm : Float := 0.0
  let mut remaining := args
  while !remaining.isEmpty do
    match remaining with
    | "--model" :: path :: rest => modelPath := path; remaining := rest
    | "--data" :: path :: rest => dataPath := path; remaining := rest
    | "--output" :: path :: rest => outputPath := path; remaining := rest
    | "--rank" :: n :: rest => rank := n.toNat!; remaining := rest
    | "--alpha" :: f :: rest => alpha := parseFloat f; remaining := rest
    | "--lr" :: f :: rest =>
      lr := parseFloat f
      remaining := rest
    | "--epochs" :: n :: rest => epochs := n.toNat!; remaining := rest
    | "--max-seq-len" :: n :: rest => maxSeqLen := n.toNat!; remaining := rest
    | "--log-every" :: n :: rest => logEvery := n.toNat!; remaining := rest
    | "--max-grad-norm" :: f :: rest => maxGradNorm := parseFloat f; remaining := rest
    | "--help" :: _ => printUsage; throw (IO.userError "")
    | unknown :: rest =>
      IO.eprintln s!"Unknown argument: {unknown}"
      remaining := rest
    | [] => remaining := []

  if modelPath.isEmpty then
    printUsage
    throw (IO.userError "Missing required --model argument")
  if dataPath.isEmpty then
    printUsage
    throw (IO.userError "Missing required --data argument")

  pure { modelPath, dataPath, outputPath, rank, alpha, lr, epochs, maxSeqLen, logEvery, maxGradNorm }

def main (args : List String) : IO Unit := do
  let args ← parseArgs args

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  Hesper: Alpaca-Style LoRA Finetuning        ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""
  IO.println s!"Model:      {args.modelPath}"
  IO.println s!"Dataset:    {args.dataPath}"
  IO.println s!"Output:     {args.outputPath}"
  IO.println s!"LoRA rank:  {args.rank}"
  IO.println s!"LoRA alpha: {args.alpha}"
  IO.println s!"LR:         {args.lr}"
  IO.println s!"Epochs:     {args.epochs}"
  IO.println s!"Max seq:    {args.maxSeqLen}"
  IO.println ""

  -- Step 1: Initialize GPU
  IO.println "[1/6] Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  -- Step 2: Load model
  IO.println "[2/6] Loading BitNet model..."
  let gguf ← loadGGUF args.modelPath
  let model ← fromGGUFObject device gguf none
  let dim := model.config.dim
  let kvDim := model.config.kvDim

  IO.println s!"  Model: {model.config.dim} dim, {model.config.numLayers} layers, {model.config.vocabSize} vocab"

  -- Step 3: Create tokenizer
  IO.println "[3/6] Creating tokenizer..."
  let tokenizer ← fromGGUF gguf true false
  let eosTokenId := (Hesper.Tokenizer.SentencePiece.eosToken tokenizer).getD 2

  -- Step 4: Load dataset
  IO.println "[4/6] Loading Alpaca dataset..."
  let examples ← AlpacaDataset.loadDataset args.dataPath
  let tokenizedExamples := AlpacaDataset.tokenizeDataset
    (fun s => encode tokenizer s) examples eosTokenId args.maxSeqLen
  AlpacaDataset.printStats tokenizedExamples

  -- Step 5: Create LoRA adapter
  IO.println "[5/6] Creating LoRA adapter..."
  let loraConfig : Hesper.LoRA.Config := { rank := args.rank, alpha := args.alpha }
  let adapter ← createAdapter device loraConfig model.config.numLayers dim kvDim

  -- Create training state and buffers
  let trainState ← TrainLoop.createTrainState device adapter dim kvDim
  let lossBuf ← createBuffer device { size := 4, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }
  let targetBuf ← createBuffer device { size := 4, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let dLogitsBuf ← createBuffer device { size := (model.config.vocabSize * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let dHiddenBuf ← createBuffer device { size := (dim * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let loraInferState ← Hesper.LoRA.Inference.createLoRATrainingState device adapter
    dim kvDim model.config.numHeads model.config.headDim model.config.maxSeqLen model.config.numLayers

  let scale := loraConfig.scale
  let startLayer := 0  -- backward through all layers
  let mut currentState := trainState
  let mut globalStep : Nat := 0

  -- Create gradient clipping buffers
  let clipBufs ← Hesper.Optimizer.GradientClip.createClipBuffers device
  let maxGradNorm := args.maxGradNorm

  -- Create LR scheduler (linear warmup + cosine decay)
  let lrScheduler := Hesper.Training.LRScheduler.create args.lr
    tokenizedExamples.size args.epochs 0.06  -- 6% warmup (PyTorch default ~10%, reduced for small datasets)

  -- Step 6: Training (GPU-optimized, PyTorch-standard)
  IO.println "[6/6] Starting training..."
  IO.println s!"  Optimizer: AdamW (lr={args.lr}, wd=0.01)"
  IO.println s!"  Gradient clipping: {if maxGradNorm > 0.0 then s!"max_norm={maxGradNorm}" else "disabled"}"
  IO.println s!"  LR schedule: warmup {lrScheduler.warmupSteps} steps + cosine decay"
  IO.println s!"  Total steps: {lrScheduler.totalSteps}"
  IO.println ""

  let cacheState ← createKVCacheState device model

  for epoch in [:args.epochs] do
    let mut epochLoss : Float := 0.0
    let mut epochTokens : Nat := 0

    for exIdx in [:tokenizedExamples.size] do
      if h : exIdx < tokenizedExamples.size then
        let ex := tokenizedExamples[exIdx]

        -- Reset caches and zero gradients
        resetPreparedDispatches model
        TrainLoop.zeroGrads device adapter currentState.grads

        let mut exampleTokens : Nat := 0

        -- Zero loss accumulator on GPU
        let zeroBytes := Hesper.WebGPU.BufferOps.uint32ToBytes 0
        writeBuffer device lossBuf 0 zeroBytes

        -- Forward + backward for ALL tokens (GPU-batched)
        for t in [:ex.seqLen - 1] do
          let tokenId := ex.tokens.getD t 0
          let targetId := ex.tokens.getD (t + 1) 0
          let isOutputToken := t >= ex.promptLen

          if isOutputToken then
            let targetBytes := Hesper.WebGPU.BufferOps.uint32ToBytes (UInt32.ofNat targetId)
            writeBuffer device targetBuf 0 targetBytes
            exampleTokens := exampleTokens + 1

          Hesper.LoRA.Inference.forwardAndBackwardBatched device model
            tokenId t cacheState adapter loraInferState
            isOutputToken targetBuf lossBuf dLogitsBuf dHiddenBuf
            currentState.grads currentState startLayer

        -- Read accumulated loss ONCE per example
        let exampleLoss ← if exampleTokens > 0 then
          TrainLoop.readLoss device lossBuf
        else pure 0.0
        epochLoss := epochLoss + exampleLoss
        epochTokens := epochTokens + exampleTokens
        globalStep := globalStep + 1

        -- === PyTorch-standard optimizer step ===
        if exampleTokens > 0 then
          -- Gradient clipping (if enabled)
          if maxGradNorm > 0.0 then
            let _gradNorm ← Hesper.Optimizer.GradientClip.clipGradNorm device adapter
              currentState.grads maxGradNorm clipBufs
          -- AdamW update
          let currentLR := Hesper.Training.LRScheduler.getLR lrScheduler globalStep
          let adamConfig : Hesper.Optimizer.AdamGPU.Config := { lr := currentLR }
          currentState ← TrainLoop.optimizerStep device currentState adamConfig

        -- Logging
        if globalStep % args.logEvery == 0 || exIdx == 0 then
          let avgLoss := if exampleTokens > 0 then exampleLoss / exampleTokens.toFloat else 0.0
          let currentLR := Hesper.Training.LRScheduler.getLR lrScheduler globalStep
          IO.println s!"[Train] Epoch {epoch + 1}, Step {globalStep}: loss={avgLoss.toString} ({exampleTokens} tokens, lr={currentLR.toString})"

    -- Epoch summary
    let avgEpochLoss := if epochTokens > 0 then epochLoss / epochTokens.toFloat else 0.0
    IO.println s!"[Train] Epoch {epoch + 1} complete: avg_loss={avgEpochLoss.toString}, tokens={epochTokens}"
    IO.println ""

  -- Check for NaN before saving
  let mut hasNaNWeights := false
  for i in [:adapter.layers.size] do
    if h : i < adapter.layers.size then
      let nanQ ← Hesper.Training.SafeBuffer.hasNaN device adapter.layers[i].loraQ.a 8  -- check first 8
      let nanB ← Hesper.Training.SafeBuffer.hasNaN device adapter.layers[i].loraQ.b 8
      if nanQ || nanB then
        IO.eprintln s!"[WARNING] Layer {i} has NaN weights — save may produce corrupt file"
        hasNaNWeights := true
        break

  -- Save LoRA weights
  if hasNaNWeights then
    IO.eprintln "Skipping save due to NaN weights. Try lower --lr or enable --max-grad-norm."
  else
    IO.println s!"Saving LoRA weights to {args.outputPath}..."
    Hesper.LoRA.IO.saveAdapter device adapter args.outputPath

  IO.println ""
  IO.println "Training complete!"
  IO.println s!"LoRA weights saved to: {args.outputPath}"
  IO.println ""
  IO.println "To use the finetuned model for inference:"
  IO.println s!"  lake exe bitnet-complete --model {args.modelPath} --lora {args.outputPath}"
