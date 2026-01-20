import Hesper

/-!
# MNIST Dataset Loader

Simple MNIST data loader that reads the standard MNIST binary format.
For this example, we'll use a simplified approach with synthetic data
to demonstrate the training pipeline. In production, you would download
the actual MNIST dataset from http://yann.lecun.com/exdb/mnist/
-/

namespace Examples.MachineLearning.MNISTData

/-- MNIST image size: 28x28 pixels = 784 features -/
def imageSize : Nat := 784

/-- Number of classes (digits 0-9) -/
def numClasses : Nat := 10

/-- Training dataset size -/
def trainSize : Nat := 60000

/-- Test dataset size -/
def testSize : Nat := 10000

/-- MNIST dataset batch -/
structure Batch where
  images : Array Float  -- Flattened images [batch_size * 784]
  labels : Array Nat    -- Class labels [batch_size]
  batchSize : Nat
  deriving Repr

/-- Generate synthetic MNIST-like data for demonstration.
    In production, replace this with actual MNIST binary file reading.
-/
def generateSyntheticBatch (batchSize : Nat) (seed : Nat := 0) : Batch :=
  let images := Array.range (batchSize * imageSize) |>.map fun i =>
    -- Simple pattern: alternate between 0.0 and 1.0 based on position
    if (i + seed) % 3 == 0 then 0.8 else 0.2
  let labels := Array.range batchSize |>.map fun i =>
    (i + seed) % numClasses
  { images, labels, batchSize }

/-- Convert labels to one-hot encoded vectors -/
def labelsToOneHot (labels : Array Nat) : Array Float :=
  let size := labels.size * numClasses
  let result := Array.mk (List.replicate size 0.0)
  (Array.range labels.size).foldl (init := result) fun acc idx =>
    let label := labels[idx]!
    acc.set! (idx * numClasses + label) 1.0

/-- Normalize pixel values to [0, 1] range -/
def normalizeImages (images : Array Float) : Array Float :=
  images.map fun pixel => pixel / 255.0

/-- Split batch into mini-batches -/
def createMiniBatches (batch : Batch) (miniBatchSize : Nat) : Array Batch :=
  let numBatches := (batch.batchSize + miniBatchSize - 1) / miniBatchSize
  Array.range numBatches |>.map fun i =>
    let startIdx := i * miniBatchSize
    let endIdx := min (startIdx + miniBatchSize) batch.batchSize
    let actualSize := endIdx - startIdx
    let images := batch.images.extract (startIdx * imageSize) (endIdx * imageSize)
    let labels := batch.labels.extract startIdx endIdx
    { images, labels, batchSize := actualSize }

/-- Simple accuracy calculation -/
def calculateAccuracy (predictions : Array Float) (labels : Array Nat) (batchSize : Nat) : Float :=
  let correct := Array.range batchSize |>.foldl (init := 0) fun acc i =>
    -- Find argmax of predictions for this sample
    let predStart := i * numClasses
    let predSlice := Array.range numClasses |>.map fun j => predictions[predStart + j]!
    let predClass := (Array.range numClasses).foldl (init := (0, predSlice[0]!)) fun (maxIdx, maxVal) j =>
      let val := predSlice[j]!
      if val > maxVal then (j, val) else (maxIdx, maxVal)
    let predLabel := predClass.1
    if predLabel == labels[i]! then acc + 1 else acc
  correct.toFloat / batchSize.toFloat

/-- Print batch statistics -/
def printBatchInfo (batch : Batch) : IO Unit := do
  IO.println s!"Batch size: {batch.batchSize}"
  IO.println s!"Images shape: [{batch.batchSize}, {imageSize}]"
  IO.println s!"Labels shape: [{batch.labels.size}]"
  IO.println s!"First 10 labels: {batch.labels.extract 0 (min 10 batch.labels.size)}"

end Examples.MachineLearning.MNISTData
