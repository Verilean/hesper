import Hesper.Validation.Metrics
import Hesper.Basic

/-!
# Reference Data Loading

Utilities for loading reference outputs from llama.cpp and transformers for validation.

## Data Format

Reference data is stored as binary files containing Float32 arrays:
```
reference_data/
├── layer_0_output.bin      # Layer 0 activation output
├── layer_1_output.bin      # Layer 1 activation output
├── ...
├── final_logits.bin        # Final model output
└── metadata.json           # Shape information
```

## Generating Reference Data

Use the provided Python script to extract reference data from llama.cpp:

```python
# scripts/extract_reference_data.py
import numpy as np
import ctypes

# Instrument llama.cpp to dump intermediate tensors
# Save as Float32 binary files
```

## Usage

```lean
-- Load reference output
let refData ← loadFloatArrayFromFile "reference_data/layer_0_output.bin"

-- Compare with Hesper output
let hesperData ← computeLayer0 device model input
let report := generateReport refData hesperData

printReport report
```
-/

namespace Hesper.Validation.ReferenceData

open Hesper.Validation.Metrics

/-! ## File I/O -/

/-- Load Float32 array from binary file

    File format: Raw Float32 values (little-endian)

    @param path Path to binary file
    @return Array of Float32 values
-/
def loadFloatArrayFromFile (path : String) : IO (Array Float) := do
  -- Read file as ByteArray
  let bytes ← IO.FS.readBinFile path

  -- Convert bytes to Float32 array (little-endian)
  let numFloats := bytes.size / 4

  let mut result := #[]
  for i in [0:numFloats] do
    let offset := i * 4
    if offset + 4 <= bytes.size then
      -- Read 4 bytes as Float32 (little-endian)
      let b0 := bytes.get! offset |>.toUInt32
      let b1 := bytes.get! (offset + 1) |>.toUInt32
      let b2 := bytes.get! (offset + 2) |>.toUInt32
      let b3 := bytes.get! (offset + 3) |>.toUInt32
      let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
      let value := Hesper.Basic.float32BitsToFloat64 bits
      result := result.push value

  return result

/-- Save Float32 array to binary file

    @param path Output file path
    @param data Array of Float32 values
-/
def saveFloatArrayToFile (path : String) (data : Array Float) : IO Unit := do
  -- Convert Float32 array to bytes (little-endian)
  let mut bytes := ByteArray.empty

  for f in data do
    let bits := f.toBits.toUInt32
    let b0 := (bits &&& 0xFF).toUInt8
    let b1 := ((bits >>> 8) &&& 0xFF).toUInt8
    let b2 := ((bits >>> 16) &&& 0xFF).toUInt8
    let b3 := ((bits >>> 24) &&& 0xFF).toUInt8
    bytes := bytes.push b0 |>.push b1 |>.push b2 |>.push b3

  -- Write to file
  IO.FS.writeBinFile path bytes

/-! ## Layer-by-Layer Validation -/

structure LayerOutput where
  layerIdx : Nat
  layerName : String  -- e.g., "layer_0", "embedding", "final_norm"
  shape : List Nat    -- [batch, seq_len, hidden_dim]
  data : Array Float
  deriving Repr, Inhabited

/-- Load layer output from reference data directory

    @param dataDir Path to reference data directory
    @param layerName Layer name (e.g., "layer_0", "embedding")
    @return Layer output data
-/
def loadLayerOutput (dataDir : String) (layerName : String) : IO LayerOutput := do
  let filePath := dataDir ++ "/" ++ layerName ++ "_output.bin"

  IO.println s!"Loading reference data: {filePath}"
  let data ← loadFloatArrayFromFile filePath

  -- TODO: Load shape from metadata.json
  -- For now, return with unknown shape
  return {
    layerIdx := 0
    layerName := layerName
    shape := []  -- Will be populated from metadata
    data := data
  }

/-- Compare two layer outputs

    @param reference Reference output (from llama.cpp)
    @param hesper Output from Hesper implementation
    @param tolerance Error tolerance (default: 1e-5)
    @return Validation report
-/
def compareLayerOutputs (reference : LayerOutput) (hesper : LayerOutput) (tolerance : Float := 1e-5) : ValidationReport :=
  generateReport reference.data hesper.data tolerance

/-- Validate multiple layers

    @param referenceLayers Reference outputs
    @param hesperLayers Hesper outputs
    @param tolerance Error tolerance
    @return Array of validation reports
-/
def validateLayers (referenceLayers : Array LayerOutput) (hesperLayers : Array LayerOutput) (tolerance : Float := 1e-5) : IO (Array ValidationReport) := do
  if referenceLayers.size != hesperLayers.size then
    IO.println s!"Warning: Layer count mismatch! Reference: {referenceLayers.size}, Hesper: {hesperLayers.size}"

  let mut reports := #[]
  for i in [0:min referenceLayers.size hesperLayers.size] do
    let ref := referenceLayers[i]!
    let hesp := hesperLayers[i]!

    IO.println ""
    IO.println s!"Validating {ref.layerName} (Layer {i})..."

    let report := compareLayerOutputs ref hesp tolerance
    reports := reports.push report

    if report.matchesWithinTolerance then
      IO.println s!"  ✓ PASS (cosine sim: {report.cosineSimilarity})"
    else
      IO.println s!"  ✗ FAIL (max error: {report.maxAbsError})"

  return reports

/-! ## Summary Statistics -/

/-- Print validation summary for all layers

    @param reports Array of validation reports
-/
def printValidationSummary (reports : Array ValidationReport) : IO Unit := do
  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Validation Summary"
  IO.println "═══════════════════════════════════════════════"

  let totalLayers := reports.size
  let passedLayers := reports.filter (·.matchesWithinTolerance) |>.size
  let failedLayers := totalLayers - passedLayers

  IO.println s!"Total layers: {totalLayers}"
  IO.println s!"Passed: {passedLayers}"
  IO.println s!"Failed: {failedLayers}"
  IO.println ""

  if failedLayers == 0 then
    IO.println "✓ All layers validated successfully!"
  else
    IO.println s!"✗ {failedLayers} layer(s) failed validation"
    IO.println ""
    IO.println "Failed layers:"
    for i in [0:reports.size] do
      let report := reports[i]!
      if !report.matchesWithinTolerance then
        IO.println s!"  Layer {i}: max error = {report.maxAbsError}"

  IO.println "═══════════════════════════════════════════════"

end Hesper.Validation.ReferenceData
