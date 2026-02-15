import Hesper.GGUF.Types
import Hesper.GGUF.Parser
import Hesper.GGUF.Quantization

/-!
# GGUF Reader - High-Level API

High-level interface for loading and accessing GGUF files.

## Usage Example
```lean
def main : IO Unit := do
  -- Load GGUF file
  let gguf ← loadGGUF "model.gguf"

  -- Query model info
  IO.println s!"Architecture: {gguf.getMetadataString "general.architecture"}"
  IO.println s!"Tensors: {gguf.header.tensorCount}"

  -- Get tensor by name
  match gguf.findTensor "blk.0.attn_q.weight" with
  | some ti =>
    IO.println s!"Found tensor: {ti.name}, shape: {ti.dimensions}"
    let data ← gguf.getTensorFloat32 ti
    IO.println s!"First 10 values: {data.extract 0 10}"
  | none => IO.println "Tensor not found"
```
-/

namespace Hesper.GGUF

/-! ## File Loading -/

/-- Load GGUF file from disk -/
def loadGGUF (path : String) : IO GGUFFile := do
  -- Read entire file into ByteArray
  let fileData ← IO.FS.readBinFile path

  -- Parse GGUF structure
  match Parser.parseGGUF fileData with
  | .ok gguf => pure gguf
  | .error msg => throw <| IO.userError s!"Failed to parse GGUF file: {msg}"

/-! ## Extended GGUFFile API -/

namespace GGUFFile

/-- Get metadata value as String (if it exists and is a string) -/
def getMetadataString (gf : GGUFFile) (key : String) : Option String :=
  match gf.findMetadata key with
  | some mv =>
    match mv.valueType with
    | .MString => String.fromUTF8? mv.data
    | _ => none
  | none => none

/-- Get metadata value as UInt32 -/
def getMetadataUInt32 (gf : GGUFFile) (key : String) : Option UInt32 :=
  match gf.findMetadata key with
  | some mv =>
    match mv.valueType with
    | .MUInt32 =>
      if mv.data.size >= 4 then
        let b0 := mv.data.get! 0
        let b1 := mv.data.get! 1
        let b2 := mv.data.get! 2
        let b3 := mv.data.get! 3
        some <| b0.toUInt32 ||| (b1.toUInt32 <<< 8) |||
                (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
      else
        none
    | _ => none
  | none => none

/-- Get metadata value as UInt64 -/
def getMetadataUInt64 (gf : GGUFFile) (key : String) : Option UInt64 :=
  match gf.findMetadata key with
  | some mv =>
    match mv.valueType with
    | .MUInt64 =>
      if mv.data.size >= 8 then
        let b0 := mv.data.get! 0
        let b1 := mv.data.get! 1
        let b2 := mv.data.get! 2
        let b3 := mv.data.get! 3
        let b4 := mv.data.get! 4
        let b5 := mv.data.get! 5
        let b6 := mv.data.get! 6
        let b7 := mv.data.get! 7
        some <| b0.toUInt64 ||| (b1.toUInt64 <<< 8) |||
                (b2.toUInt64 <<< 16) ||| (b3.toUInt64 <<< 24) |||
                (b4.toUInt64 <<< 32) ||| (b5.toUInt64 <<< 40) |||
                (b6.toUInt64 <<< 48) ||| (b7.toUInt64 <<< 56)
      else
        none
    | _ => none
  | none => none

/-- Get model architecture string -/
def getArchitecture (gf : GGUFFile) : Option String :=
  gf.getMetadataString "general.architecture"

/-- Get model name -/
def getModelName (gf : GGUFFile) : Option String :=
  gf.getMetadataString "general.name"

/-- Get number of layers -/
def getLayerCount (gf : GGUFFile) : Option UInt32 :=
  match gf.getArchitecture with
  | some arch => gf.getMetadataUInt32 s!"{arch}.block_count"
  | none => none

/-- Get hidden dimension size -/
def getHiddenSize (gf : GGUFFile) : Option UInt32 :=
  match gf.getArchitecture with
  | some arch => gf.getMetadataUInt32 s!"{arch}.embedding_length"
  | none => none

/-- Get attention head count -/
def getAttentionHeadCount (gf : GGUFFile) : Option UInt32 :=
  match gf.getArchitecture with
  | some arch => gf.getMetadataUInt32 s!"{arch}.attention.head_count"
  | none => none

/-- Get tensor data as Float32 array (automatically dequantizes) -/
def getTensorFloat32 (gf : GGUFFile) (ti : TensorInfo) : IO (Array Float) := do
  let rawData := gf.getTensorData ti
  let numElements := ti.numElements.toNat

  match Quantization.dequantize rawData ti.ggmlType numElements with
  | .ok arr => pure arr
  | .error msg => throw <| IO.userError s!"Failed to dequantize tensor '{ti.name}': {msg}"

/-- Get tensor data as raw ByteArray (no dequantization)

    ⚠️ **Use this for GPU inference!**
    This preserves the quantized format (TQ2_0, Q4_0, etc.) for direct GPU upload.
    Decode on-the-fly in WGSL compute shaders for optimal performance.
-/
def getTensorRaw (gf : GGUFFile) (ti : TensorInfo) : ByteArray :=
  gf.getTensorData ti

/-! ## GPU Upload API -/

/-- Upload quantized tensor directly to GPU buffer

    This is the **recommended approach** for inference with quantized models.

    **Performance benefits:**
    - ✅ 16x less PCIe bandwidth (for TQ2_0: 2-bit vs 32-bit)
    - ✅ Decode on-the-fly in GPU shaders (kernel fusion)
    - ✅ No intermediate CPU memory allocation

    **Example usage:**
    ```lean
    let gguf ← loadGGUF "model.gguf"
    let weightTensor := gguf.findTensor "blk.0.attn_q.weight" |>.get!
    let gpuBuf ← gguf.uploadTensorQuantized device weightTensor
    -- Now use gpuBuf in BitLinear layer
    ```

    @param device WebGPU device
    @param ti Tensor info
    @return GPU buffer containing quantized data
-/
def uploadTensorQuantized (gf : GGUFFile) (device : α) (ti : TensorInfo) : IO β := do
  -- This is a placeholder - will be implemented when WebGPU types are imported
  -- Actual implementation:
  -- let rawData := gf.getTensorRaw ti
  -- let buf ← createBuffer device {
  --   size := rawData.size.toUSize
  --   usage := [.storage, .copyDst]
  --   mappedAtCreation := false
  -- }
  -- writeBuffer device buf 0 rawData
  -- pure buf
  throw $ IO.userError "uploadTensorQuantized: WebGPU integration not yet available in this module"

/-- Find tensor by pattern (substring match) -/
def findTensorsByPattern (gf : GGUFFile) (pattern : String) : Array TensorInfo :=
  gf.tensors.filter fun ti =>
    -- Simple substring check: pattern is in name
    ti.name.startsWith pattern || (pattern.toList ⊆ ti.name.toList)

/-- Get all tensors of a specific layer -/
def getLayerTensors (gf : GGUFFile) (layerIdx : Nat) : Array TensorInfo :=
  gf.tensors.filter fun ti =>
    ti.name.startsWith s!"blk.{layerIdx}."

/-- Print file summary -/
def printSummary (gf : GGUFFile) : IO Unit := do
  IO.println "=== GGUF File Summary ==="
  IO.println s!"Version: {gf.header.version}"
  IO.println s!"Tensors: {gf.header.tensorCount}"
  IO.println s!"Metadata entries: {gf.header.metadataKVCount}"

  -- Print architecture info
  match gf.getArchitecture with
  | some arch => IO.println s!"Architecture: {arch}"
  | none => IO.println "Architecture: unknown"

  match gf.getModelName with
  | some name => IO.println s!"Model: {name}"
  | none => pure ()

  match gf.getLayerCount with
  | some count => IO.println s!"Layers: {count}"
  | none => pure ()

  match gf.getHiddenSize with
  | some size => IO.println s!"Hidden size: {size}"
  | none => pure ()

  -- Print tensor types histogram (simplified without HashMap)
  IO.println "\n=== Tensor Types ==="
  let uniqueTypes := gf.tensors.map (·.ggmlType) |>.toList.eraseDups
  for typ in uniqueTypes do
    let count := gf.tensors.filter (·.ggmlType == typ) |>.size
    IO.println s!"  {typ}: {count} tensors"

  -- Print sample tensor names
  IO.println "\n=== Sample Tensors ==="
  for ti in gf.tensors.extract 0 (min 10 gf.tensors.size) do
    IO.println s!"  {ti.name}: {ti.dimensions.toList} ({ti.ggmlType})"

  if gf.tensors.size > 10 then
    IO.println s!"  ... and {gf.tensors.size - 10} more"

end GGUFFile

/-! ## Tensor Name Mapping Utilities -/

/-- Convert HuggingFace tensor name to GGUF tensor name
    Example: "model.layers.0.self_attn.q_proj.weight" → "blk.0.attn_q.weight"
-/
def hfToGGUFTensorName (hfName : String) : Option String :=
  -- Parse layer index
  let parts := hfName.splitOn "."
  match parts with
  | ["model", "layers", layerIdx, "self_attn", component, "weight"] =>
    let ggufComponent := match component with
      | "q_proj" => "attn_q"
      | "k_proj" => "attn_k"
      | "v_proj" => "attn_v"
      | "o_proj" => "attn_output"
      | _ => component
    some s!"blk.{layerIdx}.{ggufComponent}.weight"

  | ["model", "layers", layerIdx, "mlp", component, "weight"] =>
    let ggufComponent := match component with
      | "gate_proj" => "ffn_gate"
      | "up_proj" => "ffn_up"
      | "down_proj" => "ffn_down"
      | _ => component
    some s!"blk.{layerIdx}.{ggufComponent}.weight"

  | ["model", "layers", layerIdx, "input_layernorm", "weight"] =>
    some s!"blk.{layerIdx}.attn_norm.weight"

  | ["model", "layers", layerIdx, "post_attention_layernorm", "weight"] =>
    some s!"blk.{layerIdx}.ffn_norm.weight"

  | ["model", "embed_tokens", "weight"] =>
    some "token_embd.weight"

  | ["lm_head", "weight"] =>
    some "output.weight"

  | _ => none

/-- Find tensor by HuggingFace name -/
def GGUFFile.findTensorByHFName (gf : GGUFFile) (hfName : String) : Option TensorInfo :=
  match hfToGGUFTensorName hfName with
  | some ggufName => gf.findTensor ggufName
  | none => none

end Hesper.GGUF
