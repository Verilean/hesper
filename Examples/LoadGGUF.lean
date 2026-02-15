import Hesper.GGUF

/-!
# GGUF File Loader Example

This example demonstrates how to:
1. Load a GGUF file from disk
2. Print model metadata (architecture, layer count, etc.)
3. List tensor information
4. Extract and dequantize a tensor

## Usage
```bash
lake build Examples.LoadGGUF
lake env ./build/bin/LoadGGUF <path-to-gguf-file>
```

## Example
```bash
lake env ./build/bin/LoadGGUF data/gguf/ggml-model-i2_s.gguf
```
-/

open Hesper.GGUF

def main (args : List String) : IO Unit := do
  match args with
  | [path] => do
    IO.println s!"Loading GGUF file: {path}"

    -- Check if file exists
    let filePath := System.FilePath.mk path
    let fileExists ← filePath.pathExists
    if !fileExists then
      IO.eprintln s!"Error: File not found: {path}"
      IO.Process.exit 1

    -- Load GGUF file
    let gguf ← loadGGUF path

    -- Print summary
    gguf.printSummary

    -- Try to extract first tensor
    if gguf.tensors.size > 0 then
      let ti := gguf.tensors[0]!
      IO.println s!"\n=== Extracting first tensor: {ti.name} ==="
      IO.println s!"Shape: {ti.dimensions.toList}"
      IO.println s!"Type: {ti.ggmlType}"
      IO.println s!"Elements: {ti.numElements}"

      -- For small tensors, try to dequantize
      if ti.numElements < 1000 then
        try
          let data ← gguf.getTensorFloat32 ti
          let displaySize := min 10 data.size
          let firstValues := data.extract 0 displaySize
          IO.println s!"First {displaySize} values: {firstValues.toList}"
        catch e =>
          IO.println s!"Could not dequantize: {e}"
      else
        IO.println "(Tensor too large to display)"

    IO.println "\n✅ GGUF file loaded successfully!"

  | _ => do
    IO.println "Usage: LoadGGUF <path-to-gguf-file>"
    IO.println "Example: LoadGGUF data/gguf/ggml-model-i2_s.gguf"
    IO.Process.exit 1
