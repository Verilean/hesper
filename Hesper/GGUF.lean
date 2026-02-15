import Hesper.GGUF.Types
import Hesper.GGUF.Parser
import Hesper.GGUF.Quantization
import Hesper.GGUF.Reader

/-!
# GGUF Module

GGUF (Generic Graph Universal Format) parser and utilities for BitNet model loading.

## Quick Start

```lean
import Hesper.GGUF

def main : IO Unit := do
  -- Load model
  let gguf ← GGUF.loadGGUF "path/to/model.gguf"

  -- Print summary
  gguf.printSummary

  -- Load first layer attention weights
  match gguf.findTensor "blk.0.attn_q.weight" with
  | some ti =>
    let weights ← gguf.getTensorFloat32 ti
    IO.println s!"Loaded {weights.size} weights"
  | none => IO.println "Tensor not found"
```

## Modules

- `Types`: Core data structures (GGUFFile, TensorInfo, GGMLType)
- `Parser`: Binary parsing logic
- `Quantization`: Dequantization (TQ2_0, TQ1_0)
- `Reader`: High-level API for loading and accessing tensors
-/
