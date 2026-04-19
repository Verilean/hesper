import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Tests.GoldenUnit.Common
import Tests.GoldenUnit.RMSNorm
import Tests.GoldenUnit.Linear
import Tests.GoldenUnit.Attention
import Tests.GoldenUnit.RoPE
import Tests.GoldenUnit.KVCacheWrite
import Tests.GoldenUnit.FlashAttention
import Tests.GoldenUnit.Oproj

/-!
# Gemma4 unit-test runner

Single LSpec exe `gemma4-unit-tests`.  Memory policy:
- Initialise CUDAContext ONCE (here).
- Parse GGUF ONCE (here); pass the struct to each test module.
- Each test module's helpers must free every GPU buffer they alloc.
-/

open Hesper
open Hesper.Tests.GoldenUnit.Common

unsafe def main : IO UInt32 := do
  IO.println "[Init] CUDA + GGUF..."
  let ctx ← CUDAContext.init
  let gguf ← loadGGUF
  IO.println "[Init] done.  Running tests..."
  let g1 ← Hesper.Tests.GoldenUnit.RMSNorm.allTests ctx gguf
  let g2 ← Hesper.Tests.GoldenUnit.Linear.allTests ctx gguf
  let g3 ← Hesper.Tests.GoldenUnit.Attention.allTests ctx gguf
  let g4 ← Hesper.Tests.GoldenUnit.RoPE.allTests ctx gguf
  let g5 ← Hesper.Tests.GoldenUnit.KVCacheWrite.allTests ctx gguf
  let g6 ← Hesper.Tests.GoldenUnit.FlashAttention.allTests ctx gguf
  let g7 ← Hesper.Tests.GoldenUnit.Oproj.allTests ctx gguf
  LSpec.lspecIO (.ofList (g1 ++ g2 ++ g3 ++ g4 ++ g5 ++ g6 ++ g7)) ([] : List String)
