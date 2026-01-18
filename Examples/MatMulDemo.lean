import Hesper.Tensor.Types
import Hesper.Tensor.MatMul

/-!
# Matrix Multiplication Demo

Demonstrates GPU matrix multiplication using both naive and tiled implementations.
This example shows shader generation only (no actual GPU execution).
-/

namespace Examples.MatMulDemo

open Hesper.Tensor
open Hesper.Tensor.MatMul

/-- Demo: Small matrix multiplication (4×4 × 4×4) -/
def demo1_small : IO Unit := do
  IO.println "=== Demo 1: Small Matrix Multiplication (4×4 × 4×4) ==="
  IO.println ""

  let config : MatMulConfig := {
    M := 4
    K := 4
    N := 4
    tileSize := 2
  }

  IO.println s!"Matrix A: {config.M} × {config.K}"
  IO.println s!"Matrix B: {config.K} × {config.N}"
  IO.println s!"Matrix C: {config.M} × {config.N}"
  IO.println s!"Tile Size: {config.tileSize}"
  IO.println ""

  -- Generate naive version
  IO.println "--- Naive Matrix Multiplication Shader ---"
  let naiveShader := generateNaiveMatMulShader config
  IO.println naiveShader
  IO.println ""

/-- Demo: Medium matrix multiplication (64×64 × 64×64) -/
def demo2_medium : IO Unit := do
  IO.println "=== Demo 2: Medium Matrix Multiplication (64×64 × 64×64) ==="
  IO.println ""

  let config : MatMulConfig := {
    M := 64
    K := 64
    N := 64
    tileSize := 8
  }

  IO.println s!"Matrix A: {config.M} × {config.K}"
  IO.println s!"Matrix B: {config.K} × {config.N}"
  IO.println s!"Matrix C: {config.M} × {config.N}"
  IO.println s!"Tile Size: {config.tileSize}"
  let (wgX, wgY, wgZ) := config.numWorkgroups
  IO.println s!"Workgroups: {wgX} × {wgY} × {wgZ}"
  IO.println ""

  -- Generate tiled version
  IO.println "--- Tiled Matrix Multiplication Shader ---"
  let tiledShader := generateTiledMatMulShader config
  IO.println tiledShader
  IO.println ""

/-- Demo: Rectangular matrix multiplication (128×256 × 256×512) -/
def demo3_rectangular : IO Unit := do
  IO.println "=== Demo 3: Rectangular Matrix Multiplication ==="
  IO.println ""

  let config : MatMulConfig := {
    M := 128
    K := 256
    N := 512
    tileSize := 16
  }

  IO.println s!"Matrix A: {config.M} × {config.K}"
  IO.println s!"Matrix B: {config.K} × {config.N}"
  IO.println s!"Matrix C: {config.M} × {config.N}"
  IO.println s!"Tile Size: {config.tileSize}"
  let (wgX, wgY, wgZ) := config.numWorkgroups
  IO.println s!"Workgroups: {wgX} × {wgY} × {wgZ}"
  IO.println ""

  IO.println "Generated shader structure:"
  IO.println "  - 3 storage buffers (matrixA, matrixB, matrixC)"
  IO.println "  - 2 shared memory tiles for optimization"
  IO.println "  - Workgroup synchronization barriers"
  IO.println "  - Tiled computation for memory efficiency"
  IO.println ""

/-- Demo: Shape and configuration information -/
def demo4_shapes : IO Unit := do
  IO.println "=== Demo 4: Tensor Shapes and Configurations ==="
  IO.println ""

  -- Various tensor shapes
  let vec := Shape.vector 1024
  let mat := Shape.matrix 32 64
  let tensor3 := Shape.tensor3D 8 8 8
  let tensor4 := Shape.tensor4D 2 4 8 16

  IO.println s!"Vector shape: {vec.dims}, size: {vec.size} elements"
  IO.println s!"Matrix shape: {mat.dims}, size: {mat.size} elements"
  IO.println s!"3D Tensor shape: {tensor3.dims}, size: {tensor3.size} elements"
  IO.println s!"4D Tensor shape: {tensor4.dims}, size: {tensor4.size} elements"
  IO.println ""

  -- Tensor descriptors with data types
  let f32Vec := TensorDesc.vector 1024 .f32
  let f16Mat := TensorDesc.matrix 32 64 .f16
  let i32Mat := TensorDesc.matrix 16 16 .i32

  IO.println s!"f32 vector: {f32Vec.sizeBytes} bytes"
  IO.println s!"f16 matrix: {f16Mat.sizeBytes} bytes"
  IO.println s!"i32 matrix: {i32Mat.sizeBytes} bytes"
  IO.println ""

  -- MatMul configurations
  let configs := [
    { M := 32, K := 32, N := 32, tileSize := 8 : MatMulConfig },
    { M := 64, K := 64, N := 64, tileSize := 16 : MatMulConfig },
    { M := 128, K := 128, N := 128, tileSize := 16 : MatMulConfig },
    { M := 256, K := 256, N := 256, tileSize := 16 : MatMulConfig }
  ]

  IO.println "Matrix Multiplication Configurations:"
  for config in configs do
    let (wgX, wgY, _) := config.numWorkgroups
    let totalThreads := wgX * wgY * config.tileSize * config.tileSize
    let msg := s!"  {config.M}×{config.K} × {config.K}×{config.N}: " ++
               s!"{wgX}×{wgY} workgroups, {totalThreads} threads"
    IO.println msg
  IO.println ""

/-- Demo: Performance considerations -/
def demo5_performance : IO Unit := do
  IO.println "=== Demo 5: Performance Considerations ==="
  IO.println ""

  IO.println "Naive vs Tiled Matrix Multiplication:"
  IO.println ""

  IO.println "**Naive Implementation:**"
  IO.println "  - Each thread computes one output element"
  IO.println "  - Loads data directly from global memory"
  IO.println "  - Simple but high memory bandwidth"
  IO.println "  - Good for: small matrices, debugging"
  IO.println ""

  IO.println "**Tiled Implementation:**"
  IO.println "  - Uses workgroup shared memory"
  IO.println "  - Loads tiles cooperatively"
  IO.println "  - Reduces global memory traffic"
  IO.println "  - Requires synchronization barriers"
  IO.println "  - Good for: large matrices, production"
  IO.println ""

  IO.println "Memory Access Patterns:"
  let config : MatMulConfig := { M := 1024, K := 1024, N := 1024, tileSize := 16 }
  let naiveAccesses := config.M * config.N * config.K * 2  -- 2 loads per multiply-add
  let tiledAccesses := config.M * config.N * config.K * 2 / config.tileSize  -- Reduced by tile reuse

  IO.println s!"  Naive global memory accesses: ~{naiveAccesses / 1000000}M"
  IO.println s!"  Tiled global memory accesses: ~{tiledAccesses / 1000000}M"
  IO.println s!"  Reduction factor: ~{naiveAccesses / tiledAccesses}×"
  IO.println ""

-- Run all examples
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper Matrix Multiplication Demo         ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  demo1_small
  demo2_medium
  demo3_rectangular
  demo4_shapes
  demo5_performance

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Matrix multiplication demos complete!      ║"
  IO.println "╚══════════════════════════════════════════════╝"

end Examples.MatMulDemo

-- Top-level main for executable
def main : IO Unit := Examples.MatMulDemo.main
