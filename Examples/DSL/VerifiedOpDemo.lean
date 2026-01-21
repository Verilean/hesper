import Hesper
import Hesper.Core.VerifiedOp
import Hesper.Op.MatMul

/-!
# Verified Operator Pattern Demo

Demonstrates the Verified Operator Pattern in Hesper:
1. Define operators with CPU spec + GPU impl
2. Run verification tests
3. Use operators in computation graphs

## Key Concepts

**Separation of Concerns**:
- Specification (CPU): Pure math, used for proofs
- Implementation (GPU): Optimized WGSL kernels

**Verification**:
- Compare CPU vs GPU results within tolerance
- Catch GPU kernel bugs automatically

**Composability**:
- Operators compose naturally
- Build complex networks from simple ops
-/

namespace Examples.DSL.VerifiedOpDemo

open Hesper.Core
open Hesper.Op.MatMul
open Hesper.Tensor (Shape)

/-! ## Demo 1: Basic Matrix Multiplication -/

def demo_basic_matmul : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verified Operator Demo: Matrix Multiply   ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Create test matrices
  let M := 3
  let K := 4
  let N := 2

  IO.println s!"Matrix dimensions: A=[{M}×{K}], B=[{K}×{N}]"
  IO.println ""

  -- Create matrix A
  let A := testMatrix M K
  IO.println s!"Matrix A ({M}×{K}):"
  for i in [0:M] do
    let row := (List.range K).map fun j => A.data[i * K + j]!
    IO.println s!"  {row}"

  -- Create matrix B
  let B := testMatrix K N
  IO.println s!"Matrix B ({K}×{N}):"
  for i in [0:K] do
    let row := (List.range N).map fun j => B.data[i * N + j]!
    IO.println s!"  {row}"
  IO.println ""

  -- Create input
  let input : MatMulInput := {
    A := A
    B := B
  }

  -- Run CPU forward pass
  IO.println "🖥️  Running CPU forward pass (spec)..."
  let cpu_result : MatMulOutput := VerifiedOp.spec_forward input
  IO.println s!"Result C ({M}×{N}):"
  for i in [0:M] do
    let row := (List.range N).map fun j => cpu_result.C.data[i * N + j]!
    IO.println s!"  {row}"
  IO.println ""

  -- Run GPU forward pass (currently uses CPU as placeholder)
  IO.println "🎮 Running GPU forward pass (impl)..."
  let gpu_result : MatMulOutput ← VerifiedOp.impl_forward input
  IO.println s!"Result C ({M}×{N}):"
  for i in [0:M] do
    let row := (List.range N).map fun j => gpu_result.C.data[i * N + j]!
    IO.println s!"  {row}"
  IO.println ""

  -- Verify consistency
  IO.println "🔍 Verifying CPU vs GPU consistency..."
  let inst : VerifiedOp MatMulInput MatMulOutput := inferInstance
  let _ ← inst.verify_consistency input 1e-5
  IO.println ""

/-! ## Demo 2: Backward Pass (Gradients) -/

def demo_backward : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verified Operator Demo: Backward Pass     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Small test case
  let M := 2
  let K := 2
  let N := 2

  -- Create simple matrices
  let A_data := #[1.0, 2.0, 3.0, 4.0]  -- [[1, 2], [3, 4]]
  let B_data := #[0.5, 0.0, 0.0, 0.5]  -- [[0.5, 0], [0, 0.5]]

  let A : TensorData := { shape := Shape.matrix M K, data := A_data }
  let B : TensorData := { shape := Shape.matrix K N, data := B_data }

  IO.println "Matrix A:"
  IO.println "  [1.0, 2.0]"
  IO.println "  [3.0, 4.0]"
  IO.println ""
  IO.println "Matrix B:"
  IO.println "  [0.5, 0.0]"
  IO.println "  [0.0, 0.5]"
  IO.println ""

  let input : MatMulInput := {
    A := A
    B := B
  }

  -- Forward pass
  let output : MatMulOutput := VerifiedOp.spec_forward input
  IO.println "Forward: C = A @ B ="
  for i in [0:M] do
    let row := (List.range N).map fun j => output.C.data[i * N + j]!
    IO.println s!"  {row}"
  IO.println ""

  -- Backward pass: assume gradient of loss w.r.t. C is all ones
  let grad_C_data := Array.mk (List.replicate (M * N) 1.0)
  let grad_output : MatMulOutput := {
    C := { shape := Shape.matrix M N, data := grad_C_data }
  }

  IO.println "Gradient of loss w.r.t. C (all ones):"
  IO.println "  [1.0, 1.0]"
  IO.println "  [1.0, 1.0]"
  IO.println ""

  -- Compute gradients
  IO.println "🔙 Computing backward pass..."
  let grads := VerifiedOp.spec_backward input grad_output

  IO.println "Gradient w.r.t. A (dL/dA = dL/dC @ B^T):"
  for i in [0:M] do
    let row := (List.range K).map fun j => grads.A.data[i * K + j]!
    IO.println s!"  {row}"
  IO.println ""

  IO.println "Gradient w.r.t. B (dL/dB = A^T @ dL/dC):"
  for i in [0:K] do
    let row := (List.range N).map fun j => grads.B.data[i * N + j]!
    IO.println s!"  {row}"
  IO.println ""

/-! ## Demo 3: Identity Matrix Test -/

def demo_identity : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verified Operator Demo: Identity Matrix   ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Test: A @ I = A
  let n := 3
  let A := testMatrix n n
  let I := identityMatrix n

  IO.println s!"Testing: A @ I = A (n={n})"
  IO.println ""

  let input : MatMulInput := {
    A := A
    B := I
  }

  let result : MatMulOutput := VerifiedOp.spec_forward input

  IO.println "Matrix A:"
  for i in [0:n] do
    let row := (List.range n).map fun j => A.data[i * n + j]!
    IO.println s!"  {row}"
  IO.println ""

  IO.println "Result A @ I:"
  for i in [0:n] do
    let row := (List.range n).map fun j => result.C.data[i * n + j]!
    IO.println s!"  {row}"
  IO.println ""

  -- Verify they match
  let isEqual := A.approxEq result.C 1e-5
  if isEqual then
    IO.println "✅ Success: A @ I = A"
  else
    IO.println "❌ Failed: A @ I ≠ A"
  IO.println ""

/-! ## Main Demo Runner -/

def main : IO Unit := do
  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Hesper Verified Operator Pattern"
  IO.println "  CPU Specification + GPU Implementation"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Initialize WebGPU (for future GPU demos)
  IO.println "🚀 Initializing WebGPU..."
  let _ ← Hesper.init
  IO.println ""

  -- Run demos
  demo_basic_matmul
  demo_backward
  demo_identity

  IO.println "═══════════════════════════════════════════════"
  IO.println "  All demos completed!"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  IO.println "📝 Next Steps:"
  IO.println "  1. Implement GPU kernels (WGSL) for matmul"
  IO.println "  2. Add more verified operators (conv, pooling)"
  IO.println "  3. Build neural networks from composed ops"
  IO.println "  4. Prove correctness properties in Lean"
  IO.println ""

end Examples.DSL.VerifiedOpDemo

-- Top-level main to export
def main : IO Unit := Examples.DSL.VerifiedOpDemo.main
