import Hesper.Core.VerifiedOp
import Hesper.Tensor.Types
import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer

/-!
# Matrix Multiplication Verified Operator

Implements matrix multiplication as a `VerifiedOp` instance with:
- CPU specification for correctness
- GPU implementation via WGSL (placeholder for now)
- Backward pass for automatic differentiation

## Mathematical Definition

Forward: C = A @ B
  - A: [M × K] matrix
  - B: [K × N] matrix
  - C: [M × N] matrix
  - C[i,j] = Σₖ A[i,k] * B[k,j]

Backward:
  - Given: dL/dC (gradient of loss w.r.t. output)
  - Compute: dL/dA = dL/dC @ Bᵀ
  - Compute: dL/dB = Aᵀ @ dL/dC
-/

namespace Hesper.Op.MatMul

open Hesper.Core
open Hesper.Tensor
open Hesper.WebGPU

/-! ## Input/Output Types -/

/-- Matrix multiplication input: two matrices A and B -/
structure MatMulInput where
  /-- Left matrix A: [M × K] -/
  A : TensorData
  /-- Right matrix B: [K × N] -/
  B : TensorData
  deriving Inhabited

/-- Matrix multiplication output: result matrix C -/
structure MatMulOutput where
  /-- Result matrix C: [M × N] -/
  C : TensorData
  deriving Inhabited

/-- GPU matrix multiplication input -/
structure MatMulInputGPU where
  /-- Left matrix A on GPU -/
  A : GPUHandle
  /-- Right matrix B on GPU -/
  B : GPUHandle

/-- GPU matrix multiplication output -/
structure MatMulOutputGPU where
  /-- Result matrix C on GPU -/
  C : GPUHandle

/-! ## CPU Specification (Reference Implementation) -/

/-- Naive matrix multiplication on CPU.
    Used as the mathematical specification.
    Time complexity: O(M * N * K) -/
def cpuMatMul (input : MatMulInput) : MatMulOutput :=
  let M := input.A.shape.dims[0]!
  let K := input.A.shape.dims[1]!
  let N := input.B.shape.dims[1]!

  -- Compute C[i,j] = Σₖ A[i,k] * B[k,j]
  let C_data := Array.range (M * N) |>.map fun idx =>
    let i := idx / N
    let j := idx % N
    Array.range K |>.foldl (fun acc k =>
      let a_ik := input.A.data[i * K + k]!
      let b_kj := input.B.data[k * N + j]!
      acc + a_ik * b_kj
    ) 0.0

  { C := { shape := Shape.matrix M N, data := C_data } }

/-- Backward pass for matrix multiplication (CPU).

    Given dL/dC, compute:
    - dL/dA = dL/dC @ Bᵀ
    - dL/dB = Aᵀ @ dL/dC

    For simplicity, we return dL/dA in the same input structure.
    In a full implementation, this would return both gradients. -/
def cpuMatMulBackward (input : MatMulInput) (grad_output : MatMulOutput) : MatMulInput :=
  let M := input.A.shape.dims[0]!
  let K := input.A.shape.dims[1]!
  let N := input.B.shape.dims[1]!

  -- Compute dL/dA = dL/dC @ Bᵀ
  -- dA[i,k] = Σⱼ dC[i,j] * B[k,j]
  let dA_data := Array.range (M * K) |>.map fun idx =>
    let i := idx / K
    let k := idx % K
    Array.range N |>.foldl (fun acc j =>
      let dc_ij := grad_output.C.data[i * N + j]!
      let b_kj := input.B.data[k * N + j]!
      acc + dc_ij * b_kj
    ) 0.0

  -- Compute dL/dB = Aᵀ @ dL/dC
  -- dB[k,j] = Σᵢ A[i,k] * dC[i,j]
  let dB_data := Array.range (K * N) |>.map fun idx =>
    let k := idx / N
    let j := idx % N
    Array.range M |>.foldl (fun acc i =>
      let a_ik := input.A.data[i * K + k]!
      let dc_ij := grad_output.C.data[i * N + j]!
      acc + a_ik * dc_ij
    ) 0.0

  -- Return gradients in input structure format
  { A := { shape := input.A.shape, data := dA_data }
    B := { shape := input.B.shape, data := dB_data } }

/-! ## GPU Implementation (Optimized WGSL Kernels) -/

/-- GPU matrix multiplication using WGSL compute shaders.

    TODO: Integrate with existing WGSL matmul from Examples/MainMatmul.lean
    For now, this is a placeholder that:
    1. Uploads input to GPU
    2. Performs computation (currently falling back to CPU for simplicity)
    3. Downloads result

    Future: Use subgroup matrix operations or tiled WGSL kernel -/
def gpuMatMul (input : MatMulInput) : IO MatMulOutput := do
  -- Placeholder: In production, this would:
  -- 1. Get device
  -- 2. Upload A, B to GPU buffers
  -- 3. Create WGSL compute shader
  -- 4. Execute shader
  -- 5. Download result

  -- For now, just use CPU spec
  return cpuMatMul input

/-- GPU backward pass for matrix multiplication.

    TODO: Custom WGSL kernel for backpropagation
    Future optimizations:
    - Fused gradient kernels
    - Tiled memory access
    - Subgroup operations -/
def gpuMatMulBackward (input : MatMulInput) (grad_output : MatMulOutput) : IO MatMulInput := do
  -- Placeholder: Use CPU backward for now
  return cpuMatMulBackward input grad_output

/-! ## VerifiedOp Instance -/

instance : VerifiedOp MatMulInput MatMulOutput where
  spec_forward := cpuMatMul
  impl_forward := gpuMatMul
  spec_backward := cpuMatMulBackward
  impl_backward := gpuMatMulBackward

  verify_consistency := fun input tolerance => do
    -- Run both CPU and GPU forward passes
    let cpu_output := cpuMatMul input
    let gpu_output ← gpuMatMul input

    -- Compare results
    let isMatch := cpu_output.C.approxEq gpu_output.C tolerance

    if isMatch then
      IO.println s!"  ✓ CPU and GPU results match (tolerance: {tolerance})"
    else
      IO.println s!"  ✗ CPU and GPU results differ!"
      -- Print some debug info
      IO.println s!"    CPU result (first 10): {cpu_output.C.data.toList.take 10}"
      IO.println s!"    GPU result (first 10): {gpu_output.C.data.toList.take 10}"

    return isMatch

/-! ## Helper Functions -/

/-- Create a matrix multiplication input from raw data and dimensions -/
def mkMatMulInput (M K N : Nat) (A_data B_data : Array Float)
    (h : A_data.size = M * K ∧ B_data.size = K * N) : MatMulInput :=
  { A := { shape := Shape.matrix M K, data := A_data }
    B := { shape := Shape.matrix K N, data := B_data } }

/-- Create identity matrix as TensorData -/
def identityMatrix (n : Nat) : TensorData :=
  let data := Array.range (n * n) |>.map fun idx =>
    let i := idx / n
    let j := idx % n
    if i == j then 1.0 else 0.0
  { shape := Shape.matrix n n, data := data }

/-- Create a simple test matrix with sequential values -/
def testMatrix (m n : Nat) : TensorData :=
  let data := Array.range (m * n) |>.map fun idx =>
    (idx.toFloat + 1.0) / 10.0
  { shape := Shape.matrix m n, data := data }

end Hesper.Op.MatMul
