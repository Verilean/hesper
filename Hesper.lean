import Hesper.Basic

-- WGSL DSL modules
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Shader
import Hesper.WGSL.Kernel
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WGSL.Execute
import Hesper.WGSL.Helpers

-- Profiling modules
import Hesper.Profile
import Hesper.Profile.Trace

-- WebGPU API modules
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline

-- High-level Compute API
import Hesper.Compute

-- Core abstractions
-- Note: VerifiedOp and VerifiedOpFusion both define TensorData, so we only import one
-- import Hesper.Core.VerifiedOp  -- Original immediate execution (use for legacy code)
import Hesper.Core.VerifiedOpFusion  -- New fusion-enabled abstraction

-- Core array types (opaque, zero-copy, in-place mutable)
import Hesper.Core.Float32Array
import Hesper.Core.Float16Array
-- import Hesper.Core.BFloat16Array  -- TODO: Implement C++ side

-- Tensor operations
import Hesper.Tensor.Types
import Hesper.Tensor.MatMul

-- Verified operators
-- Note: MatMul uses old VerifiedOp, MatMulFusion uses new VerifiedOpFusion
-- import Hesper.Op.MatMul  -- Legacy immediate execution (use VerifiedOpDemo for examples)
import Hesper.Op.MatMulFusion  -- Fusion-enabled matmul
import Hesper.Op.Activation

-- Neural network operations
import Hesper.NN.Activation
import Hesper.NN.Conv
import Hesper.NN.MLP

-- Automatic differentiation
import Hesper.AD.Reverse

-- Optimizers
import Hesper.Optimizer.SGD
import Hesper.Optimizer.Adam

-- Async operations
import Hesper.Async

-- SIMD CPU Backend
import Hesper.Simd
import Hesper.Float32
import Hesper.Float16

-- GLFW window management and rendering
import Hesper.GLFW.Types
import Hesper.GLFW.Internal
import Hesper.GLFW

/-!
# Hesper: Verified WebGPU Inference Engine

This module serves as the root of the Hesper library, a type-safe GPU programming framework
for Lean 4 with formal verification capabilities.

## Features

- Type-Safe WGSL DSL: Embedded shader language with compile-time type checking
- WebGPU Backend: Cross-platform GPU compute via Google Dawn (Vulkan, Metal, D3D12)
- Verified Computation: Numerical accuracy testing and formal verification support
- High-Performance: Matrix multiplication, neural networks, automatic differentiation
- Multi-Backend: GPU compute with optional SIMD CPU fallback

## Module Organization

- Core: Verified operator pattern (CPU spec + GPU impl)
- WGSL: Type-safe shader DSL and code generation
- WebGPU: Low-level WebGPU API bindings
- Compute: High-level compute API
- Tensor: Linear algebra operations (MatMul)
- Op: Verified operator instances (MatMul, etc.)
- NN: Neural network layers (Conv, Activation)
- AD: Automatic differentiation
- Optimizer: Training optimizers (SGD, Adam)
- Profile: Chrome tracing and performance profiling
- Simd: CPU SIMD backend (Google Highway)
- GLFW: Window management and rendering

## Production Readiness

- 151 DSL test cases (operators, control flow, functions)
- CPU vs GPU numerical accuracy tests
- Cross-platform CI/CD (Linux/Vulkan, Windows/D3D12, macOS/Metal)

## References

- Organization: Verilean (github.com/verilean)
- Repository: github.com/verilean/hesper
-/

namespace Hesper

/-- Initialize the Hesper WebGPU engine.

This function must be called before any GPU operations. It performs the following:
1. Initializes the Dawn WebGPU implementation and procedure table
2. Creates a Dawn native instance
3. Discovers all available GPU adapters on the system (Vulkan, Metal, D3D12)
4. Prints adapter information to stdout

**Returns**: A WebGPU Instance handle for subsequent GPU operations.

**Example**:

    def main : IO Unit := do
      let inst ← Hesper.init
      -- Use inst for GPU operations

**Backend Support**:
- **Linux**: Vulkan 1.3+
- **macOS**: Metal 3
- **Windows**: D3D12

**Note**: This is an FFI function implemented in `native/bridge.cpp`.
-/
@[extern "lean_hesper_init"]
opaque init : IO WebGPU.Instance

/-- Run GPU vector addition (Hello World compute example).

Demonstrates basic GPU compute by adding two vectors element-wise:
`C[i] = A[i] + B[i]` for i in 0..size

**Parameters**:
- `inst`: WebGPU instance from `Hesper.init`
- `size`: Number of elements in each vector (must be positive)

**Implementation**: Creates GPU buffers, uploads random test data, executes a compute
shader, and downloads results for verification.

**Example**:

    def main : IO Unit := do
      let inst ← Hesper.init
      Hesper.vectorAdd inst 1024  -- Add two 1024-element vectors

**Performance**: Executed on GPU with workgroup size of 64 threads.

**Note**: This is an FFI function implemented in `native/bridge.cpp`.
It's primarily for testing and demonstration purposes.
-/
@[extern "lean_hesper_vector_add"]
opaque vectorAdd (inst : @& WebGPU.Instance) (size : UInt32) : IO Unit

end Hesper
