# Hesper

**Write GPU programs in Lean 4. Prove them correct. Run on WebGPU.**

> [!IMPORTANT]
> **This is Alpha Software.**
> The APIs, verification features, and compiler are under active development and subject to breaking changes. While core functionality works, this project is primarily for research and experimentation.

Hesper is a verified GPU programming framework that brings the power of formal verification to GPU computing. Write type-safe shaders, execute tensor operations, and build graphics applications—all in Lean 4.

```lean
import Hesper.WGSL.DSL

-- Type-safe shader expressions with compile-time verification
let x : Exp (.scalar .f32) := var "x"
let y : Exp (.scalar .f32) := var "y"
let result := sqrt (x * x + y * y)  -- Generates: sqrt(x * x + y * y)

-- Cannot mix types (compile error!)
-- let wrong := x + (var "i" : Exp (.scalar .i32))  ✗ Type error!
```

## BitNet b1.58 Inference: 125 TPS on M4 Max

Hesper includes a complete **BitNet b1.58 2B** inference engine running entirely on WebGPU, achieving **125 tokens/second** on Apple M4 Max:

```
$ lake exe bitnet-complete --stats
> Hello, world!
Hello, world! I'm a 20-year-old college student...

Performance: 125.6 TPS (8.0 ms/token)
  Model: BitNet b1.58 2B (30 layers, 2560 dim, i2_s ternary weights)
```

**Key optimizations:**
- **Flash Attention**: fused score + online softmax + apply in 1 kernel (3 kernels → 1)
- Ternary weight kernel (i2_s): 2-bit packed weights, addition-only matmul
- Kernel fusion: fused gate+up+ReLU²×mul and fused KV cache write (150 fewer dispatches/token)
- Shared memory F16 matmul for LM head (128K vocab)
- PreparedDispatch graph capture: ~99% pipeline cache hit rate
- Command buffer batching: single GPU submit per token
- KV cache with grouped-query attention (20 heads, 5 KV heads)

**Also: 40 TPS on RTX 4070 Ti (Vulkan)**

See [bitnet.lean](https://github.com/Verilean/bitnet.lean) for the full inference pipeline.

### LoRA Finetuning (Alpaca-style Instruction Tuning)

Hesper supports LoRA finetuning with a **verified backward pass**:

```bash
# Train on Alpaca-format dataset
lake exe alpaca-finetune --model model.gguf --data alpaca_data.json --epochs 50 --rank 8

# Inference with LoRA adapter
lake exe bitnet-complete model.gguf "What is Hesper?" 60 --lora lora_weights.bin
```

**Training features:**
- Complete backward chain: 13/13 ops (attention 7 + FFN 6)
- Verified AD: each backward op numerically checked against CPU spec
- GPU ↔ CPU consistency: all backward kernels match CPU spec (error = 0.0)
- Type-safe backward chain: missing ops cause compile-time error
- AdamW optimizer with gradient clipping, LR scheduling (cosine + warmup)
- GPU-batched forward + backward (1 GPU submit per token)

### Verified Automatic Differentiation

Every backward operation is verified correct:

```bash
$ lake exe verified-ad
  PASS Softmax, RoPE, RMSNorm, ScaledDot, ReLU²×Mul  (numerical gradient check)
  PASS Chain rule composition: error = 0.0
  ✓ All AD verifications PASSED

$ lake exe gpu-vs-cpu-test
  ✓ SoftmaxBackward, RMSNormBackward, RoPEBackward, ReLU²×Mul  (GPU matches CPU spec)

$ lake exe chain-completeness
  ✓ Backward chain is COMPLETE (13/13 ops)
```

## Why Hesper?

Modern GPU programming lacks safety guarantees. Hesper provides:

- **Type Safety**: Shaders are type-checked at compile time, preventing type mismatches
- **Formal Verification**: Prove correctness properties about your GPU programs
- **Verified Training**: Backward ops numerically checked, GPU kernels match CPU specs
- **WebGPU Backend**: Cross-platform GPU access via Dawn (Metal, Vulkan, D3D12)
- **Lean Integration**: Use Lean's powerful theorem proving alongside GPU computation
- **Multi-GPU Support**: Select and coordinate across multiple GPU adapters

## Quick Start

### Prerequisites

- **Platform**: macOS (Metal), Linux (Vulkan), or Windows (D3D12/Vulkan)

### 🐳 Docker Environment (Recommended for Linux/CI)

For a reproducible build environment, especially on Linux, you can use the provided Docker image:

```bash
# Build the image
docker build -t hesper-ci .

# Run build and tests inside container
docker run -it hesper-ci lake test-all
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Verilean/hesper.git
cd hesper

# Build native dependencies (this will take a while on first build)
lake run buildNative

# Build and run a demo
lake build dsl-basics
./.lake/build/bin/dsl-basics
```

### Your First Hesper Program

Create `MyFirst.lean`:

```lean
import Hesper
import Hesper.WebGPU.Device

def main : IO Unit := do
  -- Initialize WebGPU
  Hesper.init

  -- Get a GPU device
  let device ← Hesper.WebGPU.getDevice

  IO.println "✓ GPU ready!"
```

Build and run:

```bash
lake build myfirst
./.lake/build/bin/myfirst
```

## Features

### 🚀 Portable SIMD CPU Backend (Google Highway)

Hardware-accelerated CPU operations powered by **Google Highway**, providing high-performance SIMD across x86, ARM, and RISC-V:

```lean
import Hesper.Simd
import Hesper.Float32
import Hesper.Float16

-- Float64 (8 bytes): Native Lean Float, NEON 2/op, AVX2 4/op
let a64 := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
let c64 := Hesper.Simd.simdAdd a64 b64

-- Float32 (4 bytes): 2x memory savings, NEON 4/op, AVX2 8/op
let a32 := Float32.fromFloatArray a64
let c32 := Float32.simdAdd a32 b32

-- Float16 (2 bytes): 4x memory savings, NEON 8/op, AVX2+F16C 8/op
-- Requires ARMv8.2-A FP16 or x86_64 F16C - returns error if unavailable
let hasFP16 ← Float16.hasHardwareSupport
if hasFP16 then
  let a16 ← Float16.fromFloatArray a64
  let c16 ← Float16.simdAdd a16 b16
```

**Features:**
- **Google Highway Integration**: Portable SIMD implementation with runtime dispatch
- **Architecture Support**: NEON (ARM), AVX2/AVX-512 (x86), optional FP16 vector arithmetic
- **Multi-Precision**: Optimized paths for Float64, Float32, and Float16
- **OpenMP Support**: Optional multithreading for large tensor operations

**Zero-Conversion Architecture:**
All operations work directly on raw `ByteArray` with no automatic type conversions. Conversions are explicit only when needed.

### ⚡️ High-Level Parallel API

Inspired by `webgpu-dawn`, Hesper provides an easy-to-use API for data-parallelism that handles all GPU boilerplate (buffers, shaders, synchronization) in a single call.

#### parallelFor

Quickly execute a WGSL shader over a `Float` array:

```lean
import Hesper.Compute

-- Multiply each element by 1000 on the GPU
let result ← parallelFor device shader inputData
```

#### Device.compute

Run a computation with multiple named buffers directly on the `Device`:

```lean
device.compute myKernel [("input", inputBuf), ("output", outputBuf)] config
```

### 🎯 Type-Safe Shader DSL

Write WGSL shaders with Lean's type system guaranteeing correctness:

```lean
import Hesper.WGSL.DSL

-- Expressions are typed and checked at compile time
let x : Exp (.scalar .f32) := var "x"
let y : Exp (.scalar .f32) := var "y"

-- Arithmetic operators work naturally
let distance := sqrt (x * x + y * y)

-- Built-in functions
let clamped := Exp.clamp x (lit 0.0) (lit 1.0)
let power := Exp.pow x (lit 2.0)

-- Generate WGSL code
IO.println distance.toWGSL  -- Output: sqrt((x * x) + (y * y))
```

### 🧩 Verified Composable Kernels (Operator Fusion)

Hesper's `VerifiedOpFusion` architecture allows you to compose multiple GPU operations into a single kernel pass while maintaining formal correctness:

```lean
-- Fuses MatMul and ReLU into a single GPU kernel
-- Correctness is proven by construction
let fusedOp := matmulKernel |> reluKernel
```

**Key Advantages:**
- **Zero-Copy Fusion**: Eliminate expensive memory roundtrips between kernels.
- **Formal Correctness**: Each fused kernel is verified against a high-level CPU specification (`spec_forward`).
- **Unified Interface**: Same code runs on GPU (via WGSL) or CPU (via Google Highway) for easy debugging.

### 📈 Unified Verified Automatic Differentiation

Hesper's unique architecture unifies **formal verification** with **automatic differentiation** via a shared **Differentiable** interface. This allows the AD engine to treat complex, verified GPU kernels as first-class primitives.

#### The Differentiable Interface

All operations in Hesper—from simple scalar addition to fused ResNet blocks—implement this common trait:

```lean
class Differentiable (I O : Type) where
  /-- Primal execution (Forward pass) -/
  forward : I → O
  
  /-- Adjoint computation (Backward pass) -/
  /-- Matrix-Free Vector-Jacobian Product (Jᵀv) -/
  backward : I → O → I
```

#### Why it Matters:

- **Unified Logic**: Scalar-CPU logic and Tensor-GPU kernels share the same mathematical abstraction.
- **End-to-End Correctness**: By "lifting" `VerifiedOp` instances into the AD tape, Hesper ensures that backpropagation is as formally correct as the forward pass.
- **Zero-Copy Fusion**: The AD engine can calculate gradients across fused kernels (e.g., `MatMul |> ReLU`) without writing intermediate tensors to VRAM.

```lean
-- AD engine automatically dispatches to hand-optimized GPU kernels
let grad := diff (matmul |> relu |> crossEntropy) input 
```

**Key Features:**
- **Hybrid AD**: Seamlessly switch between CPU-scalar AD and GPU-tensor AD.
- **Verified Primitives**: Every AD node is backed by a verified `spec_forward` and `spec_backward`.
- **High Performance**: Leverages Hand-optimized WGSL and Google Highway SIMD.

### ⚙️ High-Level Optimizers

Train models using state-of-the-art optimizers that integrate with Hesper's verified tensors:

```lean
import Hesper.Optimizer.SGD

-- Configure SGD with momentum
let opt := SGDConfig.default
  |>.withLearningRate 0.01 
  |>.withMomentum 0.9

-- Perform optimization step
let (newParams, newState) := opt.step params grads state
```

### 🎮 Graphics & Windowing

Build interactive graphics applications with GLFW integration:

```lean
import Hesper.GLFW

def main : IO Unit := do
  Hesper.init

  withGLFW do
    let window ← createWindow 800 600 "Hesper Graphics"
    let device ← Hesper.WebGPU.getDevice
    let surface ← createSurface device window

    -- Render loop
    gameLoop window surface
```

### 🔌 Multi-GPU Support

Enumerate and select GPUs in multi-GPU systems:

```lean
import Hesper.WebGPU.Device

-- List all available GPUs
Hesper.WebGPU.listAdapters

-- Select specific GPU
let device0 ← getDeviceByIndex 0  -- First GPU
let device1 ← getDeviceByIndex 1  -- Second GPU

-- Get adapter information
let info ← getAdapterInfo 0
IO.println s!"GPU: {info.name} (Backend: {info.backendType})"
```

## Examples

### WebGPU Tetris

A full Tetris implementation using GLFW and WebGPU, demonstrating:
- Dynamic shader generation
- Real-time rendering
- Input handling
- Game state management

```bash
lake build tetris
./.lake/build/bin/tetris
```

**Controls**: A/D (move), S (drop), Space (rotate), ESC (exit)

### Matrix Multiplication

High-performance matrix multiplication with subgroup optimizations:

```bash
lake build matmul-demo
./.lake/build/bin/matmul-demo
```

Demonstrates:
- GPU buffer management
- Compute shader execution
- Performance profiling
- Result verification

### SIMD CPU Backend

Multi-precision SIMD operations with hardware acceleration:

```bash
# Run multi-precision test (Float64/Float32/Float16)
lake script run buildSimd
lake build multi-precision
./.lake/build/bin/multi-precision

# Run SIMD benchmarks
lake build simd-bench
./.lake/build/bin/simd-bench
```

Output:
```
Backend: NEON (ARM64) - F64: 2/op, F32: 4/op, FP16

─── Float64 (8 bytes/element) ───
Result: #[6.0, 8.0, 10.0, 12.0] ✓

─── Float32 (4 bytes/element) ───
Result: Float32[4]: [6.0, 8.0, 10.0, 12.0] ✓

─── Float16 (2 bytes/element) ───
FP16 hardware detected!
Result: Float16[4]: [6.0, 8.0, 10.0, 12.0] ✓
```

### Multi-GPU Demo

Enumerate GPUs and create devices from specific adapters:

```bash
lake build multigpu
./.lake/build/bin/multigpu
```

Output:
```
Found 2 GPU adapter(s):
  [0] NVIDIA GeForce RTX 3080 (Backend: Vulkan)
  [1] Intel UHD Graphics 630 (Backend: Vulkan)
✓ Device created from GPU 0
```

### Neural Network Training

Automatic differentiation and gradient descent on GPU:

```bash
lake build nn-gpu-demo
./.lake/build/bin/nn-gpu-demo
```

Features:
- Conv2D layers with verified gradients
- Backpropagation on GPU
- Real-time training visualization

## Building and Testing

### Building the Project

Hesper requires building both native C++ dependencies (Google Dawn) and Lean code.

**Step 1: Build Native Dependencies**

The first build will take 5-15 minutes as it compiles Google Dawn from source:

```bash
# Build the native WebGPU bridge (hesper_native library)
lake script run buildNative
```

This compiles:
- Google Dawn WebGPU implementation
- C++ FFI bridge (`native/bridge.cpp`)
- SIMD CPU backend (`c_src/simd_ops.cpp`)

**Step 2: Build Lean Code**

Once native dependencies are built, compile the Lean libraries and executables:

```bash
# Build the core library
lake build Hesper

# Or build a specific executable
lake build simple-write
```

**Clean Build** (if you encounter issues):

```bash
lake clean
lake script run buildNative
lake build
```

### Testing the Installation

#### 1. Simple GPU Test (Raw WGSL + DSL)

This test verifies both raw WGSL shaders and DSL-generated shaders execute correctly on your GPU:

```bash
lake build simple-write
./.lake/build/bin/simple-write
```

**Expected output:**
```
╔══════════════════════════════════════╗
║   GPU Double Test (DSL + Raw)        ║
╚══════════════════════════════════════╝

📝 DSL-generated WGSL:
─────────────────────────────────────
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&input)) {
        output[idx] = input[idx] * 2.0;
    }
}

🚀 Initializing WebGPU...
  ✓ Created input buffer
  ✓ Wrote input: [1.0, 2.0, 3.0, 4.0]
  ✓ Created output buffer

  🔹 Test 1: Raw WGSL shader
  ✓ Raw WGSL executed

  🔹 Test 2: DSL-generated shader
  ✓ DSL shader executed

📊 Results:
  Input → Expected → Raw WGSL → DSL WGSL
  [0] 1.0 → 2.0 → 2.0 ✓ → 2.0 ✓
  [1] 2.0 → 4.0 → 4.0 ✓ → 4.0 ✓
  [2] 3.0 → 6.0 → 6.0 ✓ → 6.0 ✓
  [3] 4.0 → 8.0 → 8.0 ✓ → 8.0 ✓

✅ SUCCESS: Both shaders work correctly!
   - Raw WGSL shader: ✓
   - DSL-generated shader (ShaderM monad): ✓
   - Both produce identical correct results
```

This test validates:
- ✓ WebGPU initialization and GPU discovery
- ✓ Buffer creation and data transfer (CPU ↔ GPU)
- ✓ Raw WGSL shader compilation and execution
- ✓ DSL shader code generation (ShaderM monad → WGSL)
- ✓ DSL shader execution on GPU
- ✓ Correct data marshalling across the FFI boundary

#### 2. FFI Boundary Tests

Test data conversion across the Lean ↔ C++ FFI boundary:

```bash
lake build ffi-tests
./.lake/build/bin/ffi-tests
```

**Expected output:**
```
╔══════════════════════════════════════╗
║   FFI Boundary Tests                 ║
╚══════════════════════════════════════╝

Test 1: Lean writes data, C++ reads
  ✓ Lean wrote: [1.0, 2.0, 3.0, 4.0]
  ✓ C++ verified byte-level accuracy

Test 2: C++ writes data, Lean reads
  ✓ GPU wrote: [10.0, 20.0, 30.0, 40.0]
  ✓ Lean verified byte-level accuracy

Test 3: Round-trip (Lean → GPU → Lean)
  ✓ Input: [5.0, 10.0, 15.0, 20.0]
  ✓ Output: [10.0, 20.0, 30.0, 40.0]
  ✓ Data integrity preserved

✅ All FFI boundary tests passed!
```

This validates:
- Lean writes ByteArray → C++ reads correct bytes
- C++ writes bytes → Lean reads correct Float values
- Round-trip data integrity across FFI boundary

#### 3. SIMD CPU Backend Test

Test multi-precision SIMD operations (Float64/Float32/Float16):

```bash
lake script run buildSimd
lake build multi-precision
./.lake/build/bin/multi-precision
```

**Expected output (on ARM64 with FP16 support):**
```
Backend: NEON (ARM64) - F64: 2/op, F32: 4/op, FP16

─── Float64 (8 bytes/element) ───
Result: #[6.0, 8.0, 10.0, 12.0] ✓

─── Float32 (4 bytes/element) ───
Result: Float32[4]: [6.0, 8.0, 10.0, 12.0] ✓

─── Float16 (2 bytes/element) ───
FP16 hardware detected!
Result: Float16[4]: [6.0, 8.0, 10.0, 12.0] ✓
```

### For Contributors: Testing Your Changes

When making changes to Hesper, run these tests to ensure you haven't broken anything:

#### 1. Core FFI Tests
```bash
# Test Lean ↔ C++ data conversion
lake build ffi-tests
./.lake/build/bin/ffi-tests
```

#### 2. GPU Shader Tests
```bash
# Test raw WGSL and DSL shader execution
lake build simple-write
./.lake/build/bin/simple-write
```

#### 3. SIMD Tests
```bash
# Rebuild SIMD library and run tests
lake script run buildSimd
lake build simd-test
./.lake/build/bin/simd-test
```

#### 4. Full Test Suite
```bash
# Run all tests
lake build test-all
./.lake/build/bin/test-all
```

### Troubleshooting

#### Issue: "Build failed: native library not found"
**Solution:** Rebuild the native library:
```bash
lake clean
lake script run buildNative
lake build
```

#### Issue: "No GPU adapters found"
**Solution:** Ensure you have proper GPU drivers:
- **macOS**: No action needed (Metal is built-in)
- **Linux**: Install Vulkan drivers (`vulkan-tools`, `mesa-vulkan-drivers`)
- **Windows**: Install latest GPU drivers with D3D12/Vulkan support

#### Issue: "SIMD library not found"
**Solution:** Build the SIMD backend:
```bash
lake script run buildSimd
```

#### Issue: "FP16 not supported"
**Solution:** This is expected on older hardware. Float16 requires:
- ARM64: ARMv8.2-A with FP16 extension (Apple M1+, AWS Graviton2+)
- x86_64: F16C extension (Intel Ivy Bridge+ / AMD Bulldozer+)

The library will gracefully fall back to Float32 operations.

#### Issue: Dawn build takes too long
**Solution:** Dawn's first build can take 10-15 minutes. Subsequent builds are incremental and much faster. To speed up:
```bash
# Use more CPU cores (adjust -j value)
lake script run buildNative -- -j 16
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Lean 4 Code                               │
│  • Type-safe shader DSL                                      │
│  • Tensor operations                                         │
│  • Formal proofs                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              WGSL Code Generation                            │
│  Exp (.scalar .f32) → WGSL shader source                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Lean FFI (C++ Bridge)                           │
│  • lean_hesper_* functions                                   │
│  • Resource management via Lean.External                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Dawn (WebGPU Native)                     │
│  • Metal (macOS)                                             │
│  • Vulkan (Linux/Windows)                                    │
│  • D3D12 (Windows)                                           │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Layers

1. **DSL Layer**: Type-safe WGSL expression builder with dependent types
2. **Tensor Layer**: High-level operations (matmul, conv2d, pooling)
3. **Compute Layer**: Shader compilation, buffer management, execution
4. **WebGPU Layer**: FFI bindings to Dawn native implementation
5. **Backend Layer**: Platform-specific GPU drivers (Metal/Vulkan/D3D12)

## Project Structure

```
Hesper/
├── Hesper/
│   ├── WGSL/          # Type-safe shader DSL
│   │   ├── Types.lean      # WGSL type system
│   │   ├── Exp.lean        # Expression AST
│   │   └── DSL.lean        # User-facing DSL
│   ├── WebGPU/        # WebGPU bindings
│   │   ├── Device.lean     # GPU device management
│   │   ├── Buffer.lean     # GPU buffers
│   │   ├── Shader.lean     # Shader modules
│   │   ├── Pipeline.lean   # Compute/render pipelines
│   │   └── Errors.lean     # Comprehensive error handling
│   ├── Tensor/        # Tensor operations
│   │   └── MatMul.lean     # Matrix multiplication
│   ├── NN/            # Neural network layers
│   │   └── Conv.lean       # Convolution layers
│   ├── GLFW/          # Windowing and graphics
│   │   └── GLFW.lean       # GLFW bindings
│   ├── Simd.lean      # SIMD Float64 operations
│   ├── Float32.lean   # SIMD Float32 operations
│   ├── Float16.lean   # SIMD Float16 operations
│   └── Compute.lean   # High-level compute API
├── Examples/          # Example programs
│   ├── Tetris.lean         # Full game demo
│   ├── MultiGPU.lean       # Multi-GPU support
│   ├── DSLBasics.lean      # DSL tutorial
│   └── ...
├── native/            # C++ WebGPU bridge
│   ├── bridge.cpp          # FFI implementation
│   └── CMakeLists.txt      # Build configuration
├── c_src/             # SIMD CPU backend
│   └── simd_ops.cpp        # NEON/AVX2 implementations
├── Tests/             # Comprehensive test suite
│   ├── ErrorTests.lean     # Error handling tests
│   ├── ShaderTests.lean    # Shader monad tests
│   └── ...
└── lakefile.lean      # Lake build script
```

## Roadmap

**Current Status**: Early Development (Alpha)

- [x] **Multi-precision SIMD CPU backend (Google Highway)**
- [x] **Architecture detection (NEON/AVX2/F16C)**
- [x] **Comprehensive error handling with structured error types**
- [x] **Complete test suite (error handling, shader monad)**
- [x] **Docker-based CI environment**
- [x] **Verified Composable Kernels (VerifiedOpFusion)**

- [x] **BitNet b1.58 inference engine (125 TPS on M4 Max)**
- [x] **Kernel fusion: fused gate+up+ReLU²×mul, fused KV cache write**
- [x] **KV cache with grouped-query attention**
- [x] **PreparedDispatch graph capture (99%+ cache hit rate)**

In Progress:
- [ ] Comprehensive tensor operation library (GEMM, Conv3D)
- [ ] Gemma 3 / Transformer support
- [ ] Automatic differentiation on GPU kernels
- [ ] Formal proofs of kernel numerical stability
- [ ] Integration with Lean's tactic framework

## Contributing

Hesper is part of the **Verilean** organization's effort to bring verified computing to GPUs.

### How to Contribute

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the existing code style
3. **Run the test suite** to ensure nothing broke:
   ```bash
   # Core FFI boundary tests
   lake build ffi-tests
   ./.lake/build/bin/ffi-tests

   # GPU shader tests (raw WGSL + DSL)
   lake build simple-write
   ./.lake/build/bin/simple-write

   # SIMD tests (if you modified SIMD code)
   lake script run buildSimd
   lake build simd-test
   ./.lake/build/bin/simd-test
   ```
4. **Add tests** for new features (see `Examples/Tests/` for examples)
5. **Submit a pull request** with a clear description of changes

### Testing Guidelines

- **FFI changes**: Always run `test-ffi` to verify Lean ↔ C++ data marshalling
- **DSL changes**: Run `simple-write` to verify WGSL code generation
- **GPU operations**: Test with real GPU hardware, not just compilation
- **SIMD changes**: Test on both ARM64 (NEON) and x86_64 (AVX2) if possible
- **Cross-platform**: macOS (Metal), Linux (Vulkan), Windows (D3D12/Vulkan)

### Code Organization for Contributors

```
Hesper/
├── Hesper/               # Core library
│   ├── WGSL/            # Type-safe shader DSL
│   ├── WebGPU/          # WebGPU bindings (Device, Buffer, Shader, Pipeline)
│   ├── Compute.lean     # High-level compute API
│   ├── Simd.lean        # SIMD Float64 operations
│   ├── Float32.lean     # SIMD Float32 operations
│   └── Float16.lean     # SIMD Float16 operations
├── Examples/             # Example programs (organized by category)
│   ├── DSL/             # DSL feature demonstrations
│   ├── Compute/         # GPU compute examples
│   ├── MachineLearning/ # Neural network training
│   ├── Graphics/        # GLFW rendering demos
│   ├── SIMD/            # CPU SIMD benchmarks
│   ├── Tests/           # Integration tests
│   └── Utilities/       # Helper utilities
├── Tests/                # Unit tests
│   ├── FFIBoundaryTests.lean  # Lean ↔ C++ data conversion tests
│   └── FusionTest.lean        # Operator fusion tests
├── native/               # C++ WebGPU bridge
│   ├── bridge.cpp       # FFI implementation (lean_hesper_* functions)
│   └── CMakeLists.txt   # Dawn build configuration
├── c_src/                # SIMD CPU backend
│   └── simd_ops.cpp     # NEON/AVX2 implementations
└── lakefile.lean         # Lake build script
```

**Key files for contributors:**
- **`native/bridge.cpp`**: FFI boundary - all Lean ↔ C++ data conversion happens here
- **`Hesper/WGSL/Monad.lean`**: ShaderM monad for imperative shader construction
- **`Hesper/WGSL/Execute.lean`**: Compiles ShaderM → WGSL and executes on GPU
- **`Examples/Tests/SimpleWrite.lean`**: Reference test showing raw WGSL vs DSL execution
- **`Tests/FFIBoundaryTests.lean`**: Reference test for FFI data conversion

### Links

- **Report Issues**: [GitHub Issues](https://github.com/Verilean/hesper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Verilean/hesper/discussions)
- **Sister Project**: [Sparkle HDL](https://github.com/Verilean/sparkle) - Verified hardware design in Lean 4

## Author

**Junji Hashimoto**

Twitter/X: [@junjihashimoto3](https://twitter.com/junjihashimoto3)

## License

Apache License 2.0 - see LICENSE file for details

## Acknowledgments

- **Google Dawn** for the WebGPU native implementation
- **Lean 4** for the foundation of verified programming
- **WebGPU Working Group** for the standard
- **gpu.cpp (Answer.AI):** High-level C++ API wrapper inspiration.
- **[llama.cpp](https://github.com/ggerganov/llama.cpp) (Georgi Gerganov & contributors):** Reference implementation for Gemma 4 architecture and Q4_K_M quantization. Its per-op trace (`llama-eval-callback`) is the golden oracle for hesper's per-layer parity work, and its `ggml_gallocr` compute-buffer allocator inspired hesper's `ScratchPool` design for eliminating per-forward `cuMemAlloc` churn.

---

*Write GPU code that's not just fast—make it correct by construction.*
