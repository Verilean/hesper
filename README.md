# Hesper

**Write GPU programs in Lean 4. Prove them correct. Run on WebGPU.**

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

## Why Hesper?

Modern GPU programming lacks safety guarantees. Hesper provides:

- **Type Safety**: Shaders are type-checked at compile time, preventing type mismatches
- **Formal Verification**: Prove correctness properties about your GPU programs
- **WebGPU Backend**: Cross-platform GPU access via Dawn (Metal, Vulkan, D3D12)
- **Lean Integration**: Use Lean's powerful theorem proving alongside GPU computation
- **Multi-GPU Support**: Select and coordinate across multiple GPU adapters

## Quick Start

### Prerequisites

- **Lean 4** (latest version recommended)
- **CMake** 3.16+
- **C++17** compiler (Clang/GCC)
- **Platform**: macOS (Metal), Linux (Vulkan), or Windows (D3D12/Vulkan)

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

### 🚀 Multi-Precision SIMD CPU Backend

Hardware-accelerated CPU operations with multi-precision support:

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

**Architecture Detection:**
- ARM64: NEON (default) + optional FP16 vector arithmetic (ARMv8.2-A+)
- x86_64: AVX2 + F16C extension (Ivy Bridge+)
- Optional OpenMP multithreading support

**Zero-Conversion Architecture:**
All operations work directly on raw `ByteArray` with no automatic type conversions. Conversions are explicit only when needed.

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

### ⚙️ GPU Computation & SIMD CPU Backend

Execute compute shaders on GPU or leverage SIMD acceleration on CPU:

```lean
import Hesper.Compute
import Hesper.Simd

-- Matrix multiplication on GPU
let A : Matrix 1024 1024 := ...
let B : Matrix 1024 1024 := ...

-- Runs on GPU automatically
let C ← matmul A B

-- CPU SIMD operations with multi-precision support
let a := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
let b := FloatArray.mk #[5.0, 6.0, 7.0, 8.0]
let c := Hesper.Simd.simdAdd a b  -- NEON/AVX2 acceleration

-- Float32 (2x memory savings)
let a32 := Float32.fromFloatArray a
let c32 := Float32.simdAdd a32 b32

-- Float16 (4x memory savings, hardware-accelerated)
let a16 ← Float16.fromFloatArray a
let c16 ← Float16.simdAdd a16 b16

-- Neural network layers with automatic differentiation
let conv ← Conv2D.create inputChannels outputChannels kernelSize
let output ← conv.forward input
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

Completed:
- [x] WebGPU device initialization via Dawn
- [x] Type-safe WGSL DSL
- [x] Compute shader execution
- [x] Buffer management (GPU ↔ CPU)
- [x] GLFW windowing integration
- [x] Multi-GPU adapter enumeration
- [x] Basic matrix operations
- [x] Convolution layers
- [x] Automatic differentiation
- [x] **Multi-precision SIMD CPU backend (Float64/Float32/Float16)**
- [x] **Architecture detection (NEON/AVX2/F16C)**
- [x] **Comprehensive error handling with structured error types**
- [x] **Complete test suite (error handling, shader monad)**

In Progress:
- [ ] Comprehensive tensor operation library
- [ ] Neural network training framework
- [ ] Performance optimization (subgroup operations)
- [ ] Verification of GPU kernel correctness

Future:
- [ ] Formal proofs of numerical stability
- [ ] Compiler optimizations for shader generation
- [ ] Distributed multi-GPU training
- [ ] Integration with Lean's tactic framework
- [ ] Ray tracing support
- [ ] Multi-precision GPU tensors (Float16 for memory bandwidth optimization)

## Contributing

Hesper is part of the **Verilean** organization's effort to bring verified computing to GPUs.

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
- **Verilean Community** for support and contributions

---

*Write GPU code that's not just fast—make it correct by construction.*
