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

### ⚙️ GPU Computation

Execute compute shaders and tensor operations on the GPU:

```lean
import Hesper.Compute

-- Matrix multiplication on GPU
let A : Matrix 1024 1024 := ...
let B : Matrix 1024 1024 := ...

-- Runs on GPU automatically
let C ← matmul A B

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
│   │   └── Pipeline.lean   # Compute/render pipelines
│   ├── Tensor/        # Tensor operations
│   │   └── MatMul.lean     # Matrix multiplication
│   ├── NN/            # Neural network layers
│   │   └── Conv.lean       # Convolution layers
│   ├── GLFW/          # Windowing and graphics
│   │   └── GLFW.lean       # GLFW bindings
│   └── Compute.lean   # High-level compute API
├── Examples/          # Example programs
│   ├── Tetris.lean         # Full game demo
│   ├── MultiGPU.lean       # Multi-GPU support
│   ├── DSLBasics.lean      # DSL tutorial
│   └── ...
├── native/            # C++ WebGPU bridge
│   ├── bridge.cpp          # FFI implementation
│   └── CMakeLists.txt      # Build configuration
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
