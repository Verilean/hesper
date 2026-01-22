# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-22

### Added
- **WebGPU Backend**: Full integration with Google Dawn (Metal/Vulkan/D3D12).
- **WGSL DSL**: Type-safe embedded DSL for writing compute shaders in Lean.
- **ShaderM Monad**: Imperative-style monadic interface for easy shader construction.
- **SIMD CPU Backend**: Multi-precision support (Float64/Float32/Float16) using Google Highway.
- **Neural Network Primitives**:
    - Matrix multiplication (naive and tiled).
    - 2D Convolution and Depthwise Convolution.
    - Pooling layers (Max/Avg).
    - Activation functions (ReLU, Sigmoid, Tanh, Gelu).
- **Automatic Differentiation**: Reverse-mode AD for scalar and tensor operations.
- **GLFW Integration**: Window creation and surface management for graphics apps.
- **Examples**:
    - `Tetris`: Full game demo.
    - `MNIST`: Training demo on CPU and GPU.
    - `MultiGPU`: Adapter enumeration and selection.

### Fixed
- Float64 to Float32 FFI conversion now uses proper narrowing instead of bit truncation.
- Fixed zero-size array allocation bug in FFI bridge.
- Corrected Softmax shader implementation to fix numerical instability.

### Known Issues
- Float16 support requires hardware acceleration (ARMv8.2-A or AVX2+F16C).
