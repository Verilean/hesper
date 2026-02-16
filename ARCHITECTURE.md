# Hesper Architecture: BitNet b1.58 Inference Engine

**Date:** 2026-02-16
**Status:** ~125 TPS single-token generation (M4 Max, KV cache + kernel fusion)

## 1. System Overview

Hesper is a **Lean 4 WebGPU framework** that implements BitNet b1.58 (2B) inference. It combines:
- **Lean 4** for type-safe model definitions and shader DSL
- **WebGPU/Dawn** (Metal backend on macOS) for GPU compute
- **GGUF** format for model weight loading
- **i2_s** ternary weight format (2 bits per weight: -1, 0, +1)

```
┌─────────────────────────────────────────────────────────┐
│  Lean 4 Application Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ BitNet Model │  │ GGUF Loader  │  │ Sampling/Gen  │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                   │          │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │  Layer Library (TransformerBlock, Attention,     │    │
│  │  BitLinear, RMSNorm, RoPE, Softmax, Embedding)  │    │
│  └──────┬─────────────────────────────────────────┘    │
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │  WGSL Execution Engine                           │    │
│  │  ├─ ShaderM DSL (type-safe WGSL codegen)        │    │
│  │  ├─ Pipeline Cache (99.4% hit rate)             │    │
│  │  └─ Command Buffer Batching (18x speedup)       │    │
│  └──────┬─────────────────────────────────────────┘    │
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │  native/bridge.cpp (FFI to Dawn C++ API)         │    │
│  └──────┬─────────────────────────────────────────┘    │
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │  Dawn WebGPU (Metal / Vulkan backend)            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 2. Model Configuration

### BitNet b1.58 2B (default)

| Parameter | Value |
|-----------|-------|
| Vocabulary | 128,256 tokens |
| Embedding dimension | 2,560 |
| Transformer layers | 30 |
| Query heads | 20 |
| KV heads | 5 (GQA 4:1) |
| Head dimension | 128 |
| FFN hidden dimension | 6,912 |
| Max sequence length | 2,048 |
| RoPE base | 500,000 |
| Total parameters | ~2B |

### BitNet-Specific Design

Unlike standard LLaMA:
1. **ReLU² activation** (NOT SiLU): `hidden = relu(gate)² * up`
2. **Two sub-norms**: after attention output, after FFN gating
3. **Ternary weights**: all projection weights are {-1, 0, +1} in i2_s format
4. **Weight tying**: LM head reuses embedding table (no output.weight)
5. **RoPE variant**: NeoX split-half pairs `(x[i], x[i + headDim/2])`

## 3. Forward Pass Architecture

### 3.1 Single-Token Generation (KV Cache Path)

```
Token ID
  │
  ├─ Embedding lookup → [dim]
  │
  ├─ BEGIN BATCH (command buffer recording)
  │   │
  │   ├─ Layer 0..29: TransformerBlock.forwardSingleToken
  │   │   ├─ RMSNorm (pre-attention)
  │   │   ├─ Attention with KV cache
  │   │   │   ├─ Q,K,V projections (BitLinear)
  │   │   │   ├─ RoPE at position
  │   │   │   ├─ Append K,V to cache[pos]
  │   │   │   ├─ Q @ K_cache^T (full history)
  │   │   │   ├─ Softmax
  │   │   │   ├─ Attn @ V_cache
  │   │   │   ├─ Sub-norm + O projection
  │   │   │   └─ Output [dim]
  │   │   ├─ Residual add
  │   │   ├─ RMSNorm (pre-FFN)
  │   │   ├─ FFN: gate, up (BitLinear)
  │   │   ├─ ReLU²(gate) * up
  │   │   ├─ Sub-norm
  │   │   ├─ FFN: down (BitLinear)
  │   │   └─ Residual add
  │   │
  │   ├─ Final RMSNorm
  │   └─ LM Head (MatMul embedding^T) → [vocab]
  │
  ├─ END BATCH (single GPU submit + wait)
  │
  └─ GPU Argmax → Token ID (4-byte download)
```

### 3.2 Dispatches Per Layer (KV Cache Single-Token Mode)

| Operation | Dispatch(es) | Notes |
|-----------|-------------|-------|
| RMSNorm (pre-attention) | 1 | |
| BitLinear Q,K,V,O | 4 | M=1 warp-cooperative |
| RoPE Q, K | 2 | Dynamic params buffer |
| Fused cacheWriteKV | 1 | Was 2 (K+V separate) |
| Attention scores (Q@K^T) | 1 | |
| Softmax | 1 | |
| Attention apply (attn@V) | 1 | |
| Attention sub-norm | 1 | |
| Residual add | 1 | |
| RMSNorm (pre-FFN) | 1 | |
| **Fused gate+up+ReLU²×mul** | **1** | **Was 3 (gate+up+relu)** |
| BitLinear down | 1 | With fused residual add |
| FFN sub-norm | 1 | |
| Residual add | 1 | |
| **Total per layer** | **~18** | |

**Total per token:** ~18 × 30 layers + 2 (final norm + LM head) ≈ **~542 dispatches**

Kernel fusions save ~3 dispatches/layer (90 total): fused gate+up+ReLU²×mul (-2) and fused cacheWriteKV (-1).

All dispatched in a single command buffer via batching.

## 4. GPU Kernel Implementations

### 4.1 BitLinear (Fused i2_s Unpack + MatVec)

The dominant compute kernel. Reads packed 2-bit ternary weights directly on GPU.

**i2_s Encoding:**
```
-1 → 0b00, 0 → 0b01, +1 → 0b10
Dequant: value = (code - 1) * scale
```

**Group-of-128 Layout:**
```
group128 = elemIdx / 128
local128 = elemIdx % 128
byteIdx  = group128 * 32 + (local128 % 32)
shift    = 6 - (local128 / 32) * 2
code     = (byte >> shift) & 0x3
```

**Kernel Strategy:**
- 1 workgroup per output element, 256 threads
- Each thread accumulates partial dot product over strided input elements
- Shared memory tree reduction to produce final output
- Single scale factor applied at end

**Files:** `Hesper/Layers/BitLinear.lean`

### 4.2 RMSNorm (Fused Single-Pass)

```
y[i] = (x[i] / sqrt(mean(x²) + eps)) * scale[i]
```

- 1 workgroup per row, 256 threads
- Phase 1: Strided accumulation of x² into shared memory
- Phase 2: Tree reduction → total sum
- Phase 3: Compute RMS, normalize + scale

**Files:** `Hesper/Layers/RMSNorm.lean`

### 4.3 GPU Argmax

Single-workgroup parallel reduction over vocab_size (128,256) logits.

- 256 threads, each finds local max over strided elements
- Tree reduction in shared memory (F32 values + U32 indices)
- Downloads 4 bytes (1 u32) instead of 512KB (128K floats)
- Saves ~2.2 ms/token vs CPU download

**Files:** `Hesper/Models/BitNet.lean` (gpuArgmax)

### 4.4 Other Kernels

| Kernel | Strategy | File |
|--------|----------|------|
| Embedding | F16→F32 via `unpack2x16float` | `Layers/Embedding.lean` |
| RoPE | Per-element NeoX split-half | `Layers/RoPE.lean` |
| Softmax | Two-pass stable (max, then exp/sum) | `Layers/Softmax.lean` |
| MatMul | Tiled transpose matmul | `WGSL/MatMul.lean` |
| Elementwise | Add, ReLU²*Mul, Scale | `WGSL/Elementwise.lean` |
| Reshape | Permute for multi-head attention | `WGSL/Reshape.lean` |

### 4.5 Fused Kernels

#### Fused gate+up+ReLU²×mul (M=1)
Combines three dispatches (gate BitLinear + up BitLinear + ReLU²×mul elementwise) into one:
- 1 workgroup (32 threads) per output element
- Dual accumulators: reads input once, computes both gate and up dot products
- Thread 0 applies: `output = relu(gate_total)² × up_total`
- Saves 2 dispatches per layer (60 total) and eliminates intermediate gate/up buffers

**File:** `Hesper/Layers/BitLinear.lean` (`fusedGateUpReluSqrMulM1Kernel`)

#### Fused cacheWriteKV
Writes both K and V to KV cache in a single dispatch (was 2 separate dispatches):
- Same thread writes to both K cache and V cache at `cacheIdx`
- Position read from params buffer (dynamic, no shader recompilation)

**File:** `Hesper/Layers/Attention.lean` (`fusedCacheWriteKVKernel`)

**Note:** Fused RMSNorm+BitLinear was attempted but reverted — each of outDim workgroups independently recomputes the RMS (O(outDim×inDim) redundant reads), making it slower than the 2-dispatch approach.

## 5. Execution Engine

### 5.1 ShaderM DSL

Type-safe WGSL code generation monad in Lean 4.

```lean
def myKernel : ShaderM Unit := do
  let gid ← getGlobalId .x
  let x ← Exp.load inputBuf gid
  ShaderM.loop 0 n fun i => do  -- WGSL for loop (not compile-time unroll)
    ...
  Exp.store outputBuf gid result
```

Key patterns:
- `ShaderM.loop` → WGSL `for` loops (Lean `for` unrolls at compile time)
- `Exp.select` → conditional (no `ifThen`)
- `Exp.vecX/vecY` → vector component access
- `Exp.unpack2x16float` → hardware F16→F32

### 5.2 Pipeline Cache

Caches compiled GPU pipelines by WGSL source hash (SHA-256 of source string).

```
Cache entry: (hash, ShaderModule, BindGroupLayout, ComputePipeline, declaredNames)
```

- **Hit rate:** 99.4% (1280 hits, 8 misses in benchmark)
- First forward pass compiles all 8 unique shaders (~5ms each)
- Subsequent dispatches reuse cached pipelines

### 5.3 Command Buffer Batching

```lean
beginBatch device          -- Start recording
  -- All executeShaderNamed calls record to encoder (no submit/wait)
  for layer in layers do
    TransformerBlock.forward ...
endBatch device            -- Single submit + wait
```

- **Speedup:** 18x (0.006 ms/dispatch batched vs 0.10 ms unbatched)
- Eliminates per-dispatch command encoder creation, queue submit, and device wait
- All ~542 dispatches per token recorded into one command buffer

### 5.4 Buffer Pre-allocation

Each layer's intermediate buffers are allocated once at model load time:

```lean
structure LayerBuffers where
  normedBuf, attnOutBuf, residual1Buf, normed2Buf : Buffer
  gateBuf, upBuf, hiddenBuf, ffnOutBuf, ffnNormedBuf : Buffer
  rmsTempBuf : Buffer
  attnBufs : AttentionBuffers  -- 12+ attention intermediates
```

Model-level buffers (reused across generate steps):
- `buf1, buf2` — Ping-pong buffers [dim]
- `logitsBuf` — [vocab_size]
- `argmaxBuf` — [1] u32
- `tokenBuf` — [1] u32

## 6. Model Loading (GGUF)

### Loading Pipeline

```
GGUF File → Parser → Tensor Extraction → GPU Upload
```

1. **Parse GGUF header**: magic, version, tensor count, metadata
2. **Extract config** from metadata (or use hardcoded defaults)
3. **Load embedding**: F16 data → GPU buffer (hardware unpack)
4. **Load 30 layers**: For each layer:
   - RMSNorm scales (F32) × 4 (attn_norm, attn_sub_norm, ffn_norm, ffn_sub_norm)
   - Attention weights (i2_s packed bytes + F32 scale) × 4 (Q, K, V, O)
   - FFN weights (i2_s packed bytes + F32 scale) × 3 (gate, up, down)
5. **Load final norm** (output_norm.weight)
6. **LM head**: weight-tied with embedding (no additional loading)

### GGUF Tensor Names

```
token_embd.weight                     → Embedding [vocab, dim]
blk.{i}.attn_norm.weight             → RMSNorm scale
blk.{i}.attn_q.weight                → Q projection (i2_s)
blk.{i}.attn_k.weight                → K projection (i2_s)
blk.{i}.attn_v.weight                → V projection (i2_s)
blk.{i}.attn_output.weight           → O projection (i2_s)
blk.{i}.attn_sub_norm.weight         → Attention sub-norm scale
blk.{i}.ffn_norm.weight              → FFN norm scale
blk.{i}.ffn_gate.weight              → Gate projection (i2_s)
blk.{i}.ffn_up.weight                → Up projection (i2_s)
blk.{i}.ffn_down.weight              → Down projection (i2_s)
blk.{i}.ffn_sub_norm.weight          → FFN sub-norm scale
output_norm.weight                    → Final norm scale
```

## 7. Text Generation

### Generate Loop

```
Prefill: Process prompt tokens one at a time
  for i in [0, prompt.size):
    forwardSingleToken(prompt[i], position=i, kvCache)

Generate: Decode new tokens
  for step in [0, maxTokens):
    forwardSingleToken(lastToken, position, kvCache)
    if greedy:
      nextToken ← gpuArgmax(logits)      // 4-byte download
    else:
      logits ← downloadFloatArray(logits) // 512KB download
      nextToken ← cpuSample(strategy, logits)
```

### Sampling Strategies
- **Greedy**: GPU argmax (default, fastest)
- **Top-K**: CPU sort + sample from top-k
- **Nucleus (Top-P)**: CPU cumulative probability threshold

## 8. Performance Profile

### Microbenchmark (per-operation, seq_len=1, 10 iterations averaged)

| Operation | ms | Notes |
|-----------|----|-------|
| RMSNorm (dim=2560) | 0.45 | Fused single-pass |
| BitLinear (2560→2560) | 0.35 | Workgroup-cooperative |
| BitLinear (2560→6912) | 0.85 | Gate/Up projection |
| BitLinear (6912→2560) | 0.59 | Down projection |
| Elementwise Add | 0.37 | Residual connection |
| ReLU²*Mul | 0.13 | Gated activation |
| MatMul LM Head | 3.93 | 1×2560 @ 2560×128256 |
| GPU Argmax | 0.40 | vs 2.61 ms CPU download |

### Estimated Forward Pass (30 layers + LM head)

| Component | ms/layer | × layers | Total ms |
|-----------|----------|----------|----------|
| RMSNorm (4/layer) | 1.78 | 30 | 53.4 |
| BitLinear (7/layer) | 3.68 | 30 | 110.4 |
| Elementwise (3/layer) | 0.86 | 30 | 25.8 |
| LM Head MatMul | — | 1 | 3.9 |
| **Subtotal (measured)** | | | **193.6 ms** |

**Note:** Excludes attention matmuls (2/layer), reshape (4/layer), softmax (1/layer), RoPE (1/layer), sub-norms (2/layer). These add ~10 dispatches/layer.

### Execution Engine Stats

| Metric | Value |
|--------|-------|
| Pipeline cache hit rate | 99.4% |
| Unique shaders | 8 |
| Dispatches per token (KV cache + fusion) | ~542 |
| Batch speedup | 18x |
| Per-dispatch overhead (batched) | 0.006 ms |
| Per-dispatch overhead (unbatched) | 0.10 ms |

### End-to-End Generation

| Metric | Value |
|--------|-------|
| Measured TPS (M4 Max, with KV cache + fusion) | ~125 TPS |
| Dispatches per token (after fusion) | ~421 (was ~571) |
| Kernel fusions | gate+up+ReLU²×mul, cacheWriteKV |

## 9. Validation & Testing

### Test Suite

```bash
lake exe test-all        # All LSpec tests (0 failures)
lake exe micro-bench     # Per-operation latency benchmarks
lake exe kvcache-validation  # KV cache correctness vs full forward
lake exe bitnet-validation   # Model inference validation
```

### KV Cache Validation
- Compares single-token cached inference vs full-sequence forward
- Cosine similarity > 0.999
- Top-1 token prediction match

### Golden Values
- Extracted from llama.cpp via `scripts/extract_golden_values.cpp`
- Stored in `Tests/golden-values/`
- Used for per-layer numerical validation

## 10. Build System

```bash
lake run buildNative     # Compile bridge.cpp + Dawn (once)
lake clean               # Required after bridge.cpp changes
lake build               # Compile all Lean code
lake exe <target>        # Run specific executable
```

### Build Targets

| Target | Purpose |
|--------|---------|
| `bitnet-complete` | Full model inference |
| `kvcache-validation` | KV cache correctness |
| `bitnet-validation` | Model validation |
| `micro-bench` | Performance benchmarks |
| `test-all` | Full test suite |
| `test-compute` | GPU compute tests |
| `test-gguf` | GGUF parser tests |

### Dependencies

- **Dawn** (Google's WebGPU): Downloaded and built by `buildNative`
- **Metal framework** (macOS): GPU backend
- **Lean 4** (v4.x): Language and build system
- **Highway SIMD** (optional): CPU SIMD operations

## 11. File Map

```
refs/hesper/
├── native/bridge.cpp              # FFI: Lean ↔ Dawn C++ API
├── Hesper/
│   ├── Models/BitNet.lean         # Model config, forward, generate, KV cache
│   ├── Layers/
│   │   ├── TransformerBlock.lean  # Per-layer forward pass orchestration
│   │   ├── Attention.lean         # Multi-head attention + KV cache
│   │   ├── BitLinear.lean         # Fused i2_s unpack + matvec kernel
│   │   ├── RMSNorm.lean          # Fused single-pass normalization
│   │   ├── RoPE.lean             # Rotary positional embeddings
│   │   ├── Softmax.lean          # Two-pass stable softmax
│   │   └── Embedding.lean        # F16→F32 embedding lookup
│   ├── WGSL/
│   │   ├── Execute.lean          # Pipeline cache + batch execution
│   │   ├── Monad.lean            # ShaderM monad (WGSL codegen)
│   │   ├── MatMul.lean           # Tiled matrix multiplication
│   │   ├── Elementwise.lean      # Add, ReLU²*Mul, Scale
│   │   └── Reshape.lean          # Multi-head reshape/permute
│   ├── GGUF/
│   │   ├── Reader.lean           # GGUF binary format parser
│   │   └── Loader.lean           # Tensor extraction (F16, i2_s, TQ2_0)
│   └── WebGPU/
│       ├── Types.lean            # Opaque FFI types
│       ├── Device.lean           # Device creation + limits
│       ├── Buffer.lean           # Buffer creation + management
│       └── BufferOps.lean        # Upload/download operations
├── Tests/
│   ├── golden-values/            # Reference outputs from llama.cpp
│   └── *.lean                    # LSpec test modules
├── Bench/MicroBenchmark.lean     # Performance benchmarks
└── lakefile.lean                 # Build configuration
```
