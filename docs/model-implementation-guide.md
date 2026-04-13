# Hesper Model Implementation Guide

Best practices for adding new models and kernels, derived from BitNet and Gemma 4 implementation experience.

## 1. Architecture: GPUBackend Typeclass

All model code MUST use `[GPUBackend β]` — never hardcode `Device` or `CUDAContext`.

```lean
-- GOOD: generic, works on any backend
def forward [GPUBackend β] (ctx : β) (model : MyModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) ...

-- BAD: WebGPU-only
def forward (device : Device) (model : MyModel Buffer PreparedDispatch) ...
```

### Structure parameterization

```lean
-- GOOD: parameterized
structure MyLayer (BufT CacheT : Type) where
  weight : BufT
  prepared : IO.Ref (Option CacheT)

-- BAD: concrete types
structure MyLayer where
  weight : Buffer
  prepared : IO.Ref (Option PreparedDispatch)
```

### Loading (fromGGUF)

`fromGGUF` should also be `[GPUBackend β]` so both WebGPU and CUDA can load the same model:

```lean
def MyModel.fromGGUF [GPUBackend β] (ctx : β) (path : String) : IO (MyModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
```

Use `GPUBackend.allocBuffer` + `GPUBackend.writeBuffer`, not WebGPU-specific `createBuffer`.

## 2. Dispatch Caching (Critical for Performance)

**Every kernel dispatch MUST use a cacheRef.** Without caching, each dispatch regenerates PTX from scratch (90-330μs overhead). With caching, 2nd+ calls are ~5μs.

### Pattern: Layer-level cacheRef

```lean
structure MyLayer (BufT CacheT : Type) where
  weight : BufT
  prepared : IO.Ref (Option CacheT)  -- cache for executeWithConfigCached

def MyLayer.forward [GPUBackend β] (ctx : β) (layer : MyLayer ...) (input output : GPUBackend.Buf β) : IO Unit := do
  -- Fast path: replay cached dispatch (no PTX gen, no hash, no buffer lookup)
  if let some p ← layer.prepared.get then
    GPUBackend.replayCached ctx p (numWorkgroups, 1, 1)
    return
  -- Slow path: first call — compile + cache
  GPUBackend.executeWithConfigCached ctx myKernel
    [("input", input), ("weight", layer.weight), ("output", output)]
    config cacheKey layer.prepared
```

### Pattern: Inline kernel cacheRef (for forwardBlock)

When kernels are dispatched inline (not through a layer's `.forward`), use a `KernelCacheRefs` structure:

```lean
structure KernelCacheRefs (CacheT : Type) where
  refs : Std.HashMap String (IO.Ref (Option CacheT))

-- In forwardBlock:
let ce := fun (name : String) (shader : ShaderM Unit) (bufs config) => do
  match kcr with
  | some k =>
    let ref ← k.getRef name
    GPUBackend.executeWithConfigCached ctx shader bufs config (hash name) ref
  | none => GPUBackend.execute ctx shader bufs config
```

### Anti-pattern: Bare GPUBackend.execute

```lean
-- BAD: regenerates PTX every call (90-330μs)
GPUBackend.execute ctx myKernel [("a", buf)] (.dispatch1D n)

-- GOOD: caches after first call (~5μs on hit)
GPUBackend.executeWithConfigCached ctx myKernel [("a", buf)] (.dispatch1D n) cacheKey ref
```

## 3. Kernel Safety (CUDA Compatibility)

### Bounds checking: `if_` not `select`

CUDA has no robust buffer access. Out-of-bounds reads crash with `CUDA_ERROR_ILLEGAL_ADDRESS`. WebGPU silently returns 0.

```lean
-- BAD: reads happen before select, crash on CUDA for OOB threads
let val ← readBuffer "input" idx
let result := Exp.select inBounds val (Exp.litF32 0.0)
writeBuffer "output" idx result

-- GOOD: guard all reads/writes with if_
ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
  let val ← readBuffer "input" idx
  writeBuffer "output" idx val
) (pure ())
```

### Exp.select type awareness

`Exp.select` on u32 values requires the PTX codegen to emit `selp.u32`, not `selp.f32`. The current codegen handles this, but be aware:

```lean
-- This works correctly on both f32 and u32 branches:
Exp.select condition (Exp.litU32 10) (Exp.litU32 20)  -- → selp.u32
Exp.select condition (Exp.litF32 1.0) (Exp.litF32 0.0) -- → selp.f32
```

### subgroups extension

Any kernel using `Exp.subgroupAdd` requires `extensions := ["subgroups"]` in the ExecConfig. The WGSL codegen auto-detects this, but CUDA ignores extensions. Always include for WebGPU compatibility.

## 4. PTX CodeGen Pitfalls

### Silent fallbacks

The PTX codegen has `| _, _ => (ra, s)` fallback arms for type-mismatched arithmetic. These silently drop operations. If you see unexpected zeros or pass-through values, check for type mismatches in Exp trees.

Known safe: f32+f32, u32+u32. Known dangerous: f32+u32 → silently returns first operand.

### Covered and tested operations

All these have PTX instruction-level tests (Tests/CUDA/CUDAPTXInstTest.lean):
- f32: add, sub, mul, div, neg, sqrt, abs, max, min, exp, exp2, log, log2, sin, cos, tanh, floor, ceil, inverseSqrt, pow, fma, clamp, select
- u32: add, sub, mul, div, mod, bitAnd, bitOr, bitXor, shiftRight, shiftLeft, select
- Conversions: toF32, toU32
- F16: unpack2x16float (vecX/vecY)
- Comparisons: lt, le, eq, gt, ge, ne
- Boolean: and, or, not
- Memory: shared f32/u32, global f32/u32
- Control: forLoop, if/else
- Warp: subgroupAdd

### Adding new Exp constructors

1. Add the constructor to `Hesper/WGSL/Exp.lean`
2. Add WGSL rendering in `Exp.toWGSL`
3. Add PTX codegen in `Hesper/CUDA/CodeGen.lean` — handle both f32 and u32 cases
4. **Add a test** in `Tests/CUDA/CUDAPTXInstTest.lean`
5. Verify: `lake exe cuda-ptx-inst-test`

## 5. Testing Strategy

### Three-tier testing

1. **PTX instruction tests** (`cuda-ptx-inst-test`): Every Exp constructor, CPU expected value
2. **Golden value tests** (`cuda-fa-golden-test`): CPU ref ↔ WebGPU ↔ CUDA for composite kernels
3. **Model-level tests** (`cuda-bitnet-golden-test`): Full model logits comparison, top-5 ranking

### When to add tests

- New Exp constructor → PTX instruction test
- New composite kernel (FlashAttention, Q6K dequant, etc.) → Golden value test with CPU reference
- New model backend → Model-level logits comparison

### Golden value test pattern

```lean
-- 1. Compute CPU reference
let cpuOut := cpuReference qArr kArr vArr scale

-- 2. Run on WebGPU
let wOut ← runOnGPU device ...

-- 3. Run on CUDA
let cOut ← runOnGPU cuda ...

-- 4. Compare all three
compareArrays "CPU↔WebGPU" cpuOut wOut
compareArrays "CPU↔CUDA" cpuOut cOut
compareArrays "WebGPU↔CUDA" wOut cOut
```

## 6. Performance Checklist

Before claiming TPS numbers:

- [ ] All inline kernels use `executeWithConfigCached` (not bare `execute`)
- [ ] `beginBatch`/`endBatch` wraps the hot loop (if supported by backend)
- [ ] GPU-side argmax (not CPU logits download)
- [ ] Profile with timing instrumentation to find actual bottlenecks
- [ ] Compare against llama.cpp on same hardware/model

### Dispatch overhead budget

| Method | Cost/dispatch | 760 dispatches |
|--------|-------------|----------------|
| `GPUBackend.execute` (uncached) | 90-330μs | 114ms (8 TPS) |
| `executeWithConfigCached` (hit) | ~5μs | 3.8ms (>100 TPS) |
| `replayCached` | ~3μs | 2.3ms (>130 TPS) |
| `dispatchCompiledKernel` | ~5μs | 3.8ms (>100 TPS) |

## 7. Model Loading Performance

- Use `GPUBackend.allocBuffer` + `GPUBackend.writeBuffer` (not `IO.FS.readBinFile` into ByteArray)
- Future: mmap GGUF + direct GPU upload (eliminates 5GB copy)
- Token embedding: F16→F32 unpack on GPU, not CPU

## 8. Common Type Signatures

```lean
-- Model structure
structure MyModel (BufT CacheT : Type) where
  config : Config
  layers : Array (MyLayer BufT CacheT)

-- Forward pass
def forward [GPUBackend β] (ctx : β)
    (model : MyModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (input : GPUBackend.Buf β)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    : IO Unit

-- Loading
def fromGGUF [GPUBackend β] (ctx : β) (path : String)
    : IO (MyModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))

-- Kernel cache
structure KernelCacheRefs (CacheT : Type) where
  refs : Std.HashMap String (IO.Ref (Option CacheT))
```
