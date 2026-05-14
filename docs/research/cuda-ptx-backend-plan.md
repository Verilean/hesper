# CUDA PTX JIT Backend — Implementation Plan

## Goal

Add a CUDA PTX JIT backend to Hesper, parallel to the existing WebGPU/Dawn
backend. All existing ShaderM kernels work on both backends unchanged.
Target: 80-115 TPS on Gemma 4 e4b (vs 23 TPS on WebGPU).

## Architecture

```
ShaderM Unit (kernel definition, unchanged)
  │
  ├─→ toWGSL → Dawn (createShaderModule) → GPU      [existing]
  │
  └─→ toPTX  → cuModuleLoadData (JIT) → GPU         [new]
```

Both paths share the same ShaderM kernel definitions. Backend is selected
at runtime (env var or config).

## Implementation Phases

### Phase 1: CUDA Driver API FFI (1-2 days)

**New file: `Hesper/CUDA/FFI.lean`**

Minimal CUDA Driver API bindings via Lean FFI:

```lean
-- Device management
@[extern "lean_hesper_cuda_init"] opaque cuInit : IO Unit
@[extern "lean_hesper_cuda_get_device"] opaque cuDeviceGet : IO CUdevice
@[extern "lean_hesper_cuda_ctx_create"] opaque cuCtxCreate : CUdevice → IO CUcontext

-- Module (JIT compilation)
@[extern "lean_hesper_cuda_module_load_data"]
opaque cuModuleLoadData : @& String → IO CUmodule  -- PTX string → compiled module

@[extern "lean_hesper_cuda_module_get_function"]
opaque cuModuleGetFunction : @& CUmodule → @& String → IO CUfunction

-- Memory
@[extern "lean_hesper_cuda_malloc"] opaque cuMalloc : USize → IO CUdeviceptr
@[extern "lean_hesper_cuda_free"] opaque cuFree : CUdeviceptr → IO Unit
@[extern "lean_hesper_cuda_memcpy_h2d"] opaque cuMemcpyHtoD : CUdeviceptr → @& ByteArray → IO Unit
@[extern "lean_hesper_cuda_memcpy_d2h"] opaque cuMemcpyDtoH : CUdeviceptr → USize → IO ByteArray

-- Kernel launch
@[extern "lean_hesper_cuda_launch_kernel"]
opaque cuLaunchKernel : CUfunction → Nat → Nat → Nat    -- grid
                      → Nat → Nat → Nat                  -- block
                      → Nat                               -- shared mem
                      → Array CUdeviceptr → IO Unit       -- args
```

**New file: `hesper_cuda_ffi.cpp`** (C++ bridge)

~200 lines of `cuDriverAPI` calls wrapped in Lean external functions.
No CUDA runtime (`cudart`) dependency — driver API only.

### Phase 2: PTX Codegen (`Exp.toPTX`, `Stmt.toPTX`) (2-3 days)

**New file: `Hesper/CUDA/CodeGen.lean`**

Parallel to `Hesper/WGSL/CodeGen.lean`. Key function:

```lean
def generatePTX
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize)
    (computation : ShaderM Unit)
    : String
```

#### PTX Mapping (core subset for Gemma 4 kernels)

**Module header:**
```ptx
.version 7.0
.target sm_89          // Ada Lovelace (RTX 4070 Ti)
.address_size 64

.entry main(
    .param .u64 param_buf0,    // buffer 0
    .param .u64 param_buf1,    // buffer 1
    ...
) {
```

**Exp → PTX mappings (core):**

| Exp | WGSL | PTX |
|-----|------|-----|
| `litF32 x` | `x` | `mov.f32 %fN, x;` |
| `litU32 n` | `nu` | `mov.u32 %rN, n;` |
| `add a b` | `(a + b)` | `add.f32 %fN, %fA, %fB;` |
| `mul a b` | `(a * b)` | `mul.f32 %fN, %fA, %fB;` |
| `fma a b c` | `fma(a,b,c)` | `fma.rn.f32 %fN, %fA, %fB, %fC;` |
| `toF32 e` | `f32(e)` | `cvt.rn.f32.u32 %fN, %rA;` |
| `workgroupId` | `workgroup_id` | `mov.u32 %r, %ctaid.x;` |
| `localId` | `local_invocation_id` | `mov.u32 %r, %tid.x;` |
| `readBuffer buf idx` | `buf[idx]` | `ld.global.f32 %fN, [%ptr + idx*4];` |
| `writeBuffer buf idx val` | `buf[idx] = val` | `st.global.f32 [%ptr + idx*4], %fN;` |
| `sharedNamed name ty` | `var<workgroup>` | `.shared .f32 name[N];` |
| `readWorkgroup` | `shared_x[idx]` | `ld.shared.f32 %fN, [name + idx*4];` |
| `writeWorkgroup` | `shared_x[idx] = v` | `st.shared.f32 [name + idx*4], %fN;` |
| `barrier` | `workgroupBarrier()` | `bar.sync 0;` |
| `subgroupAdd val` | `subgroupAdd(val)` | warp shuffle tree (5 `shfl.sync.bfly.b32`) |

**Stmt → PTX:**

| Stmt | WGSL | PTX |
|------|------|-----|
| `varDecl name f32 init` | `var name: f32 = init;` | `mov.f32 %name, init;` |
| `assign name val` | `name = val;` | `mov.f32 %name, %val;` |
| `forLoop` | `for (var i = ...) { }` | `LOOP: ... @%p bra LOOP;` |
| `ifStmt` | `if (cond) { }` | `@!%p bra ELSE; ... ELSE:` |

#### Register allocation

PTX uses virtual registers (`%f0, %f1, ...` for f32, `%r0, %r1, ...`
for u32). NVIDIA's `ptxas` handles register allocation from virtual to
physical. We just emit virtual registers sequentially.

```lean
structure PTXGenState where
  fRegCount : Nat := 0    -- float register counter
  rRegCount : Nat := 0    -- int register counter
  labelCount : Nat := 0   -- label counter for branches
  output : String := ""   -- accumulated PTX text
```

#### subgroupAdd → warp shuffle tree

```ptx
// subgroupAdd(val) for warp size 32
shfl.sync.bfly.b32 %f_tmp, %f_val, 16, 31, 0xFFFFFFFF;
add.f32 %f_val, %f_val, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_val, 8, 31, 0xFFFFFFFF;
add.f32 %f_val, %f_val, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_val, 4, 31, 0xFFFFFFFF;
add.f32 %f_val, %f_val, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_val, 2, 31, 0xFFFFFFFF;
add.f32 %f_val, %f_val, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_val, 1, 31, 0xFFFFFFFF;
add.f32 %f_val, %f_val, %f_tmp;
```

### Phase 3: CUDA Execute Backend (1-2 days)

**New file: `Hesper/CUDA/Execute.lean`**

Parallel to `Hesper/WGSL/Execute.lean`:

```lean
def executeShaderCUDA
    (computation : ShaderM Unit)
    (buffers : Array CUdeviceptr)
    (gridDim : Nat × Nat × Nat)
    (blockDim : Nat × Nat × Nat)
    (sharedMem : Nat := 0)
    : IO Unit := do
  let ptxSource := generatePTX "main" blockDim computation
  let module ← cuModuleLoadData ptxSource
  let func ← cuModuleGetFunction module "main"
  cuLaunchKernel func gridDim blockDim sharedMem buffers
```

With pipeline caching (same hash-based approach as WebGPU):
```lean
initialize ptxCacheRef : IO.Ref (Array (UInt64 × CUfunction)) ← IO.mkRef #[]
```

### Phase 4: CUDABuffer type + bridge (1 day)

**New file: `Hesper/CUDA/Buffer.lean`**

```lean
structure CUDABuffer where
  ptr : CUdeviceptr
  size : USize

def createCUDABuffer (size : USize) : IO CUDABuffer
def writeCUDABuffer (buf : CUDABuffer) (data : ByteArray) : IO Unit
def readCUDABuffer (buf : CUDABuffer) (size : USize) : IO ByteArray
def freeCUDABuffer (buf : CUDABuffer) : IO Unit
```

### Phase 5: Backend abstraction (1 day)

**New file: `Hesper/Backend.lean`**

```lean
inductive Backend where
  | WebGPU
  | CUDA
  deriving BEq

def getBackend : IO Backend := do
  let env ← IO.getEnv "HESPER_BACKEND"
  match env with
  | some "cuda" => return .CUDA
  | _ => return .WebGPU
```

Model-level code checks backend once and dispatches:
```lean
match ← getBackend with
| .WebGPU => executeShaderNamed device shader bufs config
| .CUDA   => executeShaderCUDA shader cudaBufs gridDim blockDim
```

### Phase 6: Validation (2-3 days)

1. **Unit tests**: Each Exp/Stmt PTX output vs expected PTX string
2. **Kernel correctness**: Run Q4_K matmul on CUDA, compare output with WebGPU
3. **Gemma 4 inference**: Same model, same prompt → same tokens on both backends
4. **Performance**: TPS comparison WebGPU vs CUDA

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `Hesper/CUDA/FFI.lean` | CUDA Driver API Lean bindings | ~100 |
| `hesper_cuda_ffi.cpp` | C++ FFI bridge | ~250 |
| `Hesper/CUDA/CodeGen.lean` | `generatePTX` (Exp.toPTX, Stmt.toPTX) | ~500 |
| `Hesper/CUDA/Execute.lean` | Pipeline cache + dispatch | ~200 |
| `Hesper/CUDA/Buffer.lean` | GPU memory management | ~100 |
| `Hesper/Backend.lean` | Backend selection | ~30 |
| `Tests/CUDA/PTXCodeGenTest.lean` | Codegen unit tests | ~200 |

## Files to Modify

| File | Change | Impact |
|------|--------|--------|
| `lakefile.lean` | Add CUDA FFI compile, link `-lcuda` | Build only |
| `shell.nix` | Add `cudaPackages.cuda_cudart` | Dev env |

## NOT Modified

- `Hesper/WGSL/*.lean` — unchanged
- `Hesper/Models/*.lean` — unchanged (backend-agnostic via Backend.lean)
- `Hesper/TTT/*.lean` — unchanged
- All existing kernel definitions — unchanged

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| PTX codegen bugs | Validate against WebGPU output on same inputs |
| Register pressure | PTX virtual regs → ptxas handles allocation |
| Shared memory bank conflicts | Same layout as WGSL (stride access) |
| CUDA driver version compat | Target PTX 7.0+ (CUDA 11.0+, covers RTX 30xx/40xx) |
| Lean FFI stability | Same pattern as existing Dawn FFI (proven) |

## Timeline

| Phase | Days | Deliverable |
|-------|------|------------|
| 1. CUDA FFI | 1-2 | `cuInit` through `cuLaunchKernel` working |
| 2. PTX Codegen | 2-3 | `toPTX` for core Exp/Stmt subset |
| 3. Execute | 1-2 | `executeShaderCUDA` with caching |
| 4. Buffer | 1 | CUDABuffer create/read/write |
| 5. Backend | 1 | `HESPER_BACKEND=cuda` env switch |
| 6. Validation | 2-3 | Gemma 4 same-output proof |
| **Total** | **8-12 days** | **CUDA backend fully operational** |
