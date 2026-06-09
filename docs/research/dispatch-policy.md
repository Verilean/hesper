# Dispatch Policy: Enforcing Cached Dispatch

## Goals
1. Production code (forwardBlock, forwardSingleToken, generate) MUST use cached dispatch
2. Test/debug code CAN use uncached dispatch for convenience
3. Violations should be caught early (ideally at compile time, at worst in CI)

## Design: Two-Tier API

### Tier 1: Production API (default, cached)

```lean
-- The ONLY dispatch method used in model forward passes
GPUBackend.cachedExec [GPUBackend β] (ctx : β)
    (kernel : CachedKernel β)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit
```

`CachedKernel` bundles the ShaderM computation with its IO.Ref cache:

```lean
structure CachedKernel (β : Type) [GPUBackend β] where
  computation : ShaderM Unit
  cacheKey : UInt64
  cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β))
```

Model structures hold `CachedKernel` instead of raw `ShaderM Unit`:

```lean
structure InferenceState (BufT CacheT : Type) where
  -- ...
  residAddKernel : CachedKernel β  -- pre-created at init time
  qNormKernel : CachedKernel β
```

**Benefit**: Impossible to forget the cache. The type system enforces it.

### Tier 2: Debug API (opt-in, uncached)

```lean
-- Available in Tests/ and debug contexts. Requires explicit import.
namespace GPUBackend.Debug

def executeOnce [GPUBackend β] (ctx : β) (computation : ShaderM Unit)
    (namedBuffers : List (String × GPUBackend.Buf β))
    (config : ExecConfig) : IO Unit :=
  -- Creates ephemeral cacheRef, caches nothing
  let ref ← IO.mkRef none
  GPUBackend.executeWithConfigCached ctx computation namedBuffers config 0 ref

end GPUBackend.Debug
```

**Benefit**: Tests can still easily dispatch one-off kernels. But production code doesn't import `GPUBackend.Debug`, so it can't use the uncached path.

## Enforcement

### Level 1: Naming convention (immediate)
- Rename `GPUBackend.execute` → `GPUBackend.Debug.executeOnce`
- Grep in CI: `grep -r "Debug.executeOnce" Hesper/Models/ Hesper/Layers/` → fail if found

### Level 2: Module visibility (medium-term)
- Move `Debug.executeOnce` to a separate module `Hesper.Backend.Debug`
- Production code (Hesper.Models.*, Hesper.Layers.*) does NOT import it
- Lean compiler catches any use as "unknown identifier"

### Level 3: Custom linter (long-term)
- Lean 4 supports custom linters via `@[linter]`
- Check: any `executeWithConfig` call without a persistent cacheRef → warning

## Migration Path

### Step 1: Add CachedKernel (no breaking changes)
```lean
structure CachedKernel (β : Type) [GPUBackend β] where
  computation : ShaderM Unit
  cacheKey : UInt64
  cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β))

def CachedKernel.create [GPUBackend β] (computation : ShaderM Unit) (name : String) : IO (CachedKernel β) := do
  pure { computation, cacheKey := hash name, cacheRef := ← IO.mkRef none }

def CachedKernel.exec [GPUBackend β] (k : CachedKernel β) (ctx : β)
    (bufs : List (String × GPUBackend.Buf β)) (config : ExecConfig) : IO Unit :=
  GPUBackend.executeWithConfigCached ctx k.computation bufs config k.cacheKey k.cacheRef
```

### Step 2: Migrate model code to CachedKernel
Replace:
```lean
GPUBackend.execute ctx (residualAddKernel dim) [("a", a), ("b", b), ("output", o)] (.dispatch1D dim)
```
With:
```lean
state.residAddKernel.exec ctx [("a", a), ("b", b), ("output", o)] (.dispatch1D dim)
```

### Step 3: Rename/hide bare execute
- `GPUBackend.execute` → `GPUBackend.Debug.executeOnce`
- Production code compilation fails if it uses the old name

## FAQ

**Q: What about kernels whose parameters change per-call (e.g., cacheLen)?**
A: Same ShaderM with different parameters generates different PTX → different cacheKey. Each unique (computation, funcName) pair gets its own CUfunction. The cacheRef handles this correctly — different PTX → cache miss → new compilation → cached for next identical call.

**Q: What about kernels called from different buffer configurations?**
A: cacheRef caches the CUfunction (compiled PTX), not the buffer pointers. Buffer pointers are resolved fresh on every call from the named buffer list. This is safe — same kernel code, different data.

**Q: Does this hurt WebGPU performance?**
A: No. WebGPU's `executeWithConfigCached` uses the same pipeline cache as `executeShaderNamed`. The cacheRef just avoids redundant WGSL string regeneration + shader module creation. Both backends benefit equally.
