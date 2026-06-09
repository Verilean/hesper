# Type-Level Safety Proposals

Design proposals for catching common bugs at compile time via Lean 4's type system.

## Problem 1: Silent codegen fallbacks

**Current**: `| _, _ => (ra, s)` silently drops operations when types mismatch.

**Proposal**: Return `Option` or `Except` from `expToPTX`, making unhandled cases visible.

```lean
-- Current (silent bug)
| .add a b =>
    match ra, rb with
    | .f32 a, .f32 b => ...
    | .u32 a, .u32 b => ...
    | _, _ => (ra, s)  -- SILENTLY DROPS THE ADD

-- Proposed (compile-time error if unmatched)
partial def expToPTX (e : Exp t) (s : GenState) : Except String ExpResult :=
  | .add a b =>
    match ra, rb with
    | .f32 a, .f32 b => .ok (...)
    | .u32 a, .u32 b => .ok (...)
    | _, _ => .error s!"add: type mismatch {ra.typeName} + {rb.typeName}"
```

**Cost**: Propagating `Except` through all callers. Medium refactor.

**Alternative**: Keep pure, but add a `#check_codegen_coverage` test that exercises every Exp constructor and verifies no fallback was taken. (Already partially done in CUDAPTXInstTest.)

## Problem 2: Uncached dispatch

**Current**: Nothing prevents calling `GPUBackend.execute` instead of `executeWithConfigCached`. Both type-check. The performance difference (90μs vs 5μs) is invisible at compile time.

**Proposal A**: Remove `GPUBackend.execute` entirely, require cacheRef always.

```lean
-- Only API: must provide cacheRef
class GPUBackend (β : Type) where
  executeWithConfigCached : β → ShaderM Unit → List (String × Buf) → ExecConfig → UInt64 → IO.Ref (Option CachedDispatch) → IO Unit
  -- No bare "execute" method
```

**Cost**: Every callsite needs a cacheRef. For one-off dispatches (tests), provide `IO.mkRef none`.

**Proposal B**: Wrapper type `CachedKernel` that bundles ShaderM + cacheRef.

```lean
structure CachedKernel (β : Type) [GPUBackend β] where
  computation : ShaderM Unit
  config : ExecConfig
  cacheKey : UInt64
  cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β))

def CachedKernel.dispatch (k : CachedKernel β) (ctx : β) (bufs : List (String × GPUBackend.Buf β)) : IO Unit :=
  GPUBackend.executeWithConfigCached ctx k.computation bufs k.config k.cacheKey k.cacheRef
```

All layers would hold `CachedKernel` instead of raw `ShaderM Unit`. **Cannot forget the cache.**

## Problem 3: Bounds safety (select vs if_)

**Current**: `Exp.select` compiles fine but causes CUDA crash when guard expression is used to mask OOB reads.

**Proposal**: Lint/typeclass for "safe kernel" that checks all readBuffer calls are dominated by an if_ guard.

This is hard to enforce statically because Exp is a value-level DSL. Best approach: runtime check in CUDA codegen that warns when select is used with a readBuffer in the same expression tree.

**Practical alternative**: Coding convention + grep-based CI check:

```bash
# CI check: no Exp.select with readBuffer in the same ShaderM do-block
grep -B5 "Exp.select.*inBounds" Hesper/ -r | grep "readBuffer" && echo "WARNING: select-based bounds check"
```

## Problem 4: Subgroups extension

**Current**: Auto-detected by scanning WGSL output for "subgroupAdd". Works but fragile.

**Proposal**: Track in ShaderState. When `Exp.subgroupAdd` is emitted, set flag.

```lean
structure ShaderState where
  ...
  needsSubgroups : Bool := false  -- already added, just needs wiring
```

Wire it: when any Stmt contains `Exp.subgroupAdd`, the ShaderM sets `needsSubgroups := true`. WGSL codegen checks this flag.

## Problem 5: Buffer type safety

**Current**: Named buffer system (`[("input", buf)]`) has no compile-time check that buffer order matches kernel declaration order.

**Proposal**: Typed buffer binding via dependent types.

```lean
-- Kernel declares its buffers at the type level
structure KernelSig where
  inputs : List (String × WGSLType)
  outputs : List (String × WGSLType)

-- Dispatch requires matching buffer list
def dispatch [GPUBackend β] (ctx : β) (kernel : Kernel sig) (bufs : MatchingBuffers sig β) : IO Unit
```

**Cost**: Major DSL redesign. Not practical short-term.

**Practical alternative**: Runtime assertion in CUDA backend that declared buffer count == provided buffer count. Already partially done (throws on missing buffer).

## Recommended Priority

| Proposal | Impact | Cost | Priority |
|----------|--------|------|----------|
| Remove bare `execute` (2A) | High — eliminates #1 perf bug class | Low | **Do now** |
| CachedKernel wrapper (2B) | High — makes cache unforgettable | Medium | Next |
| Codegen Except (1) | Medium — catches silent bugs | Medium | After tests cover all paths |
| ShaderState needsSubgroups (4) | Low — already auto-detected | Low | Easy win |
| Runtime buffer count check (5) | Low — already catches mismatches | Low | Already done |
| select lint (3) | Medium — hard to enforce | High | Keep as convention |
