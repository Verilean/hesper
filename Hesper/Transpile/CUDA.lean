import Hesper.Transpile.CUDA.Lex
import Hesper.Transpile.CUDA.AST
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.Lower
import Hesper.WGSL.ExpRepr

/-! # CUDA → ShaderM transpiler — top-level entry points

One-shot helpers that combine `lex → parse → lower` so callers can
ergonomically convert a CUDA source string into a typed Hesper `Exp`.

```
import Hesper.Transpile.CUDA

-- u32 (Phase 1)
let v0i := cudaU32! "v0 & 0x0F0F0F0F"

-- i32 (Phase 2: __dp4a)
let dp  := cudaI32! "__dp4a(v0, u0, acc)"

-- f32 (Phase 2: float arith + casts)
let s   := cudaF32! "(float) dot * scale"
```

Each variant has three forms:

  cudaXxx           : Except-returning, total
  cudaXxxWithEnv    : env-aware, Except-returning
  cudaXxx!          : panicking (compile-time-fixed snippets)
-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL

/-! ## u32 entry points (Phase 1) -/

/-- Parse + lower a CUDA expression to `Exp (.scalar .u32)`. -/
def cudaU32WithEnv (env : Env) (src : String) : Except String (Exp (.scalar .u32)) := do
  let cexpr ← parseExprStr src
  lowerU32 env cexpr

/-- Phase 1 alias: takes a u32-only `IdEnv`. -/
def cudaU32WithIdEnv (env : IdEnv) (src : String) : Except String (Exp (.scalar .u32)) :=
  cudaU32WithEnv (Env.ofU32 env) src

def cudaU32 (src : String) : Except String (Exp (.scalar .u32)) :=
  cudaU32WithEnv .empty src

def cudaU32! (src : String) : Exp (.scalar .u32) :=
  match cudaU32 src with
  | .ok e => e
  | .error msg => panic! s!"cudaU32!: {msg}\n  source: {src}"

/-! ## i32 entry points (Phase 2) -/

def cudaI32WithEnv (env : Env) (src : String) : Except String (Exp (.scalar .i32)) := do
  let cexpr ← parseExprStr src
  lowerI32 env cexpr

def cudaI32 (src : String) : Except String (Exp (.scalar .i32)) :=
  cudaI32WithEnv .empty src

def cudaI32! (src : String) : Exp (.scalar .i32) :=
  match cudaI32 src with
  | .ok e => e
  | .error msg => panic! s!"cudaI32!: {msg}\n  source: {src}"

/-! ## f32 entry points (Phase 2) -/

def cudaF32WithEnv (env : Env) (src : String) : Except String (Exp (.scalar .f32)) := do
  let cexpr ← parseExprStr src
  lowerF32 env cexpr

def cudaF32 (src : String) : Except String (Exp (.scalar .f32)) :=
  cudaF32WithEnv .empty src

def cudaF32! (src : String) : Exp (.scalar .f32) :=
  match cudaF32 src with
  | .ok e => e
  | .error msg => panic! s!"cudaF32!: {msg}\n  source: {src}"

end Hesper.Transpile.CUDA
