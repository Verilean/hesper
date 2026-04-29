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

#eval Hesper.Transpile.CUDA.cudaU32! "(v >> 4) & 0x0F0F0F0F"
-- Exp.bitAnd (Exp.shiftRight (Exp.var "v") (Exp.litU32 4)) (Exp.litU32 252645135)

example : Exp (.scalar .u32) :=
  Hesper.Transpile.CUDA.cudaU32! "(v >> 4) & 0x0F"
```

The `!` variant (`cudaU32!`) panics on parse/lower failure, suitable
for compile-time fixed CUDA snippets. The total variant (`cudaU32`)
returns `Except String _` for runtime / dynamic CUDA strings.
-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL

/-- Parse + lower a CUDA expression string to `Exp (.scalar .u32)`,
    using the supplied identifier environment. Returns `Except` so
    callers can handle errors. -/
def cudaU32WithEnv (env : IdEnv) (src : String)
    : Except String (Exp (.scalar .u32)) := do
  let cexpr ← parseExprStr src
  lowerU32 env cexpr

/-- Parse + lower a CUDA expression string to `Exp (.scalar .u32)`
    with an empty environment. -/
def cudaU32 (src : String) : Except String (Exp (.scalar .u32)) :=
  cudaU32WithEnv emptyEnv src

/-- Panicking variant: returns the lowered `Exp`, panics on parse/lower
    error. Use for compile-time-fixed CUDA snippets where any failure
    indicates a transpiler bug rather than user input. -/
def cudaU32! (src : String) : Exp (.scalar .u32) :=
  match cudaU32 src with
  | .ok e => e
  | .error msg => panic! s!"cudaU32!: {msg}\n  source: {src}"

/-- `cudaU32!` with a custom env. -/
def cudaU32EnvOrPanic! (env : IdEnv) (src : String) : Exp (.scalar .u32) :=
  match cudaU32WithEnv env src with
  | .ok e => e
  | .error msg => panic! s!"cudaU32EnvOrPanic!: {msg}\n  source: {src}"

end Hesper.Transpile.CUDA
