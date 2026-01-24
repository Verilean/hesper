import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

namespace Hesper.WGSL.Templates

/-!
# WGSL Shader Templates

Template functions for generating common WGSL shader patterns.
These are pure Lean functions with no FFI dependencies - perfect for examples and testing.
-/

open Hesper.WGSL

/-- Generate WGSL shader code for a simple unary operation.
    This takes a function body expression and wraps it in a complete shader.

    Example:
    ```lean
    let shader := generateUnaryShader (fun x => x * lit 2.0)
    ```
-/
def generateUnaryShader (f : Exp (.scalar .f32) → Exp (.scalar .f32)) : String :=
  let computation : Monad.ShaderM Unit := do
    let gid ← Monad.ShaderM.globalId
    let i := Exp.vec3X gid

    let dataBuf ← Monad.ShaderM.declareInputBuffer "data" (.scalar .f32)
    let _outputBuf ← Monad.ShaderM.declareOutputBuffer "data" (.scalar .f32)

    let len := Exp.arrayLength (t := .scalar .f32) "&data"
    Monad.ShaderM.if_ (Exp.lt i len) (do
      let x ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 1024) dataBuf i
      let result := f x
      Monad.ShaderM.writeBuffer (ty := .scalar .f32) dataBuf i result
    ) (pure ())

  CodeGen.generateWGSL "main" {x := 256, y := 1, z := 1} ([] : List String) ([] : List (String × String)) computation

/-- Generate WGSL shader code for a binary operation (combining two arrays).

    Example:
    ```lean
    let shader := generateBinaryShader (fun a b => a + b)  -- Vector addition
    ```
-/
def generateBinaryShader (f : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32)) : String :=
  let computation : Monad.ShaderM Unit := do
    let gid ← Monad.ShaderM.globalId
    let i := Exp.vec3X gid

    let dataABuf ← Monad.ShaderM.declareInputBuffer "dataA" (.scalar .f32)
    let dataBBuf ← Monad.ShaderM.declareInputBuffer "dataB" (.scalar .f32)
    let dataCBuf ← Monad.ShaderM.declareOutputBuffer "dataC" (.scalar .f32)

    let len := Exp.arrayLength (t := .scalar .f32) "&dataA"
    Monad.ShaderM.if_ (Exp.lt i len) (do
      let a ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 1024) dataABuf i
      let b ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 1024) dataBBuf i
      let result := f a b
      Monad.ShaderM.writeBuffer (ty := .scalar .f32) dataCBuf i result
    ) (pure ())

  CodeGen.generateWGSL "main" {x := 256, y := 1, z := 1} ([] : List String) ([] : List (String × String)) computation

end Hesper.WGSL.Templates
