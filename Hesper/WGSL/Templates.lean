import Hesper.WGSL.Types
import Hesper.WGSL.Exp

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
  let x : Exp (.scalar .f32) := Exp.var "x"
  let body := f x
  let bodyCode := body.toWGSL
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\n" ++
  "@compute @workgroup_size(256)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let i = gid.x;\n" ++
  "  if (i < arrayLength(&data)) {\n" ++
  "    let x = data[i];\n" ++
  s!"    data[i] = {bodyCode};\n" ++
  "  }\n" ++
  "}"

/-- Generate WGSL shader code for a binary operation (combining two arrays).

    Example:
    ```lean
    let shader := generateBinaryShader (fun a b => a + b)  -- Vector addition
    ```
-/
def generateBinaryShader (f : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32)) : String :=
  let a : Exp (.scalar .f32) := Exp.var "a"
  let b : Exp (.scalar .f32) := Exp.var "b"
  let body := f a b
  let bodyCode := body.toWGSL
  "@group(0) @binding(0) var<storage, read_write> dataA: array<f32>;\n" ++
  "@group(0) @binding(1) var<storage, read_write> dataB: array<f32>;\n" ++
  "@group(0) @binding(2) var<storage, read_write> dataC: array<f32>;\n\n" ++
  "@compute @workgroup_size(256)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let i = gid.x;\n" ++
  "  if (i < arrayLength(&dataA)) {\n" ++
  "    let a = dataA[i];\n" ++
  "    let b = dataB[i];\n" ++
  s!"    dataC[i] = {bodyCode};\n" ++
  "  }\n" ++
  "}"

end Hesper.WGSL.Templates
