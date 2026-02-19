import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.Monad

namespace Hesper.WGSL.CodeGen

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM ShaderState)

/-!
# WGSL Code Generation

Complete code generation from ShaderM monad to WGSL modules.

Features:
- Automatic binding assignment for storage buffers
- Workgroup variable declarations
- Struct definitions
- Function generation with attributes
- Extension and diagnostic directives
- Complete module assembly

Usage:
```lean
def myShader : ShaderM Unit := do
  -- ... shader code ...

let module := generateModule "myKernel" {x := 256, y := 1, z := 1} [] myShader
IO.println module.toWGSL
```
-/

/-- Function parameter with optional built-in binding -/
structure FunctionParam where
  name : String
  ty : WGSLType
  builtin : Option BuiltinBinding := none

namespace FunctionParam

/-- Generate WGSL parameter declaration -/
def toWGSL (param : FunctionParam) : String :=
  match param.builtin with
  | none => s!"{param.name}: {param.ty.toWGSL}"
  | some builtin =>
    let builtinStr := match builtin with
      | .globalInvocationId => "global_invocation_id"
      | .localInvocationId => "local_invocation_id"
      | .workgroupId => "workgroup_id"
    s!"@builtin({builtinStr}) {param.name}: {param.ty.toWGSL}"

end FunctionParam

/-- Function declaration with attributes, parameters, and body -/
structure FunctionDecl where
  name : String
  attributes : List String := []  -- e.g., ["@compute", "@workgroup_size(256, 1, 1)"]
  params : List FunctionParam
  returnType : Option WGSLType := none
  body : List Stmt

namespace FunctionDecl

/-- Generate complete WGSL function -/
def toWGSL (func : FunctionDecl) : String :=
  let attrs := func.attributes.map (· ++ "\n") |> String.join
  let params := func.params.map FunctionParam.toWGSL |> String.intercalate ",\n        "
  let retType := match func.returnType with
    | none => ""
    | some ty => s!" -> {ty.toWGSL}"
  let bodyStr := func.body.map (Stmt.toWGSL 2) |> String.join
  s!"{attrs}fn {func.name}({params}){retType} " ++ "{\n" ++ bodyStr ++ "}\n"

end FunctionDecl

/-- Complete shader module with all declarations -/
structure ShaderModule where
  extensions : List String := []
  diagnostics : List (String × String) := []  -- (severity, rule)
  structs : List StructDef := []
  storageBuffers : List StorageBuffer := []
  workgroupVars : List WorkgroupVar := []
  functions : List FunctionDecl

namespace ShaderModule

/-- Generate storage buffer declaration with binding -/
def storageBufferDecl (buf : StorageBuffer) : String :=
  let accessMode := if buf.readWrite then "read_write" else "read"
  s!"@group({buf.group}) @binding({buf.binding})\nvar<storage, {accessMode}> {buf.name}: {buf.elemType.toWGSL};\n"

/-- Generate workgroup variable declaration -/
def workgroupVarDecl (wvar : WorkgroupVar) : String :=
  s!"var<workgroup> {wvar.name}: {wvar.type.toWGSL};\n"

/-- Generate struct definition -/
def structDecl (struct : StructDef) : String :=
  let fields := struct.fields.map fun field =>
    s!"  {field.name}: {field.type.toWGSL},"
  "struct " ++ struct.name ++ " {\n" ++ String.intercalate "\n" fields ++ "\n}\n"

/-- Generate complete WGSL module -/
def toWGSL (module : ShaderModule) : String :=
  -- Extensions section
  let extensionSection :=
    if module.extensions.isEmpty && module.diagnostics.isEmpty then ""
    else
      let exts := module.extensions.map (fun ext => s!"enable {ext};") |> String.intercalate "\n"
      let diags := module.diagnostics.map (fun (sev, rule) => s!"diagnostic({sev}, {rule});") |> String.intercalate "\n"
      let combined := [exts, diags].filter (· ≠ "") |> String.intercalate "\n"
      if combined.isEmpty then "" else combined ++ "\n\n"

  -- Struct definitions
  let structSection :=
    if module.structs.isEmpty then ""
    else (module.structs.map structDecl |> String.join) ++ "\n"

  -- Global storage buffers
  let storageSection :=
    if module.storageBuffers.isEmpty then ""
    else (module.storageBuffers.map storageBufferDecl |> String.join) ++ "\n"

  -- Workgroup variables
  let workgroupSection :=
    if module.workgroupVars.isEmpty then ""
    else (module.workgroupVars.map workgroupVarDecl |> String.join) ++ "\n"

  -- Functions
  let functionSection := module.functions.map FunctionDecl.toWGSL |> String.join

  extensionSection ++ structSection ++ storageSection ++ workgroupSection ++ functionSection

end ShaderModule

/-- Generate a compute shader module from ShaderM monad -/
def generateComputeModule
    (funcName : String)
    (workgroupSize : WorkgroupSize)
    (extensions : List String := [])
    (computation : ShaderM Unit)
    : ShaderModule :=
  let state : ShaderState := ShaderM.exec computation

  -- Convert declared buffers to StorageBuffer list with automatic binding
  let storageBuffers := state.declaredBuffers.mapIdx fun i (name, ty, mode) =>
    { group := 0
      binding := i
      name := name
      elemType := ty
      readWrite := match mode with | .readWrite => true | .read => false }

  -- Convert shared vars to WorkgroupVar list
  let workgroupVars := state.sharedVars.map fun (name, ty) =>
    { name := name, type := ty }

  -- Create function parameters for built-ins
  let params : List FunctionParam := [
    { name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId },
    { name := "local_invocation_id", ty := .vec3 .u32, builtin := some .localInvocationId },
    { name := "workgroup_id", ty := .vec3 .u32, builtin := some .workgroupId }
  ]

  -- Create function attributes
  let attrs := [
    "@compute",
    s!"@workgroup_size({workgroupSize.x}, {workgroupSize.y}, {workgroupSize.z})"
  ]

  -- Create main compute function
  let mainFunc : FunctionDecl := {
    name := funcName
    attributes := attrs
    params := params
    returnType := none
    body := state.stmts
  }

  { extensions := extensions
    diagnostics := []
    structs := []
    storageBuffers := storageBuffers
    workgroupVars := workgroupVars
    functions := [mainFunc] }

/-- Generate compute shader module with diagnostics support -/
def generateComputeModuleWithDiagnostics
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (extensions : List String := [])
    (diagnostics : List (String × String) := [])
    (computation : ShaderM Unit)
    : ShaderModule :=
  let state : ShaderState := ShaderM.exec computation

  -- Convert declared buffers to StorageBuffer list with automatic binding
  let storageBuffers := state.declaredBuffers.mapIdx fun i (name, ty, mode) =>
    { group := 0
      binding := i
      name := name
      elemType := ty
      readWrite := match mode with | .readWrite => true | .read => false }

  -- Convert shared vars to WorkgroupVar list
  let workgroupVars := state.sharedVars.map fun (name, ty) =>
    { name := name, type := ty }

  -- Create function parameters for built-ins
  let params : List FunctionParam := [
    { name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId },
    { name := "local_invocation_id", ty := .vec3 .u32, builtin := some .localInvocationId },
    { name := "workgroup_id", ty := .vec3 .u32, builtin := some .workgroupId }
  ]

  -- Create function attributes
  let attrs := [
    "@compute",
    s!"@workgroup_size({workgroupSize.x}, {workgroupSize.y}, {workgroupSize.z})"
  ]

  -- Create main compute function
  let mainFunc : FunctionDecl := {
    name := funcName
    attributes := attrs
    params := params
    returnType := none
    body := state.stmts
  }

  { extensions := extensions
    diagnostics := diagnostics
    structs := []
    storageBuffers := storageBuffers
    workgroupVars := workgroupVars
    functions := [mainFunc] }

/-- Convenience function: Generate WGSL string from ShaderM computation -/
def generateWGSL
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (extensions : List String := [])
    (diagnostics : List (String × String) := [])
    (computation : ShaderM Unit)
    : String :=
  (generateComputeModuleWithDiagnostics funcName workgroupSize extensions diagnostics computation).toWGSL

/-- Generate WGSL with default parameters -/
def generateWGSLSimple (computation : ShaderM Unit) : String :=
  generateWGSL "main" {x := 256, y := 1, z := 1} [] [] computation

end Hesper.WGSL.CodeGen
