import Hesper.WGSL.Types
import Hesper.WGSL.Exp

namespace Hesper.WGSL

/-! Statement-level DSL for complete WGSL shaders.
    Supports variable declarations, assignments, loops, conditionals, and shader structure. -/

/-- WGSL Statements -/
inductive Stmt where
  -- Variable declaration with type and initial value
  | varDecl (name : String) (ty : WGSLType) (init : Option (Σ t, Exp t)) : Stmt

  -- Assignment statement
  | assign (name : String) (ty : WGSLType) (value : Exp ty) : Stmt

  -- Array/variable index assignment
  | assignIndex (arrName : String) (index : Exp (.scalar .u32)) (ty : WGSLType) (value : Exp ty) : Stmt

  -- For loop
  | forLoop
      (varName : String)  -- Loop variable
      (init : Exp (.scalar .u32))  -- Initial value
      (cond : Exp (.scalar .bool))  -- Condition
      (update : Exp (.scalar .u32))  -- Update expression
      (body : List Stmt)  -- Loop body
      : Stmt

  -- If statement
  | ifStmt (cond : Exp (.scalar .bool)) (thenBody : List Stmt) (elseBody : List Stmt) : Stmt

  -- Expression statement (for function calls like workgroupBarrier)
  | exprStmt {t : WGSLType} (e : Exp t) : Stmt

  -- Block (sequence of statements)
  | block (stmts : List Stmt) : Stmt

/-- Storage buffer binding -/
structure StorageBuffer where
  group : Nat
  binding : Nat
  name : String
  elemType : WGSLType
  readWrite : Bool  -- true for read_write, false for read-only

/-- Workgroup size specification -/
structure WorkgroupSize where
  x : Nat
  y : Nat := 1
  z : Nat := 1

/-- Built-in variable bindings in compute shader -/
inductive BuiltinBinding where
  | workgroupId : BuiltinBinding
  | localInvocationId : BuiltinBinding
  | globalInvocationId : BuiltinBinding

structure BuiltinParam where
  builtin : BuiltinBinding
  name : String

/-- Shader function parameter -/
structure FunctionParam where
  name : String
  ty : WGSLType
  builtin : Option BuiltinBinding := none

/-- Workgroup variable declaration (shared memory) -/
structure WorkgroupVar where
  name : String
  type : WGSLType
  deriving Repr, BEq

/-- Complete compute shader structure -/
structure ComputeShader where
  -- Extensions to enable
  extensions : List String := []

  -- Diagnostics to disable
  diagnostics : List (String × String) := []  -- (severity, rule)

  -- Struct definitions
  structs : List StructDef := []

  -- Storage buffers
  buffers : List StorageBuffer

  -- Workgroup variables (shared memory)
  workgroupVars : List WorkgroupVar := []

  -- Workgroup size
  workgroupSize : WorkgroupSize

  -- Built-in parameters
  builtins : List BuiltinParam

  -- Shader body statements
  body : List Stmt

/-- Convert builtin binding to WGSL string -/
def BuiltinBinding.toWGSL : BuiltinBinding → String
  | .workgroupId => "workgroup_id"
  | .localInvocationId => "local_invocation_id"
  | .globalInvocationId => "global_invocation_id"

/-- Convert statement to WGSL string with indentation -/
partial def Stmt.toWGSL (indent : Nat := 0) : Stmt → String
  | .varDecl name ty none =>
    let ind := String.ofList (List.replicate indent ' ')
    s!"{ind}var {name}: {ty.toWGSL};\n"

  | .varDecl name ty (some ⟨_t, init⟩) =>
    let ind := String.ofList (List.replicate indent ' ')
    s!"{ind}var {name}: {ty.toWGSL} = {init.toWGSL};\n"

  | .assign name _ty value =>
    let ind := String.ofList (List.replicate indent ' ')
    s!"{ind}{name} = {value.toWGSL};\n"

  | .assignIndex arrName index _ty value =>
    let ind := String.ofList (List.replicate indent ' ')
    s!"{ind}{arrName}[{index.toWGSL}] = {value.toWGSL};\n"

  | .forLoop varName init cond update body =>
    let ind := String.ofList (List.replicate indent ' ')
    let bodyStr := String.join (body.map (·.toWGSL (indent + 2)))
    s!"{ind}for (var {varName}: u32 = {init.toWGSL}; {cond.toWGSL}; {varName} = {update.toWGSL}) " ++ "{\n" ++ bodyStr ++ s!"{ind}" ++ "}\n"

  | .ifStmt cond thenBody elseBody =>
    let ind := String.ofList (List.replicate indent ' ')
    let thenStr := String.join (thenBody.map (·.toWGSL (indent + 2)))
    if elseBody.isEmpty then
      s!"{ind}if ({cond.toWGSL}) " ++ "{\n" ++ thenStr ++ s!"{ind}" ++ "}\n"
    else
      let elseStr := String.join (elseBody.map (·.toWGSL (indent + 2)))
      s!"{ind}if ({cond.toWGSL}) " ++ "{\n" ++ thenStr ++ s!"{ind}" ++ "} else {\n" ++ elseStr ++ s!"{ind}" ++ "}\n"

  | .exprStmt e =>
    let ind := String.ofList (List.replicate indent ' ')
    s!"{ind}{e.toWGSL};\n"

  | .block stmts =>
    String.join (stmts.map (·.toWGSL indent))

/-- Generate WGSL struct definition -/
def StructDef.toWGSL (structDef : StructDef) : String :=
  let fields := structDef.fields.map fun field =>
    s!"  {field.name}: {field.type.toWGSL},"
  "struct " ++ structDef.name ++ " {\n" ++ String.intercalate "\n" fields ++ "\n};\n"

/-- Generate complete WGSL shader code from ComputeShader -/
def ComputeShader.toWGSL (shader : ComputeShader) : String :=
  let extensions := shader.extensions.foldl (fun acc ext => acc ++ s!"enable {ext};\n") ""
  let extensions := if !shader.extensions.isEmpty then extensions ++ "\n" else extensions

  let diagnostics := shader.diagnostics.foldl (fun acc (severity, rule) => acc ++ s!"diagnostic({severity}, {rule});\n") ""
  let diagnostics := if !shader.diagnostics.isEmpty then diagnostics ++ "\n" else diagnostics

  let structs := shader.structs.foldl (fun acc struct => acc ++ struct.toWGSL ++ "\n") ""

  let buffers := shader.buffers.foldl (fun acc buf =>
    let access := if buf.readWrite then "read_write" else "read"
    acc ++ s!"@group({buf.group}) @binding({buf.binding}) var<storage, {access}> {buf.name}: array<{buf.elemType.toWGSL}>;\n") ""
  let buffers := if !shader.buffers.isEmpty then buffers ++ "\n" else buffers

  let workgroupVars := shader.workgroupVars.foldl (fun acc wgVar =>
    acc ++ s!"var<workgroup> {wgVar.name}: {wgVar.type.toWGSL};\n") ""
  let workgroupVars := if !shader.workgroupVars.isEmpty then workgroupVars ++ "\n" else workgroupVars

  let header := s!"@compute @workgroup_size({shader.workgroupSize.x}, {shader.workgroupSize.y}, {shader.workgroupSize.z})\n"

  let params := shader.builtins.map fun p =>
    s!"@builtin({p.builtin.toWGSL}) {p.name}: vec3<u32>"
  let paramStr := "fn main(" ++ String.intercalate ",\n        " params ++ ") {\n"

  let body := shader.body.foldl (fun acc stmt => acc ++ stmt.toWGSL 2) ""

  extensions ++ diagnostics ++ structs ++ buffers ++ workgroupVars ++ header ++ paramStr ++ body ++ "}\n"

-- ============================================================================
-- Smart Constructors for Statements
-- ============================================================================

/-- Declare a variable with optional initial value -/
def declareVar (name : String) (ty : WGSLType) (init : Option (Σ t, Exp t) := none) : Stmt :=
  Stmt.varDecl name ty init

/-- Assign to a variable -/
def assign {ty : WGSLType} (name : String) (value : Exp ty) : Stmt :=
  Stmt.assign name ty value

/-- Assign to array element -/
def assignAt {ty : WGSLType} (arrName : String) (index : Exp (.scalar .u32)) (value : Exp ty) : Stmt :=
  Stmt.assignIndex arrName index ty value

/-- For loop statement -/
def forLoop
    (varName : String)
    (init : Exp (.scalar .u32))
    (cond : Exp (.scalar .bool))
    (update : Exp (.scalar .u32))
    (body : List Stmt)
    : Stmt :=
  Stmt.forLoop varName init cond update body

/-- Expression statement (e.g., function calls) -/
def expr {t : WGSLType} (e : Exp t) : Stmt :=
  Stmt.exprStmt e

-- ============================================================================
-- Helper Functions for Common Patterns
-- ============================================================================

/-- Initialize array elements with a generator function -/
def initArray {ty : WGSLType} (arrName : String) (count : Nat) (gen : Nat → Exp ty) : List Stmt :=
  (List.range count).map fun (i : Nat) =>
    let val : Exp ty := gen i
    let idx : Exp (.scalar .u32) := Exp.litU32 i
    Stmt.assignIndex arrName idx ty val

/-- Nested loops for 2D iteration -/
def nestedMap2D {α : Type} (rows cols : Nat) (f : Nat → Nat → α) : List α :=
  (List.range rows).foldl (fun acc i =>
    acc ++ (List.range cols).map (fun j => f i j)) []

-- ============================================================================
-- HOAS-style Loop Helpers (Inspired by Haskell DSL)
-- ============================================================================

/-- HOAS-style loop construct.
    Instead of requiring a string variable name, takes a function that receives
    the loop variable as an Exp and returns the loop body.

    Example:
      loop "i" 0 n 1 (fun i => [...statements using i...])
    -/
def loop
    (varName : String)
    (start : Exp (.scalar .u32))
    (end_ : Exp (.scalar .u32))
    (step : Exp (.scalar .u32))
    (body : Exp (.scalar .u32) → List Stmt)
    : Stmt :=
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  Stmt.forLoop varName start condition update (body loopVar)

/-- Compile-time iteration over a list, generating statements for each element.
    Similar to Haskell's `staticFor` - unrolls the loop at DSL compile time.

    Example:
      staticFor [0, 1, 2, 3] (fun i => assignAt "arr" i someValue)
    -/
def staticFor {α : Type} (items : List α) (f : α → List Stmt) : List Stmt :=
  items.foldl (fun acc item => acc ++ f item) []

/-- Helper to iterate over a range with a body function -/
def staticForRange (start : Nat) (end_ : Nat) (f : Nat → List Stmt) : List Stmt :=
  staticFor (List.range (end_ - start) |>.map (fun x => x + start)) f

/-- Helper to iterate with index and value from a list -/
def staticForIndexed {α : Type} (items : List α) (f : Nat → α → List Stmt) : List Stmt :=
  let indexed := List.zip (List.range items.length) items
  staticFor indexed (fun (i, item) => f i item)

-- ============================================================================
-- Matrix Operation Helpers
-- ============================================================================

/-- Load multiple matrices into an array variable -/
def loadMatricesLeft
    {st : ScalarType} {m k : Nat}
    (arrName : String)
    (count : Nat)
    (bufferRef : String)
    (getOffset : Nat → Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : List Stmt :=
  let matTy := WGSLType.subgroupMatrixLeft st m k
  staticForRange 0 count (fun i =>
    [ Stmt.assignIndex arrName (Exp.litU32 i) matTy
        (Exp.subgroupMatrixLoad bufferRef (getOffset i) (Exp.litBool false) stride) ])

/-- Load multiple matrices (right operand) into an array variable -/
def loadMatricesRight
    {st : ScalarType} {k n : Nat}
    (arrName : String)
    (count : Nat)
    (bufferRef : String)
    (getOffset : Nat → Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : List Stmt :=
  let matTy := WGSLType.subgroupMatrixRight st k n
  staticForRange 0 count (fun i =>
    [ Stmt.assignIndex arrName (Exp.litU32 i) matTy
        (Exp.subgroupMatrixLoadRight bufferRef (getOffset i) (Exp.litBool false) stride) ])

/-- Perform multiply-accumulate on 2D grid of matrices -/
def matrixMulAccGrid
    {st : ScalarType} {m k n tm tn : Nat}
    (axArrName bxArrName accArrName : String)
    : List Stmt :=
  let leftMatTy := WGSLType.subgroupMatrixLeft st m k
  let rightMatTy := WGSLType.subgroupMatrixRight st k n
  let resultMatTy := WGSLType.subgroupMatrixResult st m n
  let axVar : Exp (.array leftMatTy tm) := Exp.var axArrName
  let bxVar : Exp (.array rightMatTy tn) := Exp.var bxArrName
  let accVar : Exp (.array resultMatTy (tm * tn)) := Exp.var accArrName
  staticForRange 0 tn (fun j =>
    staticForRange 0 tm (fun i =>
      let idx := j * tm + i
      let resultTy := WGSLType.subgroupMatrixResult st m n
      [ Stmt.assignIndex accArrName (Exp.litU32 idx) resultTy
          (Exp.subgroupMatrixMultiplyAccumulate
            (Exp.index axVar (Exp.litU32 i))
            (Exp.index bxVar (Exp.litU32 j))
            (Exp.index accVar (Exp.litU32 idx))) ]))

/-- Store multiple result matrices from array to buffer -/
def storeMatricesResult
    {st : ScalarType} {m n tm tn : Nat}
    (arrName : String)
    (bufferRef : String)
    (getOffset : Nat → Nat → Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : List Stmt :=
  let resultMatTy := WGSLType.subgroupMatrixResult st m n
  let accVar : Exp (.array resultMatTy (tm * tn)) := Exp.var arrName
  staticForRange 0 tm (fun i =>
    staticForRange 0 tn (fun j =>
      let idx := j * tm + i
      [ expr (Exp.subgroupMatrixStore bufferRef (getOffset i j)
                (Exp.index accVar (Exp.litU32 idx)) (Exp.litBool false) stride) ]))

end Hesper.WGSL
