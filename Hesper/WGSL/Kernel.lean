import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader

namespace Hesper.WGSL

/-! Composable kernel abstraction for kernel fusion.

This module provides a high-level abstraction for composable GPU kernels.
Instead of writing imperative procedures, we treat kernels as composable
functions (Input → Output), enabling kernel fusion optimizations.

Key benefits:
  - Compose operations with monadic bind (>>=) and sequencing (>>)
  - Fuse multiple operations into single shader pass
  - Reduce global memory traffic
  - Type-safe workgroup size tracking

Example:
  -- Fuse: Load → Multiply → Add → ReLU → Store into single pass
  fusedKernel : Kernel 256 1 1 (Exp (.scalar .u32)) Unit
  fusedKernel =
    loadVec inputPtr
    |> andThen (mapK (· * Exp.litF32 2.0))
    |> andThen (mapK (· + Exp.litF32 1.0))
    |> andThen (mapK relu)
    |> andThen (storeVec outputPtr)
-/

/-- ShaderM monad - builds up statement lists while computing a value.
    This is a state monad that accumulates statements. -/
abbrev ShaderM (α : Type) := StateM (List Stmt) α

/-- A composable kernel function running on a specific workgroup size.

Type parameters:
  wX, wY, wZ: Workgroup dimensions (natural numbers)
  i: Input type (e.g., Exp (.scalar .f32), or tuples)
  o: Output type

This abstraction allows us to compose operations and perform kernel fusion.
Multiple operations can be fused into a single shader pass, reducing
global memory roundtrips. -/
structure Kernel (wX wY wZ : Nat) (i o : Type) where
  unKernel : i → ShaderM o

namespace Kernel

/-- Identity kernel: passes input through unchanged -/
def id {wX wY wZ : Nat} {α : Type} : Kernel wX wY wZ α α :=
  ⟨fun x => pure x⟩

/-- Compose two kernels: g ∘ f means "f first, then g"
    The composition is performed in the ShaderM monad, so side effects
    (like memory operations) are properly sequenced. -/
def comp {wX wY wZ : Nat} {a b c : Type}
    (g : Kernel wX wY wZ b c)
    (f : Kernel wX wY wZ a b)
    : Kernel wX wY wZ a c :=
  ⟨fun x => do
    let y ← f.unKernel x
    g.unKernel y⟩

/-- Infix operator for kernel composition (like >>> in Haskell Category) -/
infixl:90 " |> " => comp

/-- Monadic bind for kernels -/
def andThen {wX wY wZ : Nat} {a b : Type}
    (k : Kernel wX wY wZ a b)
    (f : b → Kernel wX wY wZ b b)
    : Kernel wX wY wZ a b :=
  ⟨fun x => do
    let y ← k.unKernel x
    let k' := f y
    k'.unKernel y⟩

end Kernel

-- ============================================================================
-- Kernel Construction Helpers
-- ============================================================================

/-- Lift a pure DSL expression transformation into a Kernel.

This allows you to turn any pure expression transformation
(like (· * 2.0) or (· + 1.0)) into a composable kernel.

Example:
  mapK (· * Exp.litF32 2.0)  -- Multiply by 2
  mapK (· + Exp.litF32 1.0)  -- Add 1
  mapK relu                  -- Apply ReLU

These can be composed:
  mapK (· * 2.0) |> mapK (· + 1.0) |> mapK relu
-/
def mapK {wX wY wZ : Nat} {ty : WGSLType}
    (f : Exp ty → Exp ty)
    : Kernel wX wY wZ (Exp ty) (Exp ty) :=
  ⟨fun x => pure (f x)⟩

/-- Emit a statement (side effect) in a kernel.
    This adds the statement to the accumulated list. -/
def emit {wX wY wZ : Nat} (s : Stmt) : Kernel wX wY wZ Unit Unit :=
  ⟨fun _ => do
    modify (· ++ [s])
    pure ()⟩

/-- Emit multiple statements -/
def emitMany {wX wY wZ : Nat} (stmts : List Stmt) : Kernel wX wY wZ Unit Unit :=
  ⟨fun _ => do
    modify (· ++ stmts)
    pure ()⟩

-- ============================================================================
-- Memory Operations
-- ============================================================================

/-- Load operation: reads from a buffer at given index.

Input: index expression
Output: loaded value

Note: The buffer must be an array type. The size parameter n is for type checking. -/
def loadBuffer {wX wY wZ : Nat} {ty : WGSLType} {n : Nat}
    (bufferName : String)
    : Kernel wX wY wZ (Exp (.scalar .u32)) (Exp ty) :=
  ⟨fun idx => do
    -- Reference the buffer and index into it
    let bufferVar : Exp (.array ty n) := Exp.var bufferName
    let loadExp : Exp ty := Exp.index bufferVar idx
    -- For fusion, we don't create intermediate variables - just return the expression
    pure loadExp⟩

/-- Store operation: writes to a buffer at given index.

Input: (index, value) pair
Output: unit (side effect only) -/
def storeBuffer {wX wY wZ : Nat} {ty : WGSLType}
    (bufferName : String)
    : Kernel wX wY wZ (Exp (.scalar .u32) × Exp ty) Unit :=
  ⟨fun (idx, val) => do
    modify (· ++ [Stmt.assignIndex bufferName idx ty val])
    pure ()⟩

/-- Pair an expression with an index for storage -/
def pairWithIndex {wX wY wZ : Nat} {ty : WGSLType}
    (idx : Exp (.scalar .u32))
    : Kernel wX wY wZ (Exp ty) (Exp (.scalar .u32) × Exp ty) :=
  ⟨fun val => pure (idx, val)⟩

-- ============================================================================
-- Kernel Execution
-- ============================================================================

/-- Execute a kernel and extract the generated statements.

This runs the kernel computation and returns both the result value
and the list of statements that were generated. -/
def runKernel {wX wY wZ : Nat} {i o : Type}
    (k : Kernel wX wY wZ i o)
    (input : i)
    : o × List Stmt :=
  let (result, stmts) := k.unKernel input []
  (result, stmts)

/-- Execute a kernel and return only the statements (for side-effect kernels) -/
def execKernel {wX wY wZ : Nat} {i o : Type}
    (k : Kernel wX wY wZ i o)
    (input : i)
    : List Stmt :=
  let (_, stmts) := runKernel k input
  stmts

end Hesper.WGSL
