import Hesper.WGSL.Types

/-!
# Tensor Type Definitions

Core types for tensor operations in Hesper.
Supports multi-dimensional tensors with shape information.
-/

namespace Hesper.Tensor

/-- Tensor shape represented as a list of dimensions -/
structure Shape where
  dims : List Nat
  deriving Inhabited, Repr

-- Common tensor shapes
namespace Shape

  def scalar : Shape := ⟨[]⟩

  def vector (n : Nat) : Shape := ⟨[n]⟩

  def matrix (rows cols : Nat) : Shape := ⟨[rows, cols]⟩

  def tensor3D (d1 d2 d3 : Nat) : Shape := ⟨[d1, d2, d3]⟩

  def tensor4D (d1 d2 d3 d4 : Nat) : Shape := ⟨[d1, d2, d3, d4]⟩

  /-- Total number of elements in the tensor -/
  def size (s : Shape) : Nat :=
    s.dims.foldl (· * ·) 1

  /-- Rank (number of dimensions) -/
  def rank (s : Shape) : Nat :=
    s.dims.length

end Shape

/-- Tensor data type (scalar type of elements) -/
inductive DType
  | f32
  | f16
  | i32
  | u32
  deriving Inhabited, Repr, BEq

namespace DType

  def toWGSL : DType → WGSL.ScalarType
    | f32 => .f32
    | f16 => .f16
    | i32 => .i32
    | u32 => .u32

  def sizeBytes : DType → Nat
    | f32 => 4
    | f16 => 2
    | i32 => 4
    | u32 => 4

end DType

/-- Tensor descriptor with shape and data type -/
structure TensorDesc where
  shape : Shape
  dtype : DType
  deriving Inhabited, Repr

namespace TensorDesc

  /-- Total size in bytes -/
  def sizeBytes (desc : TensorDesc) : Nat :=
    desc.shape.size * desc.dtype.sizeBytes

  /-- Create a matrix descriptor -/
  def matrix (rows cols : Nat) (dtype : DType := .f32) : TensorDesc :=
    { shape := Shape.matrix rows cols, dtype := dtype }

  /-- Create a vector descriptor -/
  def vector (n : Nat) (dtype : DType := .f32) : TensorDesc :=
    { shape := Shape.vector n, dtype := dtype }

end TensorDesc

/-- Matrix multiplication configuration -/
structure MatMulConfig where
  /-- Matrix A dimensions: M × K -/
  M : Nat
  K : Nat
  /-- Matrix B dimensions: K × N -/
  N : Nat
  /-- Workgroup tile size -/
  tileSize : Nat := 16
  /-- Data type -/
  dtype : DType := .f32
  deriving Inhabited, Repr

namespace MatMulConfig

  /-- Shape of matrix A -/
  def shapeA (c : MatMulConfig) : Shape :=
    Shape.matrix c.M c.K

  /-- Shape of matrix B -/
  def shapeB (c : MatMulConfig) : Shape :=
    Shape.matrix c.K c.N

  /-- Shape of result matrix C -/
  def shapeC (c : MatMulConfig) : Shape :=
    Shape.matrix c.M c.N

  /-- Workgroup size for compute shader -/
  def workgroupSize (c : MatMulConfig) : Nat × Nat × Nat :=
    (c.tileSize, c.tileSize, 1)

  /-- Number of workgroups needed -/
  def numWorkgroups (c : MatMulConfig) : Nat × Nat × Nat :=
    let wgX := (c.N + c.tileSize - 1) / c.tileSize
    let wgY := (c.M + c.tileSize - 1) / c.tileSize
    (wgX, wgY, 1)

end MatMulConfig

end Hesper.Tensor
