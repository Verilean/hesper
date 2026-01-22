import Hesper.Tensor.Types

namespace Hesper.Tensor

/--
Evaluate DType to its corresponding Lean native type.
-/
def DType.toLeanType : DType → Type
  | .f32 => Float
  | .f16 => Float
  | .i32 => Int
  | .u32 => UInt32

-- Helper instances to help Lean's typeclass synthesis with computed types
instance : OfScientific DType.f32.toLeanType := inferInstanceAs (OfScientific Float)
instance : OfScientific DType.f16.toLeanType := inferInstanceAs (OfScientific Float)
instance : OfNat DType.i32.toLeanType n := inferInstanceAs (OfNat Int n)
instance : OfNat DType.u32.toLeanType n := inferInstanceAs (OfNat UInt32 n)

instance : HAdd DType.f32.toLeanType DType.f32.toLeanType DType.f32.toLeanType := inferInstanceAs (HAdd Float Float Float)
instance : HAdd DType.f16.toLeanType DType.f16.toLeanType DType.f16.toLeanType := inferInstanceAs (HAdd Float Float Float)
instance : HAdd DType.i32.toLeanType DType.i32.toLeanType DType.i32.toLeanType := inferInstanceAs (HAdd Int Int Int)
instance : HAdd DType.u32.toLeanType DType.u32.toLeanType DType.u32.toLeanType := inferInstanceAs (HAdd UInt32 UInt32 UInt32)

instance : HSub DType.f32.toLeanType DType.f32.toLeanType DType.f32.toLeanType := inferInstanceAs (HSub Float Float Float)
instance : HSub DType.f16.toLeanType DType.f16.toLeanType DType.f16.toLeanType := inferInstanceAs (HSub Float Float Float)
instance : HSub DType.i32.toLeanType DType.i32.toLeanType DType.i32.toLeanType := inferInstanceAs (HSub Int Int Int)
instance : HSub DType.u32.toLeanType DType.u32.toLeanType DType.u32.toLeanType := inferInstanceAs (HSub UInt32 UInt32 UInt32)

instance : HMul DType.f32.toLeanType DType.f32.toLeanType DType.f32.toLeanType := inferInstanceAs (HMul Float Float Float)
instance : HMul DType.f16.toLeanType DType.f16.toLeanType DType.f16.toLeanType := inferInstanceAs (HMul Float Float Float)
instance : HMul DType.i32.toLeanType DType.i32.toLeanType DType.i32.toLeanType := inferInstanceAs (HMul Int Int Int)
instance : HMul DType.u32.toLeanType DType.u32.toLeanType DType.u32.toLeanType := inferInstanceAs (HMul UInt32 UInt32 UInt32)

/-- Zero value -/
def DType.zero : (dt : DType) → dt.toLeanType
  | .f32 => (0.0 : Float)
  | .f16 => (0.0 : Float)
  | .i32 => (0 : Int)
  | .u32 => (0 : UInt32)

/-- Addition -/
def DType.add : (dt : DType) → dt.toLeanType → dt.toLeanType → dt.toLeanType
  | .f32, a, b => (a : Float) + (b : Float)
  | .f16, a, b => (a : Float) + (b : Float)
  | .i32, a, b => (a : Int) + (b : Int)
  | .u32, a, b => (a : UInt32) + (b : UInt32)

/-- Subtraction -/
def DType.sub : (dt : DType) → dt.toLeanType → dt.toLeanType → dt.toLeanType
  | .f32, a, b => (a : Float) - (b : Float)
  | .f16, a, b => (a : Float) - (b : Float)
  | .i32, a, b => (a : Int) - (b : Int)
  | .u32, a, b => (a : UInt32) - (b : UInt32)

/-- Multiplication -/
def DType.mul : (dt : DType) → dt.toLeanType → dt.toLeanType → dt.toLeanType
  | .f32, a, b => (a : Float) * (b : Float)
  | .f16, a, b => (a : Float) * (b : Float)
  | .i32, a, b => (a : Int) * (b : Int)
  | .u32, a, b => (a : UInt32) * (b : UInt32)

/-- Absolute difference -/
def DType.absDiff : (dt : DType) → dt.toLeanType → dt.toLeanType → Float
  | .f32, a, b => Float.abs ((a : Float) - (b : Float))
  | .f16, a, b => Float.abs ((a : Float) - (b : Float))
  | .i32, a, b => Float.abs (Float.ofInt ((a : Int) - (b : Int)))
  | .u32, a, b => Float.abs (Float.ofNat (a : UInt32).toNat - Float.ofNat (b : UInt32).toNat)

/--
Typed Tensor Structure.
Encodes Shape and DType in the type signature for verification.
-/
structure TypedTensor (shape : Shape) (dtype : DType) where
  data : Array (dtype.toLeanType)
  h_size : data.size = shape.size

namespace TypedTensor

  /-- Create a tensor filled with zeros -/
  def zeros (shape : Shape) (dtype : DType := .f32) : TypedTensor shape dtype :=
    let data := Array.replicate shape.size dtype.zero
    { data := data, h_size := by simp [data, Array.size_replicate] }

  /-- Map function (preserves shape) -/
  def map {s : Shape} {dt : DType} (f : dt.toLeanType → dt.toLeanType) (t : TypedTensor s dt) : TypedTensor s dt :=
    let newData := t.data.map f
    have h_size : newData.size = s.size := by
      rw [Array.size_map, t.h_size]
    { data := newData, h_size := h_size }

  /-- Element-wise operation (ZipWith) -/
  def zipWith {s : Shape} {dt : DType} (f : dt.toLeanType → dt.toLeanType → dt.toLeanType) (a b : TypedTensor s dt) : TypedTensor s dt :=
    let newData := Array.zipWith f a.data b.data
    have h_size : newData.size = s.size := by
      rw [Array.size_zipWith, a.h_size, b.h_size, Nat.min_self]
    { data := newData, h_size := h_size }

  /-- Add -/
  def add {s : Shape} {dt : DType} (a b : TypedTensor s dt) : TypedTensor s dt :=
    zipWith dt.add a b

  /-- Multiply -/
  def mul {s : Shape} {dt : DType} (a b : TypedTensor s dt) : TypedTensor s dt :=
    zipWith dt.mul a b

  /-- Get element at flattened index (safe) -/
  def get {s : Shape} {dt : DType} (t : TypedTensor s dt) (i : Fin s.size) : dt.toLeanType :=
    have h : i.val < t.data.size := by rw [t.h_size]; exact i.isLt
    t.data[i.val]'h

  /-- Approximate Equality Check -/
  def approxEq {s : Shape} {dt : DType} (a b : TypedTensor s dt) (tolerance : Float := 1e-5) : Bool :=
    let size := s.size
    (List.range size).all fun i =>
      if h : i < size then
        let idx : Fin s.size := ⟨i, h⟩
        let valA := a.get idx
        let valB := b.get idx
        dt.absDiff valA valB <= tolerance
      else
        false

  /-- Matrix Multiplication (Signature only for prototype) -/
  def matmul {M K N : Nat} {dt : DType}
    (_a : TypedTensor (Shape.matrix M K) dt)
    (_b : TypedTensor (Shape.matrix K N) dt)
    : TypedTensor (Shape.matrix M N) dt :=
    TypedTensor.zeros (Shape.matrix M N) dt

end TypedTensor

end Hesper.Tensor
