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

/-- Zero value -/
def DType.zero (dt : DType) : dt.toLeanType :=
  match dt with | .f32 => 0.0 | .f16 => 0.0 | .i32 => 0 | .u32 => 0

/-- Ops for DType values -/
def DType.add {dt : DType} (a b : dt.toLeanType) : dt.toLeanType :=
  match dt with | .f32 => a+b | .f16 => a+b | .i32 => a+b | .u32 => a+b

def DType.sub {dt : DType} (a b : dt.toLeanType) : dt.toLeanType :=
  match dt with | .f32 => a-b | .f16 => a-b | .i32 => a-b | .u32 => a-b

def DType.mul {dt : DType} (a b : dt.toLeanType) : dt.toLeanType :=
  match dt with | .f32 => a*b | .f16 => a*b | .i32 => a*b | .u32 => a*b

/-- Absolute difference (always positive float for verification, or same type) -/
def DType.absDiff {dt : DType} (a b : dt.toLeanType) : Float :=
  match dt with
  | .f32 => Float.abs (a - b)
  | .f16 => Float.abs (a - b)
  | .i32 => (Int.sub a b).toFloat.abs -- Int difference converted to float
  | .u32 => -- Compare UInt32 via float
     let af := a.toNat.toFloat; let bf := b.toNat.toFloat
     Float.abs (af - bf)

/--
Typed Tensor Structure.
Encodes Shape and DType in the type signature for verification.
-/
structure TypedTensor (shape : Shape) (dtype : DType) where
  data : Array (dtype.toLeanType)
  h_size : data.size = shape.size
  deriving Repr

namespace TypedTensor

  /-- Create a tensor filled with zeros -/
  def zeros (shape : Shape) (dtype : DType := .f32) : TypedTensor shape dtype :=
    let size := shape.size
    let z := dtype.zero
    let data := Array.mk (List.replicate size z)
    have h_size : data.size = shape.size := by
      simp [data]
      exact List.length_replicate size z
    { data := data, h_size := h_size }

  /-- Map function (preserves shape) -/
  def map {s : Shape} {dt : DType} (f : dt.toLeanType → dt.toLeanType) (t : TypedTensor s dt) : TypedTensor s dt :=
    let newData := t.data.map f
    have h_size : newData.size = s.size := by
      rw [Array.size_map, t.h_size]
    { data := newData, h_size := h_size }

  /-- Element-wise operation (ZipWith) -/
  def zipWith {s : Shape} {dt : DType} (f : dt.toLeanType → dt.toLeanType → dt.toLeanType) (a b : TypedTensor s dt) : TypedTensor s dt :=
    let newData := Array.zipWith a.data b.data f
    have h_size : newData.size = s.size := by
      rw [Array.size_zipWith]
      rw [a.h_size, b.h_size]
      simp
    { data := newData, h_size := h_size }

  /-- Add -/
  def add {s : Shape} {dt : DType} (a b : TypedTensor s dt) : TypedTensor s dt :=
    zipWith DType.add a b

  /-- Multiply -/
  def mul {s : Shape} {dt : DType} (a b : TypedTensor s dt) : TypedTensor s dt :=
    zipWith DType.mul a b

  /-- Get element at flattened index (safe) -/
  def get {s : Shape} {dt : DType} (t : TypedTensor s dt) (i : Fin s.size) : dt.toLeanType :=
    -- Since t.data.size = s.size, i is valid index for t.data
    have h : i.val < t.data.size := by rw [t.h_size]; exact i.isLt
    t.data.get ⟨i.val, h⟩

  /-- Approximate Equality Check -/
  def approxEq {s : Shape} {dt : DType} (a b : TypedTensor s dt) (tolerance : Float := 1e-5) : Bool :=
    -- Checks if all elements are within tolerance
    -- We zip indices, or just zip data
    let size := s.size
    (List.range size).all fun i =>
      have h : i < size := by sorry -- List.range property
      let idx : Fin s.size := ⟨i, h⟩
      let valA := a.get idx
      let valB := b.get idx
      DType.absDiff valA valB <= tolerance

  /-- Matrix Multiplication (Signature only for prototype) -/
  def matmul {M K N : Nat} {dt : DType}
    (a : TypedTensor (Shape.matrix M K) dt)
    (b : TypedTensor (Shape.matrix K N) dt)
    : TypedTensor (Shape.matrix M N) dt :=
    TypedTensor.zeros (Shape.matrix M N) dt

end TypedTensor

end Hesper.Tensor
