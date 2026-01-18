import Hesper.WGSL.Types
import Hesper.WGSL.Exp

namespace Hesper.WGSL

/-! High-level DSL with operator overloading for natural syntax.
    Provides literals, arithmetic, comparison, and other operators
    that compile to type-safe WGSL expressions. -/

open Exp

-- ============================================================================
-- Literal Constructors
-- ============================================================================

/-- Create an f32 literal -/
def litF32 (f : Float) : Exp (.scalar .f32) := Exp.litF32 f

/-- Create an f16 literal -/
def litF16 (f : Float) : Exp (.scalar .f16) := Exp.litF16 f

/-- Create an i32 literal -/
def litI32 (i : Int) : Exp (.scalar .i32) := Exp.litI32 i

/-- Create a u32 literal -/
def litU32 (u : Nat) : Exp (.scalar .u32) := Exp.litU32 u

/-- Create a bool literal -/
def litBool (b : Bool) : Exp (.scalar .bool) := Exp.litBool b

/-- Convenience function: Create f32 literal (most common type) -/
def lit (f : Float) : Exp (.scalar .f32) := litF32 f

/-- Create a variable reference -/
def var (name : String) : Exp t := Exp.var name

-- ============================================================================
-- Numeric Type Class Instances
-- ============================================================================

/-- Addition operator for WGSL expressions -/
instance : HAdd (Exp t) (Exp t) (Exp t) where
  hAdd := Exp.add

/-- Subtraction operator for WGSL expressions -/
instance : HSub (Exp t) (Exp t) (Exp t) where
  hSub := Exp.sub

/-- Multiplication operator for WGSL expressions -/
instance : HMul (Exp t) (Exp t) (Exp t) where
  hMul := Exp.mul

/-- Division operator for WGSL expressions -/
instance : HDiv (Exp t) (Exp t) (Exp t) where
  hDiv := Exp.div

/-- Modulo operator for WGSL expressions -/
instance : HMod (Exp t) (Exp t) (Exp t) where
  hMod := Exp.mod

/-- Negation operator for WGSL expressions -/
instance : Neg (Exp t) where
  neg := Exp.neg

/-- Integer division for WGSL expressions (./.) -/
def divExp {t : WGSLType} (a b : Exp t) : Exp t := Exp.div a b

/-- Modulo for WGSL expressions (./.) -/
def modExp {t : WGSLType} (a b : Exp t) : Exp t := Exp.mod a b

-- Allow numeric literals to be used directly
instance : OfNat (Exp (.scalar .f32)) n where
  ofNat := litF32 (OfNat.ofNat n : Float)

instance : OfNat (Exp (.scalar .i32)) n where
  ofNat := litI32 (OfNat.ofNat n : Int)

instance : OfNat (Exp (.scalar .u32)) n where
  ofNat := litU32 n

-- Coercions for ergonomic DSL
instance : Coe Nat (Exp (.scalar .u32)) where
  coe n := litU32 n

instance : Coe Int (Exp (.scalar .i32)) where
  coe i := litI32 i

instance : Coe Float (Exp (.scalar .f32)) where
  coe f := litF32 f

instance : Coe Bool (Exp (.scalar .bool)) where
  coe b := litBool b

-- String coercion for variables (polymorphic in return type)
instance {t : WGSLType} : Coe String (Exp t) where
  coe s := Exp.var s

-- ============================================================================
-- Comparison Operators (custom to avoid clash with Lean's BEq/Ord)
-- ============================================================================

/-- Custom equality class for WGSL expressions -/
class WGSLEq (α : Type) where
  weq : α → α → Exp (.scalar .bool)
  wne : α → α → Exp (.scalar .bool)

/-- Custom ordering class for WGSL expressions -/
class WGSLOrd (α : Type) where
  wlt : α → α → Exp (.scalar .bool)
  wle : α → α → Exp (.scalar .bool)
  wgt : α → α → Exp (.scalar .bool)
  wge : α → α → Exp (.scalar .bool)

instance : WGSLEq (Exp t) where
  weq := Exp.eq
  wne := Exp.ne

instance : WGSLOrd (Exp t) where
  wlt := Exp.lt
  wle := Exp.le
  wgt := Exp.gt
  wge := Exp.ge

-- Convenient infix operators
infixl:50 " .==. " => WGSLEq.weq
infixl:50 " .!=. " => WGSLEq.wne
infixl:50 " .<. " => WGSLOrd.wlt
infixl:50 " .<=. " => WGSLOrd.wle
infixl:50 " .>. " => WGSLOrd.wgt
infixl:50 " .>=. " => WGSLOrd.wge

-- ============================================================================
-- Boolean Operators
-- ============================================================================

/-- Boolean AND -/
infixl:35 " .&&. " => Exp.and

/-- Boolean OR -/
infixl:30 " .||. " => Exp.or

/-- Boolean NOT -/
prefix:40 ".!." => Exp.not

-- ============================================================================
-- Bitwise Operators (u32 only)
-- ============================================================================

/-- Bitwise left shift -/
infixl:60 " .<<. " => Exp.shiftLeft

/-- Bitwise right shift -/
infixl:60 " .>>. " => Exp.shiftRight

/-- Bitwise AND -/
infixl:55 " .&. " => Exp.bitAnd

/-- Bitwise OR -/
infixl:50 " .|. " => Exp.bitOr

/-- Bitwise XOR -/
infixl:50 " .^. " => Exp.bitXor

-- ============================================================================
-- Type Conversions
-- ============================================================================

/-- Convert to f32 -/
def toF32 {t : WGSLType} (e : Exp t) : Exp (.scalar .f32) := Exp.toF32 e

/-- Convert to f16 -/
def toF16 {t : WGSLType} (e : Exp t) : Exp (.scalar .f16) := Exp.toF16 e

/-- Convert to i32 -/
def toI32 {t : WGSLType} (e : Exp t) : Exp (.scalar .i32) := Exp.toI32 e

/-- Convert to u32 -/
def toU32 {t : WGSLType} (e : Exp t) : Exp (.scalar .u32) := Exp.toU32 e

-- Convenient notation
notation "f32(" e ")" => toF32 e
notation "f16(" e ")" => toF16 e
notation "i32(" e ")" => toI32 e
notation "u32(" e ")" => toU32 e

-- ============================================================================
-- Array/Vector Access
-- ============================================================================

/-- Array indexing operator -/
infixl:90 " ! " => Exp.index

/-- Vector component accessors -/
def vecX {st : ScalarType} (v : Exp (.vec2 st)) : Exp (.scalar st) := Exp.vecX v
def vecY {st : ScalarType} (v : Exp (.vec2 st)) : Exp (.scalar st) := Exp.vecY v
def vecZ {st : ScalarType} (v : Exp (.vec3 st)) : Exp (.scalar st) := Exp.vecZ v
def vecW {st : ScalarType} (v : Exp (.vec4 st)) : Exp (.scalar st) := Exp.vecW v

-- ============================================================================
-- Vector Constructors
-- ============================================================================

/-- Create vec2 -/
def mkVec2 {st : ScalarType} (x y : Exp (.scalar st)) : Exp (.vec2 st) := Exp.vec2 x y

/-- Create vec3 -/
def mkVec3 {st : ScalarType} (x y z : Exp (.scalar st)) : Exp (.vec3 st) := Exp.vec3 x y z

/-- Create vec4 -/
def mkVec4 {st : ScalarType} (x y z w : Exp (.scalar st)) : Exp (.vec4 st) := Exp.vec4 x y z w

-- ============================================================================
-- Math Functions (with prime to avoid conflicts)
-- ============================================================================

/-- Square root -/
def sqrt' {t : WGSLType} (e : Exp t) : Exp t := Exp.sqrt e

/-- Absolute value -/
def abs' {t : WGSLType} (e : Exp t) : Exp t := Exp.abs e

/-- Minimum of two values -/
def min' {t : WGSLType} (a b : Exp t) : Exp t := Exp.min a b

/-- Maximum of two values -/
def max' {t : WGSLType} (a b : Exp t) : Exp t := Exp.max a b

/-- Clamp value between min and max -/
def clamp' {t : WGSLType} (e lo hi : Exp t) : Exp t := Exp.clamp e lo hi

/-- Exponential function -/
def exp' {t : WGSLType} (e : Exp t) : Exp t := Exp.exp e

/-- Sine function -/
def sin' {t : WGSLType} (e : Exp t) : Exp t := Exp.sin e

/-- Cosine function -/
def cos' {t : WGSLType} (e : Exp t) : Exp t := Exp.cos e

/-- Power function -/
def pow' {t : WGSLType} (base exponent : Exp t) : Exp t := Exp.pow base exponent

/-- Hyperbolic tangent -/
def tanh' {t : WGSLType} (e : Exp t) : Exp t := Exp.tanh e

-- ============================================================================
-- Conditional (Ternary Operator)
-- ============================================================================

/-- Select between two values based on condition (WGSL select function) -/
def select' {t : WGSLType} (cond : Exp (.scalar .bool)) (ifTrue ifFalse : Exp t) : Exp t :=
  Exp.select cond ifTrue ifFalse

-- ============================================================================
-- Example Usage Documentation
-- ============================================================================

/-!
## Example Usage

```lean
-- Create variables
def x : Exp (.scalar .f32) := Exp.var "x"
def y : Exp (.scalar .f32) := Exp.var "y"

-- Natural arithmetic syntax
def expr1 := x + y * 2.0  -- Multiplication binds tighter

-- Comparisons
def cond := x .>. 0.0 .&&. y .<. 10.0

-- Type conversions
def asInt := i32(x)

-- Math functions
def length := sqrt'(x * x + y * y)

-- Conditional
def result := select' (x .>. 0.0) x (-x)  -- absolute value
```
-/

-- ============================================================================
-- Convenience Wrappers for Common Functions
-- ============================================================================

/-- Square root -/
def sqrt (x : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.sqrt x

/-- Absolute value -/
def abs (x : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.abs x

/-- Minimum of two values -/
def min (x y : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.min x y

/-- Maximum of two values -/
def max (x y : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.max x y

/-- Exponential function -/
def exp (x : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.exp x

/-- Clamp value to range -/
def clamp (x minVal maxVal : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.clamp x minVal maxVal

/-- Power function -/
def pow (x y : Exp (.scalar .f32)) : Exp (.scalar .f32) := Exp.pow x y

/-- Select (ternary operator): select(cond, trueVal, falseVal) -/
def select (cond : Exp (.scalar .bool)) (trueVal falseVal : Exp t) : Exp t := Exp.select cond trueVal falseVal

-- ============================================================================
-- Subgroup Matrix Operations (Smart Constructors)
-- ============================================================================

/-- Load subgroup matrix (left operand) from buffer -/
def matLoadLeft {st : ScalarType} {m k : Nat}
    (ptrRef : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    (transpose : Exp (.scalar .bool) := false)
    : Exp (.subgroupMatrixLeft st m k) :=
  Exp.subgroupMatrixLoad ptrRef offset transpose stride

/-- Load subgroup matrix (right operand) from buffer -/
def matLoadRight {st : ScalarType} {k n : Nat}
    (ptrRef : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    (transpose : Exp (.scalar .bool) := false)
    : Exp (.subgroupMatrixRight st k n) :=
  Exp.subgroupMatrixLoadRight ptrRef offset transpose stride

/-- Multiply-accumulate for subgroup matrices -/
def matMulAcc {st : ScalarType} {m k n : Nat}
    (a : Exp (.subgroupMatrixLeft st m k))
    (b : Exp (.subgroupMatrixRight st k n))
    (acc : Exp (.subgroupMatrixResult st m n))
    : Exp (.subgroupMatrixResult st m n) :=
  Exp.subgroupMatrixMultiplyAccumulate a b acc

/-- Store subgroup matrix result to buffer -/
def matStore {st : ScalarType} {m n : Nat}
    (ptrRef : String)
    (offset : Exp (.scalar .u32))
    (mat : Exp (.subgroupMatrixResult st m n))
    (stride : Exp (.scalar .u32))
    (transpose : Exp (.scalar .bool) := false)
    : Exp (.scalar .u32) :=
  Exp.subgroupMatrixStore ptrRef offset mat transpose stride

/-- Zero-initialized subgroup matrix (left) -/
def matZeroLeft {st : ScalarType} {m k : Nat} : Exp (.subgroupMatrixLeft st m k) :=
  Exp.subgroupMatrixZeroLeft

/-- Zero-initialized subgroup matrix (right) -/
def matZeroRight {st : ScalarType} {k n : Nat} : Exp (.subgroupMatrixRight st k n) :=
  Exp.subgroupMatrixZeroRight

/-- Zero-initialized subgroup matrix (result) -/
def matZeroResult {st : ScalarType} {m n : Nat} : Exp (.subgroupMatrixResult st m n) :=
  Exp.subgroupMatrixZeroResult

/-- Workgroup barrier -/
def barrier : Exp (.scalar .u32) := Exp.workgroupBarrier

-- ============================================================================
-- Array/Variable Access Helpers
-- ============================================================================

/-- Index into an array -/
def get {elemTy : WGSLType} {n : Nat}
    (arr : Exp (.array elemTy n))
    (idx : Exp (.scalar .u32))
    : Exp elemTy :=
  Exp.index arr idx

-- Notation for array indexing
notation:max arr "[" idx "]" => get arr idx

-- ============================================================================
-- Atomic Operations
-- ============================================================================

/-- Atomically add to i32, returns old value -/
def atomicAdd {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicAdd ptr value

/-- Atomically add to u32, returns old value -/
def atomicAddU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicAddU ptr value

/-- Atomically subtract from i32, returns old value -/
def atomicSub {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicSub ptr value

/-- Atomically subtract from u32, returns old value -/
def atomicSubU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicSubU ptr value

/-- Atomically compute minimum with i32, returns old value -/
def atomicMin {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicMin ptr value

/-- Atomically compute minimum with u32, returns old value -/
def atomicMinU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicMinU ptr value

/-- Atomically compute maximum with i32, returns old value -/
def atomicMax {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicMax ptr value

/-- Atomically compute maximum with u32, returns old value -/
def atomicMaxU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicMaxU ptr value

/-- Atomically exchange (swap) i32 value, returns old value -/
def atomicExchange {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicExchange ptr value

/-- Atomically exchange (swap) u32 value, returns old value -/
def atomicExchangeU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicExchangeU ptr value

/-- Atomically compare-and-exchange i32 (weak version), returns old value -/
def atomicCompareExchangeWeak {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicI32)))
    (compare : Exp (.scalar .i32))
    (value : Exp (.scalar .i32))
    : Exp (.scalar .i32) :=
  Exp.atomicCompareExchangeWeak ptr compare value

/-- Atomically compare-and-exchange u32 (weak version), returns old value -/
def atomicCompareExchangeWeakU {space : MemorySpace}
    (ptr : Exp (.ptr space (.scalar .atomicU32)))
    (compare : Exp (.scalar .u32))
    (value : Exp (.scalar .u32))
    : Exp (.scalar .u32) :=
  Exp.atomicCompareExchangeWeakU ptr compare value

end Hesper.WGSL
