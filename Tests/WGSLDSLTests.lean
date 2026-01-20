import LSpec
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

/-!
# WGSL DSL Comprehensive Tests

Tests for WGSL DSL code generation covering:
- Type system (scalar, vector, matrix types)
- Expression code generation
- Operator precedence
- Built-in functions
- Type conversions
- Type safety
-/

namespace Tests.WGSLDSLTests

open Hesper.WGSL
open LSpec

-- ============================================================================
-- Type System Tests
-- ============================================================================

def testScalarTypes : TestSeq :=
  test "f32 toWGSL" (ScalarType.f32.toWGSL == "f32") ++
  test "f16 toWGSL" (ScalarType.f16.toWGSL == "f16") ++
  test "i32 toWGSL" (ScalarType.i32.toWGSL == "i32") ++
  test "u32 toWGSL" (ScalarType.u32.toWGSL == "u32") ++
  test "bool toWGSL" (ScalarType.bool.toWGSL == "bool") ++
  test "atomicI32 toWGSL" (ScalarType.atomicI32.toWGSL == "atomic<i32>") ++
  test "atomicU32 toWGSL" (ScalarType.atomicU32.toWGSL == "atomic<u32>")

def testVectorTypes : TestSeq :=
  test "vec2<f32> toWGSL" ((WGSLType.vec2 .f32).toWGSL == "vec2<f32>") ++
  test "vec3<f32> toWGSL" ((WGSLType.vec3 .f32).toWGSL == "vec3<f32>") ++
  test "vec4<f32> toWGSL" ((WGSLType.vec4 .f32).toWGSL == "vec4<f32>") ++
  test "vec2<i32> toWGSL" ((WGSLType.vec2 .i32).toWGSL == "vec2<i32>") ++
  test "vec3<u32> toWGSL" ((WGSLType.vec3 .u32).toWGSL == "vec3<u32>") ++
  test "vec4<bool> toWGSL" ((WGSLType.vec4 .bool).toWGSL == "vec4<bool>")

def testMatrixTypes : TestSeq :=
  test "mat2x2<f32> toWGSL" ((WGSLType.mat2x2 .f32).toWGSL == "mat2x2<f32>") ++
  test "mat3x3<f32> toWGSL" ((WGSLType.mat3x3 .f32).toWGSL == "mat3x3<f32>") ++
  test "mat4x4<f32> toWGSL" ((WGSLType.mat4x4 .f32).toWGSL == "mat4x4<f32>")

def testArrayTypes : TestSeq :=
  test "array<f32, 10> toWGSL" ((WGSLType.array (.scalar .f32) 10).toWGSL == "array<f32, 10>") ++
  test "array<vec4<f32>, 256> toWGSL" ((WGSLType.array (.vec4 .f32) 256).toWGSL == "array<vec4<f32>, 256>") ++
  test "array<i32, 1024> toWGSL" ((WGSLType.array (.scalar .i32) 1024).toWGSL == "array<i32, 1024>")

def testPtrTypes : TestSeq :=
  test "ptr<storage, f32> toWGSL" ((WGSLType.ptr .storage (.scalar .f32)).toWGSL == "ptr<storage, f32>") ++
  test "ptr<workgroup, i32> toWGSL" ((WGSLType.ptr .workgroup (.scalar .i32)).toWGSL == "ptr<workgroup, i32>") ++
  test "ptr<uniform, vec4<f32>> toWGSL" ((WGSLType.ptr .uniform (.vec4 .f32)).toWGSL == "ptr<uniform, vec4<f32>>")

def testByteSizes : TestSeq :=
  test "f32 byte size" (ScalarType.f32.byteSize == 4) ++
  test "f16 byte size" (ScalarType.f16.byteSize == 2) ++
  test "i32 byte size" (ScalarType.i32.byteSize == 4) ++
  test "u32 byte size" (ScalarType.u32.byteSize == 4) ++
  test "vec2<f32> byte size" ((WGSLType.vec2 .f32).byteSize == 8) ++
  test "vec4<f32> byte size" ((WGSLType.vec4 .f32).byteSize == 16) ++
  test "mat4x4<f32> byte size" ((WGSLType.mat4x4 .f32).byteSize == 64) ++
  test "array<f32, 100> byte size" ((WGSLType.array (.scalar .f32) 100).byteSize == 400)

-- ============================================================================
-- Literal Tests
-- ============================================================================

def testLiterals : TestSeq :=
  let f32Lit := litF32 3.14
  let f16Lit := litF16 2.5
  let i32Lit := litI32 (-42)
  let u32Lit := litU32 123
  let boolLitTrue := litBool true
  let boolLitFalse := litBool false

  test "f32 literal" (f32Lit.toWGSL == "3.140000") ++
  test "f16 literal" (f16Lit.toWGSL == "2.500000h") ++
  test "i32 literal" (i32Lit.toWGSL == "-42i") ++
  test "u32 literal" (u32Lit.toWGSL == "123u") ++
  test "bool literal (true)" (boolLitTrue.toWGSL == "true") ++
  test "bool literal (false)" (boolLitFalse.toWGSL == "false")

-- ============================================================================
-- Variable Tests
-- ============================================================================

def testVariables : TestSeq :=
  let varX : Exp (.scalar .f32) := var "x"
  let varY : Exp (.scalar .f32) := var "y"
  let varIdx : Exp (.scalar .u32) := var "idx"

  test "variable x" (varX.toWGSL == "x") ++
  test "variable y" (varY.toWGSL == "y") ++
  test "variable idx" (varIdx.toWGSL == "idx")

-- ============================================================================
-- Arithmetic Operator Tests
-- ============================================================================

def testArithmeticOps : TestSeq :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  let addExpr := x + y
  let subExpr := x - y
  let mulExpr := x * y
  let divExpr := x / y
  let modExpr := Exp.mod x y
  let negExpr := -x

  test "addition x + y" (addExpr.toWGSL == "(x + y)") ++
  test "subtraction x - y" (subExpr.toWGSL == "(x - y)") ++
  test "multiplication x * y" (mulExpr.toWGSL == "(x * y)") ++
  test "division x / y" (divExpr.toWGSL == "(x / y)") ++
  test "modulo x % y" (modExpr.toWGSL == "(x % y)") ++
  test "negation -x" (negExpr.toWGSL == "(-x)")

-- ============================================================================
-- Comparison Operator Tests
-- ============================================================================

def testComparisonOps : TestSeq :=
  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"

  let eqExpr := a .==. b
  let neExpr := a .!=. b
  let ltExpr := a .<. b
  let leExpr := a .<=. b
  let gtExpr := a .>. b
  let geExpr := a .>=. b

  test "equality a == b" (eqExpr.toWGSL == "(a == b)") ++
  test "inequality a != b" (neExpr.toWGSL == "(a != b)") ++
  test "less than a < b" (ltExpr.toWGSL == "(a < b)") ++
  test "less equal a <= b" (leExpr.toWGSL == "(a <= b)") ++
  test "greater than a > b" (gtExpr.toWGSL == "(a > b)") ++
  test "greater equal a >= b" (geExpr.toWGSL == "(a >= b)")

-- ============================================================================
-- Boolean Operator Tests
-- ============================================================================

def testBooleanOps : TestSeq :=
  let p : Exp (.scalar .bool) := var "p"
  let q : Exp (.scalar .bool) := var "q"

  let andExpr := p .&&. q
  let orExpr := p .||. q
  let notExpr := .!.p

  test "logical AND p && q" (andExpr.toWGSL == "(p && q)") ++
  test "logical OR p || q" (orExpr.toWGSL == "(p || q)") ++
  test "logical NOT !p" (notExpr.toWGSL == "(!p)")

-- ============================================================================
-- Math Function Tests (Comprehensive)
-- ============================================================================

def testMathFunctions : TestSeq :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"
  let z : Exp (.scalar .f32) := var "z"

  -- Basic math (available in DSL)
  let sqrtExpr := sqrt' x
  let absExpr := abs' x
  let minExpr := min' x y
  let maxExpr := max' x y
  let clampExpr := clamp' x (lit 0.0) (lit 1.0)

  -- Exponential
  let expExpr := exp' x

  -- Trigonometric
  let sinExpr := sin' x
  let cosExpr := cos' x

  -- Hyperbolic
  let tanhExpr := tanh' x

  -- Power
  let powExpr := pow' x y

  -- Conditional selection
  let selectExpr := select' (x .>. lit 0.0) x (lit 0.0)

  test "sqrt(x)" (sqrtExpr.toWGSL == "sqrt(x)") ++
  test "abs(x)" (absExpr.toWGSL == "abs(x)") ++
  test "min(x, y)" (minExpr.toWGSL == "min(x, y)") ++
  test "max(x, y)" (maxExpr.toWGSL == "max(x, y)") ++
  test "clamp(x, 0.0, 1.0)" (clampExpr.toWGSL == "clamp(x, 0.000000, 1.000000)") ++
  test "exp(x)" (expExpr.toWGSL == "exp(x)") ++
  test "sin(x)" (sinExpr.toWGSL == "sin(x)") ++
  test "cos(x)" (cosExpr.toWGSL == "cos(x)") ++
  test "tanh(x)" (tanhExpr.toWGSL == "tanh(x)") ++
  test "pow(x, y)" (powExpr.toWGSL == "pow(x, y)") ++
  test "select(x > 0, x, 0)" (selectExpr.toWGSL == "select(0.000000, x, (x > 0.000000))")

-- Note: Vector builtins (dot, cross, length, normalize, etc.) are not yet exposed in the DSL
-- Note: Matrix builtins (transpose, determinant) are not yet exposed in the DSL
-- Note: Bitwise operations are not yet exposed in the DSL
-- These exist in WGSL but haven't been wrapped yet

-- ============================================================================
-- Type Conversion Tests
-- ============================================================================

def testTypeConversions : TestSeq :=
  let f32Val : Exp (.scalar .f32) := var "f32Val"
  let f16Val : Exp (.scalar .f16) := var "f16Val"
  let i32Val : Exp (.scalar .i32) := var "i32Val"
  let u32Val : Exp (.scalar .u32) := var "u32Val"

  -- Convert to f32
  let f16ToF32 := toF32 f16Val
  let i32ToF32 := toF32 i32Val
  let u32ToF32 := toF32 u32Val

  -- Convert to f16
  let f32ToF16 := toF16 f32Val
  let i32ToF16 := toF16 i32Val
  let u32ToF16 := toF16 u32Val

  -- Convert to i32
  let f32ToI32 := toI32 f32Val
  let f16ToI32 := toI32 f16Val
  let u32ToI32 := toI32 u32Val

  -- Convert to u32
  let f32ToU32 := toU32 f32Val
  let f16ToU32 := toU32 f16Val
  let i32ToU32 := toU32 i32Val

  test "f16 to f32" (f16ToF32.toWGSL == "f32(f16Val)") ++
  test "i32 to f32" (i32ToF32.toWGSL == "f32(i32Val)") ++
  test "u32 to f32" (u32ToF32.toWGSL == "f32(u32Val)") ++
  test "f32 to f16" (f32ToF16.toWGSL == "f16(f32Val)") ++
  test "i32 to f16" (i32ToF16.toWGSL == "f16(i32Val)") ++
  test "u32 to f16" (u32ToF16.toWGSL == "f16(u32Val)") ++
  test "f32 to i32" (f32ToI32.toWGSL == "i32(f32Val)") ++
  test "f16 to i32" (f16ToI32.toWGSL == "i32(f16Val)") ++
  test "u32 to i32" (u32ToI32.toWGSL == "i32(u32Val)") ++
  test "f32 to u32" (f32ToU32.toWGSL == "u32(f32Val)") ++
  test "f16 to u32" (f16ToU32.toWGSL == "u32(f16Val)") ++
  test "i32 to u32" (i32ToU32.toWGSL == "u32(i32Val)")

-- ============================================================================
-- Vector Constructor Tests
-- ============================================================================

def testVectorConstructors : TestSeq :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"
  let z : Exp (.scalar .f32) := var "z"
  let w : Exp (.scalar .f32) := var "w"
  let a : Exp (.scalar .f32) := lit 1.0
  let b : Exp (.scalar .f32) := lit 2.0

  -- vec2 constructors
  let vec2FromScalars := mkVec2 x y
  let vec2Splat := mkVec2 x x
  let vec2Literals := mkVec2 a b

  -- vec3 constructors
  let vec3FromScalars := mkVec3 x y z
  let vec3Splat := mkVec3 x x x
  let vec3Mixed := mkVec3 x (lit 0.0) z

  -- vec4 constructors
  let vec4FromScalars := mkVec4 x y z w
  let vec4Splat := mkVec4 x x x x
  let vec4Mixed := mkVec4 x y (lit 0.0) (lit 1.0)

  test "vec2(x, y)" (vec2FromScalars.toWGSL == "vec2<f32>(x, y)") ++
  test "vec2(x, x) splat" (vec2Splat.toWGSL == "vec2<f32>(x, x)") ++
  test "vec2(1.0, 2.0)" (vec2Literals.toWGSL == "vec2<f32>(1.000000, 2.000000)") ++
  test "vec3(x, y, z)" (vec3FromScalars.toWGSL == "vec3<f32>(x, y, z)") ++
  test "vec3(x, x, x) splat" (vec3Splat.toWGSL == "vec3<f32>(x, x, x)") ++
  test "vec3(x, 0.0, z)" (vec3Mixed.toWGSL == "vec3<f32>(x, 0.000000, z)") ++
  test "vec4(x, y, z, w)" (vec4FromScalars.toWGSL == "vec4<f32>(x, y, z, w)") ++
  test "vec4(x, x, x, x) splat" (vec4Splat.toWGSL == "vec4<f32>(x, x, x, x)") ++
  test "vec4(x, y, 0.0, 1.0)" (vec4Mixed.toWGSL == "vec4<f32>(x, y, 0.000000, 1.000000)")

-- ============================================================================
-- Vector Accessor Tests
-- ============================================================================

def testVectorAccessors : TestSeq :=
  let v2 : Exp (.vec2 .f32) := var "v2"
  let v3 : Exp (.vec3 .f32) := var "v3"
  let v4 : Exp (.vec4 .f32) := var "v4"

  -- vec2 accessors (vecX and vecY work on vec2)
  let v2x := vecX v2
  let v2y := vecY v2

  -- vec3 accessors (only vecZ works on vec3)
  let v3z := vecZ v3

  -- vec4 accessors (only vecW works on vec4)
  let v4w := vecW v4

  -- Complex accessor chains
  let complexZ := vecZ (mkVec3 (lit 1.0) (lit 2.0) (lit 3.0))
  let complexW := vecW (mkVec4 (lit 1.0) (lit 2.0) (lit 3.0) (lit 4.0))
  let complexX := vecX (mkVec2 (lit 5.0) (lit 6.0))
  let complexY := vecY (v2 + v2)

  test "v2.x" (v2x.toWGSL == "v2.x") ++
  test "v2.y" (v2y.toWGSL == "v2.y") ++
  test "v3.z" (v3z.toWGSL == "v3.z") ++
  test "v4.w" (v4w.toWGSL == "v4.w") ++
  test "vec3(1,2,3).z" (complexZ.toWGSL == "vec3<f32>(1.000000, 2.000000, 3.000000).z") ++
  test "vec4(1,2,3,4).w" (complexW.toWGSL == "vec4<f32>(1.000000, 2.000000, 3.000000, 4.000000).w") ++
  test "vec2(5,6).x" (complexX.toWGSL == "vec2<f32>(5.000000, 6.000000).x") ++
  test "(v2 + v2).y" (complexY.toWGSL == "(v2 + v2).y")

-- Note: Array indexing is not yet exposed in the DSL
-- Will be added in future versions

-- ============================================================================
-- Operator Precedence Tests (Comprehensive)
-- ============================================================================

def testOperatorPrecedence : TestSeq :=
  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"
  let c : Exp (.scalar .f32) := var "c"
  let d : Exp (.scalar .f32) := var "d"

  -- Multiplication before addition
  let addMul := a + b * c
  let mulAdd := a * b + c

  -- Division before subtraction
  let divSub := a - b / c
  let subDiv := (a - b) / c

  -- Multiplication/Division same precedence (left-to-right)
  let mulDiv := a * b / c
  let divMul := a / b * c

  -- Addition/Subtraction same precedence (left-to-right)
  let addSub := a + b - c
  let subAdd := a - b + c

  -- Negation (unary) before multiplication
  let negMul := -a * b
  let mulNeg := -(a * b)

  -- Complex multi-level precedence
  let complex1 := a + b * c - (a / b)
  let complex2 := (a + b) * (c - (a / b))
  let complex3 := a * b + c * d
  let complex4 := (a + b) * (c + d)
  let complex5 := a + b * c + d
  let complex6 := a * (b + c) * d
  let complex7 := a / b + c / d
  let complex8 := (a / b) + (c / d)
  let complex9 := a - b * c - d
  let complex10 := a * b - c * d

  -- Modulo with other operators
  let modAdd := (Exp.mod a b) + c
  let addMod := a + (Exp.mod b c)
  let modMul := (Exp.mod a b) * c
  let mulMod := a * (Exp.mod b c)

  test "a + b * c (mul before add)" (addMul.toWGSL == "(a + (b * c))") ++
  test "a * b + c (mul before add)" (mulAdd.toWGSL == "((a * b) + c)") ++
  test "a - b / c (div before sub)" (divSub.toWGSL == "(a - (b / c))") ++
  test "(a - b) / c (explicit parens)" (subDiv.toWGSL == "((a - b) / c)") ++
  test "a * b / c (left-to-right)" (mulDiv.toWGSL == "((a * b) / c)") ++
  test "a / b * c (left-to-right)" (divMul.toWGSL == "((a / b) * c)") ++
  test "a + b - c (left-to-right)" (addSub.toWGSL == "((a + b) - c)") ++
  test "a - b + c (left-to-right)" (subAdd.toWGSL == "((a - b) + c)") ++
  test "-a * b (negation before mul)" (negMul.toWGSL == "((-a) * b)") ++
  test "-(a * b) (explicit parens)" (mulNeg.toWGSL == "(-(a * b))") ++
  test "a + b * c - a / b" (complex1.toWGSL == "((a + (b * c)) - (a / b))") ++
  test "(a + b) * (c - a / b)" (complex2.toWGSL == "((a + b) * (c - (a / b)))") ++
  test "a * b + c * d" (complex3.toWGSL == "((a * b) + (c * d))") ++
  test "(a + b) * (c + d)" (complex4.toWGSL == "((a + b) * (c + d))") ++
  test "a + b * c + d" (complex5.toWGSL == "((a + (b * c)) + d)") ++
  test "a * (b + c) * d" (complex6.toWGSL == "((a * (b + c)) * d)") ++
  test "a / b + c / d" (complex7.toWGSL == "((a / b) + (c / d))") ++
  test "(a / b) + (c / d) explicit" (complex8.toWGSL == "((a / b) + (c / d))") ++
  test "a - b * c - d" (complex9.toWGSL == "((a - (b * c)) - d)") ++
  test "a * b - c * d" (complex10.toWGSL == "((a * b) - (c * d))") ++
  test "a % b + c (mod before add)" (modAdd.toWGSL == "((a % b) + c)") ++
  test "a + b % c (mod before add)" (addMod.toWGSL == "(a + (b % c))") ++
  test "a % b * c (mod then mul)" (modMul.toWGSL == "((a % b) * c)") ++
  test "a * b % c (mul then mod)" (mulMod.toWGSL == "(a * (b % c))")

def testComparisonPrecedence : TestSeq :=
  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"
  let c : Exp (.scalar .f32) := var "c"

  -- Arithmetic before comparison
  let addLt := (a + b) .<. c
  let mulGt := (a * b) .>. c
  let divEq := (a / b) .==. c

  test "a + b < c (arithmetic before comparison)" (addLt.toWGSL == "((a + b) < c)") ++
  test "a * b > c (arithmetic before comparison)" (mulGt.toWGSL == "((a * b) > c)") ++
  test "a / b == c (arithmetic before comparison)" (divEq.toWGSL == "((a / b) == c)")

def testLogicalPrecedence : TestSeq :=
  let p : Exp (.scalar .bool) := var "p"
  let q : Exp (.scalar .bool) := var "q"
  let r : Exp (.scalar .bool) := var "r"

  -- AND before OR
  let orAnd := p .||. (q .&&. r)
  let andOr := (p .&&. q) .||. r

  -- NOT before AND
  let notAnd := (.!.p) .&&. q
  let andNot := .!.(p .&&. q)

  test "p || q && r (AND before OR)" (orAnd.toWGSL == "(p || (q && r))") ++
  test "p && q || r (AND then OR)" (andOr.toWGSL == "((p && q) || r)") ++
  test "!p && q (NOT before AND)" (notAnd.toWGSL == "((!p) && q)") ++
  test "!(p && q) (explicit parens)" (andNot.toWGSL == "(!(p && q))")

-- ============================================================================
-- Complex Expression Tests
-- ============================================================================

def testComplexExpressions : TestSeq :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  let distance := sqrt' (x * x + y * y)
  let normalized := x / sqrt' (x * x + y * y)
  let complexCond := (x .>. lit 0.0) .&&. (y .<. lit 10.0) .&&. ((x + y) .>. lit 1.0)

  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"
  let c : Exp (.scalar .f32) := var "c"
  let polynomial := a * (x * x) + b * x + c

  test "distance sqrt(x*x + y*y)" (distance.toWGSL == "sqrt(((x * x) + (y * y)))") ++
  test "normalized x / sqrt(x*x + y*y)" (normalized.toWGSL == "(x / sqrt(((x * x) + (y * y))))") ++
  test "complex boolean condition" (complexCond.toWGSL == "(((x > 0.000000) && (y < 10.000000)) && ((x + y) > 1.000000))") ++
  test "polynomial a*x^2 + b*x + c" (polynomial.toWGSL == "(((a * (x * x)) + (b * x)) + c)")

-- ============================================================================
-- Control Flow Tests (ShaderM Monad)
-- ============================================================================

open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

def testIfStatement : TestSeq :=
  -- Simple if-then-else
  let shader1 : ShaderM Unit := do
    let x : Exp (.scalar .f32) := var "x"
    let result ← var (.scalar .f32) (lit 0.0)
    if_ (x .>. lit 0.0) (do
      assign result (lit 1.0)
    ) (do
      assign result (lit (-1.0))
    )
  let wgsl1 := generateWGSLSimple shader1

  -- Nested if statements
  let shader2 : ShaderM Unit := do
    let x : Exp (.scalar .f32) := var "x"
    let y : Exp (.scalar .f32) := var "y"
    let result ← var (.scalar .f32) (lit 0.0)
    if_ (x .>. lit 0.0) (do
      if_ (y .>. lit 0.0) (do
        assign result (lit 1.0)
      ) (do
        assign result (lit 0.5)
      )
    ) (do
      assign result (lit (-1.0))
    )
  let wgsl2 := generateWGSLSimple shader2

  -- If statement with boolean AND condition
  let shader3 : ShaderM Unit := do
    let x : Exp (.scalar .f32) := var "x"
    let y : Exp (.scalar .f32) := var "y"
    let result ← var (.scalar .f32) (lit 0.0)
    if_ ((x .>. lit 0.0) .&&. (y .<. lit 10.0)) (do
      assign result (lit 1.0)
    ) (pure ())
  let wgsl3 := generateWGSLSimple shader3

  test "if statement generates valid WGSL" (wgsl1.length > 50) ++
  test "if statement contains 'if'" ((wgsl1.splitOn "if").length >= 2) ++
  test "if statement contains 'else'" ((wgsl1.splitOn "else").length >= 2) ++
  test "nested if generates valid WGSL" (wgsl2.length > 50) ++
  test "boolean AND condition in if" ((wgsl3.splitOn "&&").length >= 2)

def testLoopStatement : TestSeq :=
  -- Simple loop using higher-order function
  let shader1 : ShaderM Unit := do
    let sum ← var (.scalar .f32) (lit 0.0)
    loop (litU32 0) (litU32 10) (litU32 1) fun i => do
      assign sum (Exp.add (Exp.var sum) (toF32 i))
  let wgsl1 := generateWGSLSimple shader1

  -- Nested loops
  let shader2 : ShaderM Unit := do
    let sum ← var (.scalar .f32) (lit 0.0)
    loop (litU32 0) (litU32 5) (litU32 1) fun i => do
      loop (litU32 0) (litU32 5) (litU32 1) fun j => do
        assign sum (Exp.add (Exp.var sum) (toF32 (Exp.add i j)))
  let wgsl2 := generateWGSLSimple shader2

  -- Loop with conditional inside
  let shader3 : ShaderM Unit := do
    let sum ← var (.scalar .f32) (lit 0.0)
    loop (litU32 0) (litU32 10) (litU32 1) fun i => do
      if_ ((toF32 i) .>. lit 5.0) (do
        assign sum (Exp.add (Exp.var sum) (toF32 i))
      ) (pure ())
  let wgsl3 := generateWGSLSimple shader3

  -- Loop with step size 2
  let shader4 : ShaderM Unit := do
    let sum ← var (.scalar .u32) (litU32 0)
    loop (litU32 0) (litU32 20) (litU32 2) fun i => do
      assign sum (Exp.add (Exp.var sum) i)
  let wgsl4 := generateWGSLSimple shader4

  test "loop generates valid WGSL" (wgsl1.length > 50) ++
  test "loop contains 'for'" ((wgsl1.splitOn "for").length >= 2) ++
  test "nested loops generate valid WGSL" (wgsl2.length > 50) ++
  test "nested loops contain multiple 'for'" ((wgsl2.splitOn "for").length >= 3) ++
  test "loop with conditional inside" (((wgsl3.splitOn "if").length >= 2) && ((wgsl3.splitOn "for").length >= 2)) ++
  test "loop with step 2" ((wgsl4.splitOn "for").length >= 2)

def testForStatement : TestSeq :=
  -- Named for loop
  let shader1 : ShaderM Unit := do
    let sum ← var (.scalar .u32) (litU32 0)
    for_ "idx" (litU32 0) (litU32 100) (litU32 1) (do
      let idx : Exp (.scalar .u32) := var "idx"
      assign sum (Exp.add (Exp.var sum) idx)
    )
  let wgsl1 := generateWGSLSimple shader1

  -- For loop with larger step
  let shader2 : ShaderM Unit := do
    let count ← var (.scalar .u32) (litU32 0)
    for_ "i" (litU32 0) (litU32 1000) (litU32 10) (do
      assign count (Exp.add (Exp.var count) (litU32 1))
    )
  let wgsl2 := generateWGSLSimple shader2

  test "for_ generates valid WGSL" (wgsl1.length > 50) ++
  test "for_ contains 'for'" ((wgsl1.splitOn "for").length >= 2) ++
  test "for_ contains variable name 'idx'" ((wgsl1.splitOn "idx").length >= 2) ++
  test "for_ with step 10" ((wgsl2.splitOn "for").length >= 2)

def testComplexControlFlow : TestSeq :=
  -- Matrix-like computation with nested loops and conditionals
  let shader1 : ShaderM Unit := do
    let sum ← var (.scalar .f32) (lit 0.0)
    loop (litU32 0) (litU32 4) (litU32 1) fun row => do
      loop (litU32 0) (litU32 4) (litU32 1) fun col => do
        -- Only accumulate diagonal elements
        if_ (row .==. col) (do
          let val := toF32 (Exp.add row col)
          assign sum (Exp.add (Exp.var sum) val)
        ) (pure ())
  let wgsl1 := generateWGSLSimple shader1

  -- Complex nested control flow
  let shader2 : ShaderM Unit := do
    let result ← var (.scalar .f32) (lit 0.0)
    let x : Exp (.scalar .f32) := var "input"

    if_ (x .>. lit 0.0) (do
      loop (litU32 0) (litU32 5) (litU32 1) fun i => do
        let fi := toF32 i
        assign result (Exp.add (Exp.var result) fi)
    ) (do
      loop (litU32 0) (litU32 3) (litU32 1) fun j => do
        let fj := toF32 j
        assign result (Exp.sub (Exp.var result) fj)
    )
  let wgsl2 := generateWGSLSimple shader2

  test "nested loops with conditional" (((wgsl1.splitOn "for").length >= 2) && ((wgsl1.splitOn "if").length >= 2)) ++
  test "matrix-like nested loops generate valid code" (wgsl1.length > 100) ++
  test "complex nested control flow" (((wgsl2.splitOn "for").length >= 2) && ((wgsl2.splitOn "if").length >= 2)) ++
  test "if-else with different loop bodies" ((wgsl2.splitOn "for").length >= 3)

def testControlFlowWithBuffers : TestSeq :=
  -- Loop reading and writing buffers
  let shader1 : ShaderM Unit := do
    let inputBuf ← declareInputBuffer "input" (.scalar .f32)
    let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

    loop (litU32 0) (litU32 64) (litU32 1) fun i => do
      let val ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf i
      let doubled := val * lit 2.0
      writeBuffer (ty := .scalar .f32) outputBuf i doubled
  let wgsl1 := generateWGSLSimple shader1

  -- Conditional buffer write
  let shader2 : ShaderM Unit := do
    let inputBuf ← declareInputBuffer "input" (.scalar .f32)
    let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

    loop (litU32 0) (litU32 64) (litU32 1) fun i => do
      let val ← readBuffer (ty := .scalar .f32) (n := 1024) inputBuf i
      if_ (val .>. lit 0.0) (do
        writeBuffer (ty := .scalar .f32) outputBuf i (val * lit 2.0)
      ) (do
        writeBuffer (ty := .scalar .f32) outputBuf i (lit 0.0)
      )
  let wgsl2 := generateWGSLSimple shader2

  test "loop with buffer operations" (((wgsl1.splitOn "for").length >= 2) && ((wgsl1.splitOn "@group").length >= 2)) ++
  test "buffer reads in loop" ((wgsl1.splitOn "input").length >= 2) ++
  test "conditional buffer writes" (((wgsl2.splitOn "if").length >= 2) && ((wgsl2.splitOn "output").length >= 2))

-- ============================================================================
-- All Tests
-- ============================================================================

def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running WGSL DSL Tests..."

  pure [
    ("Scalar Types", [testScalarTypes]),
    ("Vector Types", [testVectorTypes]),
    ("Matrix Types", [testMatrixTypes]),
    ("Array Types", [testArrayTypes]),
    ("Pointer Types", [testPtrTypes]),
    ("Byte Sizes", [testByteSizes]),
    ("Literals", [testLiterals]),
    ("Variables", [testVariables]),
    ("Arithmetic Operators", [testArithmeticOps]),
    ("Comparison Operators", [testComparisonOps]),
    ("Boolean Operators", [testBooleanOps]),
    ("Math Functions", [testMathFunctions]),
    ("Type Conversions", [testTypeConversions]),
    ("Vector Constructors", [testVectorConstructors]),
    ("Vector Accessors", [testVectorAccessors]),
    ("Operator Precedence", [testOperatorPrecedence]),
    ("Comparison Precedence", [testComparisonPrecedence]),
    ("Logical Precedence", [testLogicalPrecedence]),
    ("Complex Expressions", [testComplexExpressions]),
    ("If Statements", [testIfStatement]),
    ("Loop Statements", [testLoopStatement]),
    ("For Statements", [testForStatement]),
    ("Complex Control Flow", [testComplexControlFlow]),
    ("Control Flow with Buffers", [testControlFlowWithBuffers])
  ]

end Tests.WGSLDSLTests
