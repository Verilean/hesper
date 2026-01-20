import LSpec
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL

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
-- Math Function Tests
-- ============================================================================

def testMathFunctions : TestSeq :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  let sqrtExpr := sqrt' x
  let absExpr := abs' x
  let minExpr := min' x y
  let maxExpr := max' x y
  let clampExpr := clamp' x (lit 0.0) (lit 1.0)
  let expExpr := exp' x
  let sinExpr := sin' x
  let cosExpr := cos' x
  let powExpr := pow' x y
  let tanhExpr := tanh' x

  test "sqrt(x)" (sqrtExpr.toWGSL == "sqrt(x)") ++
  test "abs(x)" (absExpr.toWGSL == "abs(x)") ++
  test "min(x, y)" (minExpr.toWGSL == "min(x, y)") ++
  test "max(x, y)" (maxExpr.toWGSL == "max(x, y)") ++
  test "clamp(x, 0.0, 1.0)" (clampExpr.toWGSL == "clamp(x, 0.000000, 1.000000)") ++
  test "exp(x)" (expExpr.toWGSL == "exp(x)") ++
  test "sin(x)" (sinExpr.toWGSL == "sin(x)") ++
  test "cos(x)" (cosExpr.toWGSL == "cos(x)") ++
  test "pow(x, y)" (powExpr.toWGSL == "pow(x, y)") ++
  test "tanh(x)" (tanhExpr.toWGSL == "tanh(x)")

-- ============================================================================
-- Operator Precedence Tests
-- ============================================================================

def testOperatorPrecedence : TestSeq :=
  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"
  let c : Exp (.scalar .f32) := var "c"

  let addMul := a + b * c
  let mulAdd := a * b + c
  let complex1 := a + b * c - (a / b)
  let complex2 := (a + b) * (c - (a / b))

  test "a + b * c (right parentheses)" (addMul.toWGSL == "(a + (b * c))") ++
  test "a * b + c (left parentheses)" (mulAdd.toWGSL == "((a * b) + c)") ++
  test "complex expression 1" (complex1.toWGSL == "((a + (b * c)) - (a / b))") ++
  test "complex expression 2" (complex2.toWGSL == "((a + b) * (c - (a / b)))")

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
    ("Operator Precedence", [testOperatorPrecedence]),
    ("Complex Expressions", [testComplexExpressions])
  ]

end Tests.WGSLDSLTests
