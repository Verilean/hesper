import Hesper.Transpile.CUDA
import Hesper.WGSL.Exp
import Hesper.WGSL.ExpRepr

/-! # Phase 1 transpiler test

Exercises lex + parse + lower on real CUDA expressions from
llama.cpp's `vec_dot_q4_K_q8_1_impl_vmmq`. Compares the lowered
`Exp (.scalar .u32)` against a hand-written reference `Exp` for
structural equality.
-/
namespace Hesper.Transpile.CUDA.Test

open Hesper.WGSL Hesper.Transpile.CUDA

/-- Compare two Exps via their canonical S-expression rendering
    (provided by `Hesper.WGSL.ExpRepr`). -/
def assertExpEq (label : String) (got expected : Exp (.scalar .u32)) : IO Unit := do
  let g := got.toSExp
  let e := expected.toSExp
  if g = e then
    IO.println s!"PASS  {label}"
  else
    IO.println s!"FAIL  {label}"
    IO.println s!"  got      = {g}"
    IO.println s!"  expected = {e}"

/-- Run lex+parse+lower on a CUDA expression string. -/
def transpileU32 (env : IdEnv) (src : String) : Except String (Exp (.scalar .u32)) :=
  cudaU32WithIdEnv env src

/-- Compare i32-typed Exps. -/
def assertI32Eq (label : String) (got expected : Exp (.scalar .i32)) : IO Unit := do
  let g := got.toSExp; let e := expected.toSExp
  if g = e then IO.println s!"PASS  {label}"
  else IO.println s!"FAIL  {label}\n  got      = {g}\n  expected = {e}"

/-- Compare f32-typed Exps. -/
def assertF32Eq (label : String) (got expected : Exp (.scalar .f32)) : IO Unit := do
  let g := got.toSExp; let e := expected.toSExp
  if g = e then IO.println s!"PASS  {label}"
  else IO.println s!"FAIL  {label}\n  got      = {g}\n  expected = {e}"

def main : IO Unit := do
  IO.println "=== Phase 1 CUDA → ShaderM expression transpile tests ==="

  -- Test 1: integer literal
  match transpileU32 emptyEnv "42" with
  | .ok e => assertExpEq "decimal literal 42" e (Exp.litU32 42)
  | .error err => IO.println s!"FAIL  decimal literal 42: {err}"

  -- Test 2: hex literal with suffix
  match transpileU32 emptyEnv "0x0F0F0F0F" with
  | .ok e => assertExpEq "hex literal" e (Exp.litU32 0x0F0F0F0F)
  | .error err => IO.println s!"FAIL  hex literal: {err}"

  -- Test 3: identifier
  match transpileU32 emptyEnv "v" with
  | .ok e => assertExpEq "identifier 'v'" e (Exp.var "v")
  | .error err => IO.println s!"FAIL  identifier: {err}"

  -- Test 4: shift + mask (the canonical Q4_K nibble extract)
  --   (v >> 4) & 0x0F0F0F0F
  let expected4 : Exp (.scalar .u32) :=
    Exp.bitAnd (Exp.shiftRight (Exp.var "v") (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
  match transpileU32 emptyEnv "(v >> 4) & 0x0F0F0F0F" with
  | .ok e => assertExpEq "(v >> 4) & 0x0F0F0F0F" e expected4
  | .error err => IO.println s!"FAIL  shift+mask: {err}"

  -- Test 5: nested arithmetic (the qsOff calculation in MMQ kernels)
  --   4 + bq8Off * 4 + elemOff
  let expected5 : Exp (.scalar .u32) :=
    Exp.add (Exp.add (Exp.litU32 4)
                     (Exp.mul (Exp.var "bq8Off") (Exp.litU32 4)))
            (Exp.var "elemOff")
  match transpileU32 emptyEnv "4 + bq8Off * 4 + elemOff" with
  | .ok e => assertExpEq "qsOff arithmetic" e expected5
  | .error err => IO.println s!"FAIL  qsOff arith: {err}"

  -- Test 6: bit-or of two masked shifts (vih_0 in vec_dot_q6_K)
  --   (vh << 4) & 0x30303030 | vl & 0x0F0F0F0F
  let expected6 : Exp (.scalar .u32) :=
    Exp.bitOr
      (Exp.bitAnd (Exp.shiftLeft (Exp.var "vh") (Exp.litU32 4)) (Exp.litU32 0x30303030))
      (Exp.bitAnd (Exp.var "vl") (Exp.litU32 0x0F0F0F0F))
  match transpileU32 emptyEnv "((vh << 4) & 0x30303030) | (vl & 0x0F0F0F0F)" with
  | .ok e => assertExpEq "Q6_K vih+vil bit ops" e expected6
  | .error err => IO.println s!"FAIL  Q6_K vih+vil: {err}"

  -- Test 7: cast (which we drop in Phase 1)
  --   (uint32_t) raw
  match transpileU32 emptyEnv "(uint32_t) raw" with
  | .ok e => assertExpEq "cast (drop)" e (Exp.var "raw")
  | .error err => IO.println s!"FAIL  cast: {err}"

  -- Test 8: env-mapped identifier
  --   `tid` → `Exp.vec3X (Exp.var \"localId\")` semantically, but for
  --   Phase 1 just stub it as `Exp.var \"tid_lane\"`.
  let env : IdEnv := fun s =>
    if s == "tid" then some (Exp.var "tid_lane") else none
  let expected8 : Exp (.scalar .u32) :=
    Exp.bitAnd (Exp.var "tid_lane") (Exp.litU32 31)
  match transpileU32 env "tid & 31" with
  | .ok e => assertExpEq "env-mapped tid" e expected8
  | .error err => IO.println s!"FAIL  env-mapped tid: {err}"

  -- Test 9: cudaU32! ergonomics — compile-time fixed CUDA snippet,
  -- no Except to unwrap. This is the "ideal API" for kernel authors:
  --   let v0i0 := cudaU32! "v0 & 0x0F0F0F0F"
  --   let v0i1 := cudaU32! "(v0 >> 4) & 0x0F0F0F0F"
  let v0i0 := cudaU32! "v0 & 0x0F0F0F0F"
  let v0i0_ref : Exp (.scalar .u32) :=
    Exp.bitAnd (Exp.var "v0") (Exp.litU32 0x0F0F0F0F)
  assertExpEq "cudaU32! v0 & 0x0F0F0F0F" v0i0 v0i0_ref

  let v0i1 := cudaU32! "(v0 >> 4) & 0x0F0F0F0F"
  let v0i1_ref : Exp (.scalar .u32) :=
    Exp.bitAnd (Exp.shiftRight (Exp.var "v0") (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
  assertExpEq "cudaU32! (v0 >> 4) & 0x0F" v0i1 v0i1_ref

  IO.println ""
  IO.println "=== Phase 2: i32 / f32 lowering ==="

  -- Test 11: __dp4a — the canonical Q4_K inner-loop intrinsic.
  --   ggml_cuda_dp4a(v0i, u0, sumi)
  --   = sumi + dot4I8Packed(v0i, u0)
  let dp := cudaI32! "__dp4a(v0i, u0, sumi)"
  let dp_ref : Exp (.scalar .i32) :=
    Exp.add (Exp.var "sumi") (Exp.dot4I8Packed (Exp.var "v0i") (Exp.var "u0"))
  assertI32Eq "cudaI32! __dp4a(v, u, c)" dp dp_ref

  -- Test 12: chained dp4a (real llama.cpp pattern from
  -- vec_dot_q4_K_q8_1_impl_vmmq line 514):
  --   const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
  -- We test the simpler 2-level nesting.
  let dp2 := cudaI32! "__dp4a(v1i, u1, __dp4a(v0i, u0, 0))"
  let dp2_ref : Exp (.scalar .i32) :=
    Exp.add (Exp.add (Exp.litI32 0)
                     (Exp.dot4I8Packed (Exp.var "v0i") (Exp.var "u0")))
            (Exp.dot4I8Packed (Exp.var "v1i") (Exp.var "u1"))
  assertI32Eq "cudaI32! chained dp4a" dp2 dp2_ref

  -- Test 13: i32 arith (sumi_d * sc[i])
  let sci := cudaI32! "dot * sc"
  let sci_ref : Exp (.scalar .i32) := Exp.mul (Exp.var "dot") (Exp.var "sc")
  assertI32Eq "cudaI32! dot * sc" sci sci_ref

  -- Test 14: f32 arithmetic — d8 * (f32(dot) * scale)
  let f1 := cudaF32! "d8 * dot * scale"
  let f1_ref : Exp (.scalar .f32) :=
    Exp.mul (Exp.mul (Exp.var "d8") (Exp.var "dot")) (Exp.var "scale")
  assertF32Eq "cudaF32! d8 * dot * scale" f1 f1_ref

  -- Test 15: float literal
  let f2 := cudaF32! "0.5"
  let f2_ref : Exp (.scalar .f32) := Exp.litF32 0.5
  assertF32Eq "cudaF32! literal 0.5" f2 f2_ref

  -- Test 16: __fmaf_rn(a, b, c)
  let fma := cudaF32! "__fmaf_rn(a, b, c)"
  let fma_ref : Exp (.scalar .f32) :=
    Exp.fma (Exp.var "a") (Exp.var "b") (Exp.var "c")
  assertF32Eq "cudaF32! __fmaf_rn" fma fma_ref

  -- Test 17: int → float cast: (float) dot * scale
  let castMul := cudaF32! "(float) dot * scale"
  let castMul_ref : Exp (.scalar .f32) :=
    Exp.mul (Exp.toF32 (Exp.var "dot" : Exp (.scalar .i32))) (Exp.var "scale")
  assertF32Eq "cudaF32! (float) dot * scale" castMul castMul_ref

  IO.println "=== done ==="

end Hesper.Transpile.CUDA.Test

def main : IO Unit := Hesper.Transpile.CUDA.Test.main
