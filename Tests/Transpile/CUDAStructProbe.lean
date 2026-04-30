import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # POD struct member access probe

llama.cpp prefill kernels work with packed structs:
```
struct block_q4_K {
    half2 dm;          // offset 0  (4 bytes)
    uint8_t scales[12]; // offset 4  (12 bytes)
    uint8_t qs[128];   // offset 16 (128 bytes)
};                      // total 144 bytes
```

Typical access patterns from llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:
- `const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;`  (cast + offset)
- `bq4_K->qs[i]`            (struct → byte-array member, then element index)
- `bq4_K->scales`           (struct → byte-array member, returning pointer)
- `bq4_K->dm`               (struct → packed half2 scalar)

For now, we test the key patterns separately to drive the design.
The goal of this probe is to expose what the transpiler currently
*can't* lower, so we can add primitives one by one.
-/
namespace Hesper.Transpile.CUDA.StructProbe

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-! ## Probe 1: simple struct member, then array index

```
bq4_K->qs[i]
```

The cleanest way to model this in the transpiler: register `bq4_K` as
a "struct pointer" in env. Member access `bq4_K->qs[i]` lowers to
a load from the underlying byte buffer at offset
`<bq4_K base offset> + offsetof(qs) + i * sizeof(qs elem)`. -/
def probe1Body : String :=
"{
  result_lo = bq4_K_qs[i];
}"

/-! ## Probe 2: struct member returning a pointer, then cast + index

```
const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
v[0] = q4[0];
```

This combines: member access + pointer arithmetic + cast + index. -/
def probe2Body : String :=
"{
  v0 = bq4_K_qs_int[bq8_offset_div_4];
}"

/-! ## Probe 3: struct member as packed half2

```
const float2 dm4f = __half22float2(bq4_K->dm);
result = dm4f.x * sumf_d - dm4f.y * sumf_m;
```

Already works after Phase 7 (vec2 inline subst) if we model
`bq4_K->dm` as a u32 scalar value bound in env. -/
def probe3Body : String :=
"{
  const float2 dm4f = __half22float2(bq4_K_dm);
  result = dm4f.x * sumf_d - dm4f.y * sumf_m;
}"

/-! ## Probe 4: native `bq4_K->qs[i]` syntax (NOT pre-flattened)

This is the actual llama.cpp source pattern. Until we add a struct
descriptor mechanism, this should fail — but the failure mode tells
us what to add. -/
def probe4Body : String :=
"{
  result_lo = bq4_K->qs[i];
}"

/-! ## Probe 5: native struct ptr arithmetic + member

```
const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
result_lo = bq4_K->qs[i];
```

This is the more realistic pattern — base pointer is computed,
then dereferenced through the struct. -/
def probe5Body : String :=
"{
  result_lo = bq4_K->qs[i];
  result_hi = bq4_K->qs[i + 1];
}"

/-! ## Probe 6: native `bq4_K->dm` as packed half2

This is the actual llama.cpp pattern from vec_dot_q4_K_q8_1. -/
def probe6Body : String :=
"{
  const float2 dm4f = __half22float2(bq4_K->dm);
  result = dm4f.x * sumf_d - dm4f.y * sumf_m;
}"

def envFor : Env := {
  -- Pre-flattened buffer names (probes 1-3) AND struct-field resolver
  -- (probes 4-5).
  bufs := fun n => match n with
    | "bq4_K_qs"     => some { name := "x_qs_buf",     elemTy := .scalar .u32 }
    | "bq4_K_qs_int" => some { name := "x_qs_buf",     elemTy := .scalar .i32 }
    | _ => none
  u32 := fun n => match n with
    | "bq4_K_dm" => some (Exp.var "bq4_K_dm")
    | _ => none
  f32 := fun n => match n with
    | "result"     => some (Exp.var "result")
    | "result_lo"  => some (Exp.var "result_lo")
    | "result_hi"  => some (Exp.var "result_hi")
    | "sumf_d"     => some (Exp.var "sumf_d")
    | "sumf_m"     => some (Exp.var "sumf_m")
    | _ => none
  consts := fun n => match n with
    | "i" => some 5
    | "bq8_offset_div_4" => some 2
    | _ => none
  -- POD struct field resolver: `bq4_K->qs[i]` lowers to a load from
  -- `x_qs_buf` (the underlying byte buffer for the qs field).  In a
  -- real wire-up, the user would also bake an `<ibx>*sizeof(block_q4_K)
  -- + offsetof(qs)` offset into BufBinding.offset?, but for this probe
  -- we test just the syntax-level resolution.
  structFields := fun base field => match base, field with
    | "bq4_K", "qs" => some { name := "x_qs_buf", elemTy := .scalar .u32 }
    | _, _ => none
  structFieldU32 := fun base field => match base, field with
    | "bq4_K", "dm" => some (Exp.var "bq4_K_dm_packed")
    | _, _ => none
}

def probe (name : String) (body : String) : IO Unit := do
  IO.println s!"--- {name} ---"
  match parseStmtStr body with
  | .error e =>
    IO.println s!"  ✖ PARSE ERROR: {e}"
  | .ok stmt =>
    match lowerStmt envFor stmt with
    | .error e =>
      IO.println s!"  ✖ LOWER ERROR: {e}"
    | .ok _ =>
      IO.println s!"  ✓ PARSE + LOWER OK"

def main : IO Unit := do
  IO.println "═══ POD struct member-access probe ═══"
  IO.println ""
  probe "Probe 1: bq4_K->qs[i] via flattened buffer" probe1Body
  probe "Probe 2: bq4_K->qs cast (i32 view)"          probe2Body
  probe "Probe 3: bq4_K->dm as packed half2"          probe3Body
  probe "Probe 4: native obj->member[i] syntax"       probe4Body
  probe "Probe 5: native obj->member[i] (multi-use)"  probe5Body
  probe "Probe 6: native obj->dm as packed half2"     probe6Body
  IO.println ""
  IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.StructProbe

def main : IO Unit := Hesper.Transpile.CUDA.StructProbe.main
