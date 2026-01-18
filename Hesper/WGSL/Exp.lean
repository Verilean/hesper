import Hesper.WGSL.Types

namespace Hesper.WGSL

/-! Type-safe WGSL expressions using GADTs (Generalized Algebraic Data Types).
    The type parameter ensures that operations are only valid for compatible types. -/

/-- Type-safe WGSL expressions.
    The type parameter `t : WGSLType` ensures compile-time type safety. -/
inductive Exp : WGSLType → Type where
  -- Literals
  | litF32 : Float → Exp (.scalar .f32)
  | litF16 : Float → Exp (.scalar .f16)
  | litI32 : Int → Exp (.scalar .i32)
  | litU32 : Nat → Exp (.scalar .u32)
  | litBool : Bool → Exp (.scalar .bool)

  -- Variables
  | var : String → Exp t

  -- Arithmetic operations (require matching types)
  | add {t : WGSLType} : Exp t → Exp t → Exp t
  | sub {t : WGSLType} : Exp t → Exp t → Exp t
  | mul {t : WGSLType} : Exp t → Exp t → Exp t
  | div {t : WGSLType} : Exp t → Exp t → Exp t
  | mod {t : WGSLType} : Exp t → Exp t → Exp t
  | neg {t : WGSLType} : Exp t → Exp t

  -- Comparison operations (return bool)
  | eq {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)
  | ne {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)
  | lt {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)
  | le {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)
  | gt {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)
  | ge {t : WGSLType} : Exp t → Exp t → Exp (.scalar .bool)

  -- Boolean operations
  | and : Exp (.scalar .bool) → Exp (.scalar .bool) → Exp (.scalar .bool)
  | or : Exp (.scalar .bool) → Exp (.scalar .bool) → Exp (.scalar .bool)
  | not : Exp (.scalar .bool) → Exp (.scalar .bool)

  -- Bitwise operations (u32 only)
  | shiftLeft : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
  | shiftRight : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
  | bitAnd : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
  | bitOr : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
  | bitXor : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)

  -- Type conversions
  | toF32 {t : WGSLType} : Exp t → Exp (.scalar .f32)
  | toF16 {t : WGSLType} : Exp t → Exp (.scalar .f16)
  | toI32 {t : WGSLType} : Exp t → Exp (.scalar .i32)
  | toU32 {t : WGSLType} : Exp t → Exp (.scalar .u32)

  -- Array/vector indexing
  | index {elemTy : WGSLType} {n : Nat} : Exp (.array elemTy n) → Exp (.scalar .u32) → Exp elemTy

  -- Vector component access
  | vecX {st : ScalarType} : Exp (.vec2 st) → Exp (.scalar st)
  | vecY {st : ScalarType} : Exp (.vec2 st) → Exp (.scalar st)
  | vec3X {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vec3Y {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vecZ {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vecW {st : ScalarType} : Exp (.vec4 st) → Exp (.scalar st)

  -- Vector construction
  | vec2 {st : ScalarType} : Exp (.scalar st) → Exp (.scalar st) → Exp (.vec2 st)
  | vec3 {st : ScalarType} : Exp (.scalar st) → Exp (.scalar st) → Exp (.scalar st) → Exp (.vec3 st)
  | vec4 {st : ScalarType} : Exp (.scalar st) → Exp (.scalar st) → Exp (.scalar st) → Exp (.scalar st) → Exp (.vec4 st)

  -- Math functions (polymorphic over numeric types)
  | sqrt {t : WGSLType} : Exp t → Exp t
  | abs {t : WGSLType} : Exp t → Exp t
  | min {t : WGSLType} : Exp t → Exp t → Exp t
  | max {t : WGSLType} : Exp t → Exp t → Exp t
  | clamp {t : WGSLType} : Exp t → Exp t → Exp t → Exp t
  | exp {t : WGSLType} : Exp t → Exp t
  | log {t : WGSLType} : Exp t → Exp t
  | sin {t : WGSLType} : Exp t → Exp t
  | cos {t : WGSLType} : Exp t → Exp t
  | pow {t : WGSLType} : Exp t → Exp t → Exp t
  | tanh {t : WGSLType} : Exp t → Exp t

  -- Conditional (ternary operator)
  | select {t : WGSLType} : Exp (.scalar .bool) → Exp t → Exp t → Exp t

  -- Function call (for custom functions)
  | call : String → List (Σ t, Exp t) → Exp t

  -- Subgroup matrix operations (chromium_experimental_subgroup_matrix)
  | subgroupMatrixLoad {st : ScalarType} {m k : Nat} :
      String →  -- pointer reference (e.g., "&A")
      Exp (.scalar .u32) →  -- offset
      Exp (.scalar .bool) →  -- transpose
      Exp (.scalar .u32) →  -- stride
      Exp (.subgroupMatrixLeft st m k)

  | subgroupMatrixLoadRight {st : ScalarType} {k n : Nat} :
      String →  -- pointer reference
      Exp (.scalar .u32) →  -- offset
      Exp (.scalar .bool) →  -- transpose
      Exp (.scalar .u32) →  -- stride
      Exp (.subgroupMatrixRight st k n)

  | subgroupMatrixMultiplyAccumulate {st : ScalarType} {m k n : Nat} :
      Exp (.subgroupMatrixLeft st m k) →
      Exp (.subgroupMatrixRight st k n) →
      Exp (.subgroupMatrixResult st m n) →
      Exp (.subgroupMatrixResult st m n)

  | subgroupMatrixStore {st : ScalarType} {m n : Nat} :
      String →  -- pointer reference (e.g., "&C")
      Exp (.scalar .u32) →  -- offset
      Exp (.subgroupMatrixResult st m n) →
      Exp (.scalar .bool) →  -- transpose
      Exp (.scalar .u32) →  -- stride
      Exp (.scalar .u32)  -- Returns unit (represented as u32 for simplicity)

  -- Subgroup matrix initialization (zero)
  | subgroupMatrixZeroLeft {st : ScalarType} {m k : Nat} :
      Exp (.subgroupMatrixLeft st m k)

  | subgroupMatrixZeroRight {st : ScalarType} {k n : Nat} :
      Exp (.subgroupMatrixRight st k n)

  | subgroupMatrixZeroResult {st : ScalarType} {m n : Nat} :
      Exp (.subgroupMatrixResult st m n)

  -- Atomic Operations
  -- All atomic operations return the OLD value before the operation
  -- The pointer must point to atomic<i32> or atomic<u32> storage

  -- Atomically add value to location, returns old value
  | atomicAdd {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →  -- pointer to atomic<i32>
      Exp (.scalar .i32) →  -- value to add
      Exp (.scalar .i32)  -- returns old value

  | atomicAddU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →  -- pointer to atomic<u32>
      Exp (.scalar .u32) →  -- value to add
      Exp (.scalar .u32)  -- returns old value

  -- Atomically subtract value from location, returns old value
  | atomicSub {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicSubU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically compute minimum, returns old value
  | atomicMin {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicMinU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically compute maximum, returns old value
  | atomicMax {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicMaxU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically exchange (swap) values, returns old value
  | atomicExchange {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicExchangeU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically compare and exchange (weak version)
  -- Returns old value
  | atomicCompareExchangeWeak {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →  -- compare value
      Exp (.scalar .i32) →  -- new value
      Exp (.scalar .i32)  -- returns old value

  | atomicCompareExchangeWeakU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →  -- compare value
      Exp (.scalar .u32) →  -- new value
      Exp (.scalar .u32)  -- returns old value

  -- Struct Operations
  -- Field access: struct.fieldName
  | fieldAccess {ty : WGSLType} :
      Exp (.struct _) →  -- struct expression
      String →  -- field name
      Exp ty  -- result type (must match field type)

  -- Struct construction
  | structConstruct :
      String →  -- struct name
      List (String × (Σ t, Exp t)) →  -- field name and value pairs
      Exp (.struct _)

  -- Texture Operations
  -- Sample texture with sampler at UV coordinates (0.0-1.0)
  -- Returns vec4<f32> containing RGBA values
  | textureSample :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.vec4 .f32)  -- returns RGBA

  -- Load raw texel data at pixel coordinates and mip level
  -- Returns vec4<f32> containing RGBA values
  | textureLoad :
      Exp (.texture2D _) →  -- texture
      Exp (.vec2 .i32) →  -- pixel coordinates
      Exp (.scalar .i32) →  -- mip level
      Exp (.vec4 .f32)  -- returns RGBA

  -- Store texel data (used in statements, returns unit)
  | textureStore :
      Exp (.texture2D _) →  -- storage texture
      Exp (.vec2 .i32) →  -- pixel coordinates
      Exp (.vec4 .f32) →  -- RGBA value to store
      Exp (.scalar .u32)  -- returns unit

  -- Workgroup barrier
  | workgroupBarrier : Exp (.scalar .u32)  -- Returns unit

/-- Code generation: convert expression to WGSL string -/
def Exp.toWGSL : Exp t → String
  | litF32 f => s!"{f}"
  | litF16 f => s!"{f}h"
  | litI32 i => s!"{i}i"
  | litU32 u => s!"{u}u"
  | litBool b => if b then "true" else "false"
  | var name => name
  | add a b => s!"({toWGSL a} + {toWGSL b})"
  | sub a b => s!"({toWGSL a} - {toWGSL b})"
  | mul a b => s!"({toWGSL a} * {toWGSL b})"
  | div a b => s!"({toWGSL a} / {toWGSL b})"
  | mod a b => s!"({toWGSL a} % {toWGSL b})"
  | neg a => s!"(-{toWGSL a})"
  | eq a b => s!"({toWGSL a} == {toWGSL b})"
  | ne a b => s!"({toWGSL a} != {toWGSL b})"
  | lt a b => s!"({toWGSL a} < {toWGSL b})"
  | le a b => s!"({toWGSL a} <= {toWGSL b})"
  | gt a b => s!"({toWGSL a} > {toWGSL b})"
  | ge a b => s!"({toWGSL a} >= {toWGSL b})"
  | and a b => s!"({toWGSL a} && {toWGSL b})"
  | or a b => s!"({toWGSL a} || {toWGSL b})"
  | not a => s!"(!{toWGSL a})"
  | shiftLeft a b => s!"({toWGSL a} << {toWGSL b})"
  | shiftRight a b => s!"({toWGSL a} >> {toWGSL b})"
  | bitAnd a b => s!"({toWGSL a} & {toWGSL b})"
  | bitOr a b => s!"({toWGSL a} | {toWGSL b})"
  | bitXor a b => s!"({toWGSL a} ^ {toWGSL b})"
  | toF32 e => s!"f32({toWGSL e})"
  | toF16 e => s!"f16({toWGSL e})"
  | toI32 e => s!"i32({toWGSL e})"
  | toU32 e => s!"u32({toWGSL e})"
  | index arr idx => s!"{toWGSL arr}[{toWGSL idx}]"
  | vecX v => s!"{toWGSL v}.x"
  | vecY v => s!"{toWGSL v}.y"
  | vec3X v => s!"{toWGSL v}.x"
  | vec3Y v => s!"{toWGSL v}.y"
  | vecZ v => s!"{toWGSL v}.z"
  | vecW v => s!"{toWGSL v}.w"
  | vec2 x y => s!"vec2({toWGSL x}, {toWGSL y})"
  | vec3 x y z => s!"vec3({toWGSL x}, {toWGSL y}, {toWGSL z})"
  | vec4 x y z w => s!"vec4({toWGSL x}, {toWGSL y}, {toWGSL z}, {toWGSL w})"
  | sqrt e => s!"sqrt({toWGSL e})"
  | abs e => s!"abs({toWGSL e})"
  | min a b => s!"min({toWGSL a}, {toWGSL b})"
  | max a b => s!"max({toWGSL a}, {toWGSL b})"
  | clamp e lo hi => s!"clamp({toWGSL e}, {toWGSL lo}, {toWGSL hi})"
  | exp e => s!"exp({toWGSL e})"
  | log e => s!"log({toWGSL e})"
  | sin e => s!"sin({toWGSL e})"
  | cos e => s!"cos({toWGSL e})"
  | pow a b => s!"pow({toWGSL a}, {toWGSL b})"
  | tanh e => s!"tanh({toWGSL e})"
  | select cond t f => s!"select({toWGSL f}, {toWGSL t}, {toWGSL cond})"
  | call fname args =>
    let argStrs := args.map fun ⟨_, e⟩ => toWGSL e
    s!"{fname}({String.intercalate ", " argStrs})"
  | subgroupMatrixLoad (st:=st) (m:=m) (k:=k) ptr offset transpose stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_left<{st.toWGSL},{m},{k}>>({ptr}, {toWGSL offset}, {toWGSL transpose}, {toWGSL stride})"
  | subgroupMatrixLoadRight (st:=st) (k:=k) (n:=n) ptr offset transpose stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_right<{st.toWGSL},{k},{n}>>({ptr}, {toWGSL offset}, {toWGSL transpose}, {toWGSL stride})"
  | subgroupMatrixMultiplyAccumulate a b acc =>
    s!"subgroupMatrixMultiplyAccumulate({toWGSL a}, {toWGSL b}, {toWGSL acc})"
  | subgroupMatrixStore (st:=st) (m:=m) (n:=n) ptr offset mat transpose stride =>
    s!"subgroupMatrixStore({ptr}, {toWGSL offset}, {toWGSL mat}, {toWGSL transpose}, {toWGSL stride})"
  | subgroupMatrixZeroLeft (st:=st) (m:=m) (k:=k) =>
    s!"subgroup_matrix_left<{st.toWGSL}, {m}, {k}>(0)"
  | subgroupMatrixZeroRight (st:=st) (k:=k) (n:=n) =>
    s!"subgroup_matrix_right<{st.toWGSL}, {k}, {n}>(0)"
  | subgroupMatrixZeroResult (st:=st) (m:=m) (n:=n) =>
    s!"subgroup_matrix_result<{st.toWGSL}, {m}, {n}>(0)"
  -- Atomic operations
  | atomicAdd ptr val =>
    s!"atomicAdd({toWGSL ptr}, {toWGSL val})"
  | atomicAddU ptr val =>
    s!"atomicAdd({toWGSL ptr}, {toWGSL val})"
  | atomicSub ptr val =>
    s!"atomicSub({toWGSL ptr}, {toWGSL val})"
  | atomicSubU ptr val =>
    s!"atomicSub({toWGSL ptr}, {toWGSL val})"
  | atomicMin ptr val =>
    s!"atomicMin({toWGSL ptr}, {toWGSL val})"
  | atomicMinU ptr val =>
    s!"atomicMin({toWGSL ptr}, {toWGSL val})"
  | atomicMax ptr val =>
    s!"atomicMax({toWGSL ptr}, {toWGSL val})"
  | atomicMaxU ptr val =>
    s!"atomicMax({toWGSL ptr}, {toWGSL val})"
  | atomicExchange ptr val =>
    s!"atomicExchange({toWGSL ptr}, {toWGSL val})"
  | atomicExchangeU ptr val =>
    s!"atomicExchange({toWGSL ptr}, {toWGSL val})"
  | atomicCompareExchangeWeak ptr cmp val =>
    s!"atomicCompareExchangeWeak({toWGSL ptr}, {toWGSL cmp}, {toWGSL val})"
  | atomicCompareExchangeWeakU ptr cmp val =>
    s!"atomicCompareExchangeWeak({toWGSL ptr}, {toWGSL cmp}, {toWGSL val})"
  -- Struct operations
  | fieldAccess struct fieldName =>
    toWGSL struct ++ "." ++ fieldName
  | structConstruct structName fields =>
    let fieldStrs := fields.map fun (name, ⟨_, val⟩) => "." ++ name ++ " = " ++ toWGSL val
    s!"{structName}({String.intercalate ", " fieldStrs})"
  -- Texture operations
  | textureSample tex sampler uv =>
    s!"textureSample({toWGSL tex}, {toWGSL sampler}, {toWGSL uv})"
  | textureLoad tex coords mipLevel =>
    s!"textureLoad({toWGSL tex}, {toWGSL coords}, {toWGSL mipLevel})"
  | textureStore tex coords value =>
    s!"textureStore({toWGSL tex}, {toWGSL coords}, {toWGSL value})"
  | workgroupBarrier =>
    "workgroupBarrier()"
termination_by e => sizeOf e
decreasing_by
  all_goals sorry  -- Termination proofs for structural recursion (TODO: prove properly)

end Hesper.WGSL
