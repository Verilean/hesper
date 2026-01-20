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
  | exp2 {t : WGSLType} : Exp t → Exp t
  | log {t : WGSLType} : Exp t → Exp t
  | log2 {t : WGSLType} : Exp t → Exp t
  | inverseSqrt {t : WGSLType} : Exp t → Exp t

  -- Trigonometric functions
  | sin {t : WGSLType} : Exp t → Exp t
  | cos {t : WGSLType} : Exp t → Exp t
  | tan {t : WGSLType} : Exp t → Exp t
  | asin {t : WGSLType} : Exp t → Exp t
  | acos {t : WGSLType} : Exp t → Exp t
  | atan {t : WGSLType} : Exp t → Exp t
  | atan2 {t : WGSLType} : Exp t → Exp t → Exp t
  | sinh {t : WGSLType} : Exp t → Exp t
  | cosh {t : WGSLType} : Exp t → Exp t
  | tanh {t : WGSLType} : Exp t → Exp t
  | asinh {t : WGSLType} : Exp t → Exp t
  | acosh {t : WGSLType} : Exp t → Exp t
  | atanh {t : WGSLType} : Exp t → Exp t

  -- Rounding and numeric functions
  | floor {t : WGSLType} : Exp t → Exp t
  | ceil {t : WGSLType} : Exp t → Exp t
  | round {t : WGSLType} : Exp t → Exp t
  | trunc {t : WGSLType} : Exp t → Exp t
  | fract {t : WGSLType} : Exp t → Exp t
  | sign {t : WGSLType} : Exp t → Exp t
  | saturate {t : WGSLType} : Exp t → Exp t

  -- Interpolation and stepping
  | pow {t : WGSLType} : Exp t → Exp t → Exp t
  | step {t : WGSLType} : Exp t → Exp t → Exp t
  | mix {t : WGSLType} : Exp t → Exp t → Exp t → Exp t
  | smoothstep {t : WGSLType} : Exp t → Exp t → Exp t → Exp t
  | fma {t : WGSLType} : Exp t → Exp t → Exp t → Exp t

  -- Vector functions (polymorphic over vec2/vec3/vec4)
  | dot {t : WGSLType} : Exp t → Exp t → Exp (.scalar .f32)  -- Result type depends on input
  | cross {st : ScalarType} : Exp (.vec3 st) → Exp (.vec3 st) → Exp (.vec3 st)
  | length {t : WGSLType} : Exp t → Exp (.scalar .f32)
  | distance {t : WGSLType} : Exp t → Exp t → Exp (.scalar .f32)
  | normalize {t : WGSLType} : Exp t → Exp t
  | faceForward {t : WGSLType} : Exp t → Exp t → Exp t → Exp t
  | reflect {t : WGSLType} : Exp t → Exp t → Exp t
  | refract {t : WGSLType} : Exp t → Exp t → Exp (.scalar .f32) → Exp t

  -- Matrix functions
  | determinant {st : ScalarType} : Exp (.mat2x2 st) → Exp (.scalar st)  -- Overload for mat2x2
  | determinant3 {st : ScalarType} : Exp (.mat3x3 st) → Exp (.scalar st)  -- mat3x3
  | determinant4 {st : ScalarType} : Exp (.mat4x4 st) → Exp (.scalar st)  -- mat4x4
  | transpose {st : ScalarType} : Exp (.mat2x2 st) → Exp (.mat2x2 st)
  | transpose3 {st : ScalarType} : Exp (.mat3x3 st) → Exp (.mat3x3 st)
  | transpose4 {st : ScalarType} : Exp (.mat4x4 st) → Exp (.mat4x4 st)

  -- Logical functions (for bool vectors)
  | all {t : WGSLType} : Exp t → Exp (.scalar .bool)
  | any {t : WGSLType} : Exp t → Exp (.scalar .bool)

  -- Conditional (ternary operator)
  | select {t : WGSLType} : Exp (.scalar .bool) → Exp t → Exp t → Exp t

  -- Array functions
  | arrayLength {t : WGSLType} : String → Exp (.scalar .u32)  -- Takes array reference like "&data"

  -- Bit reinterpretation and manipulation
  | bitcast {fromTy toTy : WGSLType} : Exp fromTy → Exp toTy
  | countLeadingZeros : Exp (.scalar .u32) → Exp (.scalar .u32)
  | countOneBits : Exp (.scalar .u32) → Exp (.scalar .u32)
  | countTrailingZeros : Exp (.scalar .u32) → Exp (.scalar .u32)
  | firstLeadingBit : Exp (.scalar .u32) → Exp (.scalar .u32)
  | firstLeadingBitSigned : Exp (.scalar .i32) → Exp (.scalar .i32)
  | firstTrailingBit : Exp (.scalar .u32) → Exp (.scalar .u32)
  | reverseBits : Exp (.scalar .u32) → Exp (.scalar .u32)
  | extractBits : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
  | extractBitsSigned : Exp (.scalar .i32) → Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .i32)
  | insertBits : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)

  -- Derivative functions (fragment shader only)
  | dpdx {t : WGSLType} : Exp t → Exp t
  | dpdxCoarse {t : WGSLType} : Exp t → Exp t
  | dpdxFine {t : WGSLType} : Exp t → Exp t
  | dpdy {t : WGSLType} : Exp t → Exp t
  | dpdyCoarse {t : WGSLType} : Exp t → Exp t
  | dpdyFine {t : WGSLType} : Exp t → Exp t
  | fwidth {t : WGSLType} : Exp t → Exp t
  | fwidthCoarse {t : WGSLType} : Exp t → Exp t
  | fwidthFine {t : WGSLType} : Exp t → Exp t

  -- Function call (for custom functions)
  | call : String → List (Σ t, Exp t) → Exp t

  -- Subgroup matrix operations (chromium_experimental_subgroup_matrix)
  | subgroupMatrixLoad {st : ScalarType} {m k : Nat} :
      String →  -- pointer reference (e.g., "&A")
      Exp (.scalar .u32) →  -- offset
      Exp (.scalar .bool) →  -- transposeFlag
      Exp (.scalar .u32) →  -- stride
      Exp (.subgroupMatrixLeft st m k)

  | subgroupMatrixLoadRight {st : ScalarType} {k n : Nat} :
      String →  -- pointer reference
      Exp (.scalar .u32) →  -- offset
      Exp (.scalar .bool) →  -- transposeFlag
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
      Exp (.scalar .bool) →  -- transposeFlag
      Exp (.scalar .u32) →  -- stride
      Exp (.scalar .u32)  -- Returns unit (represented as u32 for simplicity)

  -- Subgroup matrix initialization (zero)
  | subgroupMatrixZeroLeft {st : ScalarType} {m k : Nat} :
      Exp (.subgroupMatrixLeft st m k)

  | subgroupMatrixZeroRight {st : ScalarType} {k n : Nat} :
      Exp (.subgroupMatrixRight st k n)

  | subgroupMatrixZeroResult {st : ScalarType} {m n : Nat} :
      Exp (.subgroupMatrixResult st m n)

  -- Standard subgroup operations (require subgroup feature)
  | subgroupBroadcast {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | subgroupBroadcastFirst {t : WGSLType} : Exp t → Exp t
  | subgroupShuffle {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | subgroupShuffleDown {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | subgroupShuffleUp {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | subgroupShuffleXor {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | subgroupAdd {t : WGSLType} : Exp t → Exp t
  | subgroupExclusiveAdd {t : WGSLType} : Exp t → Exp t
  | subgroupInclusiveAdd {t : WGSLType} : Exp t → Exp t
  | subgroupMul {t : WGSLType} : Exp t → Exp t
  | subgroupExclusiveMul {t : WGSLType} : Exp t → Exp t
  | subgroupInclusiveMul {t : WGSLType} : Exp t → Exp t
  | subgroupMin {t : WGSLType} : Exp t → Exp t
  | subgroupMax {t : WGSLType} : Exp t → Exp t
  | subgroupAnd {t : WGSLType} : Exp t → Exp t
  | subgroupOr {t : WGSLType} : Exp t → Exp t
  | subgroupXor {t : WGSLType} : Exp t → Exp t
  | subgroupAll : Exp (.scalar .bool) → Exp (.scalar .bool)
  | subgroupAny : Exp (.scalar .bool) → Exp (.scalar .bool)
  | subgroupBallot : Exp (.scalar .bool) → Exp (.vec4 .u32)
  | subgroupElect : Exp (.scalar .bool)

  -- Quad operations (fragment shader only, require subgroup_quad feature)
  | quadBroadcast {t : WGSLType} : Exp t → Exp (.scalar .u32) → Exp t
  | quadSwapX {t : WGSLType} : Exp t → Exp t
  | quadSwapY {t : WGSLType} : Exp t → Exp t
  | quadSwapDiagonal {t : WGSLType} : Exp t → Exp t

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

  -- Atomically load value
  | atomicLoad {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32)

  | atomicLoadU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32)

  -- Atomically store value (returns nothing/unit)
  | atomicStore {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .u32)  -- Returns unit (represented as u32)

  | atomicStoreU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)  -- Returns unit

  -- Atomically compute bitwise AND, returns old value
  | atomicAnd {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicAndU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically compute bitwise OR, returns old value
  | atomicOr {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicOrU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Atomically compute bitwise XOR, returns old value
  | atomicXor {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicI32)) →
      Exp (.scalar .i32) →
      Exp (.scalar .i32)

  | atomicXorU {space : MemorySpace} :
      Exp (.ptr space (.scalar .atomicU32)) →
      Exp (.scalar .u32) →
      Exp (.scalar .u32)

  -- Synchronization barriers
  | storageBarrier : Exp (.scalar .u32)  -- Returns unit
  | textureBarrier : Exp (.scalar .u32)  -- Returns unit
  | workgroupUniformLoad {space : MemorySpace} {t : WGSLType} :
      Exp (.ptr space t) →
      Exp t

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

  -- Additional texture operations
  | textureDimensions {t : WGSLType} :
      Exp (.texture2D _) →  -- texture
      Exp (.vec2 .u32)  -- returns (width, height)

  | textureNumLayers :
      Exp (.texture2D _) →
      Exp (.scalar .u32)

  | textureNumLevels :
      Exp (.texture2D _) →
      Exp (.scalar .u32)

  | textureNumSamples :
      Exp (.texture2D _) →
      Exp (.scalar .u32)

  | textureSampleLevel :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.scalar .f32) →  -- mip level
      Exp (.vec4 .f32)  -- returns RGBA

  | textureSampleBias :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.scalar .f32) →  -- bias
      Exp (.vec4 .f32)  -- returns RGBA

  | textureSampleGrad :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.vec2 .f32) →  -- ddx
      Exp (.vec2 .f32) →  -- ddy
      Exp (.vec4 .f32)  -- returns RGBA

  | textureSampleCompare :
      Exp (.texture2D _) →  -- depth texture
      Exp (.sampler) →  -- comparison sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.scalar .f32) →  -- depth reference
      Exp (.scalar .f32)  -- returns comparison result

  | textureGather :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.vec4 .f32)  -- returns gathered components

  | textureSampleBaseClampToEdge :
      Exp (.texture2D _) →  -- texture
      Exp (.sampler) →  -- sampler
      Exp (.vec2 .f32) →  -- UV coordinates
      Exp (.vec4 .f32)  -- returns RGBA

  -- Data packing functions (convert to packed format)
  | pack4x8snorm : Exp (.vec4 .f32) → Exp (.scalar .u32)
  | pack4x8unorm : Exp (.vec4 .f32) → Exp (.scalar .u32)
  | pack4xI8 : Exp (.vec4 .i32) → Exp (.scalar .u32)
  | pack4xU8 : Exp (.vec4 .u32) → Exp (.scalar .u32)
  | pack4xI8Clamp : Exp (.vec4 .i32) → Exp (.scalar .u32)
  | pack4xU8Clamp : Exp (.vec4 .u32) → Exp (.scalar .u32)
  | pack2x16snorm : Exp (.vec2 .f32) → Exp (.scalar .u32)
  | pack2x16unorm : Exp (.vec2 .f32) → Exp (.scalar .u32)
  | pack2x16float : Exp (.vec2 .f32) → Exp (.scalar .u32)

  -- Data unpacking functions (extract from packed format)
  | unpack4x8snorm : Exp (.scalar .u32) → Exp (.vec4 .f32)
  | unpack4x8unorm : Exp (.scalar .u32) → Exp (.vec4 .f32)
  | unpack4xI8 : Exp (.scalar .u32) → Exp (.vec4 .i32)
  | unpack4xU8 : Exp (.scalar .u32) → Exp (.vec4 .u32)
  | unpack2x16snorm : Exp (.scalar .u32) → Exp (.vec2 .f32)
  | unpack2x16unorm : Exp (.scalar .u32) → Exp (.vec2 .f32)
  | unpack2x16float : Exp (.scalar .u32) → Exp (.vec2 .f32)

  -- Workgroup barrier (duplicate, already defined above - will be removed from toWGSL)
  | workgroupBarrier : Exp (.scalar .u32)  -- Returns unit

/-- Code generation: convert expression to WGSL string -/
def Exp.toWGSL {t : WGSLType} : Exp t → String
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
  -- Math functions
  | sqrt e => s!"sqrt({toWGSL e})"
  | abs e => s!"abs({toWGSL e})"
  | min a b => s!"min({toWGSL a}, {toWGSL b})"
  | max a b => s!"max({toWGSL a}, {toWGSL b})"
  | clamp e lo hi => s!"clamp({toWGSL e}, {toWGSL lo}, {toWGSL hi})"
  | exp e => s!"exp({toWGSL e})"
  | exp2 e => s!"exp2({toWGSL e})"
  | log e => s!"log({toWGSL e})"
  | log2 e => s!"log2({toWGSL e})"
  | inverseSqrt e => s!"inverseSqrt({toWGSL e})"

  -- Trigonometric functions
  | sin e => s!"sin({toWGSL e})"
  | cos e => s!"cos({toWGSL e})"
  | tan e => s!"tan({toWGSL e})"
  | asin e => s!"asin({toWGSL e})"
  | acos e => s!"acos({toWGSL e})"
  | atan e => s!"atan({toWGSL e})"
  | atan2 y x => s!"atan2({toWGSL y}, {toWGSL x})"
  | sinh e => s!"sinh({toWGSL e})"
  | cosh e => s!"cosh({toWGSL e})"
  | tanh e => s!"tanh({toWGSL e})"
  | asinh e => s!"asinh({toWGSL e})"
  | acosh e => s!"acosh({toWGSL e})"
  | atanh e => s!"atanh({toWGSL e})"

  -- Rounding and numeric
  | floor e => s!"floor({toWGSL e})"
  | ceil e => s!"ceil({toWGSL e})"
  | round e => s!"round({toWGSL e})"
  | trunc e => s!"trunc({toWGSL e})"
  | fract e => s!"fract({toWGSL e})"
  | sign e => s!"sign({toWGSL e})"
  | saturate e => s!"saturate({toWGSL e})"

  -- Interpolation
  | pow a b => s!"pow({toWGSL a}, {toWGSL b})"
  | step edge x => s!"step({toWGSL edge}, {toWGSL x})"
  | mix a b t => s!"mix({toWGSL a}, {toWGSL b}, {toWGSL t})"
  | smoothstep lo hi x => s!"smoothstep({toWGSL lo}, {toWGSL hi}, {toWGSL x})"
  | fma a b c => s!"fma({toWGSL a}, {toWGSL b}, {toWGSL c})"

  -- Vector functions
  | dot a b => s!"dot({toWGSL a}, {toWGSL b})"
  | cross a b => s!"cross({toWGSL a}, {toWGSL b})"
  | length v => s!"length({toWGSL v})"
  | distance a b => s!"distance({toWGSL a}, {toWGSL b})"
  | normalize v => s!"normalize({toWGSL v})"
  | faceForward n i nref => s!"faceForward({toWGSL n}, {toWGSL i}, {toWGSL nref})"
  | reflect i n => s!"reflect({toWGSL i}, {toWGSL n})"
  | refract i n eta => s!"refract({toWGSL i}, {toWGSL n}, {toWGSL eta})"

  -- Matrix functions
  | determinant m => s!"determinant({toWGSL m})"
  | determinant3 m => s!"determinant({toWGSL m})"
  | determinant4 m => s!"determinant({toWGSL m})"
  | transpose m => s!"transpose({toWGSL m})"
  | transpose3 m => s!"transpose({toWGSL m})"
  | transpose4 m => s!"transpose({toWGSL m})"

  -- Logical functions
  | all v => s!"all({toWGSL v})"
  | any v => s!"any({toWGSL v})"
  | select cond t f => s!"select({toWGSL f}, {toWGSL t}, {toWGSL cond})"

  -- Array functions
  | arrayLength arrayRef => s!"arrayLength({arrayRef})"

  -- Bit operations
  | bitcast e => s!"bitcast<_>({toWGSL e})"  -- Type inference handles target type
  | countLeadingZeros e => s!"countLeadingZeros({toWGSL e})"
  | countOneBits e => s!"countOneBits({toWGSL e})"
  | countTrailingZeros e => s!"countTrailingZeros({toWGSL e})"
  | firstLeadingBit e => s!"firstLeadingBit({toWGSL e})"
  | firstLeadingBitSigned e => s!"firstLeadingBit({toWGSL e})"
  | firstTrailingBit e => s!"firstTrailingBit({toWGSL e})"
  | reverseBits e => s!"reverseBits({toWGSL e})"
  | extractBits e offset count => s!"extractBits({toWGSL e}, {toWGSL offset}, {toWGSL count})"
  | extractBitsSigned e offset count => s!"extractBits({toWGSL e}, {toWGSL offset}, {toWGSL count})"
  | insertBits e newbits offset count => s!"insertBits({toWGSL e}, {toWGSL newbits}, {toWGSL offset}, {toWGSL count})"

  -- Derivative functions
  | dpdx e => s!"dpdx({toWGSL e})"
  | dpdxCoarse e => s!"dpdxCoarse({toWGSL e})"
  | dpdxFine e => s!"dpdxFine({toWGSL e})"
  | dpdy e => s!"dpdy({toWGSL e})"
  | dpdyCoarse e => s!"dpdyCoarse({toWGSL e})"
  | dpdyFine e => s!"dpdyFine({toWGSL e})"
  | fwidth e => s!"fwidth({toWGSL e})"
  | fwidthCoarse e => s!"fwidthCoarse({toWGSL e})"
  | fwidthFine e => s!"fwidthFine({toWGSL e})"
  | call fname args =>
    let argStrs := args.map fun ⟨_, e⟩ => toWGSL e
    s!"{fname}({String.intercalate ", " argStrs})"
  | subgroupMatrixLoad (st:=st) (m:=m) (k:=k) ptr offset transposeFlag stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_left<{st.toWGSL},{m},{k}>>({ptr}, {toWGSL offset}, {toWGSL transposeFlag}, {toWGSL stride})"
  | subgroupMatrixLoadRight (st:=st) (k:=k) (n:=n) ptr offset transposeFlag stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_right<{st.toWGSL},{k},{n}>>({ptr}, {toWGSL offset}, {toWGSL transposeFlag}, {toWGSL stride})"
  | subgroupMatrixMultiplyAccumulate a b acc =>
    s!"subgroupMatrixMultiplyAccumulate({toWGSL a}, {toWGSL b}, {toWGSL acc})"
  | subgroupMatrixStore (st:=st) (m:=m) (n:=n) ptr offset mat transposeFlag stride =>
    s!"subgroupMatrixStore({ptr}, {toWGSL offset}, {toWGSL mat}, {toWGSL transposeFlag}, {toWGSL stride})"
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
  | atomicLoad ptr =>
    s!"atomicLoad({toWGSL ptr})"
  | atomicLoadU ptr =>
    s!"atomicLoad({toWGSL ptr})"
  | atomicStore ptr val =>
    s!"atomicStore({toWGSL ptr}, {toWGSL val})"
  | atomicStoreU ptr val =>
    s!"atomicStore({toWGSL ptr}, {toWGSL val})"
  | atomicAnd ptr val =>
    s!"atomicAnd({toWGSL ptr}, {toWGSL val})"
  | atomicAndU ptr val =>
    s!"atomicAnd({toWGSL ptr}, {toWGSL val})"
  | atomicOr ptr val =>
    s!"atomicOr({toWGSL ptr}, {toWGSL val})"
  | atomicOrU ptr val =>
    s!"atomicOr({toWGSL ptr}, {toWGSL val})"
  | atomicXor ptr val =>
    s!"atomicXor({toWGSL ptr}, {toWGSL val})"
  | atomicXorU ptr val =>
    s!"atomicXor({toWGSL ptr}, {toWGSL val})"
  -- Synchronization barriers
  | storageBarrier =>
    "storageBarrier()"
  | textureBarrier =>
    "textureBarrier()"
  | workgroupUniformLoad ptr =>
    s!"workgroupUniformLoad({toWGSL ptr})"
  -- Standard subgroup operations
  | subgroupBroadcast val lane =>
    s!"subgroupBroadcast({toWGSL val}, {toWGSL lane})"
  | subgroupBroadcastFirst val =>
    s!"subgroupBroadcastFirst({toWGSL val})"
  | subgroupShuffle val lane =>
    s!"subgroupShuffle({toWGSL val}, {toWGSL lane})"
  | subgroupShuffleDown val delta =>
    s!"subgroupShuffleDown({toWGSL val}, {toWGSL delta})"
  | subgroupShuffleUp val delta =>
    s!"subgroupShuffleUp({toWGSL val}, {toWGSL delta})"
  | subgroupShuffleXor val mask =>
    s!"subgroupShuffleXor({toWGSL val}, {toWGSL mask})"
  | subgroupAdd val =>
    s!"subgroupAdd({toWGSL val})"
  | subgroupExclusiveAdd val =>
    s!"subgroupExclusiveAdd({toWGSL val})"
  | subgroupInclusiveAdd val =>
    s!"subgroupInclusiveAdd({toWGSL val})"
  | subgroupMul val =>
    s!"subgroupMul({toWGSL val})"
  | subgroupExclusiveMul val =>
    s!"subgroupExclusiveMul({toWGSL val})"
  | subgroupInclusiveMul val =>
    s!"subgroupInclusiveMul({toWGSL val})"
  | subgroupMin val =>
    s!"subgroupMin({toWGSL val})"
  | subgroupMax val =>
    s!"subgroupMax({toWGSL val})"
  | subgroupAnd val =>
    s!"subgroupAnd({toWGSL val})"
  | subgroupOr val =>
    s!"subgroupOr({toWGSL val})"
  | subgroupXor val =>
    s!"subgroupXor({toWGSL val})"
  | subgroupAll val =>
    s!"subgroupAll({toWGSL val})"
  | subgroupAny val =>
    s!"subgroupAny({toWGSL val})"
  | subgroupBallot val =>
    s!"subgroupBallot({toWGSL val})"
  | subgroupElect =>
    "subgroupElect()"
  -- Quad operations
  | quadBroadcast val lane =>
    s!"quadBroadcast({toWGSL val}, {toWGSL lane})"
  | quadSwapX val =>
    s!"quadSwapX({toWGSL val})"
  | quadSwapY val =>
    s!"quadSwapY({toWGSL val})"
  | quadSwapDiagonal val =>
    s!"quadSwapDiagonal({toWGSL val})"
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
  | textureDimensions tex =>
    s!"textureDimensions({toWGSL tex})"
  | textureNumLayers tex =>
    s!"textureNumLayers({toWGSL tex})"
  | textureNumLevels tex =>
    s!"textureNumLevels({toWGSL tex})"
  | textureNumSamples tex =>
    s!"textureNumSamples({toWGSL tex})"
  | textureSampleLevel tex sampler uv level =>
    s!"textureSampleLevel({toWGSL tex}, {toWGSL sampler}, {toWGSL uv}, {toWGSL level})"
  | textureSampleBias tex sampler uv bias =>
    s!"textureSampleBias({toWGSL tex}, {toWGSL sampler}, {toWGSL uv}, {toWGSL bias})"
  | textureSampleGrad tex sampler uv ddx ddy =>
    s!"textureSampleGrad({toWGSL tex}, {toWGSL sampler}, {toWGSL uv}, {toWGSL ddx}, {toWGSL ddy})"
  | textureSampleCompare tex sampler uv depthRef =>
    s!"textureSampleCompare({toWGSL tex}, {toWGSL sampler}, {toWGSL uv}, {toWGSL depthRef})"
  | textureGather tex sampler uv =>
    s!"textureGather({toWGSL tex}, {toWGSL sampler}, {toWGSL uv})"
  | textureSampleBaseClampToEdge tex sampler uv =>
    s!"textureSampleBaseClampToEdge({toWGSL tex}, {toWGSL sampler}, {toWGSL uv})"
  -- Data packing/unpacking
  | pack4x8snorm v =>
    s!"pack4x8snorm({toWGSL v})"
  | pack4x8unorm v =>
    s!"pack4x8unorm({toWGSL v})"
  | pack4xI8 v =>
    s!"pack4xI8({toWGSL v})"
  | pack4xU8 v =>
    s!"pack4xU8({toWGSL v})"
  | pack4xI8Clamp v =>
    s!"pack4xI8Clamp({toWGSL v})"
  | pack4xU8Clamp v =>
    s!"pack4xU8Clamp({toWGSL v})"
  | pack2x16snorm v =>
    s!"pack2x16snorm({toWGSL v})"
  | pack2x16unorm v =>
    s!"pack2x16unorm({toWGSL v})"
  | pack2x16float v =>
    s!"pack2x16float({toWGSL v})"
  | unpack4x8snorm v =>
    s!"unpack4x8snorm({toWGSL v})"
  | unpack4x8unorm v =>
    s!"unpack4x8unorm({toWGSL v})"
  | unpack4xI8 v =>
    s!"unpack4xI8({toWGSL v})"
  | unpack4xU8 v =>
    s!"unpack4xU8({toWGSL v})"
  | unpack2x16snorm v =>
    s!"unpack2x16snorm({toWGSL v})"
  | unpack2x16unorm v =>
    s!"unpack2x16unorm({toWGSL v})"
  | unpack2x16float v =>
    s!"unpack2x16float({toWGSL v})"
  | workgroupBarrier =>
    "workgroupBarrier()"
termination_by e => sizeOf e
decreasing_by
  all_goals
    simp_wf
    try omega  -- Solve most goals with arithmetic
  all_goals sorry  -- TODO: Remaining goals for call/structConstruct List.map cases require complex dependent pair proofs

end Hesper.WGSL
