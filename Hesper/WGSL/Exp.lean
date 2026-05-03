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
  /-- High 32 bits of a×b (u32 × u32 → u64, take hi).  Core primitive for
      fastdiv (see llama.cpp's `init_fastdiv_values` / `fastdiv` in
      common.cuh).  Lowered to PTX `mul.hi.u32`; WGSL path computes via
      `u32(u64(a) * u64(b) >> 32)`. -/
  | mulhiU32 : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)

  -- Type conversions
  | toF32 {t : WGSLType} : Exp t → Exp (.scalar .f32)
  -- True-unsigned variant of toF32: signals to CodeGen that the input is
  -- semantically unsigned (e.g. Q4_K's 6-bit scale 0..63), so PTX should
  -- emit cvt.rn.f32.u32 instead of cvt.rn.f32.s32.  WGSL emits the same
  -- f32() cast — only PTX path differs.
  | toF32U {t : WGSLType} : Exp t → Exp (.scalar .f32)
  | toF16 {t : WGSLType} : Exp t → Exp (.scalar .f16)
  | toI32 {t : WGSLType} : Exp t → Exp (.scalar .i32)
  | toU32 {t : WGSLType} : Exp t → Exp (.scalar .u32)

  -- Array/vector indexing
  | index {elemTy : WGSLType} {n : Nat} : Exp (.array elemTy n) → Exp (.scalar .u32) → Exp elemTy

  /-- Byte-granularity load from a buffer declared as `array<u32, n>`.
      The buffer is addressed by *byte* index.  On CUDA this lowers to one
      `ld.global.u8` instruction (zero-extended into a u32 dest); on WGSL
      (where storage buffers lack byte granularity) it emulates the load
      via `(buf[byteIdx >> 2] >> ((byteIdx & 3) * 8)) & 0xFF`.

      Used by Q6_K matmul to avoid issuing one u32 load per scale byte. -/
  | loadByteFromU32Buf {n : Nat}
      : (bufName : String)
      → (byteIdx : Exp (.scalar .u32))
      → Exp (.scalar .u32)

  /-- Halfword (16-bit) load from a buffer declared as `array<u32, n>`.
      Lowers to one `ld.global.u16` on CUDA; emulated on WGSL via the same
      shift+mask pattern as `loadByteFromU32Buf`.  Used to read fp16 block
      scales from Q6_K blocks in a single load. -/
  | loadU16FromU32Buf {n : Nat}
      : (bufName : String)
      → (byteIdx : Exp (.scalar .u32))
      → Exp (.scalar .u32)

  /-- Two-level indexing into a `bufferArray elemTy n`: pick the `bufIdx`-th
      buffer, then read element `elemIdx` from it.  On CUDA this lowers to
      pointer-table load + `ld.global`.  On WGSL to `arr[bufIdx][elemIdx]`. -/
  | indexBuf {elemTy : WGSLType} {n : Nat}
      : Exp (.bufferArray elemTy n)
      → (bufIdx : Exp (.scalar .u32))
      → (elemIdx : Exp (.scalar .u32))
      → Exp elemTy

  -- Vector component access
  | vecX {st : ScalarType} : Exp (.vec2 st) → Exp (.scalar st)
  | vecY {st : ScalarType} : Exp (.vec2 st) → Exp (.scalar st)
  | vec3X {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vec3Y {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vecZ {st : ScalarType} : Exp (.vec3 st) → Exp (.scalar st)
  | vecW {st : ScalarType} : Exp (.vec4 st) → Exp (.scalar st)
  -- vec4 component access (.x/.y/.z; .w covered by vecW above).
  | vec4X {st : ScalarType} : Exp (.vec4 st) → Exp (.scalar st)
  | vec4Y {st : ScalarType} : Exp (.vec4 st) → Exp (.scalar st)
  | vec4Z {st : ScalarType} : Exp (.vec4 st) → Exp (.scalar st)

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

  /-- Mixed-precision multiply-accumulate: A and B are `inSt`, C/D are `outSt`.
      Real NVIDIA cooperativeMatrix configs typically use `(f16, f16) → f32`
      rather than the single-type variant above. WGSL + Dawn accept this
      form too, so we expose a separate constructor. -/
  | subgroupMatrixMultiplyAccumulateMixed
      {inSt outSt : ScalarType} {m k n : Nat} :
      Exp (.subgroupMatrixLeft inSt m k) →
      Exp (.subgroupMatrixRight inSt k n) →
      Exp (.subgroupMatrixResult outSt m n) →
      Exp (.subgroupMatrixResult outSt m n)

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

  /-- Packed half2 fused multiply-add: dst = a*b + c, where each operand
      holds two f16 values packed into one u32 (low half = lane 0, high = 1).
      CUDA backend lowers to a single `fma.rn.f16x2` PTX instruction.
      WGSL backend uses native vec2<f16> fma (requires `enable f16;`). -/
  | fmaF16x2 : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)
              → Exp (.scalar .u32)

  -- Data unpacking functions (extract from packed format)
  | unpack4x8snorm : Exp (.scalar .u32) → Exp (.vec4 .f32)
  | unpack4x8unorm : Exp (.scalar .u32) → Exp (.vec4 .f32)
  | unpack4xI8 : Exp (.scalar .u32) → Exp (.vec4 .i32)
  | unpack4xU8 : Exp (.scalar .u32) → Exp (.vec4 .u32)
  | unpack2x16snorm : Exp (.scalar .u32) → Exp (.vec2 .f32)
  | unpack2x16unorm : Exp (.scalar .u32) → Exp (.vec2 .f32)
  | unpack2x16float : Exp (.scalar .u32) → Exp (.vec2 .f32)

  -- Round-to-nearest-even float → signed i32, returned as u32 (same bits).
  -- Matches PTX cvt.rni.s32.f32 / C roundf. For negative values, the u32
  -- representation holds the two's-complement i32 bit pattern.
  | roundToI32 : Exp (.scalar .f32) → Exp (.scalar .u32)

  -- Packed 4x8 integer dot product (WGSL builtin, maps to dp4a on NVIDIA)
  -- dot4I8Packed(a, b): treat each u32 as 4 signed int8s, compute dot product → i32
  | dot4I8Packed : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .i32)
  -- dot4U8Packed(a, b): treat each u32 as 4 unsigned int8s, compute dot product → u32
  | dot4U8Packed : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)

  /-- Packed signed-saturating subtract per byte (CUDA `__vsubss4`).
      Each lane interprets the u32 as 4 int8 values and computes
      `clamp(a_byte - b_byte, -128, 127)` per byte. PTX lowers to
      a single `sub.sat.s8x4` instruction (sm_70+). On WGSL there is
      no native equivalent — it falls back to a per-byte sequence in
      the WGSL emitter (acceptable since WGSL is not the perf path
      for Q6_K). Used by Q6_K vec_dot to compute `(vil | vih) - 32`
      per byte without cross-byte borrow. -/
  | subSatS8x4 : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)

  -- Workgroup barrier (duplicate, already defined above - will be removed from toWGSL)
  | workgroupBarrier : Exp (.scalar .u32)  -- Returns unit

  /-- Warp-level barrier (CUDA `__syncwarp()`). On PTX backends lowers
      to `bar.warp.sync 0xFFFFFFFF` (much cheaper than block barrier).
      On WGSL backends falls back to `workgroupBarrier()` since WGSL
      lacks a dedicated warp-sync primitive — slightly over-syncs but
      preserves correctness. -/
  | warpBarrier : Exp (.scalar .u32)

  /-- Raw u64 pointer to element `idx` of a global buffer, **without
      dereferencing**. CUDA-only. Used as the global-address operand
      for `cpAsyncCgSharedGlobal`. `elemSize` is the byte size of
      one element (typically 4 for u32/f32 buffers). WGSL backend has
      no notion of raw pointer Exps — only used in the CUDA path. -/
  | bufferAddr : (bufName : String) → (elemSize : Nat) → Exp (.scalar .u32) → Exp (.scalar .u64)

  /-- Raw u32 byte-address of element `idx` in a shared-memory array.
      Lowers to `mov.u32 + add.u32` on CUDA. Used as the smem-address
      operand for `cpAsyncCgSharedGlobal`. `elemSize` is byte size of
      one element (typically 4 for u32 smem arrays). -/
  | sharedSymAddr : (smemName : String) → (elemSize : Nat) → Exp (.scalar .u32) → Exp (.scalar .u32)

  /-- ── cp.async (sm_80+) ── async global→shared copy.
      **`.cg` only supports `bytes = 16`.** Use `cpAsyncCaSharedGlobal`
      for 4- or 8-byte transfers. -/
  | cpAsyncCgSharedGlobal : Exp (.scalar .u32) → Exp (.scalar .u64) → Nat → Exp (.scalar .u32)
  /-- `cp.async.ca` cache-all variant. Supports `bytes ∈ {4, 8, 16}`. -/
  | cpAsyncCaSharedGlobal : Exp (.scalar .u32) → Exp (.scalar .u64) → Nat → Exp (.scalar .u32)
  /-- `cp.async.commit_group` — mark all preceding cp.async issues by
      this thread as one group. -/
  | cpAsyncCommitGroup : Exp (.scalar .u32)
  /-- `cp.async.wait_group N` — block until all but the most recent N
      committed groups have completed. `N=0` waits for all. -/
  | cpAsyncWaitGroup : Nat → Exp (.scalar .u32)

/-- Convert Float to WGSL literal string with full precision.
    Uses scientific notation (e.g. `1.0e-7`) when needed to preserve
    significant digits. FP32 has ~7 significant decimal digits. -/
def floatToWGSL (f : Float) : String :=
  if f == 0.0 then "0.0"
  else if f != f then "0.0 / 0.0"  -- NaN
  else
    let abs := if f < 0.0 then 0.0 - f else f
    let sign := if f < 0.0 then "-" else ""
    -- Always use scientific notation for full precision.
    -- FP32 has ~7 significant decimal digits.
    -- Format: sign + mantissa + "e" + exponent
    let log10 := Float.log abs / Float.log 10.0
    let exp := log10.floor
    let expInt := exp.toInt64.toInt
    let mantissa := abs / Float.pow 10.0 exp
    -- Scale mantissa to 7 significant digits
    let mScaled := (mantissa * 1000000.0).round.toUInt64
    let mStr := toString mScaled
    -- Pad to at least 7 digits
    let mStr := if mStr.length < 7 then
      String.mk (List.replicate (7 - mStr.length) '0') ++ mStr
    else mStr
    let mIntPart := mStr.take 1
    let mFracPart := mStr.drop 1
    -- Trim trailing zeros from fraction for cleaner output
    let mFracTrimmed := mFracPart.dropRightWhile (· == '0')
    let mFracFinal := if mFracTrimmed.isEmpty then "0" else mFracTrimmed
    s!"{sign}{mIntPart}.{mFracFinal}e{expInt}"

/-- Code generation: convert expression to WGSL string -/
partial def Exp.toWGSL {t : WGSLType} : Exp t → String
  | litF32 f => floatToWGSL f
  | litF16 f => s!"{floatToWGSL f}h"
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
  | mulhiU32 a b =>
    -- WGSL has no mulhi; emulate via 64-bit widening.  Host-side fastdiv
    -- precompute (mp, L) keeps the u32 range safe.
    s!"u32((u64({toWGSL a}) * u64({toWGSL b})) >> 32u)"
  | toF32 e => s!"f32({toWGSL e})"
  | toF32U e => s!"f32({toWGSL e})"
  | toF16 e => s!"f16({toWGSL e})"
  | toI32 e => s!"i32({toWGSL e})"
  | toU32 e => s!"u32({toWGSL e})"
  | index arr idx => s!"{toWGSL arr}[{toWGSL idx}]"
  | indexBuf arr bufIdx elemIdx => s!"{toWGSL arr}[{toWGSL bufIdx}][{toWGSL elemIdx}]"
  | loadByteFromU32Buf name byteIdx =>
    let bi := toWGSL byteIdx
    s!"((({name}[({bi} >> 2u)]) >> (({bi} & 3u) * 8u)) & 255u)"
  | loadU16FromU32Buf name byteIdx =>
    let bi := toWGSL byteIdx
    s!"((({name}[({bi} >> 2u)]) >> (({bi} & 3u) * 8u)) & 65535u)"
  | vecX v => s!"{toWGSL v}.x"
  | vecY v => s!"{toWGSL v}.y"
  | vec3X v => s!"{toWGSL v}.x"
  | vec3Y v => s!"{toWGSL v}.y"
  | vecZ v => s!"{toWGSL v}.z"
  | vecW v => s!"{toWGSL v}.w"
  | vec4X v => s!"{toWGSL v}.x"
  | vec4Y v => s!"{toWGSL v}.y"
  | vec4Z v => s!"{toWGSL v}.z"
  | @vec2 st x y => s!"vec2<{st.toWGSL}>({toWGSL x}, {toWGSL y})"
  | @vec3 st x y z => s!"vec3<{st.toWGSL}>({toWGSL x}, {toWGSL y}, {toWGSL z})"
  | @vec4 st x y z w => s!"vec4<{st.toWGSL}>({toWGSL x}, {toWGSL y}, {toWGSL z}, {toWGSL w})"
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
  | subgroupMatrixLoad (st:=_st) (m:=_m) (k:=k) ptr offset transposeFlag stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_left<{_st.toWGSL},{_m},{k}>>(&{ptr}, {toWGSL offset}, {toWGSL transposeFlag}, {toWGSL stride})"
  | subgroupMatrixLoadRight (st:=_st) (k:=k) (n:=_n) ptr offset transposeFlag stride =>
    s!"subgroupMatrixLoad<subgroup_matrix_right<{_st.toWGSL},{k},{_n}>>(&{ptr}, {toWGSL offset}, {toWGSL transposeFlag}, {toWGSL stride})"
  | subgroupMatrixMultiplyAccumulate a b acc =>
    s!"subgroupMatrixMultiplyAccumulate({toWGSL a}, {toWGSL b}, {toWGSL acc})"
  | subgroupMatrixMultiplyAccumulateMixed a b acc =>
    s!"subgroupMatrixMultiplyAccumulate({toWGSL a}, {toWGSL b}, {toWGSL acc})"
  | subgroupMatrixStore (st:=_st) (m:=_m) (n:=_n) ptr offset mat transposeFlag stride =>
    s!"subgroupMatrixStore(&{ptr}, {toWGSL offset}, {toWGSL mat}, {toWGSL transposeFlag}, {toWGSL stride})"
  | subgroupMatrixZeroLeft (st:=_st) (m:=_m) (k:=k) =>
    s!"subgroup_matrix_left<{_st.toWGSL}, {_m}, {k}>(0)"
  | subgroupMatrixZeroRight (st:=_st) (k:=k) (n:=_n) =>
    s!"subgroup_matrix_right<{_st.toWGSL}, {k}, {_n}>(0)"
  | subgroupMatrixZeroResult (st:=_st) (m:=_m) (n:=_n) =>
    s!"subgroup_matrix_result<{_st.toWGSL}, {_m}, {_n}>(0)"
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
  | fmaF16x2 a b c =>
    -- Native packed half2 fma via WGSL `enable f16;`: each u32 holds two
    -- f16 values, bitcast to vec2<f16>, fma, bitcast back.  Compiles to
    -- a single packed-fma on backends supporting fp16 (Vulkan
    -- VK_KHR_shader_float16_int8).
    s!"bitcast<u32>(fma(bitcast<vec2<f16>>({toWGSL a}), bitcast<vec2<f16>>({toWGSL b}), bitcast<vec2<f16>>({toWGSL c})))"
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
  | roundToI32 v =>
    -- WGSL: treat i32 as u32 via bitcast (same 32-bit pattern)
    s!"bitcast<u32>(i32(round({toWGSL v})))"
  | dot4I8Packed a b =>
    s!"dot4I8Packed({toWGSL a}, {toWGSL b})"
  | dot4U8Packed a b =>
    s!"dot4U8Packed({toWGSL a}, {toWGSL b})"
  | subSatS8x4 a b =>
    -- WGSL has no native sub.sat.s8x4. Emit a placeholder helper call;
    -- if you target Web you must supply your own polyfill. CUDA path
    -- bypasses this via expToPTX → sub_sat_s8x4 instruction.
    s!"subSatS8x4({toWGSL a}, {toWGSL b})"
  | workgroupBarrier =>
    "workgroupBarrier()"
  | warpBarrier =>
    -- WGSL has no warp-sync primitive; fall back to block barrier.
    "workgroupBarrier()"
  | bufferAddr name _ _ =>
    -- WGSL has no raw-pointer Exps; CUDA-only construct.
    s!"/* bufferAddr({name}) — CUDA-only */"
  | sharedSymAddr name _ _ =>
    s!"/* sharedSymAddr({name}) — CUDA-only */"
  | cpAsyncCgSharedGlobal _ _ _ =>
    -- WGSL has no async-copy primitive; CUDA-only construct.
    "/* cp.async.cg.shared.global — CUDA-only */"
  | cpAsyncCaSharedGlobal _ _ _ =>
    "/* cp.async.ca.shared.global — CUDA-only */"
  | cpAsyncCommitGroup =>
    "/* cp.async.commit_group — CUDA-only */"
  | cpAsyncWaitGroup _ =>
    "/* cp.async.wait_group — CUDA-only */"

/-! ## Operator overloading for ergonomic Exp construction

These instances let kernel code use `+`, `-`, `*`, `/`, `%`, `<`, `==`,
`&&&` (bit-and), `|||` (bit-or), `<<<` (shift-left), `>>>` (shift-right)
on `Exp ty` values directly, instead of `Exp.add`, `Exp.mul`, etc.
Closes one of the major cognitive gaps when porting kernels from CUDA C++:
`q + k * scale` reads the same in CUDA and ShaderM.

`Exp.add q k` style still works — the operator is sugar, not a replacement.
-/

instance instHAddExp {t : WGSLType} : HAdd (Exp t) (Exp t) (Exp t) where
  hAdd := Exp.add

instance instHSubExp {t : WGSLType} : HSub (Exp t) (Exp t) (Exp t) where
  hSub := Exp.sub

instance instHMulExp {t : WGSLType} : HMul (Exp t) (Exp t) (Exp t) where
  hMul := Exp.mul

instance instHDivExp {t : WGSLType} : HDiv (Exp t) (Exp t) (Exp t) where
  hDiv := Exp.div

instance instHModExp {t : WGSLType} : HMod (Exp t) (Exp t) (Exp t) where
  hMod := Exp.mod

/-- Numeric literals in Exp context: `(0 : Exp (.scalar .u32))` becomes
    `Exp.litU32 0`, `(0 : Exp (.scalar .i32))` becomes `Exp.litI32 0`.
    For f32 / f16 literals use `Exp.litF32 0.0` directly (Lean's OfScientific
    interaction with Float-typed Exp is fragile; explicit wrapper avoids
    surprises). -/
instance : OfNat (Exp (.scalar .u32)) n where
  ofNat := Exp.litU32 n

instance : OfNat (Exp (.scalar .i32)) n where
  ofNat := Exp.litI32 (Int.ofNat n)

/-- Bit ops: `&&&` = bitwise AND, `|||` = bitwise OR.
    Names match Lean's standard `&&&` / `|||` for u32/i32. -/
instance : AndOp (Exp (.scalar .u32)) where
  and := Exp.bitAnd

instance : OrOp (Exp (.scalar .u32)) where
  or := Exp.bitOr

/-- Shift operators using Lean's `HShiftLeft`/`HShiftRight` so `x <<< n`
    and `x >>> n` work uniformly with u32 expressions. -/
instance : HShiftLeft (Exp (.scalar .u32)) (Exp (.scalar .u32)) (Exp (.scalar .u32)) where
  hShiftLeft := Exp.shiftLeft

instance : HShiftRight (Exp (.scalar .u32)) (Exp (.scalar .u32)) (Exp (.scalar .u32)) where
  hShiftRight := Exp.shiftRight

/-- Mixed-arity arithmetic with Lean literals.  Allows
    `pos + 1` instead of `Exp.add pos (Exp.litU32 1)`,
    `2 * dPair` instead of `Exp.mul (Exp.litU32 2) dPair`,
    `x * 0.5` (f32) instead of `Exp.mul x (Exp.litF32 0.5)`, etc.
    Available for u32 + Nat (both directions) and f32 + Float (both
    directions).  Matches the Hesper convention that kernel arithmetic
    keeps the typed `Exp _` wrapper visible at every site so PTX lowering
    can pick the right instruction. -/

-- u32 ↔ Nat
instance : HAdd (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hAdd x n := Exp.add x (Exp.litU32 n)
instance : HAdd Nat (Exp (.scalar .u32)) (Exp (.scalar .u32)) where
  hAdd n x := Exp.add (Exp.litU32 n) x
instance : HSub (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hSub x n := Exp.sub x (Exp.litU32 n)
instance : HSub Nat (Exp (.scalar .u32)) (Exp (.scalar .u32)) where
  hSub n x := Exp.sub (Exp.litU32 n) x
instance : HMul (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hMul x n := Exp.mul x (Exp.litU32 n)
instance : HMul Nat (Exp (.scalar .u32)) (Exp (.scalar .u32)) where
  hMul n x := Exp.mul (Exp.litU32 n) x
instance : HDiv (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hDiv x n := Exp.div x (Exp.litU32 n)
instance : HMod (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hMod x n := Exp.mod x (Exp.litU32 n)

-- f32 ↔ Float
instance : HAdd (Exp (.scalar .f32)) Float (Exp (.scalar .f32)) where
  hAdd x f := Exp.add x (Exp.litF32 f)
instance : HAdd Float (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  hAdd f x := Exp.add (Exp.litF32 f) x
instance : HSub (Exp (.scalar .f32)) Float (Exp (.scalar .f32)) where
  hSub x f := Exp.sub x (Exp.litF32 f)
instance : HSub Float (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  hSub f x := Exp.sub (Exp.litF32 f) x
instance : HMul (Exp (.scalar .f32)) Float (Exp (.scalar .f32)) where
  hMul x f := Exp.mul x (Exp.litF32 f)
instance : HMul Float (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  hMul f x := Exp.mul (Exp.litF32 f) x
instance : HDiv (Exp (.scalar .f32)) Float (Exp (.scalar .f32)) where
  hDiv x f := Exp.div x (Exp.litF32 f)
instance : HDiv Float (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  hDiv f x := Exp.div (Exp.litF32 f) x

/-- Allow shifting by a Lean Nat literal: `x >>> 3` instead of
    `x >>> Exp.litU32 3`.  Same for `<<<`. -/
instance : HShiftLeft (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hShiftLeft x n := Exp.shiftLeft x (Exp.litU32 n)

instance : HShiftRight (Exp (.scalar .u32)) Nat (Exp (.scalar .u32)) where
  hShiftRight x n := Exp.shiftRight x (Exp.litU32 n)

/-! ## Sentinel constants (Step 9f)

Common literals that appear repeatedly in attention / softmax kernels.
Centralising them makes intent explicit and prevents typos like
`-1.0e30` vs `-3.4e38` differing across files.

Two negInf flavours are exposed:
- `negInf30` = `-1.0e30`. The hesper convention; matches existing V2-V11
  call sites. Drop-in replacement for the existing literal.
- `negInfHalf` = `-FLT_MAX / 2.0f` ≈ `-1.7e38`. llama.cpp's convention.
  The "/2" leaves head-room so `expf(x - max)` doesn't overflow when
  `max == -FLT_MAX`. Use when porting from llama.cpp to keep arithmetic
  bit-identical. -/
def Exp.f32Zero    : Exp (.scalar .f32) := Exp.litF32 0.0
def Exp.f32One     : Exp (.scalar .f32) := Exp.litF32 1.0
def Exp.negInf30   : Exp (.scalar .f32) := Exp.litF32 (-1.0e30)
def Exp.negInfHalf : Exp (.scalar .f32) := Exp.litF32 (-1.7014117e38)
/-- u32 literal helpers — for slot-index and lane-mask constants
    that show up at every call site. -/
def Exp.u32Zero    : Exp (.scalar .u32) := Exp.litU32 0
def Exp.u32One     : Exp (.scalar .u32) := Exp.litU32 1

/-- Comparison operators on `Exp`. Cannot reuse Lean's `<` / `==` because
    those resolve to `Bool`, not `Exp (.scalar .bool)`. Unicode suffix
    `ᵉ` ("e" for Exp) keeps the operator visually similar to CUDA C++
    while signalling that the result is an Exp Bool, not a Lean Bool.

    Usage: `kPos <ᵉ splitEnd` instead of `Exp.lt kPos splitEnd`. -/
infixl:50 " <ᵉ "  => Exp.lt
infixl:50 " ≤ᵉ "  => Exp.le
infixl:50 " >ᵉ "  => Exp.gt
infixl:50 " ≥ᵉ "  => Exp.ge
infixl:50 " ==ᵉ " => Exp.eq
infixl:50 " !=ᵉ " => Exp.ne

end Hesper.WGSL
