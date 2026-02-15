namespace Hesper.WGSL

/-! WGSL type system using phantom types for type safety.
    These types are used at the Lean level to ensure type correctness,
    but compile down to WGSL primitive types. -/

/-- WGSL scalar types -/
inductive ScalarType where
  | f32    -- 32-bit floating point
  | f16    -- 16-bit floating point (requires f16 extension)
  | i32    -- 32-bit signed integer
  | u32    -- 32-bit unsigned integer
  | bool   -- Boolean type
  | atomicI32  -- Atomic 32-bit signed integer (for atomic operations)
  | atomicU32  -- Atomic 32-bit unsigned integer (for atomic operations)
  deriving Repr, BEq, Inhabited

/-- Memory address spaces in WGSL -/
inductive MemorySpace where
  | storage    -- Global GPU memory (large, slower)
  | uniform    -- Read-only uniform data
  | workgroup  -- Shared memory between workgroup threads (fast)
  | priv       -- Thread-local memory (registers) - "private" in WGSL
  | func       -- Function-local memory - "function" in WGSL
  deriving Repr, BEq, Inhabited

/-- WGSL composite types -/
inductive WGSLType where
  | scalar : ScalarType → WGSLType
  | vec2 : ScalarType → WGSLType
  | vec3 : ScalarType → WGSLType
  | vec4 : ScalarType → WGSLType
  | mat2x2 : ScalarType → WGSLType
  | mat3x3 : ScalarType → WGSLType
  | mat4x4 : ScalarType → WGSLType
  | array : WGSLType → Nat → WGSLType
  | runtimeArray : WGSLType → WGSLType  -- Runtime-sized array (no size specified)
  | ptr : MemorySpace → WGSLType → WGSLType
  | struct : String → WGSLType  -- Reference to a struct by name
  -- Subgroup matrix types (chromium_experimental_subgroup_matrix extension)
  | subgroupMatrixLeft : ScalarType → Nat → Nat → WGSLType    -- subgroup_matrix_left<T, M, K>
  | subgroupMatrixRight : ScalarType → Nat → Nat → WGSLType   -- subgroup_matrix_right<T, K, N>
  | subgroupMatrixResult : ScalarType → Nat → Nat → WGSLType  -- subgroup_matrix_result<T, M, N>
  -- Texture types (for image processing)
  | texture2D : String → WGSLType  -- texture_2d<format>
  | sampler : WGSLType  -- Texture sampler
  deriving Repr, BEq

/-- Struct field definition -/
structure StructField where
  name : String
  type : WGSLType
  deriving Repr, BEq

/-- Struct definition -/
structure StructDef where
  name : String
  fields : List StructField
  deriving Repr, BEq

/-- Buffer usage flags (corresponds to WebGPU buffer usage) -/
inductive BufferUsage where
  | storage
  | uniform
  | copy_src
  | copy_dst
  deriving Repr, BEq

/-- Layout rules for struct alignment -/
inductive LayoutRule where
  | std140  -- Uniform buffer layout (more padding)
  | std430  -- Storage buffer layout (compact)
  deriving Repr, BEq, Inhabited

/-- Binding layout information for type-safe pipeline creation -/
structure BindingInfo where
  group : Nat
  binding : Nat
  type : WGSLType
  usage : BufferUsage

/-- Shader layout - collection of bindings with type information -/
structure Layout where
  bindings : List BindingInfo
  deriving Inhabited

/-- Convert scalar type to WGSL string -/
def ScalarType.toWGSL : ScalarType → String
  | .f32 => "f32"
  | .f16 => "f16"
  | .i32 => "i32"
  | .u32 => "u32"
  | .bool => "bool"
  | .atomicI32 => "atomic<i32>"
  | .atomicU32 => "atomic<u32>"

/-- Convert memory space to WGSL string -/
def MemorySpace.toWGSL : MemorySpace → String
  | .storage => "storage"
  | .uniform => "uniform"
  | .workgroup => "workgroup"
  | .priv => "private"
  | .func => "function"

/-- Convert WGSL type to WGSL string -/
def WGSLType.toWGSL : WGSLType → String
  | .scalar st => st.toWGSL
  | .vec2 st => s!"vec2<{st.toWGSL}>"
  | .vec3 st => s!"vec3<{st.toWGSL}>"
  | .vec4 st => s!"vec4<{st.toWGSL}>"
  | .mat2x2 st => s!"mat2x2<{st.toWGSL}>"
  | .mat3x3 st => s!"mat3x3<{st.toWGSL}>"
  | .mat4x4 st => s!"mat4x4<{st.toWGSL}>"
  | .array elemTy n => s!"array<{elemTy.toWGSL}, {n}>"
  | .runtimeArray elemTy => s!"array<{elemTy.toWGSL}>"  -- Runtime-sized array (no size)
  | .ptr space ty => s!"ptr<{space.toWGSL}, {ty.toWGSL}>"
  | .struct name => name  -- Just the struct name
  | .subgroupMatrixLeft st m k => s!"subgroup_matrix_left<{st.toWGSL}, {m}, {k}>"
  | .subgroupMatrixRight st k n => s!"subgroup_matrix_right<{st.toWGSL}, {k}, {n}>"
  | .subgroupMatrixResult st m n => s!"subgroup_matrix_result<{st.toWGSL}, {m}, {n}>"
  | .texture2D format => s!"texture_2d<{format}>"
  | .sampler => "sampler"

/-- Byte size of a scalar type -/
def ScalarType.byteSize : ScalarType → Nat
  | .f32 => 4
  | .f16 => 2
  | .i32 => 4
  | .u32 => 4
  | .bool => 4  -- bools are 4 bytes in WGSL
  | .atomicI32 => 4  -- atomic<i32> is 4 bytes
  | .atomicU32 => 4  -- atomic<u32> is 4 bytes

/-- Calculate byte size of a type (for buffer allocation) -/
def WGSLType.byteSize : WGSLType → Nat
  | .scalar st => st.byteSize
  | .vec2 st => 2 * st.byteSize
  | .vec3 st => 3 * st.byteSize
  | .vec4 st => 4 * st.byteSize
  | .mat2x2 st => 4 * st.byteSize
  | .mat3x3 st => 9 * st.byteSize
  | .mat4x4 st => 16 * st.byteSize
  | .array elemTy n => n * elemTy.byteSize
  | .runtimeArray _ => 0  -- Size determined at runtime by buffer size
  | .ptr _ _ => 8  -- Pointers are conceptual, but 8 for compatibility
  | .struct _ => 0  -- Size depends on actual struct definition (to be calculated)
  | .subgroupMatrixLeft st m k => m * k * st.byteSize  -- Approximate
  | .subgroupMatrixRight st k n => k * n * st.byteSize
  | .subgroupMatrixResult st m n => m * n * st.byteSize
  | .texture2D _ => 0  -- Textures are opaque handles
  | .sampler => 0  -- Samplers are opaque handles

/-- Alignment of a scalar type -/
def ScalarType.alignment : ScalarType → Nat
  | .f32 => 4
  | .f16 => 2
  | .i32 => 4
  | .u32 => 4
  | .bool => 4
  | .atomicI32 => 4
  | .atomicU32 => 4

/-- Calculate alignment of a type according to layout rules -/
def WGSLType.alignment (rule : LayoutRule) : WGSLType → Nat
  | .scalar st => st.alignment
  | .vec2 st => 2 * st.alignment
  | .vec3 st =>
    match rule with
    | .std140 => 16  -- vec3 aligns to 16 in std140
    | .std430 => 4 * st.alignment
  | .vec4 st => 4 * st.alignment
  | .mat2x2 _ => 16  -- Matrices align to 16
  | .mat3x3 _ => 16
  | .mat4x4 _ => 16
  | .array elemTy _ => elemTy.alignment rule
  | .runtimeArray elemTy => elemTy.alignment rule  -- Same alignment as fixed array
  | .ptr _ _ => 8
  | .struct _ => 16  -- Structs align to 16 bytes (max alignment of members)
  | .subgroupMatrixLeft _ _ _ => 16  -- Subgroup matrices are opaque, assume 16-byte alignment
  | .subgroupMatrixRight _ _ _ => 16
  | .subgroupMatrixResult _ _ _ => 16
  | .texture2D _ => 0  -- Textures don't have alignment (opaque handles)
  | .sampler => 0  -- Samplers don't have alignment (opaque handles)

end Hesper.WGSL
