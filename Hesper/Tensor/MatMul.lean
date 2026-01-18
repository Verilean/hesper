import Hesper.Tensor.Types

/-!
# Matrix Multiplication Kernels

GPU kernels for matrix multiplication - simplified template-based generation.
-/

namespace Hesper.Tensor.MatMul

open Hesper.Tensor

/-- Generate WGSL source code for naive matrix multiplication

    ⚠️  DEPRECATED: String-based WGSL generation is for debugging only.
    Prefer using ShaderM monad for type-safe shader construction.
    This function will be removed in future versions.
-/
@[deprecated "Use ShaderM monad instead of string generation"]
def generateNaiveMatMulShader (config : MatMulConfig) : String :=
  let M := toString config.M
  let K := toString config.K
  let N := toString config.N
  let ts := toString config.tileSize
  "// Naive Matrix Multiplication Shader\n" ++
  "// Matrix A: " ++ M ++ " × " ++ K ++ "\n" ++
  "// Matrix B: " ++ K ++ " × " ++ N ++ "\n" ++
  "// Matrix C: " ++ M ++ " × " ++ N ++ "\n\n" ++
  "@group(0) @binding(0) var<storage, read_write> matrixA: array<f32>;\n" ++
  "@group(0) @binding(1) var<storage, read_write> matrixB: array<f32>;\n" ++
  "@group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;\n\n" ++
  "@compute @workgroup_size(" ++ ts ++ ", " ++ ts ++ ", 1)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let row = gid.y;\n" ++
  "  let col = gid.x;\n\n" ++
  "  if (row >= " ++ M ++ "u || col >= " ++ N ++ "u) {\n" ++
  "    return;\n" ++
  "  }\n\n" ++
  "  var sum: f32 = 0.0;\n\n" ++
  "  for (var k: u32 = 0u; k < " ++ K ++ "u; k = k + 1u) {\n" ++
  "    let aIdx = row * " ++ K ++ "u + k;\n" ++
  "    let bIdx = k * " ++ N ++ "u + col;\n" ++
  "    sum = sum + matrixA[aIdx] * matrixB[bIdx];\n" ++
  "  }\n\n" ++
  "  let cIdx = row * " ++ N ++ "u + col;\n" ++
  "  matrixC[cIdx] = sum;\n" ++
  "}"

/-- Generate WGSL source code for tiled matrix multiplication

    ⚠️  DEPRECATED: String-based WGSL generation is for debugging only.
    Prefer using ShaderM monad for type-safe shader construction.
    This function will be removed in future versions.
-/
@[deprecated "Use ShaderM monad instead of string generation"]
def generateTiledMatMulShader (config : MatMulConfig) : String :=
  let M := toString config.M
  let K := toString config.K
  let N := toString config.N
  let ts := toString config.tileSize
  let tsSquared := toString (config.tileSize * config.tileSize)
  "// Tiled Matrix Multiplication Shader\n" ++
  "// Matrix A: " ++ M ++ " × " ++ K ++ "\n" ++
  "// Matrix B: " ++ K ++ " × " ++ N ++ "\n" ++
  "// Matrix C: " ++ M ++ " × " ++ N ++ "\n" ++
  "// Tile Size: " ++ ts ++ " × " ++ ts ++ "\n\n" ++
  "@group(0) @binding(0) var<storage, read_write> matrixA: array<f32>;\n" ++
  "@group(0) @binding(1) var<storage, read_write> matrixB: array<f32>;\n" ++
  "@group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;\n\n" ++
  "var<workgroup> tileA: array<f32, " ++ tsSquared ++ ">;\n" ++
  "var<workgroup> tileB: array<f32, " ++ tsSquared ++ ">;\n\n" ++
  "@compute @workgroup_size(" ++ ts ++ ", " ++ ts ++ ", 1)\n" ++
  "fn main(\n" ++
  "  @builtin(global_invocation_id) gid: vec3<u32>,\n" ++
  "  @builtin(local_invocation_id) lid: vec3<u32>\n" ++
  ") {\n" ++
  "  let globalRow = gid.y;\n" ++
  "  let globalCol = gid.x;\n" ++
  "  let localRow = lid.y;\n" ++
  "  let localCol = lid.x;\n\n" ++
  "  var sum: f32 = 0.0;\n" ++
  "  let numTiles = (" ++ K ++ "u + " ++ ts ++ "u - 1u) / " ++ ts ++ "u;\n\n" ++
  "  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {\n" ++
  "    // Load tile from A\n" ++
  "    let aCol = t * " ++ ts ++ "u + localCol;\n" ++
  "    let aRow = globalRow;\n" ++
  "    if (aRow < " ++ M ++ "u && aCol < " ++ K ++ "u) {\n" ++
  "      let aIdx = aRow * " ++ K ++ "u + aCol;\n" ++
  "      tileA[localRow * " ++ ts ++ "u + localCol] = matrixA[aIdx];\n" ++
  "    } else {\n" ++
  "      tileA[localRow * " ++ ts ++ "u + localCol] = 0.0;\n" ++
  "    }\n\n" ++
  "    // Load tile from B\n" ++
  "    let bRow = t * " ++ ts ++ "u + localRow;\n" ++
  "    let bCol = globalCol;\n" ++
  "    if (bRow < " ++ K ++ "u && bCol < " ++ N ++ "u) {\n" ++
  "      let bIdx = bRow * " ++ N ++ "u + bCol;\n" ++
  "      tileB[localRow * " ++ ts ++ "u + localCol] = matrixB[bIdx];\n" ++
  "    } else {\n" ++
  "      tileB[localRow * " ++ ts ++ "u + localCol] = 0.0;\n" ++
  "    }\n\n" ++
  "    workgroupBarrier();\n\n" ++
  "    // Compute partial dot product\n" ++
  "    for (var k: u32 = 0u; k < " ++ ts ++ "u; k = k + 1u) {\n" ++
  "      sum = sum + tileA[localRow * " ++ ts ++ "u + k] * tileB[k * " ++ ts ++ "u + localCol];\n" ++
  "    }\n\n" ++
  "    workgroupBarrier();\n" ++
  "  }\n\n" ++
  "  // Store result\n" ++
  "  if (globalRow < " ++ M ++ "u && globalCol < " ++ N ++ "u) {\n" ++
  "    let cIdx = globalRow * " ++ N ++ "u + globalCol;\n" ++
  "    matrixC[cIdx] = sum;\n" ++
  "  }\n" ++
  "}"

end Hesper.Tensor.MatMul
