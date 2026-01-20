import Hesper

/-!
# 4K Matrix Multiplication Benchmark with FLOPS Calculation

Runs 4096x4096 matrix multiplication on GPU using subgroup operations
and calculates actual GFLOPS performance.
-/

namespace Hesper

@[extern "lean_hesper_get_device_with_features"]
opaque getDeviceWithFeatures (inst : @& WebGPU.Instance) : IO WebGPU.Device

@[extern "lean_hesper_matmul_subgroup_4k"]
opaque matmulSubgroup4K (device : @& WebGPU.Device) (shaderCode : String) : IO Unit

end Hesper

-- Generate the WGSL shader for 4K matmul (same as Examples.MatmulSubgroup but for 4K)
def generateSubgroupMatmulShader
    (precision : String)
    (m k n : Nat)
    (tm tn : Nat)
    (lid0 lid1 : Nat)
    : String :=
  let enableF16 := if precision == "f16" then "enable f16;\n\n" else ""
  let header := enableF16 ++
    "enable chromium_experimental_subgroup_matrix;\n" ++
    "diagnostic(off, chromium.subgroup_matrix_uniformity);\n\n" ++
    s!"@group(0) @binding(0) var<storage, read_write> A: array<{precision}>;\n" ++
    s!"@group(0) @binding(1) var<storage, read_write> B: array<{precision}>;\n" ++
    s!"@group(0) @binding(2) var<storage, read_write> C: array<{precision}>;\n\n" ++
    s!"@compute @workgroup_size({lid0}, {lid1}, 1)\n" ++
    "fn main(@builtin(workgroup_id) wg: vec3<u32>,\n" ++
    "        @builtin(local_invocation_id) localID: vec3<u32>) {\n\n"

  let body :=
    s!"  let rowStart: u32 = wg.x * 8u * {tm}u;\n" ++
    s!"  let colStart: u32 = (wg.y * {lid1}u + localID.y) * 8u * {tn}u;\n\n" ++
    s!"  let baseA: u32 = rowStart * {k}u;\n" ++
    s!"  let baseB: u32 = colStart;\n" ++
    s!"  let cBase: u32 = rowStart * {n}u + colStart;\n\n" ++
    s!"  var Ax: array<subgroup_matrix_left<{precision}, 8, 8>, {tm}>;\n" ++
    s!"  var Bx: array<subgroup_matrix_right<{precision}, 8, 8>, {tn}>;\n" ++
    s!"  var accxx: array<subgroup_matrix_result<{precision}, 8, 8>, {tm * tn}>;\n\n"

  let initAx := String.intercalate "" ((List.range tm).map fun i =>
    s!"  Ax[{i}] = subgroup_matrix_left<{precision}, 8, 8>(0);\n")
  let initBx := String.intercalate "" ((List.range tn).map fun i =>
    s!"  Bx[{i}] = subgroup_matrix_right<{precision}, 8, 8>(0);\n")
  let initAcc := (List.range tm).foldl (fun acc i =>
    acc ++ (List.range tn).foldl (fun acc2 j =>
      acc2 ++ s!"  accxx[{i}+{j}*{tm}] = subgroup_matrix_result<{precision}, 8, 8>(0);\n") "") ""

  let loopHeader := s!"\n  for (var k: u32 = 0u; k < {k}u; k = k + 8u) " ++ "{\n" ++
    "    workgroupBarrier();\n"

  let loadAx := String.intercalate "" ((List.range tm).map fun i =>
    s!"    Ax[{i}] = subgroupMatrixLoad<subgroup_matrix_left<{precision},8,8>>(&A, baseA + k + 8u * {k}u * {i}u, false, {k}u);\n")
  let loadBx := String.intercalate "" ((List.range tn).map fun i =>
    s!"    Bx[{i}] = subgroupMatrixLoad<subgroup_matrix_right<{precision},8,8>>(&B, baseB + k * {n}u + 8u * {i}u, false, {n}u);\n")

  let multiplyAcc := (List.range tn).foldl (fun acc j =>
    acc ++ (List.range tm).foldl (fun acc2 i =>
      acc2 ++ s!"    accxx[{j}*{tm} + {i}] = subgroupMatrixMultiplyAccumulate(Ax[{i}], Bx[{j}], accxx[{j}*{tm} + {i}]);\n") "") ""

  let loopFooter := "  }\n\n  workgroupBarrier();\n\n"

  let storeResults := (List.range tm).foldl (fun acc i =>
    acc ++ (List.range tn).foldl (fun acc2 j =>
      acc2 ++ s!"  subgroupMatrixStore(&C, cBase + {i}u * 8u * {n}u + 8u * {j}u, accxx[{j}*{tm} + {i}], false, {n}u);\n") "") ""

  let footer := "}\n"

  header ++ body ++ initAx ++ "\n" ++ initBx ++ "\n" ++ initAcc ++ loopHeader ++
  loadAx ++ "\n" ++ loadBx ++ "\n" ++ multiplyAcc ++ loopFooter ++ storeResults ++ footer

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper 4K Matrix Multiplication            ║"
  IO.println "║   Subgroup Matrix Operations Benchmark       ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  let m := 4096
  let k := 4096
  let n := 4096

  -- Calculate total operations
  let totalOps := 2 * m * n * k
  IO.println s!"Matrix dimensions: {m}×{k} × {n}×{k}"
  IO.println s!"Total operations: {totalOps} ({totalOps / 1000000000} billion)"
  IO.println ""

  -- Generate shader
  let shaderCode := generateSubgroupMatmulShader "f32" m k n 4 8 32 2
  IO.println "Generated WGSL shader (first 300 chars):"
  IO.println (shaderCode.take 300 ++ "...")
  IO.println ""

  -- Initialize WebGPU
  IO.println "Initializing WebGPU with subgroup features..."
  let inst ← Hesper.init
  let device ← Hesper.getDeviceWithFeatures inst
  IO.println ""

  -- Run benchmark
  IO.println "Running 4K matrix multiplication on GPU..."
  IO.println "This will use hardware tensor cores via Metal..."
  IO.println ""

  Hesper.matmulSubgroup4K device shaderCode

  IO.println ""
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Benchmark complete!                        ║"
  IO.println "╚══════════════════════════════════════════════╝"
