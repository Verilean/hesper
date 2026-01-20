import Hesper

/-!
# Matrix Multiplication on GPU using Subgroup Operations

This executable actually runs matrix multiplication on the GPU using
the chromium_experimental_subgroup_matrix extension.

Uses the WGSL DSL to generate type-safe shader code.
-/

namespace Hesper

/-- Create device with subgroup matrix features -/
@[extern "lean_hesper_get_device_with_features"]
opaque getDeviceWithFeatures (inst : @& WebGPU.Instance) : IO WebGPU.Device

/-- Run matrix multiplication with subgroup operations -/
@[extern "lean_hesper_matmul_subgroup"]
opaque matmulSubgroup (device : @& WebGPU.Device) (shaderCode : String) (m k n : UInt32) : IO Unit

end Hesper

open Hesper.WGSL

/-- Generate subgroup matrix multiplication shader using IMPROVED DSL -/
def generateMatmulShaderDSL
    (st : ScalarType)
    (m k n : Nat)
    (tm tn : Nat)
    (lid0 lid1 : Nat)
    : ComputeShader :=
  let scalarTy := WGSLType.scalar st
  let leftMatTy := WGSLType.subgroupMatrixLeft st 8 8
  let rightMatTy := WGSLType.subgroupMatrixRight st 8 8
  let resultMatTy := WGSLType.subgroupMatrixResult st 8 8

  -- Helper variables with explicit types
  let wgX : Exp (.scalar .u32) := "wg.x"
  let wgY : Exp (.scalar .u32) := "wg.y"
  let localIDY : Exp (.scalar .u32) := "localID.y"
  let baseA : Exp (.scalar .u32) := "baseA"
  let baseB : Exp (.scalar .u32) := "baseB"
  let cBase : Exp (.scalar .u32) := "cBase"

  -- Calculate base indices using coercions
  let rowStart := wgX * (8 : Nat) * (tm : Nat)
  let colStart := (wgY * (lid1 : Nat) + localIDY) * (8 : Nat) * (tn : Nat)

  -- Main shader body using IMPROVED DSL constructs
  let body : List Stmt := [
    declareVar "rowStart" (.scalar .u32) (some ⟨_, rowStart⟩),
    declareVar "colStart" (.scalar .u32) (some ⟨_, colStart⟩),
    declareVar "baseA" (.scalar .u32) (some ⟨_, rowStart * ((k : Nat) : Exp (.scalar .u32))⟩),
    declareVar "baseB" (.scalar .u32) (some ⟨_, colStart⟩),
    declareVar "cBase" (.scalar .u32) (some ⟨_, rowStart * ((n : Nat) : Exp (.scalar .u32)) + colStart⟩),

    -- Declare matrix arrays
    declareVar "Ax" (.array leftMatTy tm),
    declareVar "Bx" (.array rightMatTy tn),
    declareVar "accxx" (.array resultMatTy (tm * tn))
  ] ++
  -- Initialize matrices using smart constructor
  initArray "Ax" tm (fun _ => matZeroLeft (st:=st) (m:=8) (k:=8)) ++
  initArray "Bx" tn (fun _ => matZeroRight (st:=st) (k:=8) (n:=8)) ++
  initArray "accxx" (tm * tn) (fun _ => matZeroResult (st:=st) (m:=8) (n:=8)) ++
  [
    -- Main compute loop using HOAS-style loop
    loop "kk" 0 (k : Nat) 8 (fun kk =>
      [expr barrier] ++
      -- Load A tiles using helper
      loadMatricesLeft (st:=st) (m:=8) (k:=8) "Ax" tm "&A"
        (fun i => baseA + kk + ((8 * k * i) : Nat))
        ((k : Nat) : Exp (.scalar .u32)) ++
      -- Load B tiles using helper
      loadMatricesRight (st:=st) (k:=8) (n:=8) "Bx" tn "&B"
        (fun i => baseB + kk * ((n : Nat) : Exp (.scalar .u32)) + ((8 * i) : Nat))
        ((n : Nat) : Exp (.scalar .u32)) ++
      -- Multiply-accumulate using helper
      matrixMulAccGrid (st:=st) (m:=8) (k:=8) (n:=8) (tm:=tm) (tn:=tn)
        "Ax" "Bx" "accxx"
    ),
    -- Final barrier before store
    expr barrier
  ] ++
  -- Store results using helper
  storeMatricesResult (st:=st) (m:=8) (n:=8) (tm:=tm) (tn:=tn)
    "accxx" "&C"
    (fun i j => cBase + ((i * 8 * n + 8 * j) : Nat))
    ((n : Nat) : Exp (.scalar .u32))

  {
    extensions := ["chromium_experimental_subgroup_matrix"] ++
      (if st == ScalarType.f16 then ["f16"] else []),
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")],
    buffers := [
      { group := 0, binding := 0, name := "A", elemType := scalarTy, readWrite := true },
      { group := 0, binding := 1, name := "B", elemType := scalarTy, readWrite := true },
      { group := 0, binding := 2, name := "C", elemType := scalarTy, readWrite := true }
    ],
    workgroupSize := { x := lid0, y := lid1, z := 1 },
    builtins := [
      { builtin := BuiltinBinding.workgroupId, name := "wg" },
      { builtin := BuiltinBinding.localInvocationId, name := "localID" }
    ],
    body := body
  }

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper GPU Matrix Multiplication           ║"
  IO.println "║   (Subgroup Matrix Operations)               ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Small test for now
  let m := 128
  let k := 128
  let n := 128

  IO.println s!"Matrix dimensions: {m}x{k} × {n}x{k}"
  IO.println s!"Total operations: {2 * m * n * k}"
  IO.println ""

  -- Generate shader using DSL
  let shaderDSL := generateMatmulShaderDSL ScalarType.f32 m k n 4 2 32 2
  let shaderCode := shaderDSL.toWGSL

  IO.println "Generated WGSL shader using DSL (first 500 chars):"
  IO.println (shaderCode.take 500 ++ "...")
  IO.println ""

  -- Initialize WebGPU with subgroup features
  IO.println "Initializing WebGPU..."
  let inst ← Hesper.init
  IO.println ""

  IO.println "Creating device with subgroup matrix support..."
  let device ← Hesper.getDeviceWithFeatures inst
  IO.println ""

  IO.println "Running matrix multiplication on GPU..."
  Hesper.matmulSubgroup device shaderCode m.toUInt32 k.toUInt32 n.toUInt32
  IO.println ""

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Matrix multiplication complete!            ║"
  IO.println "╚══════════════════════════════════════════════╝"
