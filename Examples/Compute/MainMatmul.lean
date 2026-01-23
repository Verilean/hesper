import Hesper
import Hesper.Compute
import Hesper.WGSL.Execute

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL
open Hesper.WGSL.Execute

/-- The original shader generation from the file, preserved for functional identicality -/
def generateMatmulShader
    (st : ScalarType)
    (m k n : Nat)
    (tm tn : Nat)
    (lid0 lid1 : Nat)
    : ComputeShader :=
  let scalarTy := WGSLType.scalar st
  let leftMatTy := WGSLType.subgroupMatrixLeft st 8 8
  let rightMatTy := WGSLType.subgroupMatrixRight st 8 8
  let resultMatTy := WGSLType.subgroupMatrixResult st 8 8

  let wgX : Exp (.scalar .u32) := "wg.x"
  let wgY : Exp (.scalar .u32) := "wg.y"
  let localIDY : Exp (.scalar .u32) := "localID.y"

  let rowStart := wgX * (8 : Nat) * (tm : Nat)
  let colStart := (wgY * (lid1 : Nat) + localIDY) * (8 : Nat) * (tn : Nat)

  let rowStartVar : Exp (.scalar .u32) := Exp.var "rowStart"
  let colStartVar : Exp (.scalar .u32) := Exp.var "colStart"

  let body : List Stmt := [
    declareVar "rowStart" (.scalar .u32) (some ⟨_, rowStart⟩),
    declareVar "colStart" (.scalar .u32) (some ⟨_, colStart⟩),
    declareVar "baseA" (.scalar .u32) (some ⟨_, rowStartVar * ((k : Nat) : Exp (.scalar .u32))⟩),
    declareVar "baseB" (.scalar .u32) (some ⟨_, colStartVar⟩),
    declareVar "cBase" (.scalar .u32) (some ⟨_, rowStartVar * ((n : Nat) : Exp (.scalar .u32)) + colStartVar⟩),

    declareVar "Ax" (.array leftMatTy tm),
    declareVar "Bx" (.array rightMatTy tn),
    declareVar "accxx" (.array resultMatTy (tm * tn))
  ] ++
  initArray "Ax" tm (fun _ => matZeroLeft (st:=st) (m:=8) (k:=8)) ++
  initArray "Bx" tn (fun _ => matZeroRight (st:=st) (k:=8) (n:=8)) ++
  initArray "accxx" (tm * tn) (fun _ => matZeroResult (st:=st) (m:=8) (n:=8)) ++
  [
    loop "kk" 0 (k : Nat) 8 (fun kk =>
      [expr barrier] ++
      loadMatricesLeft (st:=st) (m:=8) (k:=8) "Ax" tm "A"
        (fun i => (Exp.var "baseA" : Exp (.scalar .u32)) + kk + ((8 * k * i) : Nat))
        ((k : Nat) : Exp (.scalar .u32)) ++
      loadMatricesRight (st:=st) (k:=8) (n:=8) "Bx" tn "B"
        (fun i => (Exp.var "baseB" : Exp (.scalar .u32)) + kk * ((n : Nat) : Exp (.scalar .u32)) + ((8 * i) : Nat))
        ((n : Nat) : Exp (.scalar .u32)) ++
      matrixMulAccGrid (st:=st) (m:=8) (k:=8) (n:=8) (tm:=tm) (tn:=tn)
        "Ax" "Bx" "accxx"
    ),
    expr barrier
  ] ++
  storeMatricesResult (st:=st) (m:=8) (n:=8) (tm:=tm) (tn:=tn)
    "accxx" "C"
    (fun i j => (Exp.var "cBase" : Exp (.scalar .u32)) + ((i * 8 * n + 8 * j) : Nat))
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

  -- Matrix dimensions
  let m := 128
  let k := 128
  let n := 128

  IO.println s!"Matrix dimensions: {m}x{k} × {n}x{k}"
  IO.println s!"Total operations: {2 * m * n * k}"
  IO.println ""

  -- Initialize WebGPU with subgroup features
  IO.println "Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← getDeviceWithFeatures inst

  -- Create buffers
  IO.println "Creating GPU buffers..."
  let aBuf ← createBuffer device {
    size := (m * k * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let bBuf ← createBuffer device {
    size := (k * n * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let cBuf ← createBuffer device {
    size := (m * n * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Initialize matrix data with non-uniform values to verify correctness
  -- A[i,j] = i + 1, B[i,j] = j + 1 (simple pattern to verify computation)
  let aData ← Hesper.Basic.floatArrayToBytes (Array.range (m * k) |>.map fun idx =>
    let i := idx / k
    (i + 1).toFloat)
  let bData ← Hesper.Basic.floatArrayToBytes (Array.range (k * n) |>.map fun idx =>
    let j := idx % n
    (j + 1).toFloat)
  writeBuffer device aBuf 0 aData
  writeBuffer device bBuf 0 bData

  -- Generate shader
  let lid0 := 32
  let lid1 := 2
  let tm := 4
  let tn := 2
  let shaderDSL := generateMatmulShader .f32 m k n tm tn lid0 lid1
  let shaderCode := shaderDSL.toWGSL

  IO.println "Generated WGSL shader (first 300 chars):"
  IO.println (shaderCode.take 300 ++ "...")
  IO.println ""

  -- Create pipeline and bind group
  let shaderModule ← createShaderModule device shaderCode
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry },
    { binding := 1, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry },
    { binding := 2, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let bindEntries := #[
    { binding := 0, buffer := aBuf, offset := 0, size := (m * k * 4).toUSize : BindGroupEntry },
    { binding := 1, buffer := bBuf, offset := 0, size := (k * n * 4).toUSize : BindGroupEntry },
    { binding := 2, buffer := cBuf, offset := 0, size := (m * n * 4).toUSize : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  -- Dispatch compute
  IO.println "Running matrix multiplication on GPU..."
  let numWorkgroupsX := (m + 31) / 32
  let numWorkgroupsY := (n + 63) / 64
  dispatchCompute device pipeline bindGroup numWorkgroupsX.toUInt32 numWorkgroupsY.toUInt32 1
  deviceWait device

  -- Read back and verify multiple samples
  IO.println "Reading back results..."
  let resultBytes ← mapBufferRead device cBuf 0 (m * n * 4).toUSize
  unmapBuffer cBuf
  let results ← Hesper.Basic.bytesToFloatArray resultBytes

  -- With A[i,k] = i+1 and B[k,j] = j+1:
  -- C[i,j] = sum((i+1) * (j+1)) for k iterations = (i+1) * (j+1) * 128
  let testCases := [
    (0, 0, 1.0 * 1.0 * 128.0),   -- C[0,0] = 128
    (0, 1, 1.0 * 2.0 * 128.0),   -- C[0,1] = 256
    (1, 0, 2.0 * 1.0 * 128.0),   -- C[1,0] = 256
    (1, 1, 2.0 * 2.0 * 128.0),   -- C[1,1] = 512
    (10, 10, 11.0 * 11.0 * 128.0) -- C[10,10] = 15488
  ]

  let mut allPassed := true
  for (i, j, expected) in testCases do
    let idx := i * n + j
    let actual := results[idx]!
    let passed := (actual - expected).abs < 0.1
    if passed then
      IO.println s!"✅ C[{i},{j}] = {actual} (expected {expected})"
    else
      IO.println s!"❌ C[{i},{j}] = {actual} (expected {expected})"
      allPassed := false

  if allPassed then
    IO.println "\n✅ All verification tests passed!"
  else
    IO.println "\n❌ Some verification tests failed!"

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Matrix multiplication complete!            ║"
  IO.println "╚══════════════════════════════════════════════╝"
