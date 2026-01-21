import Lake
open Lake DSL
open System (FilePath)

package «Hesper» where
  -- add package configuration options here

require LSpec from git
  "https://github.com/argumentcomputer/LSpec.git" @ "main"

lean_lib «Hesper» where
  -- add library configuration options here

lean_lib «Examples» where
  roots := #[`Examples]
  globs := #[.submodules `Examples]

lean_lib «Tests» where
  roots := #[`Tests]
  globs := #[.submodules `Tests]

/-- Build script for native C++ library with Dawn integration -/
script buildNative do
  -- Get absolute paths
  let cwd ← IO.currentDir
  let dawnSrc := cwd / ".lake/build/dawn-src"
  let dawnBuild := cwd / ".lake/build/dawn-build"
  let dawnInstall := cwd / ".lake/build/dawn-install"
  let bridgeBuild := cwd / ".lake/build/native"
  let srcDir := cwd / "native"

  -- Step 1: Download Dawn tarball if not already present
  let dawnExists ← dawnSrc.pathExists
  if !dawnExists then
    IO.println "[Hesper] Downloading Dawn source tarball (faster than git clone)..."
    IO.FS.createDirAll dawnSrc.toString

    -- Use known working Dawn commit
    let dawnVersion := "3f79f3aefe0b0a498002564fcfb13eb21ab6c047"
    let tarballUrl := s!"https://dawn.googlesource.com/dawn/+archive/{dawnVersion}.tar.gz"
    let tarballPath := ".lake/build/dawn.tar.gz"

    IO.println s!"Downloading from: {tarballUrl}"

    -- Download tarball using curl
    let downloadRet ← IO.Process.spawn {
      cmd := "curl"
      args := #["-L", "-o", tarballPath, tarballUrl]
    } >>= (·.wait)

    if downloadRet != 0 then
      IO.eprintln "[Hesper] Failed to download Dawn tarball"
      return downloadRet

    IO.println "Extracting tarball..."
    -- Extract tarball to dawnSrc
    let extractRet ← IO.Process.spawn {
      cmd := "tar"
      args := #["-xzf", tarballPath, "-C", dawnSrc.toString]
    } >>= (·.wait)

    if extractRet != 0 then
      IO.eprintln "[Hesper] Failed to extract Dawn tarball"
      return extractRet

    IO.println s!"Dawn source extracted to: {dawnSrc}"
  else
    IO.println "[Hesper] Dawn source already present"

  -- Step 2: Build Dawn with correct flags
  -- Create a hash of Dawn version + build config to track if rebuild is needed
  let dawnVersion := "3f79f3aefe0b0a498002564fcfb13eb21ab6c047"
  let platform := if System.Platform.isOSX then "osx" else "linux"
  let buildConfig := s!"{dawnVersion}-{platform}-Release"
  let hashFile := cwd / ".lake/build/dawn-build.hash"

  let hashExists ← hashFile.pathExists
  let needsBuild ← if !hashExists then
    pure true
  else
    let storedHash ← IO.FS.readFile hashFile
    pure (storedHash.trim != buildConfig)

  if needsBuild then
    IO.println "[Hesper] Building Dawn (this will take 10-15 minutes)..."
    IO.FS.createDirAll dawnBuild.toString

    let cmakeArgs := #[
      "-S", dawnSrc.toString,
      "-B", dawnBuild.toString,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DDAWN_FETCH_DEPENDENCIES=ON",
      "-DDAWN_BUILD_SAMPLES=OFF",
      "-DDAWN_BUILD_EXAMPLES=OFF",
      "-DDAWN_BUILD_TESTS=OFF",
      "-DDAWN_ENABLE_INSTALL=ON",
      "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC",
      "-DCMAKE_INSTALL_PREFIX=" ++ dawnInstall.toString,
      "-DTINT_BUILD_TESTS=OFF",
      "-DTINT_BUILD_IR_BINARY=OFF",
      "-DTINT_BUILD_CMD_TOOLS=OFF",
      "-DDAWN_ENABLE_NULL=OFF",
      "-DDAWN_ENABLE_DESKTOP_GL=OFF",
      "-DDAWN_ENABLE_OPENGLES=OFF"
    ]

    -- Add platform-specific flags
    let cmakeArgsFinal :=
      if System.Platform.isOSX then
        cmakeArgs ++ #["-DDAWN_ENABLE_METAL=ON", "-DDAWN_ENABLE_VULKAN=OFF"]
      else
        cmakeArgs ++ #["-DDAWN_ENABLE_VULKAN=ON", "-DDAWN_ENABLE_METAL=OFF"]

    let cmakeRet ← IO.Process.spawn {
      cmd := "cmake"
      args := cmakeArgsFinal
    } >>= (·.wait)

    if cmakeRet != 0 then
      IO.eprintln "[Hesper] Dawn CMake configuration failed"
      return cmakeRet

    -- Build Dawn
    let buildRet ← IO.Process.spawn {
      cmd := "cmake"
      args := #["--build", dawnBuild.toString, "-j", "8"]
    } >>= (·.wait)

    if buildRet != 0 then
      IO.eprintln "[Hesper] Dawn build failed"
      return buildRet

    -- Install Dawn
    let installRet ← IO.Process.spawn {
      cmd := "cmake"
      args := #["--install", dawnBuild.toString]
    } >>= (·.wait)

    if installRet != 0 then
      IO.eprintln "[Hesper] Dawn install failed"
      return installRet

    -- Write hash file to mark successful build
    IO.FS.writeFile hashFile buildConfig
    IO.println s!"[Hesper] Dawn build complete, hash saved: {buildConfig}"
  else
    IO.println "[Hesper] Dawn already built (hash matches)"

  -- Step 3: Build our bridge library
  IO.println "[Hesper] Building Hesper native bridge..."
  IO.FS.createDirAll bridgeBuild

  let bridgeCmakeRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #["-S", srcDir.toString, "-B", bridgeBuild.toString,
              "-DCMAKE_BUILD_TYPE=Release",
              "-DDAWN_SRC_DIR=" ++ dawnSrc.toString,
              "-DDAWN_BUILD_DIR=" ++ dawnBuild.toString]
  } >>= (·.wait)

  if bridgeCmakeRet != 0 then
    IO.eprintln "[Hesper] Bridge CMake configuration failed"
    return bridgeCmakeRet

  let bridgeBuildRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #["--build", bridgeBuild.toString, "--target", "hesper_native", "-j", "8"]
  } >>= (·.wait)

  if bridgeBuildRet != 0 then
    IO.eprintln "[Hesper] Bridge build failed"
    return bridgeBuildRet

  IO.println "[Hesper] ✓ Native library built successfully!"
  return 0

-- Standard linker configuration for all FFI executables (based on glfw-triangle + Google Highway)
def stdLinkArgs : Array String := #[
  "-Wl,-force_load,./.lake/build/native/libhesper_native.a",
  "-L./.lake/build/dawn-build/src/dawn", "-ldawn_proc",
  "-L./.lake/build/dawn-install/lib", "-lwebgpu_dawn",
  "-L./.lake/build/dawn-build/third_party/glfw/src", "-lglfw3",
  "-Wl,-force_load,./.lake/build/dawn-build/src/dawn/libdawn_proc.a",
  "-Wl,-force_load,./.lake/build/dawn-build/src/dawn/glfw/libdawn_glfw.a",
  "./.lake/build/simd/libhesper_simd.a",
  "-lc++",
  "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
  "-F/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks",
  "-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
  "-lobjc",
  "-framework", "CoreFoundation",
  "-framework", "Metal",
  "-framework", "Foundation",
  "-framework", "QuartzCore",
  "-framework", "IOKit",
  "-framework", "IOSurface",
  "-framework", "Cocoa"
]

-- ============================================================================
-- EXAMPLES - Organized by Category
-- ============================================================================

-- ----------------------------------------------------------------------------
-- DSL Examples (Pure Lean - Type-safe WGSL DSL demonstration)
-- ----------------------------------------------------------------------------

lean_exe «dsl-basics» where
  root := `Examples.DSL.DSLBasics

lean_exe «shader-gen» where
  root := `Examples.DSL.ShaderGeneration

lean_exe «monad-demo» where
  root := `Examples.DSL.MonadDemo

lean_exe «codegen-demo» where
  root := `Examples.DSL.CodeGenDemo

lean_exe «matmul-subgroup» where
  root := `Examples.DSL.MatmulSubgroup

lean_exe «atomic-counter» where
  root := `Examples.DSL.AtomicCounter

lean_exe «kernel-fusion» where
  root := `Examples.DSL.KernelFusion

lean_exe «ad-demo» where
  root := `Examples.DSL.ADDemo

lean_exe «ad-proof» where
  root := `Examples.DSL.ADProof

lean_exe «ad-debug» where
  root := `Examples.DSL.ADDebug

lean_exe «verified-op-demo» where
  root := `Examples.DSL.VerifiedOpDemo
  moreLinkArgs := stdLinkArgs

lean_exe «fusion-test» where
  root := `Tests.FusionTest
  -- Fusion test doesn't actually need WebGPU linking, but imports modules that do
  -- For now, keep it simple and don't link WebGPU

lean_exe «neural-net-fusion» where
  root := `Examples.DSL.NeuralNetFusion

-- ----------------------------------------------------------------------------
-- GPU Compute Examples (WebGPU compute shaders and GPU programming)
-- ----------------------------------------------------------------------------

@[default_target]
lean_exe «hesper» where
  root := `Examples.Compute.Main
  moreLinkArgs := stdLinkArgs

lean_exe «hesper-simple» where
  root := `Examples.Compute.MainSimple
  moreLinkArgs := stdLinkArgs

lean_exe «execute-demo» where
  root := `Examples.Compute.ExecuteDemo
  moreLinkArgs := stdLinkArgs

lean_exe «real-gpu-demo» where
  root := `Examples.Compute.RealGPUDemo
  moreLinkArgs := stdLinkArgs

lean_exe «matmul-gpu» where
  root := `Examples.Compute.MainMatmul
  moreLinkArgs := stdLinkArgs

lean_exe «matmul-4k» where
  root := `Examples.Compute.MainMatmul4K
  moreLinkArgs := stdLinkArgs

lean_exe «async-demo» where
  root := `Examples.Compute.AsyncDemo
  moreLinkArgs := stdLinkArgs

lean_exe «multigpu» where
  root := `Examples.Compute.MultiGPU
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- Machine Learning Examples (Neural networks and training)
-- ----------------------------------------------------------------------------

lean_exe «nn-demo» where
  root := `Examples.MachineLearning.NNDemo

lean_exe «mnist-train» where
  root := `Examples.MachineLearning.MNISTTrain
  moreLinkArgs := stdLinkArgs

lean_exe «mnist-train-fused» where
  root := `Examples.MachineLearning.MNISTTrainFused

lean_exe «mnist-train-gpu» where
  root := `Examples.MachineLearning.MNISTTrainGPU
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- Graphics Examples (GLFW rendering and window management)
-- ----------------------------------------------------------------------------

lean_exe «glfw-simple» where
  root := `Examples.Graphics.GLFWSimple
  moreLinkArgs := stdLinkArgs

lean_exe «glfw-triangle» where
  root := `Examples.Graphics.GLFWTriangle
  moreLinkArgs := stdLinkArgs

lean_exe «glfw-demo» where
  root := `Examples.Graphics.GLFWDemo
  moreLinkArgs := stdLinkArgs

lean_exe «tetris» where
  root := `Examples.Graphics.Tetris
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- Utilities and Tools
-- ----------------------------------------------------------------------------

lean_exe «test-ffi» where
  root := `Examples.Utilities.MainTest
  moreLinkArgs := stdLinkArgs

lean_exe «chrome-tracing-demo» where
  root := `Examples.Utilities.ChromeTracingDemo

-- ============================================================================
-- TESTS AND BENCHMARKS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Test Executables
-- ----------------------------------------------------------------------------

@[test_driver]
lean_exe test where
  root := `Tests.ErrorHandlingMain
  moreLinkArgs := stdLinkArgs

lean_exe «test-all» where
  root := `Tests.All
  moreLinkArgs := stdLinkArgs

-- GPU Tests
lean_exe «test-device» where
  root := `Tests.DeviceTestsMain
  moreLinkArgs := stdLinkArgs

lean_exe «test-buffer» where
  root := `Tests.BufferTestsMain
  moreLinkArgs := stdLinkArgs

lean_exe «test-compute» where
  root := `Tests.ComputeTestsMain
  moreLinkArgs := stdLinkArgs

lean_exe «test-gpu-accuracy» where
  root := `Tests.GPUAccuracyTestsMain
  moreLinkArgs := stdLinkArgs

lean_exe «buffer-test» where
  root := `Tests.BufferTest
  moreLinkArgs := stdLinkArgs

lean_exe «minimal-test» where
  root := `Tests.MinimalBenchmark
  moreLinkArgs := stdLinkArgs

lean_exe «gpu-roundtrip» where
  root := `Examples.Tests.GPURoundtrip
  moreLinkArgs := stdLinkArgs

lean_exe «simple-write» where
  root := `Examples.Tests.SimpleWrite
  moreLinkArgs := stdLinkArgs

lean_exe «unit-tests» where
  root := `Tests.UnitTests
  moreLinkArgs := stdLinkArgs

lean_exe «ffi-tests» where
  root := `Tests.FFIBoundaryTests
  moreLinkArgs := stdLinkArgs

-- Pure Lean Tests (no GPU required)
lean_exe «test-wgsl-dsl» where
  root := `Tests.WGSLDSLTestsMain

lean_exe «test-numerical» where
  root := `Tests.NumericalTestsMain

lean_exe «test-shader-monad» where
  root := `Tests.ShaderMonadTestsMain

-- ----------------------------------------------------------------------------
-- Benchmarks
-- ----------------------------------------------------------------------------

lean_exe benchmark where
  root := `Benchmarks.Performance
  moreLinkArgs := stdLinkArgs

-- ============================================================================
-- SIMD CPU BACKEND (Google Highway)
-- ============================================================================

/-- Script to build SIMD library using CMake + Google Highway -/
script buildSimd do
  let cwd ← IO.currentDir
  let buildDir := cwd / ".lake" / "build" / "simd"
  let srcDir := cwd / "c_src"
  let libPath := buildDir / "libhesper_simd.a"

  -- Create build directory
  IO.FS.createDirAll buildDir

  IO.println "[Hesper] Building SIMD library with Google Highway..."

  -- Run CMake configuration
  let cmakeRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #[
      "-S", srcDir.toString,
      "-B", buildDir.toString,
      "-DCMAKE_BUILD_TYPE=Release"
    ]
  } >>= (·.wait)

  if cmakeRet != 0 then
    IO.eprintln "[Hesper] CMake configuration failed"
    return cmakeRet

  -- Build the library
  let buildRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #[
      "--build", buildDir.toString,
      "--target", "hesper_simd",
      "-j", "8"
    ]
  } >>= (·.wait)

  if buildRet != 0 then
    IO.eprintln "[Hesper] SIMD library build failed"
    return buildRet

  IO.println s!"[Hesper] SIMD library compiled: {libPath}"
  return 0

-- ----------------------------------------------------------------------------
-- SIMD Executables (Google Highway backend for CPU vectorization)
-- ----------------------------------------------------------------------------

lean_exe «simd-bench» where
  root := `MainSimdBench
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «simd-simple» where
  root := `MainSimdSimple
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «simd-test» where
  root := `MainSimdTest
  supportInterpreter := true

lean_exe «simd-debug» where
  root := `MainSimdDebug
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «simd-minimal» where
  root := `MainSimdMinimal
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «multi-precision» where
  root := `MainMultiPrecision
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «debug-conversion» where
  root := `MainDebugConversion
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «debug-f16» where
  root := `MainDebugF16
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]

lean_exe «simd-perf-bench» where
  root := `MainSimdPerfBench
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a"]
