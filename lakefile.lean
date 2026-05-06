import Lake
open Lake DSL
open System (FilePath)

/-! ## Build configuration

  `gpu` controls only the **CUDA PTX backend** — whether libhesper_cuda.a
  is built and linked into lean_exe targets. It does NOT control Dawn /
  native bridge / SIMD: those are needed by every backend (WGSL + Vulkan
  + Metal + D3D12) so they always build.

  Recognised values:
    auto  — (default) probe the host. CUDA bridge built+linked if cmake
            + libcuda/nvcc are present, otherwise skipped with a message.
            Dawn + native bridge + SIMD always build either way.
    cuda  — force CUDA bridge build. Hard-error if CUDA SDK missing.
    cpu   — skip CUDA bridge entirely. cudaExeArgs evaluate to [], so
            as long as you only ask for non-CUDA exes (gemma4-cuda etc
            won't link), the build succeeds.

  Examples:
    lake build hesper                     # auto: probe CUDA, build Dawn always
    lake -Kgpu=cuda build hesper          # force CUDA (Dawn + CUDA)
    lake -Kgpu=cpu build test-wgsl-dsl    # WGSL via Dawn, no CUDA
-/
def gpuBackend : String := (get_config? gpu).getD "auto"

/-- True iff the build is allowed to link libhesper_cuda.a / -lcuda.
    `auto` and `cuda` enable it; `cpu` disables. (Whether the linked
    library actually exists is determined at `nativeDeps` time when in
    `auto` mode.) -/
def withCuda : Bool := gpuBackend != "cpu"

/-- True iff the user *demanded* CUDA (`-Kgpu=cuda`). `auto` does NOT
    set this; it just tries CUDA and falls back. Used by `nativeDeps`
    to decide whether a missing CUDA SDK is a hard error or a warning. -/
def cudaRequired : Bool := gpuBackend == "cuda"

package «Hesper» where
  -- nativeDeps always runs (Dawn + native bridge + SIMD). The CUDA
  -- bridge inside it is gated separately by `withCuda` / probe.
  extraDepTargets := #[`nativeDeps]

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

-- KernelStub extractor (research / design tool).
-- See `tools/StubExtract/Stub.lean` for the schema and rationale.
-- NOT linked into production. NOT linked from Hesper/.
lean_lib «StubExtract» where
  srcDir := "tools"
  roots := #[`StubExtract.Stub, `StubExtract.Extract]

lean_exe «stub-extract» where
  srcDir := "tools"
  root := `StubExtract.Main
  supportInterpreter := false

-- ============================================================================
-- NATIVE DEPENDENCY BUILD (auto-triggered by `lake build`)
-- ============================================================================

/-- Run a process, printing stdout/stderr. Returns exit code. -/
private def runCmd (cmd : String) (args : Array String) (cwd : Option FilePath := none) : IO UInt32 := do
  let child ← IO.Process.spawn {
    cmd := cmd
    args := args
    cwd := cwd
    stdout := .inherit
    stderr := .inherit
  }
  child.wait

/-- Download Dawn source tarball if not present. -/
private def downloadDawn (cwd : FilePath) (dawnSrc : FilePath) (dawnVersion : String) : IO UInt32 := do
  IO.println "[Hesper] Downloading Dawn source tarball..."
  IO.FS.createDirAll dawnSrc.toString
  let tarballUrl := s!"https://dawn.googlesource.com/dawn/+archive/{dawnVersion}.tar.gz"
  let tarballPath := cwd / ".lake/build/dawn.tar.gz"
  let ret ← runCmd "curl" #["-s", "-L", "-o", tarballPath.toString, tarballUrl]
  if ret != 0 then return ret
  let ret ← runCmd "tar" #["-xzf", tarballPath.toString, "-C", dawnSrc.toString]
  if ret != 0 then return ret
  IO.println s!"[Hesper] Dawn source extracted to: {dawnSrc}"
  return 0

/-- Build Dawn with CMake. -/
private def compileDawn (dawnSrc dawnBuild dawnInstall : FilePath) : IO UInt32 := do
  IO.println "[Hesper] Building Dawn (this may take 10-15 minutes on first build)..."
  IO.FS.createDirAll dawnBuild.toString
  -- Use the platform default generator (Unix Makefiles on Linux/macOS,
  -- Visual Studio on Windows). `--config Release` is passed to
  -- `cmake --build` and `--install` below so that multi-config
  -- generators emit and install from `Release/` correctly.
  -- Visual Studio is a multi-config generator; CMAKE_BUILD_TYPE alone
  -- doesn't constrain it, and `cmake --install` then races between
  -- Release/ and Debug/ subdirs and emits the infamous
  -- "INSTALL cannot find ... Release/webgpu_dawn.lib: File exists" error.
  -- CMAKE_CONFIGURATION_TYPES=Release pins the multi-config generator
  -- to a single config, matching the single-config (Makefiles/Ninja)
  -- behavior we get on Linux/macOS.
  let cmakeArgs := #[
    "-S", dawnSrc.toString, "-B", dawnBuild.toString,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CONFIGURATION_TYPES=Release",
    "-DDAWN_FETCH_DEPENDENCIES=ON",
    "-DDAWN_BUILD_SAMPLES=OFF", "-DDAWN_BUILD_EXAMPLES=OFF", "-DDAWN_BUILD_TESTS=OFF",
    "-DDAWN_ENABLE_INSTALL=ON", "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC",
    "-DCMAKE_INSTALL_PREFIX=" ++ dawnInstall.toString,
    "-DTINT_BUILD_TESTS=OFF", "-DTINT_BUILD_IR_BINARY=OFF", "-DTINT_BUILD_CMD_TOOLS=OFF",
    "-DDAWN_ENABLE_NULL=OFF", "-DDAWN_ENABLE_DESKTOP_GL=OFF", "-DDAWN_ENABLE_OPENGLES=OFF"
  ]
  let backendArgs :=
    if System.Platform.isOSX then
      #["-DDAWN_ENABLE_METAL=ON", "-DDAWN_ENABLE_VULKAN=OFF", "-DDAWN_ENABLE_D3D12=OFF"]
    else if System.Platform.isWindows then
      #["-DDAWN_ENABLE_D3D12=ON", "-DDAWN_ENABLE_VULKAN=OFF", "-DDAWN_ENABLE_METAL=OFF"]
    else
      #["-DDAWN_ENABLE_VULKAN=ON", "-DDAWN_ENABLE_METAL=OFF", "-DDAWN_ENABLE_D3D12=OFF"]
  let cmakeArgsFinal := cmakeArgs ++ backendArgs
  let ret ← runCmd "cmake" cmakeArgsFinal
  if ret != 0 then return ret
  let ret ← runCmd "cmake" #["--build", dawnBuild.toString, "--config", "Release", "-j", "8"]
  if ret != 0 then return ret
  let ret ← runCmd "cmake" #["--install", dawnBuild.toString, "--config", "Release"]
  return ret

/-- Build Dawn from tarball, configure + compile + install. Cached via hash file. -/
private def buildDawnIfNeeded (cwd : FilePath) : IO UInt32 := do
  let dawnSrc := cwd / ".lake/build/dawn-src"
  let dawnBuild := cwd / ".lake/build/dawn-build"
  let dawnInstall := cwd / ".lake/build/dawn-install"
  let dawnVersion := "07e53299e6f6eb75a61d26e17c1ece0655e6e97e"

  -- Download Dawn tarball if not already present.
  -- If a local checkout exists at `dawn/` (e.g. a manually cloned upstream
  -- repo), seed dawn-src from it instead of downloading, so developers can
  -- test patches against the latest upstream without re-downloading.
  if !(← dawnSrc.pathExists) then
    let localDawn := cwd / "dawn"
    if (← localDawn.pathExists) then
      IO.println "[Hesper] Seeding dawn-src from local ./dawn checkout..."
      let ret ← runCmd "cp" #["-r", localDawn.toString, dawnSrc.toString]
      if ret != 0 then return ret
    else
      let ret ← downloadDawn cwd dawnSrc dawnVersion
      if ret != 0 then return ret

  -- Check if rebuild is needed via hash
  let platform := if System.Platform.isOSX then "osx" else "linux"
  let buildConfig := s!"{dawnVersion}-{platform}-Release"
  let hashFile := cwd / ".lake/build/dawn-build.hash"
  let hashExists ← hashFile.pathExists
  let storedHash ← if hashExists then IO.FS.readFile hashFile else pure ""
  if !hashExists || storedHash.trim != buildConfig then
    let ret ← compileDawn dawnSrc dawnBuild dawnInstall
    if ret != 0 then return ret
    IO.FS.writeFile hashFile buildConfig
    IO.println "[Hesper] Dawn build complete."
  else
    IO.println "[Hesper] Dawn already built (cached)."
  return 0

/-- Build the Hesper native bridge library AND the CUDA bridge.
    The CUDA bridge target always exists in CMakeLists.txt — it's the
    real implementation when CUDAToolkit is found, or a stub otherwise
    (so Lean exes link successfully on every platform). -/
private def buildBridgeIfNeeded (cwd : FilePath) : IO UInt32 := do
  let bridgeBuild := cwd / ".lake/build/native"
  let nativeLib := bridgeBuild / "libhesper_native.a"
  let cudaLib := bridgeBuild / "libhesper_cuda.a"
  let needConfigure := !(← (bridgeBuild / "CMakeCache.txt").pathExists)
  if needConfigure then
    IO.println "[Hesper] Configuring native bridge..."
    IO.FS.createDirAll bridgeBuild
    let ret ← runCmd "cmake" #[
      "-S", (cwd / "native").toString, "-B", bridgeBuild.toString,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_CONFIGURATION_TYPES=Release",
      "-DDAWN_SRC_DIR=" ++ (cwd / ".lake/build/dawn-src").toString,
      "-DDAWN_BUILD_DIR=" ++ (cwd / ".lake/build/dawn-build").toString]
    if ret != 0 then return ret
  if !(← nativeLib.pathExists) then
    IO.println "[Hesper] Building Hesper native bridge..."
    let ret ← runCmd "cmake" #["--build", bridgeBuild.toString, "--target", "hesper_native", "--config", "Release", "-j", "8"]
    if ret != 0 then return ret
  else
    IO.println "[Hesper] Native bridge already built (cached)."
  if !(← cudaLib.pathExists) then
    IO.println "[Hesper] Building Hesper CUDA bridge (real or stub depending on CUDAToolkit)..."
    let ret ← runCmd "cmake" #["--build", bridgeBuild.toString, "--target", "hesper_cuda", "--config", "Release", "-j", "8"]
    if ret != 0 then return ret
  else
    IO.println "[Hesper] CUDA bridge already built (cached)."
  return 0

/-- Build the SIMD library (Google Highway). -/
private def buildSimdIfNeeded (cwd : FilePath) : IO UInt32 := do
  let simdBuild := cwd / ".lake/build/simd"
  let libPath := simdBuild / "libhesper_simd.a"
  if (← libPath.pathExists) then
    IO.println "[Hesper] SIMD library already built (cached)."
    return 0
  else
    IO.println "[Hesper] Building SIMD library (Google Highway)..."
    IO.FS.createDirAll simdBuild
    let ret ← runCmd "cmake" #[
      "-S", (cwd / "c_src").toString, "-B", simdBuild.toString,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_CONFIGURATION_TYPES=Release"]
    if ret != 0 then return ret
    let ret ← runCmd "cmake" #["--build", simdBuild.toString, "--target", "hesper_simd", "-j", "8"]
    if ret != 0 then return ret
    IO.println "[Hesper] SIMD library built."
    return 0

/-- Returns true iff `cmd` is on PATH. Uses `which` on Unix, `where` on Windows. -/
private def hasCmd (cmd : String) : IO Bool := do
  let probe := if System.Platform.isWindows then "where" else "which"
  try
    let child ← IO.Process.spawn {
      cmd := probe, args := #[cmd], stdout := .null, stderr := .null }
    return (← child.wait) == 0
  catch _ => return false

/-- Probe for the tools/libs needed by the CUDA + Dawn build path.
    Returns `none` if everything is present, or `some reason` describing
    the first missing piece. -/
private def probeCudaEnv : IO (Option String) := do
  if System.Platform.isOSX then
    return some "macOS host (cuda_bridge.cpp is Linux-only)"
  if System.Platform.isWindows then
    -- cuda_bridge.cpp uses the Linux libcuda.so / dlopen pattern.
    -- Windows would need the bridge ported to LoadLibrary("nvcuda.dll").
    -- Until that lands, the CUDA bridge build is forced to the stub on
    -- Windows (see CMakeLists.txt: `if(NOT APPLE)` is too lax — but the
    -- stub branch still emits the symbols, so exes link). Dawn + native
    -- bridge themselves DO build on Windows now.
    return some "Windows host (cuda_bridge.cpp is Linux-only — using stub)"
  if !(← hasCmd "cmake") then
    return some "cmake not found on PATH"
  -- libcuda.so ships with the NVIDIA driver. nvcc ships with the toolkit.
  -- Either alone is enough to attempt the CUDA bridge build.
  let libcudaCandidates := #[
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so",
    "/usr/lib64/libcuda.so.1"
  ]
  let mut foundLibcuda := false
  for p in libcudaCandidates do
    if (← System.FilePath.pathExists p) then
      foundLibcuda := true
      break
  if !foundLibcuda && !(← hasCmd "nvcc") then
    return some "neither libcuda.so nor nvcc found"
  return none

/-- Lake target that builds all native dependencies before any Lean compilation.

    Builds Dawn + native bridge + CUDA bridge + SIMD on every platform.
    The CUDA bridge is real on Linux+CUDA hosts and a stub elsewhere
    (cuda_bridge_stub.cpp), so Lean exes always link successfully — but
    actually invoking CUDA at runtime errors out unless the real bridge
    is present. The `gpu` flag still controls whether `lean_exe` targets
    add `-lcuda` to their link line (cudaExeArgs).

    Visual Studio (Windows default) is a multi-config generator;
    `CMAKE_CONFIGURATION_TYPES=Release` is passed at configure time so
    that `cmake --install` doesn't try to find Debug/ subdir variants
    of webgpu_dawn.lib — the symptom reported on the dawn-graphics list. -/
target nativeDeps : Unit := do
  let cwd ← IO.currentDir
  let ret ← buildDawnIfNeeded cwd
  if ret != 0 then
    error s!"Dawn build failed (exit code {ret})"
  let ret ← buildBridgeIfNeeded cwd
  if ret != 0 then
    error s!"Native bridge build failed (exit code {ret})"
  let ret ← buildSimdIfNeeded cwd
  if ret != 0 then
    error s!"SIMD build failed (exit code {ret})"
  return .nil

-- ============================================================================
-- MANUAL BUILD SCRIPTS (kept for backward compatibility)
-- ============================================================================

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

    -- Download tarball using curl (silent mode)
    let downloadRet ← IO.Process.spawn {
      cmd := "curl"
      args := #["-s", "-L", "-o", tarballPath, tarballUrl]
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

  IO.println "[Hesper] ✓ Native bridge built successfully!"

  -- Step 4: Build SIMD library (Google Highway)
  IO.println "[Hesper] Building SIMD library (Google Highway)..."
  let simdBuildDir := cwd / ".lake" / "build" / "simd"
  -- Reuse srcDir from above ("native" is used for bridge, "c_src" is for SIMD usually, let's check buildSimd)
  -- The buildSimd script (lines 437+) uses `cwd / "c_src"`.
  let simdSrcDir := cwd / "c_src"

  IO.FS.createDirAll simdBuildDir

  let simdCmakeRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #["-S", simdSrcDir.toString, "-B", simdBuildDir.toString, "-DCMAKE_BUILD_TYPE=Release"]
  } >>= (·.wait)

  if simdCmakeRet != 0 then
    IO.eprintln "[Hesper] SIMD CMake configuration failed"
    return simdCmakeRet

  let simdBuildRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #["--build", simdBuildDir.toString, "--target", "hesper_simd", "-j", "8"]
  } >>= (·.wait)

  if simdBuildRet != 0 then
    IO.eprintln "[Hesper] SIMD library build failed"
    return simdBuildRet

  IO.println "[Hesper] ✓ SIMD library built successfully!"
  IO.println "[Hesper] All native dependencies built."
  return 0

-- Standard linker configuration for all FFI executables (based on glfw-triangle + Google Highway)
-- Platform-specific: macOS uses frameworks + force_load; Linux uses whole-archive + Vulkan/X11
-- Note: libhesper_cuda.a is ALWAYS linked because Hesper.CUDA.FFI's @[extern]
-- symbols leak into every exe via Lean's .c.o.export. The library is the real
-- bridge on Linux+CUDA hosts and a stub elsewhere; either way the file exists.
-- Only `-lcuda` (the NVIDIA driver shim) is gated by `withCuda`.
def stdLinkArgs : Array String :=
  -- Dawn installs to lib/ on macOS, lib64/ on Linux x86_64
  let dawnLibDir := if System.Platform.isOSX then "lib" else "lib64"
  -- glfw3 is needed by the Linux Vulkan/X11 path. macOS/Metal's Dawn
  -- build doesn't produce a glfw3 static lib (it links Cocoa frameworks
  -- directly), so skip -lglfw3 there.
  let glfwArgs : Array String :=
    if System.Platform.isOSX then #[]
    else #[
      -- Upstream Dawn moved glfw from third_party/glfw/src to third_party/glfw3/src/src
      "-L./.lake/build/dawn-build/third_party/glfw3/src/src", "-lglfw3"
    ]
  let commonArgs := #[
    "-L./.lake/build/dawn-build/src/dawn", "-ldawn_proc",
    "-L./.lake/build/dawn-install/" ++ dawnLibDir, "-lwebgpu_dawn"
  ] ++ glfwArgs ++ #[
    "./.lake/build/simd/libhesper_simd.a",
    "./.lake/build/simd/_deps/highway-build/libhwy.a"
  ]
  -- Always link libhesper_cuda.a (real or stub). -lcuda only when CUDA SDK is
  -- actually wanted AND we're on Linux (libcuda.so is the NVIDIA driver shim).
  let cudaArgs :=
    if !withCuda || System.Platform.isOSX then
      #["./.lake/build/native/libhesper_cuda.a"]
    else
      #["./.lake/build/native/libhesper_cuda.a", "-lcuda"]
  if System.Platform.isOSX then
    #["-Wl,-force_load,./.lake/build/native/libhesper_native.a",
      "-Wl,-force_load,./.lake/build/dawn-build/src/dawn/libdawn_proc.a",
      "-Wl,-force_load,./.lake/build/dawn-build/src/dawn/glfw/libdawn_glfw.a"]
    ++ commonArgs ++ cudaArgs ++
    #["-lc++",
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
      "-framework", "Cocoa"]
  else
    #["-Wl,--whole-archive",
      "./.lake/build/native/libhesper_native.a",
      "./.lake/build/dawn-build/src/dawn/libdawn_proc.a",
      "./.lake/build/dawn-build/src/dawn/glfw/libdawn_glfw.a",
      "-Wl,--no-whole-archive"]
    ++ commonArgs ++ cudaArgs ++
    #["-lstdc++",
      "-lvulkan",
      "-lX11",
      "-lX11-xcb",
      "-lxcb",
      "-lwayland-client",
      "-ldl",
      "-lpthread"]

/-- Per-exe CUDA link args. libhesper_cuda.a is always linked (real on
    Linux+CUDA, stub elsewhere — both expose the `lean_hesper_cuda_*`
    symbols that Lean exes reference via @[extern]). Only `-lcuda`
    (the NVIDIA driver shim) is gated. -/
def cudaExeArgs : Array String :=
  if !withCuda || System.Platform.isOSX then
    #["./.lake/build/native/libhesper_cuda.a"]
  else
    #["./.lake/build/native/libhesper_cuda.a", "-lcuda"]

-- ============================================================================
-- EXAMPLES - Organized by Category
-- ============================================================================

-- ----------------------------------------------------------------------------
-- DSL Examples (Pure Lean - Type-safe WGSL DSL demonstration)
-- ----------------------------------------------------------------------------

lean_exe «circuit-irv2-poc» where
  root := `Examples.DSL.CircuitIRv2PoC
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-qproj-parity» where
  root := `Examples.DSL.Gemma4QProjParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-q4k-mmq-parity» where
  root := `Examples.DSL.Gemma4Q4KMMQParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-qkv-parity» where
  root := `Examples.DSL.Gemma4QKVProjParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-ffn-parity» where
  root := `Examples.DSL.Gemma4FFNParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-postffn-parity» where
  root := `Examples.DSL.Gemma4PostFFNParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-kv-parity» where
  root := `Examples.DSL.Gemma4KVWriteParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-k-parity» where
  root := `Examples.DSL.Gemma4KWriteParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-kv-multi-parity» where
  root := `Examples.DSL.Gemma4KVWriteMultiParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-ropeq-parity» where
  root := `Examples.DSL.Gemma4RopeQParity
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-dispatch-count» where
  root := `Examples.DSL.Gemma4DispatchCount
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

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

lean_exe «unified-ad-demo» where
  root := `Examples.DSL.UnifiedADDemo
  moreLinkArgs := stdLinkArgs

lean_exe «verified-op-demo» where
  root := `Examples.DSL.VerifiedOpDemo
  moreLinkArgs := stdLinkArgs

lean_exe «fusion-test» where
  root := `Tests.FusionTest
  -- Fusion test doesn't actually need WebGPU linking, but imports modules that do
  -- For now, keep it simple and don't link WebGPU

lean_exe «neural-net-fusion» where
  root := `Examples.DSL.NeuralNetFusion

lean_exe «composable-mlp» where
  root := `Hesper.NN.ComposableMLP
  moreLinkArgs := stdLinkArgs

lean_exe «resnet» where
  root := `Hesper.NN.ResNet
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- GPU Compute Examples (WebGPU compute shaders and GPU programming)
-- ----------------------------------------------------------------------------

@[default_target]
lean_exe «hesper» where
  root := `Examples.Compute.Main
  moreLinkArgs := stdLinkArgs
lean_exe «parallel-demo» where
  root := `Examples.Compute.ParallelDemo
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

lean_exe «matmul-simple» where
  root := `Examples.Compute.MainMatmulSimple
  moreLinkArgs := stdLinkArgs

lean_exe «matmul-subgroup-m» where
  root := `Examples.Compute.MainMatmulSubgroupM
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

lean_exe «mnist-train-gpu-backprop» where
  root := `Examples.MachineLearning.MNISTTrainGPUBackprop
  moreLinkArgs := stdLinkArgs

lean_exe «mnist-train-gpu-full» where
  root := `Examples.MachineLearning.MNISTTrainGPUFull
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

lean_exe «embedding-test» where
  root := `Tests.EmbeddingTest
  moreLinkArgs := stdLinkArgs

lean_exe «minimal-bindgroup-test» where
  root := `Tests.MinimalBindGroupTest
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

lean_exe «opaque-array-test» where
  root := `Examples.Tests.OpaqueArrayTest
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

lean_exe «test-subgroup-codegen» where
  root := `Examples.Compute.TestSubgroupShader

-- GGUF Tests and Examples (Pure Lean - no GPU or FFI required)
lean_exe «test-gguf» where
  root := `Tests.GGUF.ParserSpec

lean_exe «load-gguf» where
  root := `Examples.LoadGGUF

lean_exe «test-tq2_0» where
  root := `Examples.Tests.TQ2_0_Test

lean_exe «test-rmsnorm» where
  root := `Examples.Tests.RMSNorm_Test

-- ----------------------------------------------------------------------------
-- Benchmarks
-- ----------------------------------------------------------------------------

lean_exe benchmark where
  root := `Benchmarks.Performance
  moreLinkArgs := stdLinkArgs

lean_exe «micro-bench» where
  root := `Bench.MicroBenchmark
  moreLinkArgs := stdLinkArgs

lean_exe «gpu-fixed-cost-bench» where
  root := `Bench.GpuFixedCost
  moreLinkArgs := stdLinkArgs

lean_exe «ttt-golden-gpu» where
  root := `Tests.TTT.TTTGoldenGPUMain
  moreLinkArgs := stdLinkArgs

lean_exe «fuse-pointwise-test» where
  root := `Tests.Circuit.FusePointwiseTest
  moreLinkArgs := stdLinkArgs

lean_exe «fuse-matmul-epilogue-test» where
  root := `Tests.Circuit.FuseMatmulEpilogueTest
  moreLinkArgs := stdLinkArgs

lean_exe «fuse-write-destination-test» where
  root := `Tests.Circuit.FuseWriteDestinationTest
  moreLinkArgs := stdLinkArgs

lean_exe «scatter-dynamic-gpu-test» where
  root := `Tests.Circuit.ScatterDynamicGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «llamacpp-ptx-load-test» where
  root := `Tests.LlamaCppPTX.LoadTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «llamacpp-mmq-launch-test» where
  root := `Tests.LlamaCppPTX.MmqLaunchTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «llamacpp-abi-test» where
  root := `Tests.LlamaCppPTX.ABITest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «hesper-vs-llamacpp-q4k» where
  root := `Tests.LlamaCppPTX.HesperVsLlamacppQ4K
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «rope-k-scatter-gpu-test» where
  root := `Tests.Circuit.RopeKScatterGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «warp-sum-gpu-test» where
  root := `Tests.Circuit.WarpSumGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «warp-dotproduct-gpu-test» where
  root := `Tests.Circuit.WarpDotProductGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «reduce-scatter-gpu-test» where
  root := `Tests.Circuit.ReduceScatterGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «fastdiv-gpu-test» where
  root := `Tests.Circuit.FastdivGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «pointwise-gpu-test» where
  root := `Tests.Circuit.PointwiseGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «broadcast-gpu-test» where
  root := `Tests.Circuit.BroadcastGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «rmsnorm-gpu-test» where
  root := `Tests.Circuit.RmsNormGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «fused-norm-q8-gpu-test» where
  root := `Tests.Circuit.FusedNormQ8GPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «fused-qkv-norm-gpu-test» where
  root := `Tests.Circuit.FusedQKVNormGPUTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «bitnet-ttt-mqar» where
  root := `Examples.BitNetTTT_MQAR
  moreLinkArgs := stdLinkArgs

lean_exe «bitnet-ttt-needle» where
  root := `Examples.BitNetTTT_Needle
  moreLinkArgs := stdLinkArgs

lean_exe «smart-kv-needle» where
  root := `Examples.SmartKV_Needle
  moreLinkArgs := stdLinkArgs

lean_exe «smart-kv-needle-gemma4» where
  root := `Examples.SmartKV_Needle_Gemma4
  moreLinkArgs := stdLinkArgs

lean_exe «ptx-codegen-test» where
  root := `Tests.CUDA.PTXCodeGenTest

def cudaLinkArgs : Array String :=
  stdLinkArgs ++ cudaExeArgs ++ #["-ldl"]

lean_exe «cuda-minimal-test» where
  root := `Tests.CUDA.CUDAMinimalTest
  moreLinkArgs := cudaExeArgs

lean_exe «cuda-execute-test» where
  root := `Tests.CUDA.CUDAExecuteTest
  moreLinkArgs := cudaLinkArgs

lean_exe «buffer-array-test» where
  root := `Tests.CUDA.BufferArrayTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «multi-layer-q4k-test» where
  root := `Tests.CUDA.MultiLayerQ4KTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-matmul-test» where
  root := `Tests.CUDA.CUDAMatMulTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-matmul-microbench» where
  root := `Tests.CUDA.CUDAMatmulMicrobench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-launch-bench» where
  root := `Tests.CUDA.CUDALaunchBench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-backend-test» where
  root := `Tests.CUDA.CUDABackendTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-bitnet-test» where
  root := `Tests.CUDA.CUDABitNetTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-benchmark» where
  root := `Tests.CUDA.CUDABenchmark
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «bitnet-cuda» where
  root := `Examples.BitNetCUDA
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-bitlinear-test» where
  root := `Tests.CUDA.CUDABitLinearTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-flash-test» where
  root := `Tests.CUDA.CUDAFlashAttnTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-fma-f16x2-test» where
  root := `Tests.CUDA.CUDAFmaF16x2Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-cp-async-smoke» where
  root := `Tests.CUDA.CUDACpAsyncTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-im2col-parity» where
  root := `Tests.CUDA.CUDAIm2colTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-conv2d-parity» where
  root := `Tests.CUDA.CUDAConv2dTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-conv-transpose-1d-parity» where
  root := `Tests.CUDA.CUDAConvTranspose1dTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-im2col-vs-llama» where
  root := `Tests.CUDA.CUDAIm2colVsLlamaTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-conv-transpose-1d-vs-llama» where
  root := `Tests.CUDA.CUDAConvTranspose1dVsLlamaTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-im2col-bench» where
  root := `Tests.CUDA.CUDAIm2colBench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-conv-transpose-1d-bench» where
  root := `Tests.CUDA.CUDAConvTranspose1dBench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-geglu-quick-vs-llama» where
  root := `Tests.CUDA.CUDAGegluQuickTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-concat-dim0-vs-llama» where
  root := `Tests.CUDA.CUDAConcatDim0Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-permute-4d-vs-llama» where
  root := `Tests.CUDA.CUDAPermute4dTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-if-branch-cse-microtest» where
  root := `Tests.CUDA.CUDAIfBranchCSEMicrotest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-concat-no-hoist-repro» where
  root := `Tests.CUDA.CUDAConcatNoHoistTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-fa-v11-parity» where
  root := `Tests.CUDA.V11LauncherParityTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-fa-batch-f16-parity» where
  root := `Tests.CUDA.BatchAttnF16Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-rope-kv-batch-f16-parity» where
  root := `Tests.CUDA.RopeKVBatchF16Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-ptx-inst-test» where
  root := `Tests.CUDA.CUDAPTXInstTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-dp4a-test» where
  root := `Tests.CUDA.CUDADP4ATest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-quantize-test» where
  root := `Tests.CUDA.CUDAQuantizeTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-simple-test» where
  root := `Tests.CUDA.CUDASimpleTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q6k-dp4a-test» where
  root := `Tests.CUDA.CUDAQ6KDP4ATest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q6k-4warp-parity» where
  root := `Tests.CUDA.CUDAQ6K4WarpParityTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q6k-4warp-graphs» where
  root := `Tests.CUDA.CUDAQ6K4WarpGraphsTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q6k-to-f16-test» where
  root := `Tests.CUDA.Q6KToF16Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-f16-lmhead-microbench» where
  root := `Tests.CUDA.F16LmHeadMicrobench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-f16-lmhead-ptx-dump» where
  root := `Tests.CUDA.F16LmHeadPtxDump
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q6k-ptx-dump» where
  root := `Tests.CUDA.Q6KPtxDump
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-q4k-ptx-dump» where
  root := `Tests.CUDA.Q4KPtxDump
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-fa-golden-test» where
  root := `Tests.CUDA.CUDAFlashAttnGoldenTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-if-guarded-rb-test» where
  root := `Tests.CUDA.IfGuardedReadBufferTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-cse-assign-bug-test» where
  root := `Tests.CUDA.CSEAssignBugTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-rope-k-f16-test» where
  root := `Tests.CUDA.RopeKF16Test
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-bitnet-golden-test» where
  root := `Tests.CUDA.CUDABitNetGoldenTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-cuda» where
  root := `Examples.Gemma4CUDA
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «mmap-smoke» where
  root := `Tests.CUDA.MmapSmokeTest
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-llama-skeleton» where
  root := `Examples.Gemma4LlamaSkeleton
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-llama-prefill-skeleton» where
  root := `Examples.Gemma4LlamaPrefillSkeleton
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-stub-decode-bench» where
  root := `Examples.Gemma4StubDecodeBench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-unit-tests» where
  root := `Tests.GoldenUnit.Main
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «gemma4-kernel-bench» where
  root := `Tests.Perf.Gemma4KernelBench
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

lean_exe «cuda-graph-smoke» where
  root := `Examples.CUDAGraphSmoke
  moreLinkArgs := stdLinkArgs ++ cudaExeArgs

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
  root := `Examples.SIMD.MainSimdBench
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «simd-simple» where
  root := `Examples.SIMD.MainSimdSimple
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «simd-test» where
  root := `Examples.SIMD.MainSimdTest
  supportInterpreter := true

lean_exe «simd-debug» where
  root := `Examples.SIMD.MainSimdDebug
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «simd-minimal» where
  root := `Examples.SIMD.MainSimdMinimal
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «multi-precision» where
  root := `Examples.SIMD.MainMultiPrecision
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «debug-conversion» where
  root := `Examples.SIMD.MainDebugConversion
  supportInterpreter := true
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «debug-f16» where
  root := `Examples.SIMD.MainDebugF16
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «simd-perf-bench» where
  root := `Examples.SIMD.MainSimdPerfBench
  supportInterpreter := false
  moreLinkArgs := #[s!".lake/build/simd/libhesper_simd.a", s!".lake/build/simd/_deps/highway-build/libhwy.a"]

lean_exe «test-dsl-kernels» where
  root := `Examples.Tests.TestDSLKernels
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «test-gpu-backward» where
  root := `Examples.Tests.TestGPUBackward
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «test-adam-gpu» where
  root := `Examples.Tests.TestAdamGPU
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «integration-tests» where
  root := `Tests.Integration.IntegrationMain
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- BitNet Validation Tests
-- ----------------------------------------------------------------------------

lean_exe «bitlinear-equiv» where
  root := `Tests.BitLinearEquivalence
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «bitlinear-bench» where
  root := `Tests.BitLinearBench
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «wmma-matmul-equiv» where
  root := `Tests.WMMAMatMulEquiv
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «wmma-matmul-bench» where
  root := `Tests.WMMAMatMulBench
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «subgroup-matrix-f16-probe» where
  root := `Examples.Compute.SubgroupMatrixF16Probe
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «bitnet-validation» where
  root := `Tests.BitNetValidation
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «kvcache-validation» where
  root := `Tests.KVCacheValidation
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «bitnet-complete» where
  root := `Examples.BitNetComplete
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-inference» where
  root := `Examples.Gemma4Inference
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-validation» where
  root := `Examples.Gemma4Validation
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-embd-test» where
  root := `Examples.Gemma4EmbdTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-layer0-dump» where
  root := `Examples.Gemma4Layer0Dump
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-layer0-forward» where
  root := `Examples.Gemma4Layer0Forward
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-all-layers-dump» where
  root := `Examples.Gemma4AllLayersDump
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-layer5-dump» where
  root := `Examples.Gemma4Layer5Dump
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-pos1-dump» where
  root := `Examples.Gemma4Pos1Dump
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «gemma4-profile» where
  root := `Examples.Gemma4Profile
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «flash-attn-shader-dump» where
  root := `Examples.FlashAttnShaderDump
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

-- ----------------------------------------------------------------------------
-- Training (LoRA Finetuning)
-- ----------------------------------------------------------------------------

lean_exe «alpaca-finetune» where
  root := `Examples.Training.AlpacaFinetune
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «backward-verify» where
  root := `Tests.BackwardVerification
  supportInterpreter := true

lean_exe «verified-ad» where
  root := `Tests.VerifiedAD
  supportInterpreter := true

lean_exe «wrong-backward-test» where
  root := `Tests.WrongBackwardTest
  supportInterpreter := true

lean_exe «parse-float-spec» where
  root := `Tests.ParseFloatSpec
  supportInterpreter := true

lean_exe «saved-activation-test» where
  root := `Tests.SavedActivationTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «rmsnorm-backward-test» where
  root := `Tests.RMSNormBackwardGPUTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «chain-completeness» where
  root := `Tests.ChainCompletenessTest
  supportInterpreter := true

lean_exe «gpu-vs-cpu-test» where
  root := `Tests.GPUvsCPUBackwardTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «flash-attention-test» where
  root := `Tests.FlashAttentionTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe i2s_validation where
  root := `Tests.I2S_Validation
  supportInterpreter := true
lean_exe small_embedding_test where
  root := `Tests.SmallEmbeddingTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs


lean_exe «wmma-ptx-text-test» where
  root := `Tests.CUDA.WmmaPTXTextTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «wmma-shaderm-test» where
  root := `Tests.CUDA.WmmaShaderMTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «wmma-gpu-parity-test» where
  root := `Tests.CUDA.WmmaGPUParityTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs
