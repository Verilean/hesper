import Lake
open Lake DSL
open System (FilePath)

package «Hesper» where
  extraDepTargets := #[`nativeDeps]

require LSpec from git
  "https://github.com/argumentcomputer/LSpec.git" @ "main"

-- `doc-gen4` powers the API-reference rendering on the GitHub Pages
-- site (https://verilean.github.io/hesper/api/).  We pin to `main`;
-- the Pages job invokes `lake update doc-gen4 && lake build Hesper:docs`
-- on its own runner so the freshly-resolved manifest never has to
-- be committed.  Production builds DO need the require statement
-- so `lake-manifest.json` knows about the dependency tree, but
-- they never invoke `lake build Hesper:docs`, so the doc-gen4
-- libraries are only fetched on `lake update`, not on a normal
-- build of Hesper itself.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.28.0"

-- ============================================================================
-- NATIVE LINKER FLAGS
-- ============================================================================
-- Defined up here (before any `lean_lib` / `lean_exe`) so the tutorial
-- Docker image (`docker/tutorial/Dockerfile`) can reuse the same Dawn /
-- Highway / CUDA bridge link line when re-linking the xeus-lean kernel
-- against Hesper.  `stdLinkArgs` and `cudaExeArgs` are referenced by
-- every `lean_exe` further down the file.

-- Standard linker configuration for all FFI executables (based on glfw-triangle + Google Highway)
-- Platform-specific: macOS uses frameworks + force_load; Linux uses whole-archive + Vulkan/X11
-- Note: libhesper_cuda.a is unconditionally linked because Hesper.lean's CUDA FFI
-- is exported into nearly every Lean module's .c.o.export, making the symbols
-- live for any exe that imports anything from Hesper.* on Linux.
/-- `HESPER_NO_CUDA=1` (read at lakefile-load time) tells the Linux
    link line to skip `-lcuda` for environments where the CUDA user-
    mode driver shared library isn't installed (tutorial Docker image,
    CI runners with a WebGPU-only target).  The stub
    `libhesper_cuda.a` is still linked so CUDA FFI calls resolve to
    error-returning stubs instead of being undefined symbols. -/
unsafe def hesperNoCudaImpl : Bool :=
  match unsafeBaseIO (IO.getEnv "HESPER_NO_CUDA") with
  | some v => v == "1" || v == "true"
  | none   => false

@[implemented_by hesperNoCudaImpl]
def hesperNoCuda : Bool := false

def stdLinkArgs : Array String := Id.run do
  -- Windows: no native bridge is built (see the `nativeDeps` target —
  -- `System.Platform.isWindows` early-returns).  Returning an empty
  -- link line lets the per-module `.dll`s link cleanly against
  -- pure-Lean code only; the lean_exe targets that actually need
  -- WebGPU don't exist on the Windows CI path.
  if System.Platform.isWindows then return #[]
  -- Dawn lives in `libhesper_native.{so,dylib}` (a SHARED library that
  -- whole-archives libdawn_proc.a, libwebgpu_dawn.a, and libdawn_glfw.a
  -- on the inside — see `native/CMakeLists.txt`).  Every consumer (root
  -- lean_lib `Hesper`, per-module precompiled `.so`s, every lean_exe)
  -- just links `-lhesper_native` to get ONE shared copy of Dawn's
  -- runtime state in the process.  Previously each precompiled module
  -- whole-archived its own private copy of libdawn_proc.a, which gave
  -- each `.so` its own `procs` static — `Hesper.init` set procs in
  -- libHesper_Hesper.so's copy, then `getDevice` read Device.so's still
  -- NULL copy and SIGSEGV'd jumping through a NULL fn ptr.
  let commonArgs := #[
    "-L./.lake/build/native", "-lhesper_native",
    -- Highway SIMD is a small leaf — still safe to static-link directly.
    "./.lake/build/simd/libhesper_simd.a",
    "./.lake/build/simd/_deps/highway-build/libhwy.a"
  ]
  let cudaArgs :=
    if System.Platform.isOSX then #["./.lake/build/native/libhesper_cuda.a"]
    else if hesperNoCuda then #["./.lake/build/native/libhesper_cuda.a"]
    else #["./.lake/build/native/libhesper_cuda.a", "-lcuda"]
  if System.Platform.isOSX then
    -- macOS: rpath to the shared lib so dlopen finds it without
    -- DYLD_LIBRARY_PATH at run time.
    return #["-Wl,-rpath,@loader_path/../native",
             "-Wl,-rpath,@executable_path/.lake/build/native"]
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
  -- Linux: rpath into the build tree so unwrapped lean_exe / dlopen()
  -- from xlean finds libhesper_native.so without LD_LIBRARY_PATH.
  -- $ORIGIN is the directory of the consumer .so / exe.
  --
  -- C++ stdlib: NOT specified here.  The per-module precompiled .so
  -- files contain only Lean-generated C against `lean_object*` — no
  -- C++ TUs of their own.  All the libstdc++ ABI surface lives
  -- inside libhesper_native.so (built with the host compiler's
  -- default stdlib, which is libstdc++ for gcc and clang on Linux),
  -- so consumers don't have to agree on a stdlib at link time.
  return commonArgs ++ cudaArgs ++
    -- $ORIGIN relative paths.  Lay out which consumer lives where:
    --   Hesper_Hesper_*.so → .lake/build/lib/lean/Hesper_*.so       → ../../native
    --   libHesper_Hesper.so (sharedLib) → .lake/build/lib/lib*.so   → ../native
    --   lean_exe binaries    → .lake/build/bin/*                    → ../native
    -- We list all three so any consumer resolves libhesper_native.so.
    #["-Wl,-rpath,$ORIGIN/../native",
      "-Wl,-rpath,$ORIGIN/../../native",
      -- libhesper_native.so already links vulkan / X11 / wayland; we
      -- keep `-ldl -lpthread` here because Lean glue may also need them.
      "-ldl", "-lpthread"]

/-- Per-exe CUDA link args. Empty on macOS so exes that conditionally use the
    CUDA backend still link there (CUDA codegen falls back to no-op at runtime).
    Note: stdLinkArgs already includes the CUDA library on Linux for the FFI
    surface, but per-exe definitions historically appended these unconditionally.
    We centralise here so adding macOS support is a single-line change. -/
def cudaExeArgs : Array String :=
  if System.Platform.isOSX then #[]
  -- Windows: no native bridge is built (nativeDeps is a no-op there),
  -- so don't try to link the CUDA stub archive that doesn't exist.
  else if System.Platform.isWindows then #[]
  else if hesperNoCuda then #["./.lake/build/native/libhesper_cuda.a"]
  else #["./.lake/build/native/libhesper_cuda.a", "-lcuda"]

lean_lib «Hesper» where
  -- precompileModules: produces a per-module `.so` next to each `.olean`.
  -- Combined with `moreLinkArgs := stdLinkArgs` this means every
  -- `@[extern]` declaration in Hesper.* resolves at `#eval` time when
  -- the `.so` is loaded — letting notebook cells run real GPU kernels
  -- without a custom kernel re-link.
  --
  -- Disabled on Windows: nativeDeps is a no-op there (libwebgpu_dawn.a
  -- doesn't exist on MSVC), so per-module .dll links would fail with
  -- "undefined symbol: lean_glfw_*" / "lean_hesper_*" — the bridge
  -- those externs point to was never built.
  precompileModules := !System.Platform.isWindows
  moreLinkArgs := stdLinkArgs

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
  -- C++ stdlib: 2026-05-25 reverted to libstdc++ on Linux (Dawn's own
  -- default).  Previous experiments forcing libc++ via a CMake
  -- toolchain file caused Dawn's FetchContent dependencies (Abseil,
  -- SPIRV-Tools, etc.) to leak libstdc++-mangled std::__future_base
  -- symbols into libwebgpu_dawn.a anyway — those sub-projects reset
  -- CXX_FLAGS in their own `project()` call.  When libhesper_native
  -- became SHARED (so Dawn's `procs` static would be unique per
  -- process — see native/CMakeLists.txt) the unresolved libstdc++
  -- symbols started failing dlopen().  bridge.cpp never crosses
  -- `std::*` types over the Lean FFI, so we don't need ABI match
  -- with Lean's libc++.
  let cmakeArgs := #[
    "-S", dawnSrc.toString, "-B", dawnBuild.toString,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DDAWN_FETCH_DEPENDENCIES=ON",
    "-DDAWN_BUILD_SAMPLES=OFF", "-DDAWN_BUILD_EXAMPLES=OFF", "-DDAWN_BUILD_TESTS=OFF",
    "-DDAWN_ENABLE_INSTALL=ON", "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC",
    "-DCMAKE_INSTALL_PREFIX=" ++ dawnInstall.toString,
    "-DTINT_BUILD_TESTS=OFF", "-DTINT_BUILD_IR_BINARY=OFF", "-DTINT_BUILD_CMD_TOOLS=OFF",
    "-DDAWN_ENABLE_NULL=OFF", "-DDAWN_ENABLE_DESKTOP_GL=OFF", "-DDAWN_ENABLE_OPENGLES=OFF"
  ]
  let cmakeArgsFinal :=
    if System.Platform.isOSX then
      cmakeArgs ++ #["-DDAWN_ENABLE_METAL=ON", "-DDAWN_ENABLE_VULKAN=OFF"]
    else if System.Platform.isWindows then
      cmakeArgs ++ #["-DDAWN_ENABLE_D3D12=ON", "-DDAWN_ENABLE_VULKAN=OFF", "-DDAWN_ENABLE_METAL=OFF"]
    else
      cmakeArgs ++ #["-DDAWN_ENABLE_VULKAN=ON", "-DDAWN_ENABLE_METAL=OFF"]
  let ret ← runCmd "cmake" cmakeArgsFinal
  if ret != 0 then return ret
  -- `--config Release` is required on multi-config generators (e.g. Visual
  -- Studio on Windows) and silently ignored elsewhere.
  let ret ← runCmd "cmake" #["--build", dawnBuild.toString, "--config", "Release", "-j", "8"]
  if ret != 0 then return ret
  let ret ← runCmd "cmake" #["--install", dawnBuild.toString, "--config", "Release"]
  return ret

/-- Build Dawn from tarball, configure + compile + install. Cached via hash file. -/
private def buildDawnIfNeeded (cwd : FilePath) : IO UInt32 := do
  let dawnSrc := cwd / ".lake/build/dawn-src"
  let dawnBuild := cwd / ".lake/build/dawn-build"
  let dawnInstall := cwd / ".lake/build/dawn-install"
  let dawnVersion := "3f79f3aefe0b0a498002564fcfb13eb21ab6c047"

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
  let platform :=
    if System.Platform.isOSX then "osx"
    else if System.Platform.isWindows then "windows"
    else "linux"
  -- Suffix the cache key with the C++ stdlib so that flipping the Dawn
  -- ABI between libstdc++ and libc++ invalidates the cached build.
  -- 2026-05-25: switched from libc++ → libstdc++ on Linux.  See the
  -- long-form note in native/CMakeLists.txt for why.  Short version:
  -- the libc++ choice was a defensive measure that proved unnecessary
  -- (gdb showed the real bug was Dawn proc-table duplication, not a
  -- libc++ mismatch) and Dawn's FetchContent sub-projects leak
  -- libstdc++-mangled symbols even with a libcxx toolchain file.
  let cxxStdlib :=
    if System.Platform.isOSX || System.Platform.isWindows then "default"
    else "libstdcxx"
  let buildConfig := s!"{dawnVersion}-{platform}-Release-{cxxStdlib}"
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

/-- Build the Hesper native bridge library and (on Linux) the CUDA bridge.

    Linux / macOS produce a SHARED `libhesper_native.{so,dylib}` so that
    Dawn's static archives (whole-archived inside it) contribute their
    `procs` and other static globals exactly once per process — see
    `native/CMakeLists.txt` for the long-form rationale.  Windows still
    produces a `.lib`. -/
private def buildBridgeIfNeeded (cwd : FilePath) : IO UInt32 := do
  let bridgeBuild := cwd / ".lake/build/native"
  let (libPrefix, libExt) :=
    if System.Platform.isWindows then ("", "lib")
    else if System.Platform.isOSX then ("lib", "dylib")
    else ("lib", "so")
  let nativeLib := bridgeBuild / s!"{libPrefix}hesper_native.{libExt}"
  let cudaLib   := bridgeBuild / s!"{libPrefix}hesper_cuda.a"
  -- If a stale STATIC `libhesper_native.a` from a pre-shared-lib build
  -- is on disk, the next `cmake --build` would still produce only the
  -- SHARED `.so`/`.dylib`, but the cached CMakeCache.txt may be from
  -- the STATIC-era configure.  Force a reconfigure when we detect that
  -- the expected SHARED output is missing but a stale STATIC archive
  -- is sitting next to it.
  let staleStatic := bridgeBuild / s!"{libPrefix}hesper_native.a"
  let haveStale ←
    if System.Platform.isWindows then pure false
    else pure ((← staleStatic.pathExists) && !(← nativeLib.pathExists))
  let needConfigure := !(← (bridgeBuild / "CMakeCache.txt").pathExists) || haveStale
  if haveStale then
    IO.println "[Hesper] Stale STATIC libhesper_native.a detected — forcing reconfigure for SHARED build"
    -- Wipe the CMake cache so the SHARED-lib configure takes effect.
    let _ ← runCmd "rm" #["-f", (bridgeBuild / "CMakeCache.txt").toString, staleStatic.toString]
  if needConfigure then
    IO.println "[Hesper] Configuring native bridge..."
    IO.FS.createDirAll bridgeBuild
    -- 2026-05-25: libstdc++ is the default for CMake's host gcc/clang
    -- on Linux, so we no longer pass a toolchain file.  Matches Dawn's
    -- own libstdc++ build (see buildDawnIfNeeded cache key).
    let ret ← runCmd "cmake" #[
      "-S", (cwd / "native").toString, "-B", bridgeBuild.toString,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DDAWN_SRC_DIR="     ++ (cwd / ".lake/build/dawn-src").toString,
      "-DDAWN_BUILD_DIR="   ++ (cwd / ".lake/build/dawn-build").toString,
      "-DDAWN_INSTALL_DIR=" ++ (cwd / ".lake/build/dawn-install").toString]
    if ret != 0 then return ret
  if !(← nativeLib.pathExists) then
    IO.println "[Hesper] Building Hesper native bridge..."
    let ret ← runCmd "cmake" #["--build", bridgeBuild.toString, "--target", "hesper_native", "--config", "Release", "-j", "8"]
    if ret != 0 then return ret
  else
    IO.println "[Hesper] Native bridge already built (cached)."
  -- CUDA bridge: real on Linux, stub on macOS/Windows (see native/CMakeLists.txt)
  if !(← cudaLib.pathExists) then
    IO.println "[Hesper] Building Hesper CUDA bridge..."
    let ret ← runCmd "cmake" #["--build", bridgeBuild.toString, "--target", "hesper_cuda", "--config", "Release", "-j", "8"]
    if ret != 0 then return ret
  else
    IO.println "[Hesper] CUDA bridge already built (cached)."
  return 0

/-- Build the SIMD library (Google Highway). -/
private def buildSimdIfNeeded (cwd : FilePath) : IO UInt32 := do
  let simdBuild := cwd / ".lake/build/simd"
  let libExt := if System.Platform.isWindows then "lib" else "a"
  let libPrefix := if System.Platform.isWindows then "" else "lib"
  let libPath := simdBuild / s!"{libPrefix}hesper_simd.{libExt}"
  if (← libPath.pathExists) then
    IO.println "[Hesper] SIMD library already built (cached)."
    return 0
  else
    IO.println "[Hesper] Building SIMD library (Google Highway)..."
    IO.FS.createDirAll simdBuild
    let ret ← runCmd "cmake" #[
      "-S", (cwd / "c_src").toString, "-B", simdBuild.toString,
      "-DCMAKE_BUILD_TYPE=Release"]
    if ret != 0 then return ret
    let ret ← runCmd "cmake" #["--build", simdBuild.toString, "--target", "hesper_simd", "--config", "Release", "-j", "8"]
    if ret != 0 then return ret
    IO.println "[Hesper] SIMD library built."
    return 0

/-- Lake target that builds all native dependencies before any Lean compilation. -/
target nativeDeps : Unit := do
  -- Windows: skip the Dawn + native-bridge build.  Hesper has no
  -- production Windows backend yet, and `native/CMakeLists.txt`
  -- searches for the Unix-style `libwebgpu_dawn.a` / `libglfw3.a`
  -- archives that MSVC's Dawn build doesn't produce (it emits
  -- `webgpu_dawn.lib` instead).  Leaving this as a no-op keeps
  -- `lake build Hesper` (pure Lean) green on the Windows CI runner
  -- without pretending the native bridge works there.
  if System.Platform.isWindows then return .nil
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
  -- Suffix the cache key with the C++ stdlib so that flipping the Dawn
  -- ABI between libstdc++ and libc++ invalidates the cached build.
  -- 2026-05-25: switched from libc++ → libstdc++ on Linux.  See the
  -- long-form note in native/CMakeLists.txt for why.  Short version:
  -- the libc++ choice was a defensive measure that proved unnecessary
  -- (gdb showed the real bug was Dawn proc-table duplication, not a
  -- libc++ mismatch) and Dawn's FetchContent sub-projects leak
  -- libstdc++-mangled symbols even with a libcxx toolchain file.
  let cxxStdlib :=
    if System.Platform.isOSX || System.Platform.isWindows then "default"
    else "libstdcxx"
  let buildConfig := s!"{dawnVersion}-{platform}-Release-{cxxStdlib}"
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
              "-DDAWN_SRC_DIR="     ++ dawnSrc.toString,
              "-DDAWN_BUILD_DIR="   ++ dawnBuild.toString,
              "-DDAWN_INSTALL_DIR=" ++ dawnInstall.toString]
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

  -- CUDA bridge: real impl on Linux, stub on macOS (see native/CMakeLists.txt).
  -- Required because Hesper.CUDA.FFI symbols are exported by most Lean modules.
  let cudaBridgeRet ← IO.Process.spawn {
    cmd := "cmake"
    args := #["--build", bridgeBuild.toString, "--target", "hesper_cuda", "-j", "8"]
  } >>= (·.wait)

  if cudaBridgeRet != 0 then
    IO.eprintln "[Hesper] CUDA bridge build failed"
    return cudaBridgeRet

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

-- ============================================================================
-- EXAMPLES - Organized by Category
-- ============================================================================
-- (Linker flag defs `stdLinkArgs` and `cudaExeArgs` are defined near the
--  top of the file so `lean_lib Hesper` can reference them.)

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

/-- Ch11 California Housing — DataFrame demo (CPU-only data ops). -/
lean_exe «california-housing» where
  root := `Examples.Tutorial.CaliforniaHousing
  moreLinkArgs := stdLinkArgs

/-- Ch11 California Housing — same training loop, every iter dispatched
    onto WebGPU via Hesper.WGSL.MatMul + a few small pointwise kernels. -/
lean_exe «california-housing-gpu» where
  root := `Examples.Tutorial.CaliforniaHousingGPU
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

lean_exe «diffusiongemma-validation» where
  root := `Examples.DiffusionGemmaValidation
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-tiny-test» where
  root := `Examples.DiffusionGemmaTinyTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-rmsnorm-parity» where
  root := `Examples.DSL.DiffusionGemmaRMSNormParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-rope-parity» where
  root := `Examples.DSL.DiffusionGemmaRoPEParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-geglu-parity» where
  root := `Examples.DSL.DiffusionGemmaGeGLUParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-softcap-parity» where
  root := `Examples.DSL.DiffusionGemmaSoftcapParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-matmul-parity» where
  root := `Examples.DSL.DiffusionGemmaMatMulParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-softmax-parity» where
  root := `Examples.DSL.DiffusionGemmaSoftmaxParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-attn-parity» where
  root := `Examples.DSL.DiffusionGemmaAttnParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-softcap-gpu-parity» where
  root := `Examples.DSL.DiffusionGemmaSoftcapGPUParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-gpu-parity» where
  root := `Examples.DSL.DiffusionGemmaGPUParity
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

lean_exe «diffusiongemma-load» where
  root := `Examples.DiffusionGemmaLoad
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-forward-probe» where
  root := `Examples.DiffusionGemmaForwardProbe
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-q8mm-parity» where
  root := `Examples.DSL.DiffusionGemmaQ8MMParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-gqa-attn-test» where
  root := `Examples.DSL.DiffusionGemmaGQAAttnTest
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-q4kexp-parity» where
  root := `Examples.DSL.DiffusionGemmaQ4KExpertParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-q8exp-parity» where
  root := `Examples.DSL.DiffusionGemmaQ8ExpertParity
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-forward» where
  root := `Examples.DiffusionGemmaForward
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-forward-gpu» where
  root := `Examples.DiffusionGemmaForwardGPU
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs


lean_exe «diffusiongemma-q4kbatch-parity» where
  root := `Examples.DSL.DiffusionGemmaQ4KBatchParity
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-battn-parity» where
  root := `Examples.DSL.DiffusionGemmaBatchAttnParity
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-q8batch-parity» where
  root := `Examples.DSL.DiffusionGemmaQ8BatchParity
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-qknormrope-parity» where
  root := `Examples.DSL.DiffusionGemmaQKNormRopeParity
  moreLinkArgs := stdLinkArgs

lean_exe «diffusiongemma-bidir» where
  root := `Examples.DiffusionGemmaBidir
  supportInterpreter := false
  moreLinkArgs := stdLinkArgs
