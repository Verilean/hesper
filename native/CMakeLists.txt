cmake_minimum_required(VERSION 3.16)
project(HesperNative)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++ stdlib: 2026-05-25 we use whatever the host compiler defaults to
# (libstdc++ for gcc on Linux, libc++ for AppleClang on macOS).  Dawn
# is built with the same default in lakefile.lean::compileDawn — see
# the long-form note there for why a previous attempt to force libc++
# (toolchain file) didn't stick across Dawn's FetchContent deps.
if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Hesper native: using toolchain ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Expect DAWN_SRC_DIR and DAWN_BUILD_DIR to be provided
if(NOT DEFINED DAWN_SRC_DIR)
    message(FATAL_ERROR "DAWN_SRC_DIR must be specified")
endif()
if(NOT DEFINED DAWN_BUILD_DIR)
    message(FATAL_ERROR "DAWN_BUILD_DIR must be specified")
endif()

# Find Lean installation
if(NOT DEFINED LEAN_INCLUDE_DIR)
    execute_process(
        COMMAND lean --print-prefix
        OUTPUT_VARIABLE LEAN_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(LEAN_INCLUDE_DIR "${LEAN_PREFIX}/include")
endif()

message(STATUS "Lean include directory: ${LEAN_INCLUDE_DIR}")
message(STATUS "Dawn source directory: ${DAWN_SRC_DIR}")
message(STATUS "Dawn build directory: ${DAWN_BUILD_DIR}")

# Build our native bridge library as a SHARED library.
#
# Why SHARED (was STATIC before 2026-05-25):
#   With `precompileModules := true` in lakefile.lean every Hesper.* module
#   becomes its own `.so`.  If Dawn's static archives (libdawn_proc.a +
#   libwebgpu_dawn.a) were whole-archived into each per-module link line,
#   each `.so` would have its OWN private copy of Dawn's internal `procs`
#   static (the function-pointer table that `dawnProcSetProcs` fills).
#   `Hesper.init` would set `procs` in `libHesper_Hesper.so`'s copy, but
#   `getDevice` running from `Hesper_Hesper_WebGPU_Device.so` would read
#   ITS own still-NULL copy and SIGSEGV jumping through a NULL fn ptr.
#
#   Putting bridge.cpp + all Dawn archives into a single SHARED lib
#   guarantees there is exactly one `procs` static in the process, so
#   `dawnProcSetProcs(...)` is visible to every dlopen()'d Hesper module.
#
# The Dawn / GLFW static archives are pulled in with --whole-archive so
# their symbols (`dawnProcSetProcs`, `wgpuInstance*`, the entire Dawn
# native runtime) are present in libhesper_native.so.  Consumers link
# with `-lhesper_native` only — Dawn becomes an implementation detail.
add_library(hesper_native SHARED bridge.cpp glfw_bridge.cpp)

# Include Lean, Dawn, and GLFW headers (GLFW is in Dawn's third_party)
target_include_directories(hesper_native PRIVATE
  ${LEAN_INCLUDE_DIR}
  ${DAWN_SRC_DIR}/include
  ${DAWN_BUILD_DIR}/gen/include
  ${DAWN_SRC_DIR}/third_party/glfw/include
)

# Resolve the install-side libwebgpu_dawn.a — CMake's GNUInstallDirs picks
# `lib/` on Ubuntu and `lib64/` on NixOS, so probe both.
if(EXISTS "${DAWN_INSTALL_DIR}/lib64/libwebgpu_dawn.a")
  set(_WEBGPU_DAWN_ARCHIVE "${DAWN_INSTALL_DIR}/lib64/libwebgpu_dawn.a")
elseif(EXISTS "${DAWN_INSTALL_DIR}/lib/libwebgpu_dawn.a")
  set(_WEBGPU_DAWN_ARCHIVE "${DAWN_INSTALL_DIR}/lib/libwebgpu_dawn.a")
else()
  message(FATAL_ERROR
    "libwebgpu_dawn.a not found under ${DAWN_INSTALL_DIR}/lib{,64}/")
endif()
message(STATUS "Hesper native: bundling ${_WEBGPU_DAWN_ARCHIVE}")

# Resolve glfw3 — moved between Dawn revisions
if(EXISTS "${DAWN_BUILD_DIR}/third_party/glfw3/src/src/libglfw3.a")
  set(_GLFW3_ARCHIVE "${DAWN_BUILD_DIR}/third_party/glfw3/src/src/libglfw3.a")
elseif(EXISTS "${DAWN_BUILD_DIR}/third_party/glfw/src/libglfw3.a")
  set(_GLFW3_ARCHIVE "${DAWN_BUILD_DIR}/third_party/glfw/src/libglfw3.a")
else()
  message(FATAL_ERROR
    "libglfw3.a not found under ${DAWN_BUILD_DIR}/third_party/glfw{,3}/...")
endif()
message(STATUS "Hesper native: bundling ${_GLFW3_ARCHIVE}")

# Critical: --whole-archive libdawn_proc.a ONLY.  That archive is tiny
# (~150 KB, just dawn_proc.cpp.o + dawn_thread_dispatch_proc.cpp.o) and
# owns the `procs` static globals that `dawnProcSetProcs(...)` fills.
# By forcing the entire archive into libhesper_native.so we guarantee
# the `procs` static exists exactly once in the process — fixing the
# 2026-05-24 crash where each per-module precompiled `.so` had its own
# private (and uninitialised) `procs` table.
#
# libwebgpu_dawn.a is linked WITHOUT --whole-archive, so the linker
# pulls only what bridge.cpp references on-demand.  This skips
# webgpu_dawn_native_proc.cpp.o (the alternative direct-dispatch proc
# table) because the same C `wgpu*` symbols are already supplied by
# libdawn_proc.a's dispatcher.
if(APPLE)
  # libwebgpu_dawn.a + libglfw3.a contain Objective-C / Objective-C++
  # objects (Metal backend in Dawn, Cocoa window code in GLFW), so the
  # SHARED .dylib has to resolve the libobjc runtime entry points
  # (objc_msgSendSuper2, objc_opt_class, …) and the Cocoa / Metal /
  # CoreFoundation framework symbols here.  Before hesper_native became
  # SHARED, these were resolved at the final lean_exe link via
  # `stdLinkArgs` in lakefile.lean; now the dylib stands on its own
  # and must link the same set itself.
  #
  # We use `find_library` instead of plain `-framework Foo` so CMake
  # resolves the framework path against `CMAKE_OSX_SYSROOT` (the macOS
  # SDK).  Passing the framework as a raw `-framework Foo` string
  # confused the GitHub Actions linker (`ld: library 'CoreFoundation'
  # not found`) on macOS-15 because the SDK's `System/Library/Frameworks`
  # was not on the default framework search path.
  find_library(COCOA_LIB Cocoa REQUIRED)
  find_library(COREFOUNDATION_LIB CoreFoundation REQUIRED)
  find_library(FOUNDATION_LIB Foundation REQUIRED)
  find_library(METAL_LIB Metal REQUIRED)
  find_library(QUARTZCORE_LIB QuartzCore REQUIRED)
  find_library(IOKIT_LIB IOKit REQUIRED)
  find_library(IOSURFACE_LIB IOSurface REQUIRED)
  target_link_libraries(hesper_native PRIVATE
    "-Wl,-force_load,${DAWN_BUILD_DIR}/src/dawn/libdawn_proc.a"
    ${DAWN_BUILD_DIR}/src/dawn/glfw/libdawn_glfw.a
    ${_WEBGPU_DAWN_ARCHIVE}
    ${_GLFW3_ARCHIVE}
    "-lobjc"
    ${COREFOUNDATION_LIB}
    ${FOUNDATION_LIB}
    ${METAL_LIB}
    ${QUARTZCORE_LIB}
    ${IOKIT_LIB}
    ${IOSURFACE_LIB}
    ${COCOA_LIB}
  )
else()
  target_link_libraries(hesper_native PRIVATE
    "-Wl,--whole-archive"
    "${DAWN_BUILD_DIR}/src/dawn/libdawn_proc.a"
    "-Wl,--no-whole-archive"
    ${DAWN_BUILD_DIR}/src/dawn/glfw/libdawn_glfw.a
    ${_WEBGPU_DAWN_ARCHIVE}
    ${_GLFW3_ARCHIVE}
    # System libraries Dawn / GLFW depend on at run time.
    vulkan dl pthread X11 X11-xcb xcb wayland-client
  )
endif()

# ============================================================================
# CUDA bridge
# - Auto-detect: if the CUDA Toolkit is available, build the real bridge
#   that links against the CUDA driver API. Otherwise fall back to the stub
#   library so Lean modules importing Hesper.CUDA.FFI still link.
# - macOS / Windows: always stub (no CUDA support upstream).
# ============================================================================
set(HESPER_CUDA_FOUND FALSE)
if(NOT (APPLE OR WIN32))
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(HESPER_CUDA_FOUND TRUE)
    endif()
endif()

if(HESPER_CUDA_FOUND)
    message(STATUS "Hesper CUDA bridge: real (CUDA Toolkit detected)")
    add_library(hesper_cuda STATIC cuda_bridge.cpp)
    target_include_directories(hesper_cuda PRIVATE ${LEAN_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(hesper_cuda PRIVATE CUDA::cuda_driver)
else()
    message(STATUS "Hesper CUDA bridge: stub (no CUDA Toolkit)")
    # Compile as C++ rather than C: MSVC otherwise needs the experimental
    # /experimental:c11atomics flag for the <stdatomic.h> pulled in by
    # <lean/lean.h>. C++ has built-in atomics support and no such flag.
    add_library(hesper_cuda STATIC cuda_bridge_stub.cpp)
    target_include_directories(hesper_cuda PRIVATE ${LEAN_INCLUDE_DIR})
endif()

# Expose the decision to consumers via a marker file that lakefile.lean
# inspects to decide whether to add `-lcuda` to executable link lines.
file(WRITE "${CMAKE_BINARY_DIR}/hesper_cuda.found" "${HESPER_CUDA_FOUND}")
