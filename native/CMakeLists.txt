cmake_minimum_required(VERSION 3.16)
project(HesperNative)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# Build our native bridge library
add_library(hesper_native STATIC bridge.cpp glfw_bridge.cpp)

# Include Lean, Dawn, and GLFW headers (GLFW is in Dawn's third_party)
target_include_directories(hesper_native PRIVATE
  ${LEAN_INCLUDE_DIR}
  ${DAWN_SRC_DIR}/include
  ${DAWN_BUILD_DIR}/gen/include
  ${DAWN_SRC_DIR}/third_party/glfw/include
)

# Link Dawn GLFW utilities for surface creation
target_link_libraries(hesper_native PRIVATE
  ${DAWN_BUILD_DIR}/src/dawn/glfw/libdawn_glfw.a
)

# Note: GLFW and Dawn linking happens in the final executable via lakefile.lean

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
    add_library(hesper_cuda STATIC cuda_bridge_stub.c)
    target_include_directories(hesper_cuda PRIVATE ${LEAN_INCLUDE_DIR})
    # <lean/lean.h> pulls in <stdatomic.h>; MSVC requires /std:c11 for those.
    set_target_properties(hesper_cuda PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
endif()

# Expose the decision to consumers via a marker file that lakefile.lean
# inspects to decide whether to add `-lcuda` to executable link lines.
file(WRITE "${CMAKE_BINARY_DIR}/hesper_cuda.found" "${HESPER_CUDA_FOUND}")
