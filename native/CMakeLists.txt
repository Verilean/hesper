cmake_minimum_required(VERSION 3.16)
project(HesperNative)

set(CMAKE_CXX_STANDARD 17)
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
