cmake_minimum_required(VERSION 3.16)
project(HesperSIMD CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch Google Highway
include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.1.0  # Stable release
)

# Highway configuration
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(highway)

# Find Lean include directory
if(NOT DEFINED LEAN_INCLUDE_DIR)
  execute_process(
    COMMAND lean --print-prefix
    OUTPUT_VARIABLE LEAN_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(LEAN_INCLUDE_DIR "${LEAN_PREFIX}/include")
endif()

message(STATUS "Lean include directory: ${LEAN_INCLUDE_DIR}")

# Create SIMD operations library
add_library(hesper_simd STATIC simd_ops_highway.cpp)

target_include_directories(hesper_simd PRIVATE
  ${LEAN_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/..  # For c_src/simd_ops_highway.h
)

target_link_libraries(hesper_simd PRIVATE
  hwy
)

# Enable optimizations
target_compile_options(hesper_simd PRIVATE
  -O3
  -fPIC
)

# Optional: Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hesper_simd PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled for multi-threading")
else()
  message(STATUS "OpenMP not found - single-threaded only")
endif()

message(STATUS "Highway SIMD library configured")
