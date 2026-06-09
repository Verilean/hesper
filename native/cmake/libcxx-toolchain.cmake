# CMake toolchain file: force every C++ target (including FetchContent /
# add_subdirectory sub-projects) onto Clang's libc++.
#
# Why this exists: Dawn pulls in Abseil, SPIRV-Tools, Vulkan-Headers, etc.
# via FetchContent.  Passing `-DCMAKE_CXX_FLAGS=-stdlib=libc++` to the
# top-level Dawn configure only sticks for Dawn's own targets — each
# FetchContent dep re-runs `project()` and resets its own flags.  A
# toolchain file is applied at the very start of `project()` *and*
# inherited by every nested project, so a single setting reaches every
# translation unit.
#
# Activate by passing `-DCMAKE_TOOLCHAIN_FILE=<path-to-this-file>` to
# the outer `cmake` invocation.
#
# Background: hesper's host (xeus-lean's `xlean`) is built by the Lean
# toolchain, which statically embeds libc++ (LLVM).  If Dawn ends up
# linked against libstdc++ (GCC) instead, the same process has two C++
# ABIs and `std::function`, `std::string`, etc. silently corrupt across
# the FFI boundary.  Symptom: SIGSEGV deep inside
# `Adapter::CreateDevice` on the very first `getDevice()` call from a
# notebook cell.

if(__hesper_libcxx_toolchain_loaded)
  return()
endif()
set(__hesper_libcxx_toolchain_loaded TRUE)

# Force Clang — `-stdlib=libc++` is a Clang-only flag.
if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER   clang   CACHE STRING "")
endif()
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++ CACHE STRING "")
endif()

# `*_INIT` variables seed CMAKE_<lang>_FLAGS in every (sub-)project at
# `project()` time, so even FetchContent_MakeAvailable'd deps inherit
# them.  Append rather than overwrite so a sub-project's own additions
# (e.g. `-DABSL_INTERNAL_AT_LEAST_CXX20=1`) still take effect.
set(CMAKE_CXX_FLAGS_INIT             "-stdlib=libc++"  CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT      "-stdlib=libc++ -lc++abi" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT   "-stdlib=libc++ -lc++abi" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_INIT   "-stdlib=libc++ -lc++abi" CACHE STRING "" FORCE)

# Disable GCC ABI tag — libc++ does not emit the `[abi:cxx11]` tag, and
# mixing tagged TUs with untagged ones in the same archive triggers
# linker errors on some Abseil revisions.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
