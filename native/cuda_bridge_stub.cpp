// CUDA bridge stub for non-Linux platforms (macOS, Windows).
// Provides the same `lean_hesper_cuda_*` symbols as cuda_bridge.cpp so
// that Lean exes link successfully on platforms without a CUDA driver,
// but every entry point fails with an IO error at runtime.
//
// The Lean side (Hesper.CUDA.FFI) declares these via `@[extern "..."]`,
// so the symbols MUST exist at link time even when CUDA is unavailable.
//
// This file is selected by native/CMakeLists.txt when CUDAToolkit isn't
// found OR the host is non-Linux.

#include <lean/lean.h>
#include <cstdio>
#include <cstring>

static lean_obj_res cuda_unavailable(const char* func) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s: CUDA backend is not available on this build "
             "(macOS / Windows / no CUDA SDK).", func);
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
}

#define CUDA_STUB(name) \
    extern "C" lean_obj_res name() { return cuda_unavailable(#name); }

// Variadic-arg stubs: Lean FFI passes its arguments by value. Since we
// only need to satisfy the linker (the call always errors immediately),
// declare each entry as accepting any arity via varargs.
//
// Simpler approach: define each symbol explicitly with a signature that
// matches at least one valid arity. The actual call is never reached
// because Lean FFI invokes the symbol as declared in `@[extern]`, but
// our function returns immediately. Use `...` so any signature works.

// Note: C variadic functions can't appear without at least one named
// parameter. We give each one a single `lean_object*` (which always
// works as the discarded "world token"-ish slot), then `...` for the
// rest. This is enough for the linker.

#undef CUDA_STUB
#define CUDA_STUB(name) \
    extern "C" lean_obj_res name(lean_object* /*x*/, ...) { \
        return cuda_unavailable(#name); \
    }

CUDA_STUB(lean_hesper_cuda_init)
CUDA_STUB(lean_hesper_cuda_device_count)
CUDA_STUB(lean_hesper_cuda_device_get)
CUDA_STUB(lean_hesper_cuda_device_name)
CUDA_STUB(lean_hesper_cuda_compute_capability)
CUDA_STUB(lean_hesper_cuda_total_mem)
CUDA_STUB(lean_hesper_cuda_ctx_create)
CUDA_STUB(lean_hesper_cuda_ctx_destroy)
CUDA_STUB(lean_hesper_cuda_module_load_data)
CUDA_STUB(lean_hesper_cuda_module_load_data_bytes)
CUDA_STUB(lean_hesper_cuda_module_get_function)
CUDA_STUB(lean_hesper_cuda_module_unload)
CUDA_STUB(lean_hesper_cuda_func_set_max_dynamic_smem)
CUDA_STUB(lean_hesper_cuda_malloc)
CUDA_STUB(lean_hesper_cuda_free)
CUDA_STUB(lean_hesper_cuda_memcpy_htod)
CUDA_STUB(lean_hesper_cuda_memcpy_dtoh)
CUDA_STUB(lean_hesper_cuda_memset)
CUDA_STUB(lean_hesper_cuda_set_l2_persist_limit)
CUDA_STUB(lean_hesper_cuda_set_l2_access_window)
CUDA_STUB(lean_hesper_cuda_reset_l2_persisting_cache)
CUDA_STUB(lean_hesper_cuda_launch_kernel)
CUDA_STUB(lean_hesper_cuda_launch_kernel_raw)
CUDA_STUB(lean_hesper_cuda_launch_kernel_raw_on_stream)
CUDA_STUB(lean_hesper_fast_string_hash)
CUDA_STUB(lean_hesper_mmap_file)
CUDA_STUB(lean_hesper_munmap)
CUDA_STUB(lean_hesper_mmap_slice_to_bytes)
CUDA_STUB(lean_hesper_mmap_to_gpu)
CUDA_STUB(lean_hesper_mmap_open_persistent)
CUDA_STUB(lean_hesper_mmap_size)
CUDA_STUB(lean_hesper_mmap_register_region)
CUDA_STUB(lean_hesper_mmap_slice_to_bytes_persistent)
CUDA_STUB(lean_hesper_cuda_memcpy_htod_from_mmap)
CUDA_STUB(lean_hesper_read_file_fast)
CUDA_STUB(lean_hesper_cuda_stream_create)
CUDA_STUB(lean_hesper_cuda_stream_create_default)
CUDA_STUB(lean_hesper_cuda_stream_destroy)
CUDA_STUB(lean_hesper_cuda_stream_begin_capture)
CUDA_STUB(lean_hesper_cuda_stream_end_capture)
CUDA_STUB(lean_hesper_cuda_graph_instantiate)
CUDA_STUB(lean_hesper_cuda_graph_exec_destroy)
CUDA_STUB(lean_hesper_cuda_graph_destroy)
CUDA_STUB(lean_hesper_cuda_graph_launch)
CUDA_STUB(lean_hesper_cuda_stream_synchronize)
CUDA_STUB(lean_hesper_cuda_launch_kernel_on_stream)
CUDA_STUB(lean_hesper_cuda_memcpy_htod_async)
CUDA_STUB(lean_hesper_cuda_mem_alloc_host)
CUDA_STUB(lean_hesper_cuda_mem_alloc_host_mapped)
CUDA_STUB(lean_hesper_cuda_read_pinned_u32)
CUDA_STUB(lean_hesper_cuda_mem_free_host)
CUDA_STUB(lean_hesper_cuda_write_pinned)
CUDA_STUB(lean_hesper_cuda_memcpy_htod_from_pinned)
CUDA_STUB(lean_hesper_cuda_pinned_write_and_copy)
CUDA_STUB(lean_hesper_desc_reset)
CUDA_STUB(lean_hesper_desc_register)
CUDA_STUB(lean_hesper_desc_rebind)
CUDA_STUB(lean_hesper_desc_launch)
CUDA_STUB(lean_hesper_desc_launch_with_args)
