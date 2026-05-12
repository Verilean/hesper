// macOS stub for Hesper's CUDA FFI symbols.
//
// On Linux, `cuda_bridge.cpp` provides real implementations that link against
// the CUDA driver API. On macOS there is no CUDA, but every Lean module that
// transitively imports `Hesper.CUDA.FFI` references these symbols from its
// generated `.c.o.export`. Without a stub library, even non-CUDA executables
// (test-all, bitnet-complete, ...) fail to link.
//
// These stubs return `IO.Error.userError "CUDA not available on macOS"` and
// must never be invoked at runtime on macOS — they exist purely to satisfy
// the linker. Defined in plain C with unspecified-arg signatures so the
// platform ABI delivers caller arguments harmlessly into registers/stack we
// never read.

#include <lean/lean.h>

#ifdef __cplusplus
extern "C" {
#endif

static lean_obj_res hesper_cuda_unavailable(void) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("CUDA is not available on macOS")));
}

#define HESPER_CUDA_STUB(name) \
    lean_obj_res name() { return hesper_cuda_unavailable(); }

HESPER_CUDA_STUB(lean_hesper_cuda_init)
HESPER_CUDA_STUB(lean_hesper_cuda_device_count)
HESPER_CUDA_STUB(lean_hesper_cuda_device_get)
HESPER_CUDA_STUB(lean_hesper_cuda_device_name)
HESPER_CUDA_STUB(lean_hesper_cuda_compute_capability)
HESPER_CUDA_STUB(lean_hesper_cuda_total_mem)
HESPER_CUDA_STUB(lean_hesper_cuda_ctx_create)
HESPER_CUDA_STUB(lean_hesper_cuda_ctx_destroy)
HESPER_CUDA_STUB(lean_hesper_cuda_module_load_data)
HESPER_CUDA_STUB(lean_hesper_cuda_module_load_data_bytes)
HESPER_CUDA_STUB(lean_hesper_cuda_module_get_function)
HESPER_CUDA_STUB(lean_hesper_cuda_module_unload)
HESPER_CUDA_STUB(lean_hesper_cuda_func_set_max_dynamic_smem)
HESPER_CUDA_STUB(lean_hesper_fast_string_hash)
HESPER_CUDA_STUB(lean_hesper_mmap_file)
HESPER_CUDA_STUB(lean_hesper_munmap)
HESPER_CUDA_STUB(lean_hesper_mmap_slice_to_bytes)
HESPER_CUDA_STUB(lean_hesper_mmap_to_gpu)
HESPER_CUDA_STUB(lean_hesper_read_file_fast)
HESPER_CUDA_STUB(lean_hesper_mmap_open_persistent)
HESPER_CUDA_STUB(lean_hesper_mmap_size)
HESPER_CUDA_STUB(lean_hesper_mmap_slice_to_bytes_persistent)
HESPER_CUDA_STUB(lean_hesper_cuda_memcpy_htod_from_mmap)
HESPER_CUDA_STUB(lean_hesper_mmap_register_region)
HESPER_CUDA_STUB(lean_hesper_cuda_malloc)
HESPER_CUDA_STUB(lean_hesper_cuda_free)
HESPER_CUDA_STUB(lean_hesper_cuda_memcpy_htod)
HESPER_CUDA_STUB(lean_hesper_cuda_memcpy_dtoh)
HESPER_CUDA_STUB(lean_hesper_cuda_memset)
HESPER_CUDA_STUB(lean_hesper_cuda_launch_kernel)
HESPER_CUDA_STUB(lean_hesper_desc_reset)
HESPER_CUDA_STUB(lean_hesper_desc_register)
HESPER_CUDA_STUB(lean_hesper_desc_rebind)
HESPER_CUDA_STUB(lean_hesper_desc_launch)
HESPER_CUDA_STUB(lean_hesper_desc_launch_with_args)
HESPER_CUDA_STUB(lean_hesper_cuda_launch_kernel_raw)
HESPER_CUDA_STUB(lean_hesper_cuda_launch_kernel_raw_on_stream)
HESPER_CUDA_STUB(lean_hesper_cuda_set_l2_persist_limit)
HESPER_CUDA_STUB(lean_hesper_cuda_set_l2_access_window)
HESPER_CUDA_STUB(lean_hesper_cuda_reset_l2_persisting_cache)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_create)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_create_default)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_destroy)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_begin_capture)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_end_capture)
HESPER_CUDA_STUB(lean_hesper_cuda_graph_instantiate)
HESPER_CUDA_STUB(lean_hesper_cuda_graph_exec_destroy)
HESPER_CUDA_STUB(lean_hesper_cuda_graph_destroy)
HESPER_CUDA_STUB(lean_hesper_cuda_graph_launch)
HESPER_CUDA_STUB(lean_hesper_cuda_launch_kernel_on_stream)
HESPER_CUDA_STUB(lean_hesper_cuda_stream_synchronize)
HESPER_CUDA_STUB(lean_hesper_cuda_memcpy_htod_async)
HESPER_CUDA_STUB(lean_hesper_cuda_mem_alloc_host)
HESPER_CUDA_STUB(lean_hesper_cuda_mem_free_host)
HESPER_CUDA_STUB(lean_hesper_cuda_mem_alloc_host_mapped)
HESPER_CUDA_STUB(lean_hesper_cuda_read_pinned_u32)
HESPER_CUDA_STUB(lean_hesper_cuda_write_pinned)
HESPER_CUDA_STUB(lean_hesper_cuda_memcpy_htod_from_pinned)
HESPER_CUDA_STUB(lean_hesper_cuda_pinned_write_and_copy)

#ifdef __cplusplus
}
#endif
