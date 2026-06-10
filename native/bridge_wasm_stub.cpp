// bridge_wasm_stub.cpp
//
// Placeholder for the Emscripten / emdawnwebgpu port of bridge.cpp.
// The full native bridge.cpp depends on Dawn's `dawn::native::Instance`
// (the C++ object that wraps the platform backends — Vulkan / Metal /
// D3D12); none of that compiles under Emscripten because emdawnwebgpu
// exposes only the C `wgpu*` API forwarded to `navigator.gpu`.
//
// In this M1 build we ship a static archive (libhesper_wasm.a) whose
// only job is to satisfy the linker — every Lean `@[extern]` declaration
// resolves to a stub that returns an `IO.Error` (or just `0`).  This is
// enough to:
//
//   1. produce a `libhesper_wasm.a` artifact for downstream wiring
//      (xeus-lean's wasm CMake will whole-archive this into xlean),
//   2. confirm in CI that the EMSCRIPTEN branch of native/CMakeLists.txt
//      stays buildable as we iterate, and
//   3. let pure-Lean code that merely *references* `Hesper.init` &c.
//      link in xlean (even if calling them at run time will fail).
//
// The real bridge — rewritten to call emdawnwebgpu's `wgpu*` API
// directly, drop `dawn::native::*`, and pump `wgpuInstanceProcessEvents`
// between buffer-map ticks — lands in a follow-up.  See
// `docs/research/wasm-webgpu-plan.md` (M2 / M3).
//
// Why a separate file:
//   The native bridge.cpp depends transitively on `dawn/native/...`
//   headers and uses `dawn::native::Instance*` in dozens of places.
//   Guarding every single occurrence with `#ifdef __EMSCRIPTEN__` would
//   make the file unreadable.  Splitting the wasm port off cleanly lets
//   each side stay focused.

#include <lean/lean.h>
#include <cstdio>

extern "C" {

// -----------------------------------------------------------------------------
// IO.Error builder
// -----------------------------------------------------------------------------

// Wrap a C string into an Except.error/IO.Error result that matches the
// shape `Hesper.init` and friends return on the native side.  We use
// `lean_io_result_mk_error` from Lean's runtime to stay ABI-compatible.
static lean_obj_res hesper_wasm_unsupported(const char* fn_name) {
    std::fprintf(
        stderr,
        "[hesper-wasm] %s called, but the Wasm WebGPU bridge is not yet "
        "wired up.  See docs/research/wasm-webgpu-plan.md (M2/M3).\n",
        fn_name);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string(
                "Hesper Wasm bridge: real WebGPU FFI not wired yet")));
}

// Every Lean @[extern "lean_hesper_*"] declaration that the Hesper.WebGPU
// modules expose at link time gets one stub here.  Keep this list in sync
// with `native/bridge.cpp`'s `LEAN_EXPORT` set as we iterate.  Adding a
// new extern on the Lean side without listing it here will surface as an
// unresolved symbol at xlean link time — which is exactly the signal we
// want.

#define HESPER_WASM_STUB(name)                                           \
    LEAN_EXPORT lean_obj_res name(lean_obj_arg /* args... */) {          \
        return hesper_wasm_unsupported(#name);                           \
    }

// `lean_hesper_init` returns IO Instance — stub returns an IO error.
HESPER_WASM_STUB(lean_hesper_init)

// All of the device / buffer / pipeline / dispatch entrypoints fail the
// same way.  Their exact signatures vary, but Lean only checks signature
// equality at extern-resolution time when invoked, not at link time, so
// declaring them with a generic `lean_obj_arg` is fine for the linker.
HESPER_WASM_STUB(lean_hesper_get_device)
HESPER_WASM_STUB(lean_hesper_get_device_with_features)
HESPER_WASM_STUB(lean_hesper_get_device_by_index)
HESPER_WASM_STUB(lean_hesper_get_adapter_count)
HESPER_WASM_STUB(lean_hesper_get_adapter_info)
HESPER_WASM_STUB(lean_hesper_release_device)
HESPER_WASM_STUB(lean_hesper_device_tick)
HESPER_WASM_STUB(lean_hesper_device_wait)
HESPER_WASM_STUB(lean_hesper_device_has_subgroups)
HESPER_WASM_STUB(lean_hesper_device_has_subgroup_matrix)
HESPER_WASM_STUB(lean_hesper_device_has_shader_f16)
HESPER_WASM_STUB(lean_hesper_create_buffer)
HESPER_WASM_STUB(lean_hesper_write_buffer)
HESPER_WASM_STUB(lean_hesper_map_buffer_read)
HESPER_WASM_STUB(lean_hesper_unmap_buffer)
HESPER_WASM_STUB(lean_hesper_buffer_id)
HESPER_WASM_STUB(lean_hesper_hash_buffer_array)
HESPER_WASM_STUB(lean_hesper_create_shader_module)
HESPER_WASM_STUB(lean_hesper_create_bind_group_layout)
HESPER_WASM_STUB(lean_hesper_create_bind_group)
HESPER_WASM_STUB(lean_hesper_create_compute_pipeline)
HESPER_WASM_STUB(lean_hesper_dispatch_compute)

// Non-IO helpers (no IO.Error wrapper needed).  Return zero / empty.
LEAN_EXPORT uint8_t lean_hesper_set_verbose(uint8_t /* verbose */,
                                             lean_obj_arg /* unit */) {
    return 0;
}
LEAN_EXPORT lean_obj_res lean_hesper_get_time_ns(lean_obj_arg /* unit */) {
    return lean_io_result_mk_ok(lean_box_uint64(0));
}
LEAN_EXPORT double lean_hesper_bytes_to_float64(lean_obj_arg /* bytes */,
                                                 uint32_t /* offset */) {
    return 0.0;
}

#undef HESPER_WASM_STUB

}  // extern "C"
