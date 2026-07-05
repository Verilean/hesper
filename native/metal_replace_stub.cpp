// metal_replace_stub.cpp — non-Apple stubs for the metal_replacer (Metal/MPS) FFI.
//
// native/metal_replace.mm is Objective-C++ (Metal / MetalPerformanceShaders) and only compiles on
// macOS. Its symbols are declared as `@[extern]` opaques in Hesper (Hesper/WebGPU/Device.lean) and
// may be referenced by any executable linked on Linux/Windows even though the code paths that call
// them are macOS-only (guarded at run time). Without these stubs the link fails with undefined
// symbols. Each stub returns a clean IO error so a stray call fails gracefully instead of crashing.
//
// Signatures MUST match Hesper/WebGPU/Device.lean exactly (C linkage → only the symbol name and the
// ABI arg list matter; parameter names are omitted since the bodies never use them).

#include <lean/lean.h>
#include <cstdint>

static inline lean_obj_res msl_stub() {
    return lean_io_result_mk_error(lean_mk_string(
        "metal_replace (Metal/MPS) is unavailable on non-Apple platforms"));
}

extern "C" {

lean_obj_res lean_hesper_msl_busy_read(lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_mtl_device_name(b_lean_obj_arg, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_mtl_buffer_probe(b_lean_obj_arg, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_occupancy_probe(b_lean_obj_arg, b_lean_obj_arg, lean_obj_res) { return msl_stub(); }
lean_obj_res lean_hesper_msl_concurrent_probe(b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t, lean_obj_res) { return msl_stub(); }
lean_obj_res lean_hesper_msl_bench_serial(b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_metal_dispatch_mul2(
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, uint32_t, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_mps_matmul_bench(
    b_lean_obj_arg, uint32_t, uint32_t, uint32_t, uint32_t, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_q4k_bench(
    b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_q4k_dispatch(
    b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_q8down_dispatch(
    b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_q5down_dispatch(
    b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    lean_obj_res) { return msl_stub(); }

lean_obj_res lean_hesper_msl_gateup_down_onecb(
    b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg,
    b_lean_obj_arg, b_lean_obj_arg, b_lean_obj_arg,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    lean_obj_res) { return msl_stub(); }

}  // extern "C"
