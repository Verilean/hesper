/**
 * Hesper SIMD Operations - Google Highway Implementation
 */

#include <lean/lean.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

// Highway foreach_target pattern
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "c_src/simd_ops_highway.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Float64 vector addition
HWY_ATTR void AddFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                         double* HWY_RESTRICT out, size_t count) {
  HWY_FULL(double) d;

  size_t i = 0;
  const size_t N = hn::Lanes(d);
  for (; i + N <= count; i += N) {
    hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
  }

  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

// Float32 vector addition
HWY_ATTR void AddFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                         float* HWY_RESTRICT out, size_t count) {
  HWY_FULL(float) d;

  size_t i = 0;
  const size_t N = hn::Lanes(d);
  for (; i + N <= count; i += N) {
    hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
  }

  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

// Float16 vector addition
HWY_ATTR void AddFloat16(const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
                         uint16_t* HWY_RESTRICT out, size_t count) {
  HWY_FULL(float) df;
  const hn::Rebind<uint16_t, decltype(df)> du16;
  const hn::Rebind<hwy::float16_t, decltype(df)> df16;

  size_t i = 0;
  const size_t N = hn::Lanes(df);

  // SIMD loop: process N elements at a time
  for (; i + N <= count; i += N) {
    // Load F16 as uint16 then BitCast to float16_t
    const auto va_u16 = hn::LoadU(du16, a + i);
    const auto vb_u16 = hn::LoadU(du16, b + i);
    const auto va_f16 = hn::BitCast(df16, va_u16);
    const auto vb_f16 = hn::BitCast(df16, vb_u16);

    // Promote to F32 (Highway handles native conversion)
    const auto va_f32 = hn::PromoteTo(df, va_f16);
    const auto vb_f32 = hn::PromoteTo(df, vb_f16);

    // Add
    const auto sum_f32 = hn::Add(va_f32, vb_f32);

    // Demote back to F16 then BitCast to uint16
    const auto sum_f16 = hn::DemoteTo(df16, sum_f32);
    const auto sum_u16 = hn::BitCast(du16, sum_f16);
    hn::StoreU(sum_u16, du16, out + i);
  }

  // Scalar tail
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

// Export the functions
HWY_EXPORT(AddFloat64);
HWY_EXPORT(AddFloat32);
HWY_EXPORT(AddFloat16);

namespace hesper {

const char* GetTargetName() {
  return hwy::TargetName(HWY_TARGET);
}

}  // namespace hesper

// Software FP16 conversion
static inline uint16_t float_to_half(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exponent = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent == 0xFF) {
        return sign | 0x7C00 | (mantissa != 0 ? 0x0200 : 0);
    }
    if (exponent == 0) {
        return sign;
    }

    int32_t new_exp = exponent - 127 + 15;
    if (new_exp >= 31) {
        return sign | 0x7C00;
    }
    if (new_exp <= 0) {
        return sign;
    }

    uint32_t new_mantissa = (mantissa + 0x1000) >> 13;
    return sign | (new_exp << 10) | (new_mantissa & 0x3FF);
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            return *(float*)&sign;
        }
        return 0.0f;
    }
    if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&result;
    }

    uint32_t new_exp = (exponent - 15 + 127) << 23;
    uint32_t new_mantissa = mantissa << 13;
    uint32_t result = sign | new_exp | new_mantissa;
    return *(float*)&result;
}

extern "C" {

// Float64 Operations
lean_obj_res lean_simd_add_f64(b_lean_obj_arg a_array, b_lean_obj_arg b_array) {
    size_t size = lean_sarray_size(a_array);
    const double* a_data = lean_float_array_cptr(a_array);
    const double* b_data = lean_float_array_cptr(b_array);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* c_data = lean_float_array_cptr(result);

    HWY_DYNAMIC_DISPATCH(AddFloat64)(a_data, b_data, c_data, size);

    return result;
}

// Float32 Operations
lean_obj_res lean_simd_add_f32(b_lean_obj_arg a_bytes, b_lean_obj_arg b_bytes) {
    size_t byte_size_a = lean_sarray_size(a_bytes);
    size_t byte_size_b = lean_sarray_size(b_bytes);

    if (byte_size_a != byte_size_b || byte_size_a % 4 != 0) {
        return lean_alloc_sarray(1, 0, 0);
    }

    size_t size = byte_size_a / 4;
    const float* a_data = (const float*)lean_sarray_cptr(a_bytes);
    const float* b_data = (const float*)lean_sarray_cptr(b_bytes);

    lean_object* result = lean_alloc_sarray(1, byte_size_a, byte_size_a);
    float* c_data = (float*)lean_sarray_cptr(result);

    HWY_DYNAMIC_DISPATCH(AddFloat32)(a_data, b_data, c_data, size);

    return result;
}

lean_obj_res lean_f32_from_f64_array(b_lean_obj_arg f64_array) {
    size_t size = lean_sarray_size(f64_array);
    const double* f64_data = lean_float_array_cptr(f64_array);

    size_t byte_size = size * sizeof(float);
    lean_object* result_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    float* f32_data = (float*)lean_sarray_cptr(result_bytes);

    for (size_t i = 0; i < size; i++) {
        f32_data[i] = (float)f64_data[i];
    }

    lean_object* f32_struct = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(f32_struct, 0, result_bytes);
    lean_ctor_set(f32_struct, 1, lean_box(size));
    return f32_struct;
}

lean_obj_res lean_f32_to_f64_array(b_lean_obj_arg f32_struct) {
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));
    const float* f32_data = (const float*)lean_sarray_cptr(f32_bytes);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* f64_data = lean_float_array_cptr(result);

    for (size_t i = 0; i < size; i++) {
        f64_data[i] = (double)f32_data[i];
    }

    return result;
}

lean_obj_res lean_f32_get(b_lean_obj_arg f32_struct, size_t index) {
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));

    if (index >= size) {
        return lean_box_float(0.0);
    }

    const float* data = (const float*)lean_sarray_cptr(f32_bytes);
    return lean_box_float((double)data[index]);
}

lean_obj_res lean_f32_set(lean_obj_arg f32_struct, size_t index, double value) {
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));

    if (index >= size) {
        return f32_struct;
    }

    size_t byte_size = size * sizeof(float);
    lean_object* new_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    float* dest = (float*)lean_sarray_cptr(new_bytes);
    const float* src = (const float*)lean_sarray_cptr(f32_bytes);

    memcpy(dest, src, byte_size);
    dest[index] = (float)value;

    lean_object* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, new_bytes);
    lean_ctor_set(result, 1, lean_box(size));

    lean_dec(f32_struct);
    return result;
}

// Float16 Operations
lean_obj_res lean_f16_hw_check(lean_obj_arg /* unit */) {
    return lean_io_result_mk_ok(lean_box(1));
}

lean_obj_res lean_simd_add_f16(b_lean_obj_arg a_bytes, b_lean_obj_arg b_bytes) {
    size_t byte_size_a = lean_sarray_size(a_bytes);
    size_t byte_size_b = lean_sarray_size(b_bytes);

    if (byte_size_a != byte_size_b || byte_size_a % 2 != 0) {
        lean_object* err_msg = lean_mk_string("Float16 array size mismatch");
        return lean_io_result_mk_error(lean_mk_io_user_error(err_msg));
    }

    size_t size = byte_size_a / 2;
    const uint16_t* a_data = (const uint16_t*)lean_sarray_cptr(a_bytes);
    const uint16_t* b_data = (const uint16_t*)lean_sarray_cptr(b_bytes);

    lean_object* result = lean_alloc_sarray(1, byte_size_a, byte_size_a);
    uint16_t* c_data = (uint16_t*)lean_sarray_cptr(result);

    HWY_DYNAMIC_DISPATCH(AddFloat16)(a_data, b_data, c_data, size);

    return lean_io_result_mk_ok(result);
}

lean_obj_res lean_f16_from_f64_array(b_lean_obj_arg f64_array) {
    size_t size = lean_sarray_size(f64_array);
    const double* f64_data = lean_float_array_cptr(f64_array);

    size_t byte_size = size * sizeof(uint16_t);
    lean_object* result_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    uint16_t* f16_data = (uint16_t*)lean_sarray_cptr(result_bytes);

    for (size_t i = 0; i < size; i++) {
        f16_data[i] = float_to_half((float)f64_data[i]);
    }

    lean_object* f16_struct = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(f16_struct, 0, result_bytes);
    lean_ctor_set(f16_struct, 1, lean_box(size));
    return lean_io_result_mk_ok(f16_struct);
}

lean_obj_res lean_f16_to_f64_array(b_lean_obj_arg f16_struct) {
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));
    const uint16_t* f16_data = (const uint16_t*)lean_sarray_cptr(f16_bytes);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* f64_data = lean_float_array_cptr(result);

    for (size_t i = 0; i < size; i++) {
        f64_data[i] = (double)half_to_float(f16_data[i]);
    }

    return lean_io_result_mk_ok(result);
}

lean_obj_res lean_f16_get(b_lean_obj_arg f16_struct, size_t index) {
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));

    if (index >= size) {
        return lean_io_result_mk_ok(lean_box_float(0.0));
    }

    const uint16_t* data = (const uint16_t*)lean_sarray_cptr(f16_bytes);
    double result = (double)half_to_float(data[index]);
    return lean_io_result_mk_ok(lean_box_float(result));
}

lean_obj_res lean_f16_set(lean_obj_arg f16_struct, size_t index, double value) {
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));

    if (index >= size) {
        return lean_io_result_mk_ok(f16_struct);
    }

    size_t byte_size = size * sizeof(uint16_t);
    lean_object* new_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    uint16_t* dest = (uint16_t*)lean_sarray_cptr(new_bytes);
    const uint16_t* src = (const uint16_t*)lean_sarray_cptr(f16_bytes);

    memcpy(dest, src, byte_size);
    dest[index] = float_to_half((float)value);

    lean_object* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, new_bytes);
    lean_ctor_set(result, 1, lean_box(size));

    lean_dec(f16_struct);
    return lean_io_result_mk_ok(result);
}

// Backend Information
lean_obj_res lean_simd_backend_info(b_lean_obj_arg /* unit */) {
    const char* target = hesper::GetTargetName();
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "Google Highway (%s) - Multi-precision SIMD", target);
    return lean_io_result_mk_ok(lean_mk_string(buffer));
}

}  // extern "C"
#endif  // HWY_ONCE
