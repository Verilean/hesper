#include <lean/lean.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

// OpenMP support
#ifdef _OPENMP
    #include <omp.h>
#endif

// Architecture detection for SIMD intrinsics
#if defined(__AVX2__) && defined(__x86_64__)
    #define USE_AVX2
    #include <immintrin.h>
#elif defined(__ARM_NEON)
    #define USE_NEON
    #include <arm_neon.h>
#endif

// FP16 support detection
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define USE_NEON_FP16
#elif defined(__AVX2__) && defined(__F16C__)
    #define USE_F16C
#endif

extern "C" {

// ============================================================================
// Float64 (Double) Operations
// ============================================================================

/**
 * SIMD-optimized Float64 vector addition with OpenMP: c = a + b
 */
lean_obj_res lean_simd_add_f64(b_lean_obj_arg a_array, b_lean_obj_arg b_array) {
    size_t size_a = lean_sarray_size(a_array);
    size_t size_b = lean_sarray_size(b_array);

    if (size_a != size_b) {
        return lean_alloc_sarray(sizeof(double), 0, 0);
    }

    size_t size = size_a;
    const double* a_data = lean_float_array_cptr(a_array);
    const double* b_data = lean_float_array_cptr(b_array);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* c_data = lean_float_array_cptr(result);

#ifdef _OPENMP
    // Use OpenMP for large arrays (overhead not worth it for small arrays)
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
        return result;
    }
#endif

    size_t i = 0;

#ifdef USE_AVX2
    const size_t simd_width = 4;
    const size_t simd_end = (size / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&a_data[i]);
        __m256d vb = _mm256_loadu_pd(&b_data[i]);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&c_data[i], vc);
    }

#elif defined(USE_NEON)
    const size_t simd_width = 2;
    const size_t simd_end = (size / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&a_data[i]);
        float64x2_t vb = vld1q_f64(&b_data[i]);
        float64x2_t vc = vaddq_f64(va, vb);
        vst1q_f64(&c_data[i], vc);
    }
#endif

    // Scalar tail
    for (; i < size; i++) {
        c_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

// ============================================================================
// Float32 Operations
// ============================================================================

/**
 * SIMD-optimized Float32 vector addition with OpenMP: c = a + b
 *
 * Operates directly on raw ByteArrays - NO type conversion overhead.
 * Takes raw ByteArrays, returns raw ByteArray.
 */
lean_obj_res lean_simd_add_f32(b_lean_obj_arg a_bytes, b_lean_obj_arg b_bytes) {
    // Work directly with ByteArrays (no structure wrapping at this level)
    size_t byte_size_a = lean_sarray_size(a_bytes);
    size_t byte_size_b = lean_sarray_size(b_bytes);

    if (byte_size_a != byte_size_b || byte_size_a % 4 != 0) {
        return lean_alloc_sarray(1, 0, 0);  // Empty ByteArray
    }

    size_t size = byte_size_a / 4;  // Number of float32 elements
    const float* a_data = (const float*)lean_sarray_cptr(a_bytes);
    const float* b_data = (const float*)lean_sarray_cptr(b_bytes);

    lean_object* result = lean_alloc_sarray(1, byte_size_a, byte_size_a);
    float* c_data = (float*)lean_sarray_cptr(result);

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
        return result;
    }
#endif

    size_t i = 0;

#ifdef USE_AVX2
    const size_t simd_width = 8;  // 8 floats per operation
    const size_t simd_end = (size / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(&a_data[i]);
        __m256 vb = _mm256_loadu_ps(&b_data[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c_data[i], vc);
    }

#elif defined(USE_NEON)
    const size_t simd_width = 4;  // 4 floats per operation
    const size_t simd_end = (size / simd_width) * simd_width;

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(&a_data[i]);
        float32x4_t vb = vld1q_f32(&b_data[i]);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(&c_data[i], vc);
    }
#endif

    // Scalar tail
    for (; i < size; i++) {
        c_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

/**
 * Convert Float64 array to Float32Array (explicit conversion only)
 * Returns: Float32Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f32_from_f64_array(b_lean_obj_arg f64_array) {
    size_t size = lean_sarray_size(f64_array);
    const double* f64_data = lean_float_array_cptr(f64_array);

    size_t byte_size = size * sizeof(float);
    lean_object* result_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    float* f32_data = (float*)lean_sarray_cptr(result_bytes);

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            f32_data[i] = (float)f64_data[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < size; i++) {
            f32_data[i] = (float)f64_data[i];
        }
    }

    // Return as Float32Array structure { data : ByteArray, numElements : Nat }
    lean_object* f32_struct = lean_alloc_ctor(0, 2, 0);  // 2 fields
    lean_ctor_set(f32_struct, 0, result_bytes);          // field 0: data
    lean_ctor_set(f32_struct, 1, lean_box(size));        // field 1: numElements
    return f32_struct;
}

/**
 * Convert Float32Array to Float64 array (explicit conversion only)
 * Input: Float32Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f32_to_f64_array(b_lean_obj_arg f32_struct) {
    // Extract fields from Float32Array structure
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));          // numElements
    const float* f32_data = (const float*)lean_sarray_cptr(f32_bytes);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* f64_data = lean_float_array_cptr(result);

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            f64_data[i] = (double)f32_data[i];
        }
        return result;
    }
#endif

    for (size_t i = 0; i < size; i++) {
        f64_data[i] = (double)f32_data[i];
    }

    return result;
}

/**
 * Get Float32 element at index (returns as Float64)
 * Input: Float32Array { data : ByteArray, numElements : Nat }
 */
double lean_f32_get(b_lean_obj_arg f32_struct, size_t index) {
    // Extract fields from Float32Array structure
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));          // numElements

    if (index >= size) return 0.0;

    const float* data = (const float*)lean_sarray_cptr(f32_bytes);
    return (double)data[index];
}

/**
 * Set Float32 element at index (from Float64)
 * Input/Output: Float32Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f32_set(lean_obj_arg f32_struct, size_t index, double value) {
    // Extract fields from Float32Array structure
    lean_object* f32_bytes = lean_ctor_get(f32_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f32_struct, 1));          // numElements

    if (index >= size) {
        // Return original array unchanged
        return f32_struct;
    }

    // Copy-on-write: create new ByteArray
    size_t byte_size = size * sizeof(float);
    lean_object* new_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    float* dest = (float*)lean_sarray_cptr(new_bytes);
    const float* src = (const float*)lean_sarray_cptr(f32_bytes);

    memcpy(dest, src, byte_size);
    dest[index] = (float)value;

    // Create new Float32Array structure
    lean_object* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, new_bytes);
    lean_ctor_set(result, 1, lean_box(size));

    lean_dec(f32_struct);
    return result;
}

// ============================================================================
// Float16 (Half) Operations
// ============================================================================

/**
 * Check if FP16 hardware support is available
 */
lean_obj_res lean_f16_hw_check(lean_obj_arg /* unit */) {
#if defined(USE_NEON_FP16) || defined(USE_F16C)
    return lean_io_result_mk_ok(lean_box(1));  // true
#else
    return lean_io_result_mk_ok(lean_box(0));  // false
#endif
}

// Software FP16 conversion (only used for conversions, not arithmetic)
// Proper IEEE 754 half-precision conversion with correct exponent bias handling
static inline uint16_t float_to_half_sw(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exponent = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;

    // Handle special cases
    if (exponent == 0xFF) {  // Inf or NaN
        return sign | 0x7C00 | (mantissa != 0 ? 0x0200 : 0);
    }
    if (exponent == 0) {  // Zero or denormal
        return sign;
    }

    // Adjust exponent from FP32 bias (127) to FP16 bias (15)
    int32_t new_exp = exponent - 127 + 15;

    if (new_exp >= 31) {  // Overflow to infinity
        return sign | 0x7C00;
    }
    if (new_exp <= 0) {  // Underflow to zero
        return sign;
    }

    // Round mantissa from 23 bits to 10 bits
    uint32_t new_mantissa = (mantissa + 0x1000) >> 13;

    return sign | (new_exp << 10) | (new_mantissa & 0x3FF);
}

static inline float half_to_float_sw(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    // Handle special cases
    if (exponent == 0) {  // Zero or denormal
        if (mantissa == 0) {
            return *(float*)&sign;  // Zero
        }
        // Denormal - not implementing for simplicity
        return 0.0f;
    }
    if (exponent == 31) {  // Inf or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&result;
    }

    // Adjust exponent from FP16 bias (15) to FP32 bias (127)
    uint32_t new_exp = (exponent - 15 + 127) << 23;
    uint32_t new_mantissa = mantissa << 13;
    uint32_t result = sign | new_exp | new_mantissa;

    return *(float*)&result;
}

/**
 * SIMD-optimized Float16 vector addition: c = a + b
 *
 * Operates directly on raw ByteArrays - NO type conversion overhead.
 * Takes raw ByteArrays, returns IO ByteArray.
 * REQUIRES hardware FP16 support - returns error otherwise.
 */
lean_obj_res lean_simd_add_f16(b_lean_obj_arg a_bytes, b_lean_obj_arg b_bytes) {
#if !defined(USE_NEON_FP16) && !defined(USE_F16C)
    // No hardware support - return error via IO exception
    lean_object* err_msg = lean_mk_string(
        "FP16 hardware not available. "
        "Requires ARMv8.2-A FP16 or x86_64 F16C extension."
    );
    return lean_io_result_mk_error(lean_mk_io_user_error(err_msg));
#else
    // Work directly with ByteArrays (no structure wrapping at this level)
    size_t byte_size_a = lean_sarray_size(a_bytes);
    size_t byte_size_b = lean_sarray_size(b_bytes);

    if (byte_size_a != byte_size_b || byte_size_a % 2 != 0) {
        lean_object* err_msg = lean_mk_string("FP16 array size mismatch or misalignment");
        return lean_io_result_mk_error(lean_mk_io_user_error(err_msg));
    }

    size_t size = byte_size_a / 2;  // Number of float16 elements
    const uint16_t* a_data = (const uint16_t*)lean_sarray_cptr(a_bytes);
    const uint16_t* b_data = (const uint16_t*)lean_sarray_cptr(b_bytes);

    lean_object* result = lean_alloc_sarray(1, byte_size_a, byte_size_a);
    uint16_t* c_data = (uint16_t*)lean_sarray_cptr(result);

    size_t i = 0;

#ifdef USE_NEON_FP16
    // Native FP16 arithmetic on ARM
    const size_t simd_width = 8;  // 8 halfs per operation
    const size_t simd_end = (size / simd_width) * simd_width;

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t j = 0; j < simd_end; j += simd_width) {
            float16x8_t va = vld1q_f16((const __fp16*)&a_data[j]);
            float16x8_t vb = vld1q_f16((const __fp16*)&b_data[j]);
            float16x8_t vc = vaddq_f16(va, vb);
            vst1q_f16((__fp16*)&c_data[j], vc);
        }
        i = simd_end;
    } else
#endif
    {
        for (; i < simd_end; i += simd_width) {
            float16x8_t va = vld1q_f16((const __fp16*)&a_data[i]);
            float16x8_t vb = vld1q_f16((const __fp16*)&b_data[i]);
            float16x8_t vc = vaddq_f16(va, vb);
            vst1q_f16((__fp16*)&c_data[i], vc);
        }
    }

    // Scalar tail with native FP16
    for (; i < size; i++) {
        __fp16 a_h = *(const __fp16*)&a_data[i];
        __fp16 b_h = *(const __fp16*)&b_data[i];
        __fp16 c_h = a_h + b_h;
        c_data[i] = *(uint16_t*)&c_h;
    }

#elif defined(USE_F16C)
    // AVX2 + F16C: hardware conversion + FP32 arithmetic
    const size_t simd_width = 8;
    const size_t simd_end = (size / simd_width) * simd_width;

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t j = 0; j < simd_end; j += simd_width) {
            __m128i va_h = _mm_loadu_si128((__m128i*)&a_data[j]);
            __m128i vb_h = _mm_loadu_si128((__m128i*)&b_data[j]);
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vc_f = _mm256_add_ps(va_f, vb_f);
            __m128i vc_h = _mm256_cvtps_ph(vc_f, 0);
            _mm_storeu_si128((__m128i*)&c_data[j], vc_h);
        }
        i = simd_end;
    } else
#endif
    {
        for (; i < simd_end; i += simd_width) {
            __m128i va_h = _mm_loadu_si128((__m128i*)&a_data[i]);
            __m128i vb_h = _mm_loadu_si128((__m128i*)&b_data[i]);
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vc_f = _mm256_add_ps(va_f, vb_f);
            __m128i vc_h = _mm256_cvtps_ph(vc_f, 0);
            _mm_storeu_si128((__m128i*)&c_data[i], vc_h);
        }
    }

    // Scalar tail with F16C
    for (; i < size; i++) {
        __m128i va_h = _mm_set1_epi16(a_data[i]);
        __m128i vb_h = _mm_set1_epi16(b_data[i]);
        __m128 va_f = _mm_cvtph_ps(va_h);
        __m128 vb_f = _mm_cvtph_ps(vb_h);
        __m128 vc_f = _mm_add_ps(va_f, vb_f);
        __m128i vc_h = _mm_cvtps_ph(vc_f, 0);
        c_data[i] = _mm_extract_epi16(vc_h, 0);
    }
#endif

    return lean_io_result_mk_ok(result);
#endif
}

/**
 * Convert Float64 array to Float16Array (explicit conversion only)
 * Returns: IO (Float16Array { data : ByteArray, numElements : Nat })
 */
lean_obj_res lean_f16_from_f64_array(b_lean_obj_arg f64_array) {
    size_t size = lean_sarray_size(f64_array);
    const double* f64_data = lean_float_array_cptr(f64_array);

    size_t byte_size = size * sizeof(uint16_t);
    lean_object* result_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    uint16_t* f16_data = (uint16_t*)lean_sarray_cptr(result_bytes);

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            f16_data[i] = float_to_half_sw((float)f64_data[i]);
        }
    } else
#endif
    {
        for (size_t i = 0; i < size; i++) {
            f16_data[i] = float_to_half_sw((float)f64_data[i]);
        }
    }

    // Return as Float16Array structure { data : ByteArray, numElements : Nat } wrapped in IO
    lean_object* f16_struct = lean_alloc_ctor(0, 2, 0);  // 2 fields
    lean_ctor_set(f16_struct, 0, result_bytes);          // field 0: data
    lean_ctor_set(f16_struct, 1, lean_box(size));        // field 1: numElements
    return lean_io_result_mk_ok(f16_struct);
}

/**
 * Convert Float16Array to Float64 array (explicit conversion only)
 * Input: Float16Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f16_to_f64_array(b_lean_obj_arg f16_struct) {
    // Extract fields from Float16Array structure
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));          // numElements
    const uint16_t* f16_data = (const uint16_t*)lean_sarray_cptr(f16_bytes);

    lean_object* result = lean_alloc_sarray(sizeof(double), size, size);
    double* f64_data = lean_float_array_cptr(result);

#ifdef _OPENMP
    if (size >= 10000) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            f64_data[i] = (double)half_to_float_sw(f16_data[i]);
        }
        return lean_io_result_mk_ok(result);
    }
#endif

    for (size_t i = 0; i < size; i++) {
        f64_data[i] = (double)half_to_float_sw(f16_data[i]);
    }

    return lean_io_result_mk_ok(result);
}

/**
 * Get Float16 element at index (returns as Float64)
 * Input: Float16Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f16_get(b_lean_obj_arg f16_struct, size_t index) {
    // Extract fields from Float16Array structure
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));          // numElements

    if (index >= size) {
        return lean_io_result_mk_ok(lean_box_float(0.0));
    }

    const uint16_t* data = (const uint16_t*)lean_sarray_cptr(f16_bytes);
    double result = (double)half_to_float_sw(data[index]);
    return lean_io_result_mk_ok(lean_box_float(result));
}

/**
 * Set Float16 element at index (from Float64)
 * Input/Output: Float16Array { data : ByteArray, numElements : Nat }
 */
lean_obj_res lean_f16_set(lean_obj_arg f16_struct, size_t index, double value) {
    // Extract fields from Float16Array structure
    lean_object* f16_bytes = lean_ctor_get(f16_struct, 0);           // data
    size_t size = lean_unbox(lean_ctor_get(f16_struct, 1));          // numElements

    if (index >= size) {
        // Return original array unchanged
        return lean_io_result_mk_ok(f16_struct);
    }

    // Copy-on-write: create new ByteArray
    size_t byte_size = size * sizeof(uint16_t);
    lean_object* new_bytes = lean_alloc_sarray(1, byte_size, byte_size);
    uint16_t* dest = (uint16_t*)lean_sarray_cptr(new_bytes);
    const uint16_t* src = (const uint16_t*)lean_sarray_cptr(f16_bytes);

    memcpy(dest, src, byte_size);
    dest[index] = float_to_half_sw((float)value);

    // Create new Float16Array structure
    lean_object* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, new_bytes);
    lean_ctor_set(result, 1, lean_box(size));

    lean_dec(f16_struct);
    return lean_io_result_mk_ok(result);
}

// ============================================================================
// Backend Information
// ============================================================================

/**
 * Get SIMD backend information
 */
lean_obj_res lean_simd_backend_info(b_lean_obj_arg /* unit */) {
    const char* simd_info =
#ifdef USE_AVX2
        "AVX2 (x86_64) - F64: 4/op, F32: 8/op";
#elif defined(USE_NEON)
        "NEON (ARM64) - F64: 2/op, F32: 4/op";
#else
        "Scalar fallback - 1/op";
#endif

    const char* fp16_info =
#if defined(USE_F16C)
        ", F16C";
#elif defined(USE_NEON_FP16)
        ", FP16";
#else
        "";
#endif

    const char* omp_info =
#ifdef _OPENMP
        ", OpenMP enabled";
#else
        "";
#endif

    char backend[512];
    snprintf(backend, sizeof(backend), "%s%s%s", simd_info, fp16_info, omp_info);

    lean_object* str = lean_mk_string(backend);
    return lean_io_result_mk_ok(str);
}

} // extern "C"
