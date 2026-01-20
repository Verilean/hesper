/**
 * Hesper SIMD Operations - Highway Header
 *
 * This file gets compiled multiple times with different -DHWY_TARGET flags
 */

#include <lean/lean.h>
#include <cstdint>
#include "hwy/highway.h"

namespace hesper {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Float64 vector addition
HWY_ATTR void AddFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                         double* HWY_RESTRICT out, size_t count) {
  const hn::ScalableTag<double> d;

  size_t i = 0;
  HWY_FULL(double) df;
  for (; i + hn::Lanes(df) <= count; i += hn::Lanes(df)) {
    const auto va = hn::LoadU(df, a + i);
    const auto vb = hn::LoadU(df, b + i);
    hn::StoreU(hn::Add(va, vb), df, out + i);
  }

  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

// Float32 vector addition
HWY_ATTR void AddFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                         float* HWY_RESTRICT out, size_t count) {
  HWY_FULL(float) df;

  size_t i = 0;
  for (; i + hn::Lanes(df) <= count; i += hn::Lanes(df)) {
    const auto va = hn::LoadU(df, a + i);
    const auto vb = hn::LoadU(df, b + i);
    hn::StoreU(hn::Add(va, vb), df, out + i);
  }

  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

// Float16 vector addition
HWY_ATTR void AddFloat16(const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
                         uint16_t* HWY_RESTRICT out, size_t count) {
  // Use F32 promotion for FP16
  HWY_FULL(float) df;
  const hn::Rebind<uint16_t, decltype(df)> du16;

  size_t i = 0;
  const size_t N = hn::Lanes(df);

  for (; i + N <= count; i += N) {
    // Load as uint16
    const auto va_u16 = hn::LoadU(du16, a + i);
    const auto vb_u16 = hn::LoadU(du16, b + i);

    // Promote to float32 (Highway handles F16C conversion automatically)
    const auto va_f32 = hn::PromoteTo(df, va_u16);
    const auto vb_f32 = hn::PromoteTo(df, vb_u16);

    // Add
    const auto sum_f32 = hn::Add(va_f32, vb_f32);

    // Demote back to uint16
    const auto sum_u16 = hn::DemoteTo(du16, sum_f32);
    hn::StoreU(sum_u16, du16, out + i);
  }

  // Scalar tail
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];  // Simple for now
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hesper
