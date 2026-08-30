// FIXTURE (intentional FAIL): simdgroup_store of an 8x8 fragment at the last
// tile row writes rows past M — the R32 WMMA-tail heap-stomp class.
// grid 1x4x1, wg 32: rowBase up to 96; footprint 96*40+7*40+7 = 4127 >= 4000.
#include <metal_stdlib>
using namespace metal;

constant uint LD = 40u;

kernel void fail_simdgroup_tail(
    device float* c [[buffer(0)]],
    uint2 wid [[threadgroup_position_in_grid]])
{
  simdgroup_float8x8 Cx = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  const uint rowBase = wid.y * 32u;
  simdgroup_store(Cx, &c[rowBase*LD], LD);
}
