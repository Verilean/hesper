// FIXTURE (intentional FAIL): grid 4x1x1 * wg 256 = 1024 threads store
// unguarded into a 1000-element buffer — the excess-thread clamp-race class.
#include <metal_stdlib>
using namespace metal;

kernel void fail_scalar_oob(
    device float* outp [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
  outp[gid] = 1.0f;
}
