// FIXTURE (expected ok): the same store as fail_scalar_oob, guarded.
#include <metal_stdlib>
using namespace metal;

constant uint NELEM = 1000u;

kernel void ok_guarded(
    device float* outp [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid < NELEM) {
    outp[gid] = 1.0f;
  }
}
