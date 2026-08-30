// FIXTURE (intentional FAIL): the s-loop fans each thread to 8 slots but the
// guard bounds only m, not the flat store index into a smaller buffer.
#include <metal_stdlib>
using namespace metal;

constant uint COLS = 32u;

kernel void fail_excess_thread(
    device float* dst [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]])
{
  for (uint s = 0u; s < 8u; s++) {
    uint flat = tid + s*128u;
    uint m = flat/32u, k = flat % 32u;
    dst[m*COLS + k] = 0.0f;
  }
}
