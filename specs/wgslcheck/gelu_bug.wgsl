// Models the pre-54a2a60 geluMulB race: N = 262*2112 = 553,344 is not
// 256-divisible, the grid rounds up to 2162 workgroups = 553,472 threads,
// and the store is unguarded — 128 excess threads clamp-write element
// 553,343, racing its owner.
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let x = a[i];
  let g = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  outBuf[i] = g * b[i];
}
