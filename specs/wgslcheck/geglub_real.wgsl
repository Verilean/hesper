enable f16;
// Batched geglu epilogue: gu = [rows][gate(N)|up(N)] f32; yh[row*N + j] =
// gelu_tanh(gate) * up, f16. rows = M·KMUL (KMUL=1: dense per token; KMUL=K:
// per (token,slot) entry for the grouped MoE GEMM). Same gelu expression as
// q40gu (bit-matching epilogue). Params: N, KMUL, WG
@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read> mprm: array<u32>;      // [1]=M
@group(0) @binding(2) var<storage, read_write> yh: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= mprm[1] * 1u * 2112u) { return; }
  let tok = i / 2112u;
  let j = i % 2112u;
  let g = gu[tok * 2u * 2112u + j];
  let u = gu[tok * 2u * 2112u + 2112u + j];
  let gel = 0.5 * g * (1.0 + tanh(clamp(0.7978845608028654 * (g + 0.044715 * g*g*g), -20.0, 20.0)));
  yh[i] = f16(gel * u);
}
