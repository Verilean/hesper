// Strided-loop pattern (attnos style): the loop bound itself guards the
// store; the checker must bound the loop variable from the for-condition.
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read_write> probs : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  for (var t = lid.x; t < 8320u; t = t + 256u) {
    probs[t] = x[t] * 2.0;
  }
}
