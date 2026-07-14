// The fix: early-return guard (tests the checker's negative refinement).
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let blk = wid.y * 65535u * 64u + wid.x * 64u + lid.x;
  if (blk >= 2880000u) { return; }
  dst[blk] = src[blk] ^ 0x55555555u;
}
