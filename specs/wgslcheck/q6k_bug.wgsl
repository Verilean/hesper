// Models the pre-54a2a60 q6kToF16 race: 2.88M blocks dispatched as a 2D
// grid 65535 x 45 of wg64 = 188,740,800 threads; the flattened block index
// runs far past the buffer and every OOB write clamps onto the last u32
// (the tail vocab token's f16) — 65k racing writes, last-writer-wins.
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let blk = wid.y * 65535u * 64u + wid.x * 64u + lid.x;
  dst[blk] = src[blk] ^ 0x55555555u;
}
