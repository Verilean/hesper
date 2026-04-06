#!/usr/bin/env python3
"""Dump the embedding row for a given token from GGUF (dequantized)."""

import sys
import struct
import os

def dequant_q6_k_block(block_bytes, out):
    """Dequantize a single Q6_K block (210 bytes → 256 f32)."""
    ql = block_bytes[0:128]
    qh = block_bytes[128:192]
    scales = struct.unpack("<16b", block_bytes[192:208])
    d_half = struct.unpack("<H", block_bytes[208:210])[0]

    # FP16 → F32
    sign = (d_half >> 15) & 1
    exp = (d_half >> 10) & 0x1F
    mant = d_half & 0x3FF
    if exp == 0:
        d = 0.0
    else:
        d = ((-1.0) ** sign) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))

    # Dequant loop from dequantize_row_q6_K
    y_offset = 0
    ql_offset = 0
    qh_offset = 0
    sc_offset = 0
    for n_chunk in range(2):  # n = 0, 128
        for l in range(32):
            is_ = l // 16
            q1 = (((ql[ql_offset + l +  0] & 0xF) | (((qh[qh_offset + l] >> 0) & 3) << 4))) - 32
            q2 = (((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 3) << 4))) - 32
            q3 = (((ql[ql_offset + l +  0]  >> 4) | (((qh[qh_offset + l] >> 4) & 3) << 4))) - 32
            q4 = (((ql[ql_offset + l + 32]  >> 4) | (((qh[qh_offset + l] >> 6) & 3) << 4))) - 32
            out[y_offset + l +  0] = d * scales[sc_offset + is_ + 0] * q1
            out[y_offset + l + 32] = d * scales[sc_offset + is_ + 2] * q2
            out[y_offset + l + 64] = d * scales[sc_offset + is_ + 4] * q3
            out[y_offset + l + 96] = d * scales[sc_offset + is_ + 6] * q4
        y_offset += 128
        ql_offset += 64
        qh_offset += 32
        sc_offset += 8
    return d  # return d for debugging


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/gemma-4-e4b-it-Q4_K_M.gguf"
    token_id = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(model_path)
    # Find token_embd tensor
    emb = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            emb = t
            break
    if emb is None:
        print("token_embd.weight not found")
        sys.exit(1)

    print(f"token_embd: shape={emb.shape}, type={emb.tensor_type.name}")
    dim = emb.shape[0]  # 2560
    vocab = emb.shape[1]  # 262144
    blocks_per_row = dim // 256
    block_size_bytes = 210
    row_bytes = blocks_per_row * block_size_bytes

    # emb.data is a numpy uint8 array
    raw = bytes(emb.data)
    print(f"Total bytes: {len(raw)}, expected: {vocab * row_bytes}")

    # Extract row for token_id
    row_offset = token_id * row_bytes
    row_data = raw[row_offset:row_offset + row_bytes]
    print(f"Row {token_id}: offset={row_offset}, bytes={len(row_data)}")

    # Dequant all blocks in this row
    result = [0.0] * dim
    for bi in range(blocks_per_row):
        block_offset = bi * block_size_bytes
        block_bytes = row_data[block_offset:block_offset + block_size_bytes]
        out_slice = result[bi * 256:(bi + 1) * 256]
        d_val = dequant_q6_k_block(block_bytes, out_slice)
        result[bi * 256:(bi + 1) * 256] = out_slice
        if bi < 2:
            print(f"Block {bi}: d={d_val}")
            print(f"  scales: {list(struct.unpack('<16b', block_bytes[192:208]))}")
            print(f"  first 4 ql: {list(block_bytes[0:4])}")
            print(f"  first 4 qh: {list(block_bytes[128:132])}")

    print(f"\nFirst 16 dequantized values for token {token_id}:")
    for i in range(16):
        print(f"  [{i}] = {result[i]}")

    # Stats
    non_zero = sum(1 for v in result if v != 0.0)
    max_abs = max(abs(v) for v in result)
    print(f"\nStats: nonzero={non_zero}/{dim}, max_abs={max_abs}")


if __name__ == "__main__":
    main()
