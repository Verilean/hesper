// Dump deterministic input + ggml permute+cont output as raw f32.
// Tests permute axes (0,2,1,3) which is the (1↔2) swap used in SigLIP attn.

#include <ggml.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static void write_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "fopen %s: failed\n", path.c_str()); exit(2); }
    if (fwrite(data, 1, bytes, f) != bytes) { exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out_dir\n", argv[0]); return 1; }
    std::string out_dir = argv[1];

    // Match the SigLIP shape we saw in the graph: input [64, 12, 64, 1]
    // permuted to [64, 64, 12, 1] = swap axes 1 and 2 (perm = 0,2,1,3).
    const int s0 = 64, s1 = 12, s2 = 64, s3 = 1;
    const int total = s0 * s1 * s2 * s3;

    struct ggml_init_params iparams = { 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne[4] = { s0, s1, s2, s3 };
    struct ggml_tensor * in = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    float * d = (float *) in->data;
    for (int i = 0; i < total; i++) d[i] = (float)(std::sin((double)i * 0.013) * 0.4);

    // Permute axes (0,2,1,3) — swap dim 1 and dim 2.
    struct ggml_tensor * permuted = ggml_permute(ctx, in, 0, 2, 1, 3);
    // Force contiguous copy so output reflects physical permutation.
    struct ggml_tensor * out = ggml_cont(ctx, permuted);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { fprintf(stderr, "compute failed\n"); return 3; }

    std::string base = out_dir + "/";
    write_bin(base + "src.bin", in->data,  ggml_nbytes(in));
    write_bin(base + "out.bin", out->data, ggml_nbytes(out));

    {
        FILE * f = fopen((base + "meta.txt").c_str(), "w");
        fprintf(f, "s0 %d\ns1 %d\ns2 %d\ns3 %d\n", s0, s1, s2, s3);
        fprintf(f, "perm 0 2 1 3\n");
        fprintf(f, "out_ne %lld %lld %lld %lld\n",
            (long long)out->ne[0], (long long)out->ne[1],
            (long long)out->ne[2], (long long)out->ne[3]);
        fclose(f);
    }

    fprintf(stderr, "OK: out shape ne=[%lld %lld %lld %lld] bytes=%zu\n",
        (long long)out->ne[0], (long long)out->ne[1],
        (long long)out->ne[2], (long long)out->ne[3], ggml_nbytes(out));
    return 0;
}
