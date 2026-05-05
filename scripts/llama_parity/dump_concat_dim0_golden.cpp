// Dump deterministic input + ggml_concat (dim=0) output as raw f32.

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

    // Match the Lean test: matches Gemma 4 SigLIP M-RoPE shape [32, 12, 64, 1] + [32, 12, 64, 1].
    const int ne00 = 32, ne10 = 32;
    const int ne1 = 12, ne2 = 64;

    struct ggml_init_params iparams = { 32*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t neX[4] = { ne00, ne1, ne2, 1 };
    int64_t neY[4] = { ne10, ne1, ne2, 1 };
    struct ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, neX);
    struct ggml_tensor * y = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, neY);

    int xN = ne00 * ne1 * ne2;
    int yN = ne10 * ne1 * ne2;
    float * xd = (float *) x->data;
    float * yd = (float *) y->data;
    for (int i = 0; i < xN; i++) xd[i] = (float)(std::sin((double)i * 0.013) * 0.4);
    for (int i = 0; i < yN; i++) yd[i] = (float)(std::cos((double)i * 0.027) * 0.3);

    struct ggml_tensor * out = ggml_concat(ctx, x, y, 0);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { fprintf(stderr, "compute failed\n"); return 3; }

    std::string base = out_dir + "/";
    write_bin(base + "x.bin",   x->data,   ggml_nbytes(x));
    write_bin(base + "y.bin",   y->data,   ggml_nbytes(y));
    write_bin(base + "out.bin", out->data, ggml_nbytes(out));

    {
        FILE * f = fopen((base + "meta.txt").c_str(), "w");
        fprintf(f, "ne00 %d\nne10 %d\nne1 %d\nne2 %d\n", ne00, ne10, ne1, ne2);
        fprintf(f, "out_ne0 %lld\nout_bytes %zu\n",
                (long long)out->ne[0], ggml_nbytes(out));
        fclose(f);
    }

    fprintf(stderr, "OK: out shape ne=[%lld %lld %lld] bytes=%zu\n",
            (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2],
            ggml_nbytes(out));
    return 0;
}
