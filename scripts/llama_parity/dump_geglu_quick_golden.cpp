// Dump deterministic input + ggml geglu_quick (split form) output as raw f32.
// Verifies hesper's GEGLU_QUICK port against llama.cpp byte-for-byte.

#include <ggml.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void write_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "fopen %s: failed\n", path.c_str()); exit(2); }
    if (fwrite(data, 1, bytes, f) != bytes) { fprintf(stderr, "fwrite %s: short\n", path.c_str()); exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out_dir\n", argv[0]); return 1; }
    std::string out_dir = argv[1];

    // Match Tests/CUDA/CUDAGegluQuickTest.lean: n=512 (typical FFN dim row).
    const int n = 512;

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne[4] = { n, 1, 1, 1 };
    struct ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    struct ggml_tensor * g = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    // Deterministic seeds matching the Lean test.
    float * xd = (float *) x->data;
    float * gd = (float *) g->data;
    for (int i = 0; i < n; i++) {
        xd[i] = (float) (std::sin((double)i * 0.013) * 2.0);   // wider range to exercise sigmoid tails
        gd[i] = (float) (std::cos((double)i * 0.027) * 0.5);
    }

    // ggml_geglu_quick_split builds the gated path.
    struct ggml_tensor * out = ggml_geglu_quick_split(ctx, x, g);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) {
        fprintf(stderr, "graph_compute failed\n");
        return 3;
    }

    std::string base = out_dir + "/";
    write_bin(base + "x.bin",   x->data,   ggml_nbytes(x));
    write_bin(base + "g.bin",   g->data,   ggml_nbytes(g));
    write_bin(base + "out.bin", out->data, ggml_nbytes(out));

    {
        FILE * f = fopen((base + "meta.txt").c_str(), "w");
        fprintf(f, "n %d\n", n);
        fprintf(f, "out_ne0 %lld\n", (long long) out->ne[0]);
        fprintf(f, "out_bytes %zu\n", ggml_nbytes(out));
        fclose(f);
    }

    fprintf(stderr, "OK: out shape ne0=%lld bytes=%zu\n",
            (long long) out->ne[0], ggml_nbytes(out));
    return 0;
}
