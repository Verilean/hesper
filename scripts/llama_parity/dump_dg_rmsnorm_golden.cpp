// DiffusionGemma RMSNorm golden: deterministic input + ggml rms_norm * weight.
// Matches llama.cpp build_norm(LLM_NORM_RMS): out = (x / sqrt(mean(x^2)+eps)) * weight.
// Hesper's RMSNorm (CPU reference + WGSL kernel) is verified against this byte/precision-wise.
//
// Build (after ggml is built; see scripts/llama_parity/README.md):
//   g++ -O2 -std=c++17 scripts/llama_parity/dump_dg_rmsnorm_golden.cpp \
//     -I <llama.cpp>/ggml/include -I <llama.cpp>/ggml/src \
//     -L <llama.cpp>/build/bin -Wl,-rpath,<llama.cpp>/build/bin \
//     -lggml -lggml-base -lggml-cpu \
//     -o scripts/llama_parity/dump_dg_rmsnorm_golden
//
// Run:  ./scripts/llama_parity/dump_dg_rmsnorm_golden /tmp/dg_golden/rmsnorm

#include <ggml.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static void write_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "fopen %s failed\n", path.c_str()); exit(2); }
    if (fwrite(data, 1, bytes, f) != bytes) { fprintf(stderr, "short write %s\n", path.c_str()); exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out_dir\n", argv[0]); return 1; }
    std::string out_dir = argv[1];

    const int n = 64;             // representative hidden dim
    const float eps = 1e-6f;      // diffusion-gemma.attention.layer_norm_rms_epsilon

    struct ggml_init_params iparams = { 16*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne[4] = { n, 1, 1, 1 };
    struct ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    struct ggml_tensor * w = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    float * xd = (float *) x->data;
    float * wd = (float *) w->data;
    for (int i = 0; i < n; i++) {
        xd[i] = (float) (std::sin((double) i * 0.013) * 2.0);
        wd[i] = (float) (1.0 + std::cos((double) i * 0.027) * 0.1);   // Gemma-style ~1+small
    }

    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    struct ggml_tensor * out    = ggml_mul(ctx, normed, w);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { fprintf(stderr, "compute failed\n"); return 3; }

    std::string b = out_dir + "/";
    write_bin(b + "x.bin",   x->data,   ggml_nbytes(x));
    write_bin(b + "w.bin",   w->data,   ggml_nbytes(w));
    write_bin(b + "out.bin", out->data, ggml_nbytes(out));
    FILE * mf = fopen((b + "meta.txt").c_str(), "w");
    fprintf(mf, "n %d\neps %.9g\n", n, eps);
    fclose(mf);

    fprintf(stderr, "OK rmsnorm: n=%d eps=%g bytes=%zu\n", n, eps, ggml_nbytes(out));
    return 0;
}
