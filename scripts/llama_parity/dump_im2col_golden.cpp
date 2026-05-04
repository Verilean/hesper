// Dump deterministic input + im2col output produced by ggml's CPU backend.
// Used to verify hesper's port of im2col against the actual llama.cpp
// implementation (not just our re-reading of it).
//
// Build:
//   g++ -O2 -std=c++17 dump_im2col_golden.cpp \
//     -I.../llama.cpp/ggml/include -I.../llama.cpp/ggml/src \
//     -L.../llama.cpp/build/bin -lggml -lggml-base -lggml-cpu \
//     -o dump_im2col_golden
//
// Run:
//   LD_LIBRARY_PATH=.../llama.cpp/build/bin ./dump_im2col_golden out_dir
//
// Output files in out_dir/:
//   src.bin        — input tensor f32, shape [IW, IH, IC, N] (ggml row-major)
//   weight.bin     — weights f32, shape [KW, KH, IC, OC]
//   out.bin        — im2col output f32, shape [IC*KH*KW, OW*OH, N, 1]
//   meta.txt       — N IC IH IW OC KH KW s0 s1 p0 p1 d0 d1 OH OW

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void write_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "fopen %s: failed\n", path.c_str()); exit(2); }
    if (fwrite(data, 1, bytes, f) != bytes) { fprintf(stderr, "fwrite %s: short\n", path.c_str()); exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out_dir\n", argv[0]); return 1; }
    std::string out_dir = argv[1];

    // Match Tests/CUDA/CUDAIm2colTest.lean: N=2 IC=3 IH=8 IW=8 KH=KW=3 stride=1 pad=1 dilation=1.
    // ggml_im2col input layout is [W, H, C, N] (ne[0] is fastest dim). For 2D mode is_2D=true,
    // src1 (data) shape: [IW, IH, IC, N]; src0 (kernel) shape: [KW, KH, IC, OC].
    const int N = 2;
    const int IC = 3;
    const int IH = 8;
    const int IW = 8;
    const int OC = 1;  // im2col output is independent of OC; pick 1.
    const int KH = 3;
    const int KW = 3;
    const int s0 = 1, s1 = 1;
    const int p0 = 1, p1 = 1;
    const int d0 = 1, d1 = 1;
    const bool is_2D = true;
    const int OH = (IH + 2*p1 - (d1 * (KH - 1) + 1)) / s1 + 1;
    const int OW = (IW + 2*p0 - (d0 * (KW - 1) + 1)) / s0 + 1;

    // Build context.
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne_data[4]   = { IW, IH, IC, N };
    int64_t ne_kernel[4] = { KW, KH, IC, OC };
    struct ggml_tensor * data   = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_data);
    struct ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel);
    ggml_set_name(data,   "data");
    ggml_set_name(kernel, "kernel");

    // Same deterministic seed as the Lean test:
    //   src[i] = sin(i * 0.013) * 0.4
    //   w[i]   = cos(i * 0.027) * 0.3
    const int srcSize = N * IC * IH * IW;
    const int wSize   = OC * IC * KH * KW;
    float * sd = (float *) data->data;
    float * wd = (float *) kernel->data;
    for (int i = 0; i < srcSize; i++) sd[i] = (float) (std::sin((double)i * 0.013) * 0.4);
    for (int i = 0; i < wSize;   i++) wd[i] = (float) (std::cos((double)i * 0.027) * 0.3);

    // Build im2col(kernel, data, ...) graph.
    struct ggml_tensor * out = ggml_im2col(ctx, kernel, data, s0, s1, p0, p1, d0, d1, is_2D, GGML_TYPE_F32);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // CPU compute.
    int n_threads = 1;
    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != 0) {
        fprintf(stderr, "graph_compute failed\n");
        return 3;
    }

    // Dump.
    std::string base = out_dir + "/";
    write_bin(base + "src.bin",    data->data,   ggml_nbytes(data));
    write_bin(base + "weight.bin", kernel->data, ggml_nbytes(kernel));
    write_bin(base + "out.bin",    out->data,    ggml_nbytes(out));

    {
        FILE * f = fopen((base + "meta.txt").c_str(), "w");
        if (!f) { fprintf(stderr, "fopen meta.txt failed\n"); return 4; }
        fprintf(f, "N %d\nIC %d\nIH %d\nIW %d\nOC %d\nKH %d\nKW %d\n", N, IC, IH, IW, OC, KH, KW);
        fprintf(f, "s0 %d\ns1 %d\np0 %d\np1 %d\nd0 %d\nd1 %d\n", s0, s1, p0, p1, d0, d1);
        fprintf(f, "OH %d\nOW %d\n", OH, OW);
        fprintf(f, "out_ne0 %lld\nout_ne1 %lld\nout_ne2 %lld\nout_ne3 %lld\n",
                (long long) out->ne[0], (long long) out->ne[1],
                (long long) out->ne[2], (long long) out->ne[3]);
        fprintf(f, "out_nb0 %lld\nout_nb1 %lld\nout_nb2 %lld\nout_nb3 %lld\n",
                (long long) out->nb[0], (long long) out->nb[1],
                (long long) out->nb[2], (long long) out->nb[3]);
        fclose(f);
    }

    fprintf(stderr, "OK: dumped to %s, out shape [%lld, %lld, %lld, %lld]\n",
            out_dir.c_str(),
            (long long) out->ne[0], (long long) out->ne[1],
            (long long) out->ne[2], (long long) out->ne[3]);
    return 0;
}
