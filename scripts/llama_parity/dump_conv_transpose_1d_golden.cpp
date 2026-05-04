// Dump deterministic input + conv_transpose_1d output produced by ggml's
// CPU backend.  Verifies hesper's port against the actual llama.cpp impl.
//
// Build/run pattern matches dump_im2col_golden.cpp.
//
// Output files in out_dir/:
//   src.bin     — input tensor f32, shape per ggml = [IL, IC, 1, 1]
//   weight.bin  — weights f32, shape [KW, OC, IC, 1]
//   out.bin     — conv_transpose_1d output f32
//   meta.txt    — IC OC KW IL OL s0

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

    // Match Tests/CUDA/CUDAConvTranspose1dTest.lean: IC=4 OC=3 KW=4 IL=8 stride=2 → OL=18.
    // ggml ne[] order is fastest-first.
    //   src1 (data)   : ne = [IL, IC, 1, 1] = [8, 4, 1, 1]
    //   src0 (kernel) : ne = [KW, OC, IC, 1] = [4, 3, 4, 1]
    const int IC = 4;
    const int OC = 3;
    const int KW = 4;
    const int IL = 8;
    const int s0 = 2;
    const int p0 = 0;
    const int d0 = 1;
    const int OL = (IL - 1) * s0 + KW;  // 7*2 + 4 = 18

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne_data[4]   = { IL, IC, 1, 1 };
    int64_t ne_kernel[4] = { KW, OC, IC, 1 };
    struct ggml_tensor * data   = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_data);
    struct ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel);
    ggml_set_name(data,   "data");
    ggml_set_name(kernel, "kernel");

    // Match the Lean test's deterministic seed:
    //   src[i] = sin(i * 0.027) * 0.4   (note: different from im2col)
    //   w[i]   = cos(i * 0.039) * 0.3
    const int srcSize = IC * IL;
    const int wSize   = IC * OC * KW;
    float * sd = (float *) data->data;
    float * wd = (float *) kernel->data;
    for (int i = 0; i < srcSize; i++) sd[i] = (float) (std::sin((double)i * 0.027) * 0.4);
    for (int i = 0; i < wSize;   i++) wd[i] = (float) (std::cos((double)i * 0.039) * 0.3);

    struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, data, s0, p0, d0);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) {
        fprintf(stderr, "graph_compute failed\n");
        return 3;
    }

    std::string base = out_dir + "/";
    write_bin(base + "src.bin",    data->data,   ggml_nbytes(data));
    write_bin(base + "weight.bin", kernel->data, ggml_nbytes(kernel));
    write_bin(base + "out.bin",    out->data,    ggml_nbytes(out));

    {
        FILE * f = fopen((base + "meta.txt").c_str(), "w");
        fprintf(f, "IC %d\nOC %d\nKW %d\nIL %d\nOL %d\ns0 %d\np0 %d\nd0 %d\n",
                IC, OC, KW, IL, OL, s0, p0, d0);
        fprintf(f, "out_ne0 %lld\nout_ne1 %lld\nout_ne2 %lld\nout_ne3 %lld\n",
                (long long) out->ne[0], (long long) out->ne[1],
                (long long) out->ne[2], (long long) out->ne[3]);
        fprintf(f, "out_nb0 %lld\nout_nb1 %lld\n",
                (long long) out->nb[0], (long long) out->nb[1]);
        fclose(f);
    }

    fprintf(stderr, "OK: out shape [%lld, %lld, %lld, %lld] bytes=%zu\n",
            (long long) out->ne[0], (long long) out->ne[1],
            (long long) out->ne[2], (long long) out->ne[3],
            ggml_nbytes(out));
    return 0;
}
