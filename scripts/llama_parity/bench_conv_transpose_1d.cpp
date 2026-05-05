// Benchmark llama.cpp's ggml_conv_transpose_1d on the CUDA backend at
// realistic audio decoder shapes.

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cmath>
#include <cstdio>
#include <chrono>
#include <vector>

struct Shape {
    int IC, OC, KW, IL, s0, p0, d0;
};

static double bench_one(ggml_backend_t backend, const Shape & sh, int n_iter, int n_warmup) {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne_data[4]   = { sh.IL, sh.IC, 1, 1 };
    int64_t ne_kernel[4] = { sh.KW, sh.OC, sh.IC, 1 };
    struct ggml_tensor * data   = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_data);
    struct ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel);

    struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, data, sh.s0, sh.p0, sh.d0);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> data_h(ggml_nelements(data));
    std::vector<float> kernel_h(ggml_nelements(kernel));
    for (size_t i = 0; i < data_h.size();   i++) data_h[i]   = (float)(std::sin((double)i * 0.027) * 0.4);
    for (size_t i = 0; i < kernel_h.size(); i++) kernel_h[i] = (float)(std::cos((double)i * 0.039) * 0.3);
    ggml_backend_tensor_set(data,   data_h.data(),   0, ggml_nbytes(data));
    ggml_backend_tensor_set(kernel, kernel_h.data(), 0, ggml_nbytes(kernel));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < n_warmup; i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::high_resolution_clock::now();

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iter;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    std::vector<Shape> shapes = {
        // Parity test shape.
        { /*IC*/ 4, /*OC*/ 3, /*KW*/ 4, /*IL*/ 8, /*s0*/ 2, /*p0*/ 0, /*d0*/ 1 },
        // VocoderS-like upsampler stage.
        { 64, 64, 4, 512, 2, 0, 1 },
        { 128, 64, 8, 256, 4, 0, 1 },
        // BigVGAN-ish stage.
        { 512, 256, 16, 128, 8, 0, 1 },
        // Audio codec head (small kernel, long input).
        { 32, 32, 3, 4096, 1, 0, 1 },
    };

    printf("shape                                          llama_µs/call\n");
    printf("-------------------------------------------------------------\n");
    for (auto & sh : shapes) {
        double t = bench_one(backend, sh, 100, 20);
        char buf[256];
        snprintf(buf, sizeof(buf), "IC=%d OC=%d KW=%d IL=%d s=%d", sh.IC, sh.OC, sh.KW, sh.IL, sh.s0);
        printf("%-46s  %9.2f\n", buf, t);
    }
    ggml_backend_free(backend);
    return 0;
}
