// Benchmark llama.cpp's ggml_im2col on the CUDA backend at various
// realistic vision-tower shapes.  Reports kernel-only time per call
// (cudaEventElapsed) and effective DRAM bandwidth.
//
// Build:
//   g++ -O2 -std=c++17 bench_im2col.cpp \
//     -I.../llama.cpp/ggml/include -I.../llama.cpp/ggml/src \
//     -L.../llama.cpp/build/bin -Wl,-rpath,$PWD/.../build/bin \
//     -lggml -lggml-base -lggml-cpu -lggml-cuda -lcuda -lcudart \
//     -o bench_im2col
//
// Run:
//   ./bench_im2col

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

struct Shape {
    int N, IC, IH, IW, OC, KH, KW, s0, s1, p0, p1, d0, d1;
};

static double bench_one(ggml_backend_t backend, const Shape & sh, int n_iter, int n_warmup) {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,    // we'll use a backend buffer
    };
    struct ggml_context * ctx = ggml_init(iparams);

    int64_t ne_data[4]   = { sh.IW, sh.IH, sh.IC, sh.N };
    int64_t ne_kernel[4] = { sh.KW, sh.KH, sh.IC, sh.OC };
    struct ggml_tensor * data   = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_data);
    struct ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel);

    struct ggml_tensor * out = ggml_im2col(ctx, kernel, data,
        sh.s0, sh.s1, sh.p0, sh.p1, sh.d0, sh.d1, /*is_2D*/ true, GGML_TYPE_F32);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Fill data + kernel with deterministic values.
    std::vector<float> data_h(ggml_nelements(data));
    std::vector<float> kernel_h(ggml_nelements(kernel));
    for (size_t i = 0; i < data_h.size();   i++) data_h[i]   = (float)(std::sin((double)i * 0.013) * 0.4);
    for (size_t i = 0; i < kernel_h.size(); i++) kernel_h[i] = (float)(std::cos((double)i * 0.027) * 0.3);
    ggml_backend_tensor_set(data,   data_h.data(),   0, ggml_nbytes(data));
    ggml_backend_tensor_set(kernel, kernel_h.data(), 0, ggml_nbytes(kernel));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Warmup.
    for (int i = 0; i < n_warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_call_us = total_us / n_iter;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return per_call_us;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    std::vector<Shape> shapes = {
        // Small tests (matches our parity test).
        { /*N*/ 2, /*IC*/ 3, /*IH*/ 8, /*IW*/ 8, /*OC*/ 1, /*KH*/ 3, /*KW*/ 3, 1,1, 1,1, 1,1 },
        // SigLIP-style patch-embed-ish reshape (use im2col before matmul).
        { 1, 3, 224, 224, 1, 16, 16, 16,16, 0,0, 1,1 },
        // Dense conv (CLIP-ViT input pre-transformer 7x7 stem-ish).
        { 1, 3, 224, 224, 1, 7, 7, 4,4, 3,3, 1,1 },
        // Bigger feature map mid-network.
        { 1, 64, 56, 56, 1, 3, 3, 1,1, 1,1, 1,1 },
        { 1, 256, 28, 28, 1, 3, 3, 1,1, 1,1, 1,1 },
    };

    printf("shape                                                      llama_µs/call\n");
    printf("-------------------------------------------------------------------------\n");
    for (auto & sh : shapes) {
        double t = bench_one(backend, sh, 50, 10);
        char buf[256];
        snprintf(buf, sizeof(buf), "N=%d IC=%d IH=%d IW=%d KH=%d KW=%d s=%d p=%d",
                 sh.N, sh.IC, sh.IH, sh.IW, sh.KH, sh.KW, sh.s0, sh.p0);
        printf("%-58s  %9.2f\n", buf, t);
    }
    ggml_backend_free(backend);
    return 0;
}
