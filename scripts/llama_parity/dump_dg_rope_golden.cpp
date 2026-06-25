// DiffusionGemma RoPE (NeoX) golden: deterministic [hd, nHead, nTok] input + ggml_rope_ext output.
// Matches gemma4-common.h gemma4_build_q/kv: ggml_rope_ext(x, pos, freq_factors=NULL, n_rot=hd,
// GGML_ROPE_TYPE_NEOX, freq_base=theta, freq_scale=1, ext=0, attn=1, beta_fast=32, beta_slow=1).
//
// Run:  ./scripts/llama_parity/dump_dg_rope_golden /tmp/dg_golden/rope

#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static void write_bin(const std::string & p, const void * d, size_t n) {
    FILE * f = fopen(p.c_str(), "wb"); if (!f) { fprintf(stderr,"fopen %s\n",p.c_str()); exit(2);}
    if (fwrite(d,1,n,f)!=n){fprintf(stderr,"short %s\n",p.c_str());exit(2);} fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out_dir\n", argv[0]); return 1; }
    std::string out_dir = argv[1];

    const int hd = 8, nHead = 2, nTok = 4;
    const float theta = 10000.0f;   // SWA freq_base

    struct ggml_init_params ip = { 16*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne[4] = { hd, nHead, nTok, 1 };
    struct ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nTok);

    float * xd = (float *) x->data;
    for (int t = 0; t < nTok; t++)
        for (int h = 0; h < nHead; h++)
            for (int i = 0; i < hd; i++) {
                int idx = i + h*hd + t*hd*nHead;
                xd[idx] = (float) (std::sin((double) idx * 0.021) * 1.5);
            }
    int32_t * pd = (int32_t *) pos->data;
    for (int t = 0; t < nTok; t++) pd[t] = t;

    struct ggml_tensor * out = ggml_rope_ext(ctx, x, pos, NULL, hd, GGML_ROPE_TYPE_NEOX,
        262144, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { fprintf(stderr,"compute fail\n"); return 3; }

    std::string b = out_dir + "/";
    write_bin(b+"x.bin",   x->data,   ggml_nbytes(x));
    write_bin(b+"out.bin", out->data, ggml_nbytes(out));
    FILE * mf = fopen((b+"meta.txt").c_str(), "w");
    fprintf(mf, "hd %d\nnHead %d\nnTok %d\ntheta %.9g\n", hd, nHead, nTok, theta);
    fclose(mf);
    fprintf(stderr, "OK rope: hd=%d nHead=%d nTok=%d theta=%g\n", hd, nHead, nTok, theta);
    return 0;
}
