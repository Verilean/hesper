// DiffusionGemma GeGLU golden: out = gelu(gate) * up  (LLM_FFN_GELU, LLM_FFN_PAR).
// ggml's CPU gelu uses an f16 lookup table, so expect ~1e-3 vs an exact-tanh reference.
// Run:  ./scripts/llama_parity/dump_dg_geglu_golden /tmp/dg_golden/geglu

#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
static void wb(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f){exit(2);}if(fwrite(d,1,n,f)!=n)exit(2);fclose(f);}
int main(int argc,char**argv){
    if(argc!=2){fprintf(stderr,"usage: %s out_dir\n",argv[0]);return 1;}
    std::string o=argv[1];
    const int n=64;
    struct ggml_init_params ip={16*1024*1024,NULL,false};
    struct ggml_context*ctx=ggml_init(ip);
    int64_t ne[4]={n,1,1,1};
    struct ggml_tensor*gate=ggml_new_tensor(ctx,GGML_TYPE_F32,4,ne);
    struct ggml_tensor*up=ggml_new_tensor(ctx,GGML_TYPE_F32,4,ne);
    float*gd=(float*)gate->data;float*ud=(float*)up->data;
    for(int i=0;i<n;i++){gd[i]=(float)(std::sin((double)i*0.013)*2.0);ud[i]=(float)(std::cos((double)i*0.027)*1.5);}
    struct ggml_tensor*out=ggml_mul(ctx,ggml_gelu(ctx,gate),up);
    struct ggml_cgraph*gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0){fprintf(stderr,"compute fail\n");return 3;}
    std::string b=o+"/";
    wb(b+"gate.bin",gate->data,ggml_nbytes(gate));
    wb(b+"up.bin",up->data,ggml_nbytes(up));
    wb(b+"out.bin",out->data,ggml_nbytes(out));
    FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"n %d\n",n);fclose(mf);
    fprintf(stderr,"OK geglu: n=%d\n",n);
    return 0;
}
