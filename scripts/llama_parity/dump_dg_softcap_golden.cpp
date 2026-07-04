// DiffusionGemma final logit softcap golden: out = softcap * tanh(x / softcap), softcap=30.
// Run:  ./scripts/llama_parity/dump_dg_softcap_golden /tmp/dg_golden/softcap

#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
static void wb(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)exit(2);if(fwrite(d,1,n,f)!=n)exit(2);fclose(f);}
int main(int argc,char**argv){
    if(argc!=2){fprintf(stderr,"usage: %s out_dir\n",argv[0]);return 1;}
    std::string o=argv[1];
    const int n=64; const float cap=30.0f;
    struct ggml_init_params ip={16*1024*1024,NULL,false};
    struct ggml_context*ctx=ggml_init(ip);
    int64_t ne[4]={n,1,1,1};
    struct ggml_tensor*x=ggml_new_tensor(ctx,GGML_TYPE_F32,4,ne);
    float*xd=(float*)x->data;
    for(int i=0;i<n;i++) xd[i]=(float)(std::sin((double)i*0.05)*80.0); // span beyond ±cap to exercise tanh saturation
    struct ggml_tensor*out=ggml_scale(ctx,ggml_tanh(ctx,ggml_scale(ctx,x,1.0f/cap)),cap);
    struct ggml_cgraph*gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0){fprintf(stderr,"compute fail\n");return 3;}
    std::string b=o+"/";
    wb(b+"x.bin",x->data,ggml_nbytes(x));
    wb(b+"out.bin",out->data,ggml_nbytes(out));
    FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"n %d\ncap %.9g\n",n,cap);fclose(mf);
    fprintf(stderr,"OK softcap: n=%d cap=%g\n",n,cap);
    return 0;
}
