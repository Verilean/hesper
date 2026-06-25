// DiffusionGemma softmax golden (router gating + attention): ggml_soft_max.
// Run:  ./scripts/llama_parity/dump_dg_softmax_golden /tmp/dg_golden/softmax
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
  const int n=32;
  struct ggml_init_params ip={16*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  struct ggml_tensor*x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
  float*xd=(float*)x->data;for(int i=0;i<n;i++)xd[i]=(float)(std::sin((double)i*0.3)*3.0);
  struct ggml_tensor*y=ggml_soft_max(ctx,x);
  struct ggml_cgraph*gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,y);
  if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0)return 3;
  std::string b=o+"/";
  wb(b+"x.bin",x->data,ggml_nbytes(x));wb(b+"out.bin",y->data,ggml_nbytes(y));
  FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"n %d\n",n);fclose(mf);
  fprintf(stderr,"OK softmax: n=%d\n",n);return 0;
}
