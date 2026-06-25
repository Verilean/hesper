// DiffusionGemma f32 linear-projection golden: y = W·x  via ggml_mul_mat.
// a=[in,out] (ne0=in), x=[in,1]; out=[out,1]. a's memory == Reference.matVec's [out,in] row-major.
// (Q4_K-quantized matmul parity is a GPU-kernel-vs-ggml test, done at the kernel stage.)
// Run:  ./scripts/llama_parity/dump_dg_matmul_golden /tmp/dg_golden/matmul
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
  const int in=24,out=16;
  struct ggml_init_params ip={16*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  struct ggml_tensor*a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,in,out);
  struct ggml_tensor*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,in,1);
  float*ad=(float*)a->data;for(int i=0;i<in*out;i++)ad[i]=(float)(std::sin((double)i*0.011)*0.4);
  float*xd=(float*)x->data;for(int i=0;i<in;i++)xd[i]=(float)(std::cos((double)i*0.019)*0.6);
  struct ggml_tensor*y=ggml_mul_mat(ctx,a,x);
  struct ggml_cgraph*gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,y);
  if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0)return 3;
  std::string b=o+"/";
  wb(b+"a.bin",a->data,ggml_nbytes(a));wb(b+"x.bin",x->data,ggml_nbytes(x));wb(b+"out.bin",y->data,ggml_nbytes(y));
  FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"in %d\nout %d\n",in,out);fclose(mf);
  fprintf(stderr,"OK matmul: in=%d out=%d\n",in,out);return 0;
}
