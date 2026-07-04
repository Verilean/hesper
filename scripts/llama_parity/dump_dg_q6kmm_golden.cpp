// DiffusionGemma Q6_K matmul golden — reference = dequant(Q6_K weight, via ggml) · f32 input.
// Validates Hesper's Metal fusedQ6KLinearKernel / fusedQ6KBatchKernel inline dequant.
#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
static void wb(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)exit(2);if(fwrite(d,1,n,f)!=n)exit(2);fclose(f);}
int main(int argc,char**argv){
  if(argc!=2){fprintf(stderr,"usage: %s out_dir\n",argv[0]);return 1;}
  std::string o=argv[1];
  const int in=256, out=8;                 // in % 256 == 0 for Q6_K
  struct ggml_init_params ip={64*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  std::vector<float> wf((size_t)in*out);
  for(int i=0;i<in*out;i++) wf[i]=(float)(std::sin((double)i*0.017)*0.4);
  struct ggml_tensor*a=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,in,out);
  ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), a->data, 0, out, in, nullptr);
  // dequant the whole weight back via ggml's own to_float (independent reference)
  std::vector<float> wdq((size_t)in*out);
  const auto* tt = ggml_get_type_traits(GGML_TYPE_Q6_K);
  tt->to_float(a->data, wdq.data(), (int64_t)in*out);
  std::vector<float> x(in);
  for(int i=0;i<in;i++) x[i]=(float)(std::cos((double)i*0.019)*0.6);
  std::vector<float> y(out);
  for(int r=0;r<out;r++){ double acc=0.0; for(int c=0;c<in;c++) acc += (double)wdq[r*in+c]*(double)x[c]; y[r]=(float)acc; }
  std::string b=o+"/";
  wb(b+"a.bin",a->data,ggml_nbytes(a));
  wb(b+"x.bin",x.data(),in*sizeof(float));
  wb(b+"out.bin",y.data(),out*sizeof(float));
  fprintf(stderr,"OK q6kmm: in=%d out=%d q6bytes=%zu\n",in,out,ggml_nbytes(a));
  return 0;
}
