// DiffusionGemma Q8_0 matmul golden — reference = dequant(Q8_0 weight) · f32 input
// (matches Hesper's fusedQ8_0LinearKernel: inline weight dequant, full-precision activation).
// Dumps packed Q8_0 weight bytes (a.bin), f32 input (x.bin), f32 reference (out.bin).
// Run:  ./scripts/llama_parity/dump_dg_q8mm_golden /tmp/dg_golden/q8mm
#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
static void wb(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)exit(2);if(fwrite(d,1,n,f)!=n)exit(2);fclose(f);}
int main(int argc,char**argv){
  if(argc!=2){fprintf(stderr,"usage: %s out_dir\n",argv[0]);return 1;}
  std::string o=argv[1];
  const int in=64, out=32;                 // in % 32 == 0
  struct ggml_init_params ip={64*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  std::vector<float> wf((size_t)in*out);
  for(int i=0;i<in*out;i++) wf[i]=(float)(std::sin((double)i*0.017)*0.4);
  struct ggml_tensor*a=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,in,out);
  ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), a->data, 0, out, in, nullptr);
  std::vector<float> x(in);
  for(int i=0;i<in;i++) x[i]=(float)(std::cos((double)i*0.019)*0.6);
  // reference: y[r] = Σ_c dequant(W[r][c]) * x[c], with W dequantized from the Q8_0 bytes
  const uint8_t* q8 = (const uint8_t*) a->data;
  const int bpr = in/32;                   // blocks per row
  std::vector<float> y(out);
  for(int r=0;r<out;r++){
    double acc=0.0;
    for(int bk=0;bk<bpr;bk++){
      const uint8_t* blk = q8 + (size_t)(r*bpr+bk)*34;
      ggml_fp16_t dh; memcpy(&dh, blk, 2);
      float d = ggml_fp16_to_fp32(dh);
      for(int i=0;i<32;i++){
        int8_t q = (int8_t) blk[2+i];
        acc += (double)(d * (float)q) * (double)x[bk*32+i];
      }
    }
    y[r]=(float)acc;
  }
  std::string b=o+"/";
  wb(b+"a.bin",a->data,ggml_nbytes(a));
  wb(b+"x.bin",x.data(),in*sizeof(float));
  wb(b+"out.bin",y.data(),out*sizeof(float));
  FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"in %d\nout %d\nq8bytes %zu\n",in,out,ggml_nbytes(a));fclose(mf);
  fprintf(stderr,"OK q8mm: in=%d out=%d q8bytes=%zu\n",in,out,ggml_nbytes(a));
  return 0;
}
