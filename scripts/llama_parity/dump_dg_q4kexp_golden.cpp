// DiffusionGemma expert-indexed Q4_K golden: quantize [inDim, outDim*nExpert] to Q4_K,
// reference = dequant(expert e slab) · f32 input. Dumps full Q4_K bytes + x + ref(expert e).
// Run: ./scripts/llama_parity/dump_dg_q4kexp_golden /tmp/dg_golden/q4kexp
#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
static void wb(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)exit(2);if(fwrite(d,1,n,f)!=n)exit(2);fclose(f);}
int main(int argc,char**argv){
  if(argc!=2){fprintf(stderr,"usage: %s out_dir\n",argv[0]);return 1;}
  std::string o=argv[1];
  const int inD=256, outD=8, nExp=3, e=1;      // expert 1; inD%256==0
  struct ggml_init_params ip={64*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  const int rows=outD*nExp;
  std::vector<float> wf((size_t)inD*rows);
  for(int i=0;i<inD*rows;i++) wf[i]=(float)(std::sin((double)i*0.011)*0.4);
  struct ggml_tensor*a=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,inD,rows);
  ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), a->data, 0, rows, inD, nullptr);
  // dequant back to f32
  struct ggml_tensor*adeq=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,inD,rows);
  struct ggml_tensor*cpy=ggml_cpy(ctx,a,adeq);
  // expert e slab = rows [e*outD, (e+1)*outD)
  std::vector<float> x(inD);
  for(int i=0;i<inD;i++) x[i]=(float)(std::cos((double)i*0.019)*0.6);
  struct ggml_tensor*xt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,inD,1);
  memcpy(xt->data,x.data(),inD*sizeof(float));
  struct ggml_tensor*slab=ggml_view_2d(ctx,adeq,inD,outD,adeq->nb[1],(size_t)e*outD*adeq->nb[1]);
  struct ggml_tensor*y=ggml_mul_mat(ctx,slab,xt);
  struct ggml_cgraph*gf=ggml_new_graph(ctx);
  ggml_build_forward_expand(gf,cpy); ggml_build_forward_expand(gf,y);
  if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0)return 3;
  std::string b=o+"/";
  wb(b+"a.bin",a->data,ggml_nbytes(a));
  wb(b+"x.bin",x.data(),inD*sizeof(float));
  wb(b+"out.bin",y->data,outD*sizeof(float));
  FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"inD %d\noutD %d\nnExp %d\nexpert %d\nq4kbytes %zu\n",inD,outD,nExp,e,ggml_nbytes(a));fclose(mf);
  fprintf(stderr,"OK q4kexp: inD=%d outD=%d nExp=%d e=%d bytes=%zu\n",inD,outD,nExp,e,ggml_nbytes(a));
  return 0;
}
