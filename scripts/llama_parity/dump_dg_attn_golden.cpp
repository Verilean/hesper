// DiffusionGemma scaled-dot-product attention core (single head, additive mask, scale=1.0):
// scores=K^T·Q; scores+=mask; probs=softmax(scores); out=V·probs.  This is the numeric heart
// of build_attn (f_attention_scale=1.0).  Q=[hd,nq] K=V=[hd,nk]; mask=[nk,nq] (0 / -1e30).
// Run:  ./scripts/llama_parity/dump_dg_attn_golden /tmp/dg_golden/attn
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
  const int hd=8,nq=3,nk=5;
  struct ggml_init_params ip={32*1024*1024,NULL,false};
  struct ggml_context*ctx=ggml_init(ip);
  struct ggml_tensor*Q=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hd,nq);
  struct ggml_tensor*K=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hd,nk);
  struct ggml_tensor*V=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hd,nk);
  struct ggml_tensor*M=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,nk,nq); // mask[k,q]
  float*qd=(float*)Q->data;for(int i=0;i<hd*nq;i++)qd[i]=(float)(std::sin((double)i*0.07)*1.0);
  float*kd=(float*)K->data;for(int i=0;i<hd*nk;i++)kd[i]=(float)(std::cos((double)i*0.05)*1.0);
  float*vd=(float*)V->data;for(int i=0;i<hd*nk;i++)vd[i]=(float)(std::sin((double)i*0.09+1.0)*1.0);
  float*md=(float*)M->data;
  for(int q=0;q<nq;q++)for(int k=0;k<nk;k++){
     bool allow = !(q==0 && k>=3);     // query 0 masks last 2 keys; others see all
     md[q*nk+k]= allow? 0.0f : -1e30f;
  }
  struct ggml_tensor*scores=ggml_mul_mat(ctx,K,Q);        // [nk,nq]
  scores=ggml_add(ctx,scores,M);
  struct ggml_tensor*probs=ggml_soft_max(ctx,scores);     // softmax over nk
  struct ggml_tensor*Vt=ggml_cont(ctx,ggml_transpose(ctx,V)); // [nk,hd]
  struct ggml_tensor*out=ggml_mul_mat(ctx,Vt,probs);      // [hd,nq]
  struct ggml_cgraph*gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
  if(ggml_graph_compute_with_ctx(ctx,gf,1)!=0)return 3;
  std::string b=o+"/";
  wb(b+"q.bin",Q->data,ggml_nbytes(Q));wb(b+"k.bin",K->data,ggml_nbytes(K));
  wb(b+"v.bin",V->data,ggml_nbytes(V));wb(b+"mask.bin",M->data,ggml_nbytes(M));
  wb(b+"out.bin",out->data,ggml_nbytes(out));
  FILE*mf=fopen((b+"meta.txt").c_str(),"w");fprintf(mf,"hd %d\nnq %d\nnk %d\n",hd,nq,nk);fclose(mf);
  fprintf(stderr,"OK attn: hd=%d nq=%d nk=%d\n",hd,nq,nk);return 0;
}
