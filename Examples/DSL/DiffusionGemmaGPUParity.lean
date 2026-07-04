import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WebGPU.Types
import Hesper.Compute
import Hesper.Basic

/-!
# DiffusionGemma per-module GPU parity on Metal (WebGPU/Dawn)

Runs each DiffusionGemma module as a WGSL compute kernel on the
WebGPU→Metal backend and compares to the SAME ggml goldens the CPU parity
tests use.  Proves the module/kernel path runs on macOS — no CUDA.

Kernels are single-threaded loops (inputs are tiny) for clarity; they
compute the exact op, not a tuned production kernel.  The point is
numerical parity with the reference on Metal.

Run (after generating goldens):
  lake exe diffusiongemma-gpu-parity
-/

open Hesper.WebGPU

def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)    ).toUInt32
    let b1 := (bytes.get! (i*4 + 1)).toUInt32
    let b2 := (bytes.get! (i*4 + 2)).toUInt32
    let b3 := (bytes.get! (i*4 + 3)).toUInt32
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)))
  return a

def maxAbsDiff (a b : Array Float) : Float := Id.run do
  let mut m := 0.0
  for i in [0:min a.size b.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

/-- Run a WGSL kernel with `inputs.size` read_write storage buffers (bindings
    0..k-1) + one output buffer (binding k), single workgroup, return output. -/
def runKernel (device : Device) (src : String) (inputs : Array (Array Float))
    (outN : Nat) : IO (Array Float) := do
  let mkBuf (sz : Nat) : IO Buffer :=
    createBuffer device { size := (sz*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let mut layoutEntries : Array BindGroupLayoutEntry := #[]
  let mut bindEntries : Array BindGroupEntry := #[]
  let mut idx : Nat := 0
  for d in inputs do
    let buf ← mkBuf d.size
    writeBuffer device buf 0 (← Hesper.Basic.floatArrayToBytes d)
    layoutEntries := layoutEntries.push { binding := idx.toUInt32, visibility := .compute, bindingType := .buffer false }
    bindEntries := bindEntries.push { binding := idx.toUInt32, buffer := buf, offset := 0, size := (d.size*4).toUSize }
    idx := idx + 1
  let outBuf ← mkBuf outN
  layoutEntries := layoutEntries.push { binding := idx.toUInt32, visibility := .compute, bindingType := .buffer false }
  bindEntries := bindEntries.push { binding := idx.toUInt32, buffer := outBuf, offset := 0, size := (outN*4).toUSize }
  let shaderModule ← createShaderModule device src
  let layout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device { shaderModule, entryPoint := "main", bindGroupLayout := layout }
  let bindGroup ← createBindGroup device layout bindEntries
  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future
  let resultBytes ← mapBufferRead device outBuf 0 (outN*4).toUSize
  unmapBuffer outBuf
  Hesper.Basic.bytesToFloatArray resultBytes

/-! ## Per-module WGSL kernels (single-threaded, constants baked) -/

def rmsnormWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> x: array<f32>;
   @group(0) @binding(1) var<storage, read_write> w: array<f32>;
   @group(0) @binding(2) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let n = 64u; var ss = 0.0;
     for (var i=0u;i<n;i=i+1u){ ss = ss + x[i]*x[i]; }
     let inv = 1.0 / sqrt(ss/f32(n) + 1e-6);
     for (var i=0u;i<n;i=i+1u){ outp[i] = x[i]*inv*w[i]; }
   }"

def ropeWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> x: array<f32>;
   @group(0) @binding(1) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let hd=8u; let nHead=2u; let nTok=4u; let half=4u; let theta=10000.0;
     for (var t=0u;t<nTok;t=t+1u){ for (var h=0u;h<nHead;h=h+1u){
       let base=(t*nHead+h)*hd;
       for (var j=0u;j<half;j=j+1u){
         let freq = exp(-(2.0*f32(j)/f32(hd))*log(theta));
         let ang = f32(t)*freq;
         let a = x[base+j]; let b = x[base+j+half];
         outp[base+j] = a*cos(ang)-b*sin(ang);
         outp[base+j+half] = a*sin(ang)+b*cos(ang);
       } } }
   }"

def gegluWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
   @group(0) @binding(1) var<storage, read_write> up: array<f32>;
   @group(0) @binding(2) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let n=64u; let c=0.7978845608028654;
     for (var i=0u;i<n;i=i+1u){
       let g = gate[i];
       let gl = 0.5*g*(1.0+tanh(c*(g+0.044715*g*g*g)));
       outp[i] = gl*up[i];
     }
   }"

def matmulWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> a: array<f32>;
   @group(0) @binding(1) var<storage, read_write> xv: array<f32>;
   @group(0) @binding(2) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let inD=24u; let outD=16u;
     for (var r=0u;r<outD;r=r+1u){ var s=0.0;
       for (var c=0u;c<inD;c=c+1u){ s = s + a[r*inD+c]*xv[c]; }
       outp[r]=s; }
   }"

def softmaxWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> x: array<f32>;
   @group(0) @binding(1) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let n=32u; var mx=x[0];
     for (var i=1u;i<n;i=i+1u){ if (x[i]>mx){ mx=x[i]; } }
     var s=0.0;
     for (var i=0u;i<n;i=i+1u){ let e=exp(x[i]-mx); outp[i]=e; s=s+e; }
     for (var i=0u;i<n;i=i+1u){ outp[i]=outp[i]/s; }
   }"

def attnWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
   @group(0) @binding(1) var<storage, read_write> K: array<f32>;
   @group(0) @binding(2) var<storage, read_write> V: array<f32>;
   @group(0) @binding(3) var<storage, read_write> mask: array<f32>;
   @group(0) @binding(4) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let hd=8u; let nq=3u; let nk=5u;
     for (var q=0u;q<nq;q=q+1u){
       var sc: array<f32, 5>;
       var mx = -1e30;
       for (var k=0u;k<nk;k=k+1u){ var s=0.0;
         for (var d=0u;d<hd;d=d+1u){ s = s + Q[q*hd+d]*K[k*hd+d]; }
         s = s + mask[q*nk+k]; sc[k]=s; if (s>mx){ mx=s; } }
       var sum=0.0;
       for (var k=0u;k<nk;k=k+1u){ sc[k]=exp(sc[k]-mx); sum=sum+sc[k]; }
       for (var d=0u;d<hd;d=d+1u){ var acc=0.0;
         for (var k=0u;k<nk;k=k+1u){ acc = acc + (sc[k]/sum)*V[k*hd+d]; }
         outp[q*hd+d]=acc; }
     }
   }"

def G : String := "/tmp/dg_golden"

def main : IO Unit := do
  IO.println "[dg-gpu-parity] initializing WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let mut allOk := true

  let check (name : String) (out gold : Array Float) (tol : Float) : IO Bool := do
    let err := maxAbsDiff out gold
    let ok := out.size == gold.size && err < tol
    let mark := if ok then "✓" else "✗"
    IO.println s!"  {mark} {name}  maxAbsErr={err}  (tol {tol})"
    return ok

  -- RMSNorm
  let x ← readF32Bin s!"{G}/rmsnorm/x.bin"
  let w ← readF32Bin s!"{G}/rmsnorm/w.bin"
  let gold ← readF32Bin s!"{G}/rmsnorm/out.bin"
  allOk := (← check "rmsnorm" (← runKernel device rmsnormWGSL #[x, w] gold.size) gold 1e-4) && allOk
  -- RoPE
  let x ← readF32Bin s!"{G}/rope/x.bin"
  let gold ← readF32Bin s!"{G}/rope/out.bin"
  allOk := (← check "rope" (← runKernel device ropeWGSL #[x] gold.size) gold 1e-4) && allOk
  -- GeGLU
  let gate ← readF32Bin s!"{G}/geglu/gate.bin"
  let up ← readF32Bin s!"{G}/geglu/up.bin"
  let gold ← readF32Bin s!"{G}/geglu/out.bin"
  allOk := (← check "geglu" (← runKernel device gegluWGSL #[gate, up] gold.size) gold 3e-3) && allOk
  -- MatMul
  let a ← readF32Bin s!"{G}/matmul/a.bin"
  let xv ← readF32Bin s!"{G}/matmul/x.bin"
  let gold ← readF32Bin s!"{G}/matmul/out.bin"
  allOk := (← check "matmul" (← runKernel device matmulWGSL #[a, xv] gold.size) gold 1e-4) && allOk
  -- Softmax
  let x ← readF32Bin s!"{G}/softmax/x.bin"
  let gold ← readF32Bin s!"{G}/softmax/out.bin"
  allOk := (← check "softmax" (← runKernel device softmaxWGSL #[x] gold.size) gold 1e-4) && allOk
  -- Attention
  let q ← readF32Bin s!"{G}/attn/q.bin"
  let k ← readF32Bin s!"{G}/attn/k.bin"
  let v ← readF32Bin s!"{G}/attn/v.bin"
  let mask ← readF32Bin s!"{G}/attn/mask.bin"
  let gold ← readF32Bin s!"{G}/attn/out.bin"
  allOk := (← check "attn" (← runKernel device attnWGSL #[q, k, v, mask] gold.size) gold 1e-4) && allOk

  if allOk then
    IO.println "✓ ALL modules PASS on Metal — DiffusionGemma kernels run on macOS, no CUDA."
  else
    IO.println "✗ some modules FAILED"
    throw (IO.userError "gpu parity failed")
