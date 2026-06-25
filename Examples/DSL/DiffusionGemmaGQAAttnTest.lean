import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WebGPU.Types
import Hesper.Basic

/-!
# DiffusionGemma GQA attention-core test on Metal

Validates the multi-head **GQA** attention core (query head h attends via
kv head ⌊h/groupSize⌋) with **bidirectional** softmax (scale=1.0) — the
DiffusionGemma attention core — on the WebGPU→Metal backend against a CPU
reference.  The single-head math was already validated bit-exact vs ggml
(`diffusiongemma-gpu-parity` attn); this adds the GQA head mapping + a
full sequence.

Layout (ne0=hd fastest): Q[q,h,d]=Q[(q*nH+h)*hd+d], K/V[k,kvh,d]=[(k*nKV+kvh)*hd+d].

Run:  lake exe diffusiongemma-gqa-attn-test
-/

open Hesper.WebGPU

/-- Run a WGSL kernel: `inputs.size` read_write storage buffers (bindings 0..k-1)
    + one output buffer (binding k), single workgroup. -/
def runK (device : Device) (src : String) (inputs : Array (Array Float)) (outN : Nat) : IO (Array Float) := do
  let mkBuf (nbytes : Nat) : IO Buffer :=
    createBuffer device { size := nbytes.toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let mut layoutEntries : Array BindGroupLayoutEntry := #[]
  let mut bindEntries : Array BindGroupEntry := #[]
  let mut idx : Nat := 0
  for d in inputs do
    let buf ← mkBuf (d.size*4)
    writeBuffer device buf 0 (← Hesper.Basic.floatArrayToBytes d)
    layoutEntries := layoutEntries.push { binding := idx.toUInt32, visibility := .compute, bindingType := .buffer false }
    bindEntries := bindEntries.push { binding := idx.toUInt32, buffer := buf, offset := 0, size := (d.size*4).toUSize }
    idx := idx + 1
  let outBuf ← mkBuf (outN*4)
  layoutEntries := layoutEntries.push { binding := idx.toUInt32, visibility := .compute, bindingType := .buffer false }
  bindEntries := bindEntries.push { binding := idx.toUInt32, buffer := outBuf, offset := 0, size := (outN*4).toUSize }
  let shaderModule ← createShaderModule device src
  let layout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device { shaderModule, entryPoint := "main", bindGroupLayout := layout }
  let bindGroup ← createBindGroup device layout bindEntries
  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future
  let bytes ← mapBufferRead device outBuf 0 (outN*4).toUSize
  unmapBuffer outBuf
  Hesper.Basic.bytesToFloatArray bytes

-- dims: seqLen=4, nHead=4, nKV=2, headDim=8, groupSize=2 (GQA)
def gqaAttnWGSL : String :=
  "@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
   @group(0) @binding(1) var<storage, read_write> K: array<f32>;
   @group(0) @binding(2) var<storage, read_write> V: array<f32>;
   @group(0) @binding(3) var<storage, read_write> outp: array<f32>;
   @compute @workgroup_size(1) fn main() {
     let S=4u; let nH=4u; let nKV=2u; let hd=8u; let grp=2u;
     for (var q=0u;q<S;q=q+1u){ for (var h=0u;h<nH;h=h+1u){
       let kvh = h/grp;
       var sc: array<f32, 4>;
       var mx = -1e30;
       for (var k=0u;k<S;k=k+1u){ var s=0.0;
         for (var d=0u;d<hd;d=d+1u){ s = s + Q[(q*nH+h)*hd+d]*K[(k*nKV+kvh)*hd+d]; }
         sc[k]=s; if (s>mx){ mx=s; } }
       var sum=0.0;
       for (var k=0u;k<S;k=k+1u){ sc[k]=exp(sc[k]-mx); sum=sum+sc[k]; }
       for (var d=0u;d<hd;d=d+1u){ var acc=0.0;
         for (var k=0u;k<S;k=k+1u){ acc = acc + (sc[k]/sum)*V[(k*nKV+kvh)*hd+d]; }
         outp[(q*nH+h)*hd+d]=acc; }
     } }
   }"

def main : IO Unit := do
  let S := 4; let nH := 4; let nKV := 2; let hd := 8; let grp := nH/nKV
  IO.println "[dg-gqa-attn] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  -- deterministic synthetic Q/K/V
  let Q := (List.range (S*nH*hd)).toArray.map (fun i => Float.sin (i.toFloat * 0.07))
  let K := (List.range (S*nKV*hd)).toArray.map (fun i => Float.cos (i.toFloat * 0.05))
  let V := (List.range (S*nKV*hd)).toArray.map (fun i => Float.sin (i.toFloat * 0.09 + 1.0))
  let gpu ← runK device gqaAttnWGSL #[Q, K, V] (S*nH*hd)

  -- CPU reference (same GQA + bidirectional softmax math)
  let mut cpu := Array.replicate (S*nH*hd) 0.0
  for q in [0:S] do
    for h in [0:nH] do
      let kvh := h / grp
      let mut sc := Array.replicate S 0.0
      let mut mx := -1e30
      for k in [0:S] do
        let mut s := 0.0
        for d in [0:hd] do
          s := s + Q[(q*nH+h)*hd+d]! * K[(k*nKV+kvh)*hd+d]!
        sc := sc.set! k s
        if s > mx then mx := s
      let mut sum := 0.0
      for k in [0:S] do
        let e := Float.exp (sc[k]! - mx); sc := sc.set! k e; sum := sum + e
      for d in [0:hd] do
        let mut acc := 0.0
        for k in [0:S] do
          acc := acc + (sc[k]!/sum) * V[(k*nKV+kvh)*hd+d]!
        cpu := cpu.set! ((q*nH+h)*hd+d) acc

  let mut err := 0.0
  for i in [0:S*nH*hd] do
    let d := (gpu[i]! - cpu[i]!).abs
    if d > err then err := d
  IO.println s!"  S={S} nH={nH} nKV={nKV} hd={hd} grp={grp}  maxAbsErr={err}"
  IO.println s!"  gpu[0..4] = {(gpu.extract 0 4).toList}"
  IO.println s!"  cpu[0..4] = {(cpu.extract 0 4).toList}"
  if gpu.size == S*nH*hd && err < 1e-4 then
    IO.println "✓ GQA multi-head bidirectional attention core runs on Metal (matches CPU reference)"
  else
    IO.println "✗ FAIL"
    throw (IO.userError "gqa attn failed")
