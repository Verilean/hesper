import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.WebGPU
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.FlashAttention
import Hesper
import Hesper.Backend
import Hesper.Models.BitNet
import Hesper.CUDA.CodeGen
import Hesper.Layers.Embedding

/-!
# FlashAttention sub-kernel unit tests: WebGPU vs CUDA

Tests individual FlashAttention building blocks in isolation.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL (Exp)

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 : Int := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toNat.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let bits := f64ToF32Bits f
    acc.push bits.toUInt8 |>.push (bits>>>8).toUInt8 |>.push (bits>>>16).toUInt8 |>.push (bits>>>24).toUInt8
  ) ByteArray.empty

private def unpackFloat (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -fv else fv

/-- Test 1: Dot product with shared memory reduction.
    Extracted from FlashAttention's inner loop.
    q[d] * k[d] → shared_reduce → tree reduce → output[0] = sum -/
def dotProductReduceKernel (dim : Nat) (workgroupSize : Nat := 32) : ShaderM Unit := do
  let lid ← localId
  let tid := Exp.vec3X lid

  let _q ← declareInputBuffer "q" (.array (.scalar .f32) dim)
  let _k ← declareInputBuffer "k" (.array (.scalar .f32) dim)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1)

  sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)

  -- Partial dot product
  let partialVar ← var (.scalar .f32) (Exp.litF32 0.0)
  loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun d => do
    let qVal ← readBuffer (ty := .scalar .f32) (n := dim) "q" d
    let kVal ← readBuffer (ty := .scalar .f32) (n := dim) "k" d
    assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

  -- Shared memory reduction
  writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
  barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
      let b ← readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
      writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add a b)
    ) (pure ())
    barrier
    stride := stride / 2

  if_ (Exp.eq tid (Exp.litU32 0)) (do
    let result ← readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "output" (Exp.litU32 0) result
  ) (pure ())

def runDotProductTest [GPUBackend β] (ctx : β) (name : String) : IO Float := do
  let dim := 128; let wgSize := 32
  let qBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let kBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let outBuf ← GPUBackend.allocBuffer ctx (4 : USize)

  -- q = [1, 1, 1, ...], k = [1, 2, 3, ..., 128]
  let qData := Array.replicate dim 1.0
  let kData := Array.range dim |>.map (fun i => (i + 1).toFloat)
  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)

  let bufs : List (String × GPUBackend.Buf β) :=
    [("q", qBuf), ("k", kBuf), ("output", outBuf)]
  GPUBackend.execute ctx (dotProductReduceKernel dim wgSize) bufs
    { workgroupSize := { x := wgSize }, numWorkgroups := (1, 1, 1) }

  let result ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let v := unpackFloat result 0
  -- Expected: sum(1..128) = 128*129/2 = 8256
  IO.println s!"  {name} dot product: {v} (expect 8256.0)"
  GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf; GPUBackend.freeBuffer ctx outBuf
  return v

/-- Test 2: subgroupAdd-based dot product (same as FlashAttention uses) -/
def dotProductSubgroupKernel (dim : Nat) : ShaderM Unit := do
  let lid ← localId
  let tid := Exp.vec3X lid

  let _q ← declareInputBuffer "q" (.array (.scalar .f32) dim)
  let _k ← declareInputBuffer "k" (.array (.scalar .f32) dim)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1)

  let partialVar ← var (.scalar .f32) (Exp.litF32 0.0)
  loop tid (Exp.litU32 dim) (Exp.litU32 32) fun d => do
    let qVal ← readBuffer (ty := .scalar .f32) (n := dim) "q" d
    let kVal ← readBuffer (ty := .scalar .f32) (n := dim) "k" d
    assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

  -- subgroupAdd
  varNamed "dot" (.scalar .f32) (Exp.subgroupAdd (Exp.var partialVar))
  let dot : Exp (.scalar .f32) := Exp.var "dot"

  if_ (Exp.eq tid (Exp.litU32 0)) (do
    writeBuffer (ty := .scalar .f32) "output" (Exp.litU32 0) dot
  ) (pure ())

def runSubgroupDotTest [GPUBackend β] (ctx : β) (name : String) : IO Float := do
  let dim := 128
  let qBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let kBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let outBuf ← GPUBackend.allocBuffer ctx (4 : USize)

  let qData := Array.replicate dim 1.0
  let kData := Array.range dim |>.map (fun i => (i + 1).toFloat)
  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)

  let bufs : List (String × GPUBackend.Buf β) :=
    [("q", qBuf), ("k", kBuf), ("output", outBuf)]
  GPUBackend.execute ctx (dotProductSubgroupKernel dim) bufs
    { workgroupSize := { x := 32 }, numWorkgroups := (1, 1, 1),
      extensions := ["subgroups"] }

  let result ← GPUBackend.readBuffer ctx outBuf (4 : USize)
  let v := unpackFloat result 0
  IO.println s!"  {name} subgroup dot: {v} (expect 8256.0)"
  GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf; GPUBackend.freeBuffer ctx outBuf
  return v

private def unpackFloats (ba : ByteArray) (n : Nat) : Array Float :=
  Array.range n |>.map (fun i => unpackFloat ba i)

/-- Test 3: Full subgroup FlashAttention kernel with known Q, K, V.
    1 head, 1 KV head, headDim=32 (1 dim per lane), seqLen=4. -/
def runFullFlashAttnTest [GPUBackend β] (ctx : β) (name : String) : IO (Array Float) := do
  let numHeads := 1; let numKVHeads := 1; let headDim := 32; let maxSeqLen := 8; let cacheLen := 4
  let scale := 1.0 / Float.sqrt headDim.toFloat  -- 1/sqrt(32)

  -- Allocate buffers
  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let outSize := (numHeads * headDim * 4).toUSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let outBuf ← GPUBackend.allocBuffer ctx outSize

  -- Q = [1, 0, 0, ...] (first dim = 1, rest = 0)
  let qData := Array.range headDim |>.map (fun i => if i == 0 then 1.0 else 0.0)
  -- K cache: 4 positions, each has k[0]=pos_idx+1, rest=0
  -- Layout: kvHead * maxSeqLen * headDim, so k[pos][d] at index pos*headDim+d
  let mut kData := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for pos in [0:cacheLen] do
    kData := kData.set! (pos * headDim) (pos + 1).toFloat  -- k[pos][0] = pos+1
  -- V cache: 4 positions with distinct patterns
  let mut vData := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for pos in [0:cacheLen] do
    for d in [0:headDim] do
      vData := vData.set! (pos * headDim + d) ((pos * headDim + d).toFloat * 0.01)

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)
  GPUBackend.writeBuffer ctx vBuf (packFloats vData)

  let kernel := Hesper.WGSL.FlashAttention.flashAttentionSubgroupKernel
    numHeads numKVHeads maxSeqLen headDim cacheLen scale
  let bufs : List (String × GPUBackend.Buf β) :=
    [("q", qBuf), ("k_cache", kBuf), ("v_cache", vBuf), ("output", outBuf)]
  GPUBackend.execute ctx kernel bufs
    { workgroupSize := { x := 32 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  let result ← GPUBackend.readBuffer ctx outBuf outSize
  let output := unpackFloats result headDim

  IO.println s!"  {name} FlashAttn output[0..3]: {output[0]!}, {output[1]!}, {output[2]!}, {output[3]!}"

  GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf
  GPUBackend.freeBuffer ctx vBuf; GPUBackend.freeBuffer ctx outBuf
  return output

private def packNatAsU32 (ba : ByteArray) (v : Nat) : ByteArray :=
  ba.push (v % 256).toUInt8 |>.push ((v / 256) % 256).toUInt8
    |>.push ((v / 65536) % 256).toUInt8 |>.push ((v / 16777216) % 256).toUInt8

/-- Test 4: Production flashAttentionParamsKernel (shared mem reduction + params buffer).
    This is the actual kernel used in BitNet inference. -/
def runParamsFlashAttnTest [GPUBackend β] (ctx : β) (name : String) : IO (Array Float) := do
  let numHeads := 1; let numKVHeads := 1; let headDim := 128; let maxSeqLen := 8; let cacheLen := 4
  let scale := 1.0 / Float.sqrt headDim.toFloat
  let wgSize := 256

  -- Allocate buffers (q_output is read-write, k/v are read, params is read u32)
  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let paramSize := (2 * 4).toUSize  -- 2 x u32
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let paramsBuf ← GPUBackend.allocBuffer ctx paramSize

  -- Q = [1.0, 0.5, 0.25, ...] repeating
  let qData := Array.range headDim |>.map (fun i => 1.0 / (i + 1).toFloat)
  -- K cache: 4 positions with increasing values
  let mut kData := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for pos in [0:cacheLen] do
    for d in [0:headDim] do
      kData := kData.set! (pos * headDim + d) ((pos * headDim + d).toFloat * 0.001)
  -- V cache
  let mut vData := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for pos in [0:cacheLen] do
    for d in [0:headDim] do
      vData := vData.set! (pos * headDim + d) ((pos + 1).toFloat * 0.1 + d.toFloat * 0.001)
  -- Params: [currentPos=3, cacheLen=4]
  let paramsData := packNatAsU32 (packNatAsU32 ByteArray.empty 3) cacheLen

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)
  GPUBackend.writeBuffer ctx vBuf (packFloats vData)
  GPUBackend.writeBuffer ctx paramsBuf paramsData

  let kernel := Hesper.WGSL.FlashAttention.flashAttentionParamsKernel
    numHeads numKVHeads maxSeqLen headDim scale wgSize
  let bufs : List (String × GPUBackend.Buf β) :=
    [("q_output", qBuf), ("k_cache", kBuf), ("v_cache", vBuf), ("params", paramsBuf)]
  GPUBackend.execute ctx kernel bufs
    { workgroupSize := { x := wgSize }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  let result ← GPUBackend.readBuffer ctx qBuf qSize
  let output := unpackFloats result headDim

  IO.println s!"  {name} ParamsFA output[0..3]: {output[0]!}, {output[1]!}, {output[2]!}, {output[3]!}"

  GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf
  GPUBackend.freeBuffer ctx vBuf; GPUBackend.freeBuffer ctx paramsBuf
  return output

def main : IO Unit := do
  IO.println "═══ FlashAttention Sub-Kernel Tests ═══"

  -- WebGPU
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let wDot ← runDotProductTest device "WebGPU"
  let wSub ← runSubgroupDotTest device "WebGPU"

  -- CUDA
  let ctx ← Hesper.CUDAContext.init
  let cDot ← runDotProductTest ctx "CUDA"
  let cSub ← runSubgroupDotTest ctx "CUDA"

  IO.println "\nComparison (dot products):"
  IO.println s!"  Shared reduce: WebGPU={wDot}, CUDA={cDot}, diff={(wDot - cDot).abs}"
  IO.println s!"  SubgroupAdd:   WebGPU={wSub}, CUDA={cSub}, diff={(wSub - cSub).abs}"

  if (wDot - cDot).abs < 1.0 && (wSub - cSub).abs < 1.0 then
    IO.println "✓ Dot product sub-kernels match!"
  else
    IO.println "✗ Dot product DIVERGENCE"

  -- Test 3: Full FlashAttention
  IO.println "\n── Test 3: Full Subgroup FlashAttention ──"
  let wFA ← runFullFlashAttnTest device "WebGPU"
  let cFA ← runFullFlashAttnTest ctx "CUDA"

  IO.println "\nFull FlashAttn comparison (first 8 dims):"
  let mut maxDiff := 0.0
  for i in [0:min wFA.size 8] do
    let w := wFA[i]!; let c := cFA[i]!; let d := (w - c).abs
    IO.println s!"  [{i}] WebGPU={w}, CUDA={c}, diff={d}"
    if d > maxDiff then maxDiff := d

  if maxDiff < 0.01 then
    IO.println s!"✓ FlashAttention match! (max diff={maxDiff})"
  else
    IO.println s!"✗ FlashAttention DIVERGENCE (max diff={maxDiff})"

  -- Test 4: Production params kernel
  IO.println "\n── Test 4: Production ParamsFlashAttention (wg=256, shared reduce) ──"
  let wPFA ← runParamsFlashAttnTest device "WebGPU"
  let cPFA ← runParamsFlashAttnTest ctx "CUDA"

  IO.println "\nParamsFA comparison (first 8 dims):"
  let mut maxDiffP := 0.0
  for i in [0:min wPFA.size 8] do
    let w := wPFA[i]!; let c := cPFA[i]!; let d := (w - c).abs
    IO.println s!"  [{i}] WebGPU={w}, CUDA={c}, diff={d}"
    if d > maxDiffP then maxDiffP := d

  if maxDiffP < 0.01 then
    IO.println s!"✓ ParamsFlashAttention match! (max diff={maxDiffP})"
  else
    IO.println s!"✗ ParamsFlashAttention DIVERGENCE (max diff={maxDiffP})"

  -- Test 5: Argmax kernel
  IO.println "\n── Test 5: Argmax Kernel ──"
  let vocabSize := 1024
  let mut logitsData := Array.replicate vocabSize 0.0
  logitsData := logitsData.set! 264 10.0
  logitsData := logitsData.set! 500 5.0

  -- WebGPU
  let wLogitsBuf ← GPUBackend.allocBuffer device (vocabSize * 4).toUSize
  let wArgmaxBuf ← GPUBackend.allocBuffer device (4 : USize)
  GPUBackend.writeBuffer device wLogitsBuf (packFloats logitsData)
  let wArgmax ← Hesper.Models.BitNet.gpuArgmax device wLogitsBuf wArgmaxBuf vocabSize
  IO.println s!"  WebGPU argmax: {wArgmax} (expect 264)"
  GPUBackend.freeBuffer device wLogitsBuf; GPUBackend.freeBuffer device wArgmaxBuf

  -- CUDA
  let cLogitsBuf ← GPUBackend.allocBuffer ctx (vocabSize * 4).toUSize
  let cArgmaxBuf ← GPUBackend.allocBuffer ctx (4 : USize)
  GPUBackend.writeBuffer ctx cLogitsBuf (packFloats logitsData)
  let cArgmax ← Hesper.Models.BitNet.gpuArgmax ctx cLogitsBuf cArgmaxBuf vocabSize
  IO.println s!"  CUDA argmax: {cArgmax} (expect 264)"
  GPUBackend.freeBuffer ctx cLogitsBuf; GPUBackend.freeBuffer ctx cArgmaxBuf

  if wArgmax == 264 && cArgmax == 264 then
    IO.println "✓ Argmax match!"
  else
    IO.println s!"✗ Argmax DIVERGENCE: WebGPU={wArgmax}, CUDA={cArgmax}"

  -- Test 6: Embedding lookup PTX dump (small config for inspection)
  IO.println "\n── Test 6: Embedding Lookup ──"
  let embConfig : Hesper.Layers.Embedding.Config := { vocabSize := 16, dim := 4 }
  let embKernel := Hesper.Layers.Embedding.embeddingLookupKernel embConfig 1 1
  let ptx := Hesper.CUDA.CodeGen.generatePTX "main" ⟨256, 1, 1⟩ embKernel
  IO.println "Generated PTX for embedding lookup (vocab=16, dim=4):"
  IO.println ptx

  -- Functional test: embedding lookup with known data
  let embTableData := Array.range (16 * 4) |>.map (fun i => i.toFloat * 0.1)
  -- Token 3 should give [1.2, 1.3, 1.4, 1.5] (indices 12-15)

  -- WebGPU
  let wEmbBuf ← GPUBackend.allocBuffer device (16 * 4 * 4).toUSize
  let wTokBuf ← GPUBackend.allocBuffer device (4 : USize)
  let wOutBuf ← GPUBackend.allocBuffer device (4 * 4).toUSize
  GPUBackend.writeBuffer device wEmbBuf (packFloats embTableData)
  GPUBackend.writeBuffer device wTokBuf (packNatAsU32 ByteArray.empty 3)
  let wBufs : List (String × GPUBackend.Buf Hesper.WebGPU.Device) :=
    [("token_ids", wTokBuf), ("embedding_table", wEmbBuf), ("output", wOutBuf)]
  GPUBackend.execute device embKernel wBufs (ExecConfig.dispatch1D 1)
  let wResult ← GPUBackend.readBuffer device wOutBuf (4 * 4).toUSize
  let wEmb := unpackFloats wResult 4
  IO.println s!"  WebGPU embedding[3]: {wEmb}"

  -- CUDA
  let cEmbBuf ← GPUBackend.allocBuffer ctx (16 * 4 * 4).toUSize
  let cTokBuf ← GPUBackend.allocBuffer ctx (4 : USize)
  let cOutBuf ← GPUBackend.allocBuffer ctx (4 * 4).toUSize
  GPUBackend.writeBuffer ctx cEmbBuf (packFloats embTableData)
  GPUBackend.writeBuffer ctx cTokBuf (packNatAsU32 ByteArray.empty 3)
  let cBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [("token_ids", cTokBuf), ("embedding_table", cEmbBuf), ("output", cOutBuf)]
  GPUBackend.execute ctx embKernel cBufs (ExecConfig.dispatch1D 1)
  let cResult ← GPUBackend.readBuffer ctx cOutBuf (4 * 4).toUSize
  let cEmb := unpackFloats cResult 4
  IO.println s!"  CUDA embedding[3]:  {cEmb}"

  if (wEmb[0]! - cEmb[0]!).abs < 0.001 && (wEmb[3]! - cEmb[3]!).abs < 0.001 then
    IO.println "✓ Embedding match!"
  else
    IO.println "✗ Embedding DIVERGENCE"
