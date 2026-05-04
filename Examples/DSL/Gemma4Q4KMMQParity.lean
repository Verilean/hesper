import Hesper.Models.Gemma4
import Hesper.Layers.Linear
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic
import Hesper.WGSL.Monad

/-!
# Q4_K batched matmul parity test (baseline vs MMQ2 vs MMQ5)

Compares hesper's batched Q4_K matmul kernels on real Gemma 4 weights:

  REFERENCE (1-warp):  `q4kMatmulBatchKernel`        block=32,  grid=(outDim, seqLen)
  MMQ2 (default):      `q4kMatmulBatchMMQ2Kernel`    block=256, grid=(outDim/32, seqLen/8) — X smem
  MMQ5 (llama-shape):  `q4kMatmulBatchMMQ5Kernel`    block=256, grid=(outDim/128, seqLen/64) — X+Y smem,
                                                     mmq_y=128, mmq_x=64, nwarps=8 (matches llama.cpp's
                                                     `mul_mat_q<Q4_K, mmq_x=64>`)

All three consume the same Q4_K weight tensor and pre-quantized Q8_1
input. They must produce near-bit-identical (≤1e-3) outputs of shape
`[outDim, seqLen]`.

Build: `lake build gemma4-q4k-mmq-parity`
Run:   `HESPER_PARITY_SEQLEN=64 lake exe gemma4-q4k-mmq-parity`
       (default seqLen=64 to engage MMQ5 mmq_x=64 tile)
-/

open Hesper
open Hesper.WGSL.Monad

def maxAbsDiffMMQ (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def main : IO Unit := do
  IO.println "=== Q4_K batched matmul parity (baseline vs MMQ2 vs MMQ5) ==="

  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  Hesper.Layers.Linear.dp4aEnabled.set true
  let ctx ← Hesper.CUDAContext.init

  IO.println s!"[Load] {modelPath}"
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF ctx modelPath
  let block0 ← match model.blocks[0]? with
    | some b => pure b
    | none   => do
      IO.println "FAIL: model has no blocks"; IO.Process.exit 1

  -- HESPER_PARITY_SHAPE selects which Q4_K weight matrix to test.
  -- Defaults to wQ. Possible values: wQ | wK | wO | gate | down.
  -- (wV ≡ wK shape; up ≡ gate shape — skipped to keep tests minimal.)
  let shapeName := (← IO.getEnv "HESPER_PARITY_SHAPE").getD "wQ"
  let wQ ← match shapeName with
    | "wQ"   => pure block0.attention.wQ
    | "wK"   => pure block0.attention.wK
    | "wO"   => pure block0.attention.wO
    | "gate" => pure block0.ffn.gate
    | "down" => pure block0.ffn.down
    | other  => do
      IO.println s!"FAIL: unknown HESPER_PARITY_SHAPE '{other}' (use wQ/wK/wO/gate/down)"
      IO.Process.exit 1
  if wQ.quantFormat != .Q4_K then
    IO.println s!"FAIL: {shapeName} is not Q4_K (got {repr wQ.quantFormat})"; IO.Process.exit 1
  let inDim  := wQ.config.inDim
  let outDim := wQ.config.outDim

  let seqLen := match (← IO.getEnv "HESPER_PARITY_SEQLEN") with
    | some s => (s.toNat?).getD 64
    | none => 64
  IO.println s!"[Test] shape={shapeName}  inDim={inDim} outDim={outDim} seqLen={seqLen}"
  if outDim % 64 != 0 then
    IO.println s!"FAIL: outDim {outDim} not divisible by 64 (MMQ5/6/7 require mmq_y=64)"; IO.Process.exit 1
  if inDim % 256 != 0 then
    IO.println s!"FAIL: inDim {inDim} not divisible by 256"; IO.Process.exit 1

  -- 1. Build a deterministic f32 input.
  let totalIn := inDim * seqLen
  let inArr : Array Float :=
    (List.range totalIn).toArray.map (fun i =>
      Float.sin (i.toFloat * 0.013) * 0.4)
  let inBytes ← Hesper.Basic.floatArrayToBytes inArr

  let inputBuf  ← GPUBackend.allocBuffer ctx ((totalIn * 4).toUSize)
  GPUBackend.writeBuffer ctx inputBuf inBytes

  -- 2. Quantize input → Q8_1 batched.
  let nQ8Blocks := inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
  let q8Buf ← GPUBackend.allocBuffer ctx q8BufBytes
  let q8Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.quantizeQ8_1BatchKernel inDim seqLen)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("test-q8-quant", inDim, seqLen)) q8Ref

  let outBufSz : USize := (outDim * seqLen * 4).toUSize

  -- 3. Reference: 1-warp baseline.
  let outBufRef ← GPUBackend.allocBuffer ctx outBufSz
  let refRef ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchKernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufRef)]
    { numWorkgroups := (outDim, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("test-1warp", inDim, outDim, seqLen)) refRef

  let outBytesRef ← GPUBackend.readBuffer ctx outBufRef outBufSz
  let mut outArrRef : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesRef (i * 4)
    outArrRef := outArrRef.push f
  IO.println s!"[Ref ] out[0..4]   = {outArrRef.toList.take 4}"
  IO.println s!"[Ref ] out[col1,0..3] = {(outArrRef.toList.drop outDim).take 4}"

  -- 4. MMQ2 (smem-staged X).
  IO.println ""
  IO.println "=== MMQ2 (smem-staged X) ==="
  let outBufMMQ2 ← GPUBackend.allocBuffer ctx outBufSz
  let mmq2Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  let nTileColsMMQ2 := (seqLen + 7) / 8
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchMMQ2Kernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufMMQ2)]
    { numWorkgroups := (outDim / 32, nTileColsMMQ2, 1),
      workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("test-mmq2", inDim, outDim, seqLen)) mmq2Ref

  let outBytesMMQ2 ← GPUBackend.readBuffer ctx outBufMMQ2 outBufSz
  let mut outArrMMQ2 : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesMMQ2 (i * 4)
    outArrMMQ2 := outArrMMQ2.push f
  IO.println s!"[MMQ2] out[0..4]   = {outArrMMQ2.toList.take 4}"
  IO.println s!"[MMQ2] out[col1,0..3] = {(outArrMMQ2.toList.drop outDim).take 4}"

  let err2 := maxAbsDiffMMQ outArrRef outArrMMQ2
  IO.println s!"[Parity MMQ2] max |err| = {err2}"
  if err2 < 1.0e-3 then
    IO.println s!"PASS: MMQ2 matches baseline within 1e-3"
  else
    IO.println s!"FAIL: MMQ2 differs from baseline (max |err| = {err2})"
    IO.Process.exit 1

  -- 5. MMQ5 (full llama-shape: mmq_y=128, mmq_x=64, X+Y smem).
  IO.println ""
  IO.println "=== MMQ5 (mmq_y=64, mmq_x=32, X+Y smem — half tile rev. 2026-05-02) ==="
  let outBufMMQ5 ← GPUBackend.allocBuffer ctx outBufSz
  let mmq5Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  let nTileColsMMQ5 := (seqLen + 31) / 32
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchMMQ5Kernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufMMQ5)]
    { numWorkgroups := (outDim / 64, nTileColsMMQ5, 1),
      workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("test-mmq5", inDim, outDim, seqLen)) mmq5Ref

  let outBytesMMQ5 ← GPUBackend.readBuffer ctx outBufMMQ5 outBufSz
  let mut outArrMMQ5 : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesMMQ5 (i * 4)
    outArrMMQ5 := outArrMMQ5.push f
  IO.println s!"[MMQ5] out[0..4]   = {outArrMMQ5.toList.take 4}"
  IO.println s!"[MMQ5] out[col1,0..3] = {(outArrMMQ5.toList.drop outDim).take 4}"

  let err5 := maxAbsDiffMMQ outArrRef outArrMMQ5
  IO.println s!"[Parity MMQ5] max |err| = {err5}"

  -- Locate first major mismatch.
  let mut firstBigIdx5 : Int := -1
  for i in [0:outDim * seqLen] do
    let d := (outArrRef[i]! - outArrMMQ5[i]!).abs
    if d > 0.01 && firstBigIdx5 == -1 then
      firstBigIdx5 := i.toInt32.toInt
      let row := i % outDim
      let col := i / outDim
      IO.println s!"[Diff MMQ5] first |err|>0.01 at i={i} (row={row}, col={col}): ref={outArrRef[i]!} mmq5={outArrMMQ5[i]!}"

  if err5 < 1.0e-3 then
    IO.println s!"PASS: MMQ5 matches baseline within 1e-3"
  else
    IO.println s!"FAIL: MMQ5 differs from baseline (max |err| = {err5})"
    IO.Process.exit 1

  -- 6. MMQ6 (cp.async loads, single-stage). Same shape and arithmetic as
  -- MMQ5 but Phase A/A' use cp.async.ca instead of ld→reg→st.shared.
  IO.println ""
  IO.println "=== MMQ6 (cp.async X+Y loads, single-stage) ==="
  let outBufMMQ6 ← GPUBackend.allocBuffer ctx outBufSz
  let mmq6Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  let nTileColsMMQ6 := (seqLen + 31) / 32
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchMMQ6Kernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufMMQ6)]
    { numWorkgroups := (outDim / 64, nTileColsMMQ6, 1),
      workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("test-mmq6", inDim, outDim, seqLen)) mmq6Ref

  let outBytesMMQ6 ← GPUBackend.readBuffer ctx outBufMMQ6 outBufSz
  let mut outArrMMQ6 : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesMMQ6 (i * 4)
    outArrMMQ6 := outArrMMQ6.push f
  IO.println s!"[MMQ6] out[0..4]   = {outArrMMQ6.toList.take 4}"
  IO.println s!"[MMQ6] out[col1,0..3] = {(outArrMMQ6.toList.drop outDim).take 4}"

  let err6 := maxAbsDiffMMQ outArrRef outArrMMQ6
  IO.println s!"[Parity MMQ6] max |err| = {err6}"

  -- Locate first major mismatch.
  let mut firstBigIdx6 : Int := -1
  for i in [0:outDim * seqLen] do
    let d := (outArrRef[i]! - outArrMMQ6[i]!).abs
    if d > 0.01 && firstBigIdx6 == -1 then
      firstBigIdx6 := i.toInt32.toInt
      let row := i % outDim
      let col := i / outDim
      IO.println s!"[Diff MMQ6] first |err|>0.01 at i={i} (row={row}, col={col}): ref={outArrRef[i]!} mmq6={outArrMMQ6[i]!}"

  if err6 < 1.0e-3 then
    IO.println s!"PASS: MMQ6 matches baseline within 1e-3"
  else
    IO.println s!"FAIL: MMQ6 differs from baseline (max |err| = {err6})"
    IO.Process.exit 1

  -- 7. MMQ7 (cp.async multi-stage prefetch pipeline).
  IO.println ""
  IO.println "=== MMQ7 (cp.async multi-stage pipeline, double-buffer smem) ==="
  let outBufMMQ7 ← GPUBackend.allocBuffer ctx outBufSz
  let mmq7Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  let nTileColsMMQ7 := (seqLen + 31) / 32
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchMMQ7Kernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufMMQ7)]
    { numWorkgroups := (outDim / 64, nTileColsMMQ7, 1),
      workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("test-mmq7", inDim, outDim, seqLen)) mmq7Ref

  let outBytesMMQ7 ← GPUBackend.readBuffer ctx outBufMMQ7 outBufSz
  let mut outArrMMQ7 : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesMMQ7 (i * 4)
    outArrMMQ7 := outArrMMQ7.push f
  IO.println s!"[MMQ7] out[0..4]   = {outArrMMQ7.toList.take 4}"
  IO.println s!"[MMQ7] out[col1,0..3] = {(outArrMMQ7.toList.drop outDim).take 4}"

  let err7 := maxAbsDiffMMQ outArrRef outArrMMQ7
  IO.println s!"[Parity MMQ7] max |err| = {err7}"

  let mut firstBigIdx7 : Int := -1
  for i in [0:outDim * seqLen] do
    let d := (outArrRef[i]! - outArrMMQ7[i]!).abs
    if d > 0.01 && firstBigIdx7 == -1 then
      firstBigIdx7 := i.toInt32.toInt
      let row := i % outDim
      let col := i / outDim
      IO.println s!"[Diff MMQ7] first |err|>0.01 at i={i} (row={row}, col={col}): ref={outArrRef[i]!} mmq7={outArrMMQ7[i]!}"

  if err7 < 1.0e-3 then
    IO.println s!"PASS: MMQ7 matches baseline within 1e-3"
  else
    IO.println s!"FAIL: MMQ7 differs from baseline (max |err| = {err7})"
    IO.Process.exit 1

  -- 8. MMQ8 (vectorized 16-byte cp.async.cg, padded smem stride).
  IO.println ""
  IO.println "=== MMQ8 (cp.async.cg 16-byte vec, stride 40/76) ==="
  let outBufMMQ8 ← GPUBackend.allocBuffer ctx outBufSz
  let mmq8Ref ← IO.mkRef (none : Option (GPUBackend.CachedDispatch _))
  let nTileColsMMQ8 := (seqLen + 31) / 32
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.q4kMatmulBatchMMQ8Kernel wQ.config seqLen)
    [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", outBufMMQ8)]
    { numWorkgroups := (outDim / 64, nTileColsMMQ8, 1),
      workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("test-mmq8", inDim, outDim, seqLen)) mmq8Ref

  let outBytesMMQ8 ← GPUBackend.readBuffer ctx outBufMMQ8 outBufSz
  let mut outArrMMQ8 : Array Float := Array.mkEmpty (outDim * seqLen)
  for i in [0:outDim * seqLen] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesMMQ8 (i * 4)
    outArrMMQ8 := outArrMMQ8.push f
  IO.println s!"[MMQ8] out[0..4]   = {outArrMMQ8.toList.take 4}"
  IO.println s!"[MMQ8] out[col1,0..3] = {(outArrMMQ8.toList.drop outDim).take 4}"

  let err8 := maxAbsDiffMMQ outArrRef outArrMMQ8
  IO.println s!"[Parity MMQ8] max |err| = {err8}"

  let mut firstBigIdx8 : Int := -1
  for i in [0:outDim * seqLen] do
    let d := (outArrRef[i]! - outArrMMQ8[i]!).abs
    if d > 0.01 && firstBigIdx8 == -1 then
      firstBigIdx8 := i.toInt32.toInt
      let row := i % outDim
      let col := i / outDim
      IO.println s!"[Diff MMQ8] first |err|>0.01 at i={i} (row={row}, col={col}): ref={outArrRef[i]!} mmq8={outArrMMQ8[i]!}"

  if err8 < 1.0e-3 then
    IO.println s!"PASS: MMQ8 matches baseline within 1e-3"
  else
    IO.println s!"FAIL: MMQ8 differs from baseline (max |err| = {err8})"
    IO.Process.exit 1
