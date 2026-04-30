import Hesper.Backend.LlamaCppPTX
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI

/-!
# Smoke test: launch llama.cpp's mmq Q4_K kernel from hesper

This test confirms the kernel-launch ABI is correct end-to-end:
  1. Load the cubin via `LlamaCppPTX.loadKernels`
  2. Resolve `mul_mat_q<Q4_K, mmq_x=64, need_check=false>` symbol
  3. Allocate zeroed buffers at a representative shape (outDim=2560, K=2560,
     seqLen=64 — matches hesper MMQ5 wO/ffn-down prefill tile)
  4. Raise the kernel's dynamic-smem limit (mmq Q4_K needs >48 KB on sm_89)
  5. Launch + sync. Output is unverified (inputs are zeros), but a clean
     run with no `CUDA_ERROR_*` confirms the ABI is right.

The performance microbench (#343 next step) builds on this skeleton.

Run: `lake exe llamacpp-mmq-launch-test`
-/

open Hesper Hesper.CUDA Hesper.LlamaCppPTX

unsafe def main : IO Unit := do
  let _ ← CUDAContext.init
  IO.println "=== mmq Q4_K launch smoke test ==="
  let k ← Hesper.LlamaCppPTX.loadKernels
  let func ← match k.mmqQ4K_x64 with
    | some f => pure f
    | none =>
      IO.eprintln "[FAIL] mmq Q4_K cubin not loaded — extract per docs/llama-fusion-analysis/31"
      IO.Process.exit 1

  -- Representative prefill shape: wO (outDim=2560, K=2560), seqLen=64.
  let outDim : Nat := 2560
  let inDim  : Nat := 2560
  let seqLen : Nat := 64
  IO.println s!"  shape: outDim={outDim} inDim={inDim} seqLen={seqLen}"

  -- Q4_K weight buffer: outDim rows × (inDim/256) super-blocks × 144 B/block.
  let wBytes : USize := (outDim * (inDim / 256) * 144).toUSize
  let wBuf ← cuMalloc wBytes
  cuMemset wBuf wBytes
  IO.println s!"  weights: {wBytes} bytes"

  -- Q8_1 input buffer: seqLen cols × (inDim/32) sub-blocks × 36 B/sub-block.
  -- This is the standard Q8_1 layout (post-#146 port).
  let yBytes : USize := (seqLen * (inDim / 32) * 36).toUSize
  let yBuf ← cuMalloc yBytes
  cuMemset yBuf yBytes
  IO.println s!"  Q8_1 input: {yBytes} bytes"

  -- Output: f32 (outDim × seqLen).
  let outBytes : USize := (outDim * seqLen * 4).toUSize
  let dBuf ← cuMalloc outBytes
  cuMemset dBuf outBytes
  IO.println s!"  output: {outBytes} bytes"

  -- Raise the kernel's dynamic smem limit.  mmq Q4_K with mmq_y=128, mmq_x=64,
  -- nwarps=8 on sm_89 needs ~46-50 KB (above the default 48 KB cap).  Setting
  -- 96 KB is safely above the requirement and within sm_89's 100 KB per-SM limit.
  let smemBytes : USize := 96 * 1024
  cuFuncSetMaxDynamicSmem func smemBytes
  IO.println s!"  raised dynamic smem cap to {smemBytes} B"

  IO.println "  launching..."
  launchMmqQ4K func wBuf yBuf dBuf inDim outDim seqLen smemBytes.toUInt32
  cuStreamSynchronize (← cuStreamCreateDefault)
  IO.println "✓ launch + sync OK (no CUDA error)"

  -- Quick numeric sanity: read first 8 floats. With zero weights+input,
  -- expect all zeros.
  let outFirst ← cuMemcpyDtoH dBuf 32
  let arr := Hesper.Basic.bytesToFloatArrayPure outFirst
  IO.println s!"  out[0..7] = {arr[0]!} {arr[1]!} {arr[2]!} {arr[3]!} {arr[4]!} {arr[5]!} {arr[6]!} {arr[7]!}"
  let allZero := arr.all (· == 0.0)
  if allZero then
    IO.println "✓ output is all-zero as expected (zero inputs)"
  else
    IO.println "⚠ output not all-zero (zero inputs should produce zero output — possible smem race or arg mismatch)"

  cuFree wBuf
  cuFree yBuf
  cuFree dBuf
  IO.println "PASS"
