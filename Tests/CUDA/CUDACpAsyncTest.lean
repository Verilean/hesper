import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper

/-!
# Smoke test for cp.async (sm_80+) Inst variants

Verifies that hesper's `cp.async.cg.shared.global` lowering produces
working PTX. Two layers of check:

1. **PTX dump check**: emitted PTX contains `cp.async.cg.shared.global`,
   `cp.async.commit_group`, and `cp.async.wait_group 0`.
2. **Bit-parity vs naïve copy**: 256 u32 elements are copied from a
   global input buffer to a global output buffer via cp.async into smem
   then a synchronous st from smem to output. Must match the original
   input element-for-element.

Kernel: 256 threads in 1 block. Thread `i` issues
  `cp.async.cg.shared.global [&s_buf[i]], [&input[i]], 4`
followed by `cp.async.commit_group; cp.async.wait_group 0; bar.sync`,
then `output[i] = s_buf[i]`.

This is the smallest possible exercise of the new Inst variants —
single-stage, no pipelining, no MMQ. If this fails, the multi-stage
MMQ5 rewrite cannot work.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

private def packU32 (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) (fun acc (x : UInt32) =>
    acc.push x.toUInt8
       |>.push (x >>> 8).toUInt8
       |>.push (x >>> 16).toUInt8
       |>.push (x >>> 24).toUInt8)

private def unpackU32 (ba : ByteArray) (n : Nat) : Array UInt32 := Id.run do
  let mut arr := #[]
  for i in [0:n] do
    let o := i * 4
    let b0 := ba.get! o |>.toUInt32
    let b1 := ba.get! (o+1) |>.toUInt32
    let b2 := ba.get! (o+2) |>.toUInt32
    let b3 := ba.get! (o+3) |>.toUInt32
    arr := arr.push (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))
  return arr

/-- 64 threads, each copies 16 bytes (4 u32s) from `input` to `s_buf`
    via cp.async (the only size cp.async.cg supports), then 4 scalar
    stores from smem to `output`. -/
def cpAsyncSmokeKernel : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let _input  ← ShaderM.declareInputBuffer  "input"  (.array (.scalar .u32) 256)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 256)
  ShaderM.sharedNamed "s_buf" (.array (.scalar .u32) 256)

  -- Each thread handles 4 u32s starting at index tid*4.
  -- Phase 1: issue async copy: s_buf[tid*4..(tid*4+4)] ← input[tid*4..(tid*4+4)] (16 bytes)
  let baseIdx := Exp.shiftLeft tid (Exp.litU32 2)  -- tid * 4
  let smemAddr   := ShaderM.sharedSymAddr "s_buf" 4 baseIdx
  let globalAddr := ShaderM.bufferAddr "input" 4 baseIdx
  ShaderM.cpAsync smemAddr globalAddr 16
  ShaderM.cpAsyncCommit
  ShaderM.cpAsyncWait 0
  ShaderM.barrier

  -- Phase 2: read 4 u32s from smem and write to output.
  for j in [0:4] do
    let off := Exp.add baseIdx (Exp.litU32 j)
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 256) "s_buf" off
    ShaderM.writeBuffer (ty := .scalar .u32) "output" off v

unsafe def main : IO Unit := do
  IO.println "═══ cp.async smoke test ═══"

  -- 1. PTX dump check
  let ptx := Hesper.CUDA.CodeGen.generatePTX "cp_async_smoke"
               { x := 64, y := 1, z := 1 } cpAsyncSmokeKernel
  IO.FS.writeFile "/tmp/cp_async_smoke.ptx" ptx
  IO.println s!"  (PTX written to /tmp/cp_async_smoke.ptx, {ptx.length} bytes)"

  let mut allPtxOk := true
  for marker in ["cp.async.cg.shared.global", "cp.async.commit_group", "cp.async.wait_group"] do
    if (ptx.splitOn marker).length < 2 then
      IO.println s!"✗ PTX missing `{marker}`"; allPtxOk := false
    else
      IO.println s!"✓ PTX contains `{marker}`"
  if !allPtxOk then IO.Process.exit 1

  -- 2. Numeric check: deterministic input pattern, expect output == input.
  let ctx ← Hesper.CUDAContext.init
  let inputArr : Array UInt32 := (List.range 256).map (fun i => (i.toUInt32 * 0x01010101) ^^^ 0xDEADBEEF) |>.toArray

  let inBuf  ← GPUBackend.allocBuffer ctx (256 * 4)
  let outBuf ← GPUBackend.allocBuffer ctx (256 * 4)
  GPUBackend.writeBuffer ctx inBuf (packU32 inputArr)

  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("input", inBuf), ("output", outBuf) ]
  GPUBackend.execute ctx cpAsyncSmokeKernel bufs
    { workgroupSize := { x := 64, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx outBuf (256 * 4)
  let resultArr := unpackU32 resultBytes 256

  let mut allOk := true
  for i in [0:256] do
    if resultArr[i]! != inputArr[i]! then
      allOk := false
      IO.println s!"  ✗ idx {i}: expected {inputArr[i]!}, got {resultArr[i]!}"

  GPUBackend.freeBuffer ctx inBuf
  GPUBackend.freeBuffer ctx outBuf

  if allOk then
    IO.println "✓ all 256 elements bit-identical to input"
    IO.println "═══ cp.async SMOKE PASS ═══"
  else
    IO.println "═══ cp.async SMOKE FAIL ═══"
    IO.Process.exit 1
