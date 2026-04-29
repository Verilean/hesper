import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad

/-!
# WMMA Phase 2 — ShaderM-driven PTX gen smoke

A 16×16×16 fp16/fp16/fp32 matmul, written using the new
`loadFragmentLeft` / `loadFragmentRight` / `declareFragmentResultZero` /
`fragmentMultiplyAccumulate` / `storeFragmentResult` ShaderM helpers.
We dump the resulting PTX text and assert it contains the expected
wmma instruction sequence.

This is the missing link from Phase 1: Phase 1 verified the *Inst →
PTX text* path with hand-built Insts; this verifies the *ShaderM →
Inst* path through the codegen dispatch.

Run: `lake exe wmma-shaderm-test`
-/
namespace Hesper.CUDA.WmmaShaderMTest

open Hesper.WGSL Hesper.WGSL.Monad Hesper.CUDA Hesper.CUDA.CodeGen

/-- The kernel: C[16,16] += A[16,16] * B[16,16] (fp16 inputs, fp32 acc).
    All three matrices laid out 16×16 contiguous, row-major (B loaded col-major
    via the `LoadRight` instruction's transpose hardware). -/
def wmmaMatmul16x16 : ShaderM Unit := do
  let _ ← ShaderM.declareReadOnlyBuffer "A" (.array (.scalar .f16) 256)
  let _ ← ShaderM.declareReadOnlyBuffer "B" (.array (.scalar .f16) 256)
  let _ ← ShaderM.declareOutputBuffer  "C" (.array (.scalar .f32) 256)
  -- Three single-fragment locals + one MMA + one store.
  ShaderM.loadFragmentLeft  .f16 16 16 "a_frag" "&A" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.loadFragmentRight .f16 16 16 "b_frag" "&B" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.declareFragmentResultZero .f32 16 16 "c_frag"
  ShaderM.fragmentMultiplyAccumulate .f32 16 16 16 "c_frag" "a_frag" "b_frag"
  ShaderM.storeFragmentResult .f32 16 16 "c_frag" "&C" (Exp.litU32 0) (Exp.litU32 16)

def assertContains (label : String) (text needle : String) : IO Unit := do
  if (text.splitOn needle).length > 1 then
    IO.println s!"PASS  {label}"
  else
    IO.println s!"FAIL  {label}"
    IO.println s!"  needle: {needle}"

def main : IO Unit := do
  IO.println "=== WMMA Phase 2: ShaderM → PTX gen ==="
  let ptx := generatePTX (funcName := "wmma_matmul_16x16")
                              (workgroupSize := {x := 32, y := 1, z := 1})
                              (computation := wmmaMatmul16x16)
                              (targetArch := "sm_80")
  IO.println "--- generated PTX (head 80 lines) ---"
  let lines := (ptx.splitOn "\n").take 80
  for l in lines do IO.println l
  IO.println "--- WMMA assertions ---"
  assertContains "load A f16" ptx "wmma.load.a.sync.aligned.row.m16n16k16.f16"
  assertContains "load B f16" ptx "wmma.load.b.sync.aligned.col.m16n16k16.f16"
  assertContains "mma sync"   ptx "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"
  assertContains "store D f32" ptx "wmma.store.d.sync.aligned.row.m16n16k16.f32"
  assertContains "zero D init" ptx "mov.f32 %f0, 0f00000000;"
  IO.println "=== done ==="

end Hesper.CUDA.WmmaShaderMTest

def main : IO Unit := Hesper.CUDA.WmmaShaderMTest.main
