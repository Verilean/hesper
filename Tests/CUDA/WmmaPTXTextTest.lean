import Hesper.CUDA.PTX

/-!
# WMMA PTX text-emission smoke test

Constructs a minimal WMMA inst stream by hand (Load A, Load B, Zero D,
MMA, Store) and checks the rendered PTX text is the expected
m16n16k16.row.col.f32.f16.f16.f32 sequence.

This validates the **text emission** half of WMMA codegen — the
Inst variants and `Inst.toString` cases. It does NOT exercise the
ShaderM → Inst lowering (`expToPTX` cases for `subgroupMatrix*`),
which needs a fuller test fixture.

Run: `lake exe wmma-ptx-text-test`
-/
namespace Hesper.CUDA.WmmaTest

open Hesper.CUDA Hesper.CUDA.PTX

/-- Build register groups for an m16n16k16 fp16/fp16/fp32 mma. -/
def buildInsts : Array Inst := Id.run do
  let mkU (n : Nat) : RegU32 := ⟨n, 0⟩
  let mkF (n : Nat) : RegF32 := ⟨n, 0⟩
  -- A fragment: 8 b32 regs (per nvcc reference; m16n16k16 .f16 expands
  -- to 8 b32 per thread on PTX 8.x with 32-thread warp).
  let a : Array RegU32 := #[mkU 0, mkU 1, mkU 2, mkU 3, mkU 100, mkU 101, mkU 102, mkU 103]
  -- B fragment: 8 b32 regs
  let b : Array RegU32 := #[mkU 4, mkU 5, mkU 6, mkU 7, mkU 200, mkU 201, mkU 202, mkU 203]
  -- C/D fragments: 8 f32 regs
  let c : Array RegF32 := #[mkF 0, mkF 1, mkF 2, mkF 3, mkF 4, mkF 5, mkF 6, mkF 7]
  let d : Array RegF32 := #[mkF 8, mkF 9, mkF 10, mkF 11, mkF 12, mkF 13, mkF 14, mkF 15]
  let addrA : RegU64 := ⟨0, 0⟩
  let addrB : RegU64 := ⟨1, 0⟩
  let addrD : RegU64 := ⟨2, 0⟩
  let stride : RegU32 := ⟨16, 0⟩
  pure #[
    .wmma_load_a_f16 a addrA stride,
    .wmma_load_b_f16 b addrB stride,
    .wmma_zero_d_f32 c,
    .wmma_mma_f32_f16_f16_f32 d a b c,
    .wmma_store_d_f32 addrD d stride
  ]

def assertContains (label : String) (text : String) (needle : String) : IO Unit := do
  let parts := text.splitOn needle
  if parts.length > 1 then
    IO.println s!"PASS  {label}"
  else
    IO.println s!"FAIL  {label}"
    IO.println s!"  needle: {needle}"
    IO.println s!"  text:\n{text}"

def main : IO Unit := do
  IO.println "=== WMMA PTX text emission test ==="
  let insts := buildInsts
  let lines := insts.map (·.toString)
  let dump := String.intercalate "\n" lines.toList
  IO.println dump
  IO.println "---"
  assertContains "wmma.load.a"  dump "wmma.load.a.sync.aligned.row.m16n16k16.f16"
  assertContains "wmma.load.b"  dump "wmma.load.b.sync.aligned.col.m16n16k16.f16"
  assertContains "wmma.mma"     dump "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"
  assertContains "wmma.store.d" dump "wmma.store.d.sync.aligned.row.m16n16k16.f32"
  assertContains "zero d"       dump "mov.f32 %f0, 0f00000000;"
  -- Brace lists with all 4/8 regs
  assertContains "A reg list"   dump "{%r0, %r1, %r2, %r3, %r100, %r101, %r102, %r103}"
  assertContains "D reg list"   dump "{%f8, %f9, %f10, %f11, %f12, %f13, %f14, %f15}"
  IO.println "=== done ==="

end Hesper.CUDA.WmmaTest

def main : IO Unit := Hesper.CUDA.WmmaTest.main
