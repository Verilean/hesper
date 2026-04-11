import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad

open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.CUDA.CodeGen

def vectorDoubleKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vec3X gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  writeBuffer (ty := .scalar .f32) "output" idx result

def main : IO Unit := do
  IO.println "═══ CUDA End-to-End Test ═══"
  let ptx := generatePTX "vectorDouble" {x := 256} vectorDoubleKernel
  IO.println s!"PTX: {ptx.length} chars"
  let (_dev, _ctx) ← initCUDA
  let cudaMod ← cuModuleLoadData ptx
  let func ← cuModuleGetFunction cudaMod "vectorDouble"
  IO.println "JIT: OK"
  let n : Nat := 16
  let inputBuf ← createCUDABuffer (n * 4).toUSize
  let outputBuf ← createCUDABuffer (n * 4).toUSize
  let mut data := ByteArray.empty
  for _ in List.range n do
    data := data.push 0 |>.push 0 |>.push 128 |>.push 63  -- 1.0f
  writeCUDABuffer inputBuf data
  cuLaunchKernel func 1 1 1 n.toUInt32 1 1 0 #[inputBuf.ptr, outputBuf.ptr]
  let result ← readCUDABufferFull outputBuf
  let b0 := result.get! 0; let b1 := result.get! 1
  let b2 := result.get! 2; let b3 := result.get! 3
  IO.println s!"out[0] = [{b0},{b1},{b2},{b3}] (expect [0,0,0,64] = 2.0f)"
  freeCUDABuffer inputBuf; freeCUDABuffer outputBuf
  if b0 == 0 && b1 == 0 && b2 == 0 && b3 == 64 then
    IO.println "✓ CUDA E2E PASSED — ShaderM → PTX DSL → JIT → GPU → correct result"
  else
    IO.println "✗ FAILED"
    IO.Process.exit 1
