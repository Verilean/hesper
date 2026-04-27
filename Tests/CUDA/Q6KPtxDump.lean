import Hesper.CUDA.CodeGen
import Hesper.Layers.Linear

set_option maxRecDepth 2048

/-- Standalone PTX dumper for hesper Q6_K dp4a kernels.
    Writes 1-row, 2-row, 4-row, and 4-warp variants for the lm_head shape.
    Doesn't require a CUDA context. -/
def main (args : List String) : IO Unit := do
  let outDir := args.headD "/tmp/q6k_ptx"
  -- ffn_down shape: K=10240 (inDim), N=2560 (outDim).
  -- Layout: inDim must be % 256 == 0 (Q6_K block size).
  let inDim := 10240
  let outDim := 2560

  let dumpKernel1D (name : String) (block : Nat) (k : Hesper.WGSL.Monad.ShaderM Unit) : IO Unit := do
    let ptx := Hesper.CUDA.CodeGen.generatePTX name
      { x := block, y := 1, z := 1 } k
    IO.FS.writeFile s!"{outDir}/{name}.ptx" ptx
    IO.println s!"  {name}.ptx: {ptx.length} chars"

  let dumpKernel2D (name : String) (bx byDim : Nat) (k : Hesper.WGSL.Monad.ShaderM Unit) : IO Unit := do
    let ptx := Hesper.CUDA.CodeGen.generatePTX name
      { x := bx, y := byDim, z := 1 } k
    IO.FS.writeFile s!"{outDir}/{name}.ptx" ptx
    IO.println s!"  {name}.ptx: {ptx.length} chars"

  IO.println s!"Dumping hesper Q6_K dp4a kernels (inDim={inDim} outDim={outDim}) to {outDir}"
  dumpKernel1D "q6k_dp4a_1row_b32"   32  (Hesper.Layers.Linear.fusedQ6KLinearDP4AKernel inDim outDim 0)
  dumpKernel1D "q6k_dp4a_2row_b64"   64  (Hesper.Layers.Linear.fusedQ6KLinearDP4A2RowKernel inDim outDim 0)
  dumpKernel1D "q6k_dp4a_4row_b128"  128 (Hesper.Layers.Linear.fusedQ6KLinearDP4A4RowKernel inDim outDim 0)
  -- 4-warp 1-row variant: 2-D workgroup (32, 4, 1) = 128 thread.  Matches
  -- llama.cpp's `mul_mat_vec_q<Q6_K>` dispatch shape exactly.
  dumpKernel2D "q6k_dp4a_4warp_b32x4" 32 4 (Hesper.Layers.Linear.fusedQ6KLinearDP4A4WarpKernel inDim outDim 0)
  -- Also dump 4-warp at PLE shape (works under graphs ON in production)
  -- to diff against ffn_down shape (broken under graphs ON).
  dumpKernel2D "q6k_dp4a_4warp_PLE_b32x4" 32 4 (Hesper.Layers.Linear.fusedQ6KLinearDP4A4WarpKernel 2560 512 0)
  IO.println "done."
