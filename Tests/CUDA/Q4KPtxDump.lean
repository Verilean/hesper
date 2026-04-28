import Hesper.CUDA.CodeGen
import Hesper.Layers.Linear

set_option maxRecDepth 2048

/-- Standalone PTX dumper for hesper Q4_K dp4a kernels at the production
    Gemma 4 shapes.  Doesn't require a CUDA context.

    Shapes (Gemma 4 E4B):
      ffn_gate / ffn_up fused : K=2560, N=8192  (top-1 hot kernel @ 68 µs/call)
      wO                       : K=2560, N=2560
      wQ                       : K=2560, N=2048
      wK / wV                  : K=2560, N=1024
-/
def main (args : List String) : IO Unit := do
  let outDir := args.headD "/tmp/q4k_ptx"

  let dumpKernel1D (name : String) (block : Nat)
      (k : Hesper.WGSL.Monad.ShaderM Unit) : IO Unit := do
    let ptx := Hesper.CUDA.CodeGen.generatePTX name
      { x := block, y := 1, z := 1 } k
    IO.FS.writeFile s!"{outDir}/{name}.ptx" ptx
    IO.println s!"  {name}.ptx: {ptx.length} chars"

  let _dumpKernel2D (name : String) (bx byDim : Nat)
      (k : Hesper.WGSL.Monad.ShaderM Unit) : IO Unit := do
    let ptx := Hesper.CUDA.CodeGen.generatePTX name
      { x := bx, y := byDim, z := 1 } k
    IO.FS.writeFile s!"{outDir}/{name}.ptx" ptx
    IO.println s!"  {name}.ptx: {ptx.length} chars"

  IO.println s!"Dumping hesper Q4_K dp4a kernels to {outDir}"

  -- Top-1 hot kernel: ffn_gate+up fused matmul, K=2560 N=8192.
  let cfgGateUp : Hesper.Layers.Linear.Config := { inDim := 2560, outDim := 8192 }
  dumpKernel1D "q4k_dp4a_4warp_gateup_b128"  128
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4A4WarpKernel cfgGateUp)
  dumpKernel1D "q4k_dp4a_4row_gateup_b128"   128
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4A4RowKernel cfgGateUp)
  dumpKernel1D "q4k_dp4a_1row_gateup_b32"    32
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4AKernel cfgGateUp)

  -- wO: K=2560 N=2560.  Same K, smaller N.
  let cfgWO : Hesper.Layers.Linear.Config := { inDim := 2560, outDim := 2560 }
  dumpKernel1D "q4k_dp4a_4warp_wO_b128"      128
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4A4WarpKernel cfgWO)

  -- wQ: K=2560 N=2048.
  let cfgWQ : Hesper.Layers.Linear.Config := { inDim := 2560, outDim := 2048 }
  dumpKernel1D "q4k_dp4a_4warp_wQ_b128"      128
    (Hesper.Layers.Linear.fusedQ4KMLinearDP4A4WarpKernel cfgWQ)

  IO.println "done."
