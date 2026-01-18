import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Shader
import Hesper.WGSL.Kernel

/-!
# Kernel Fusion Example - Map Fusion

This demonstrates the Kernel abstraction that enables kernel fusion.
Instead of multiple GPU kernel launches, we fuse operations into a single pass.

WITHOUT FUSION (3 separate kernels):
  Kernel 1: output1[i] = input[i] * 2.0
  Kernel 2: output2[i] = output1[i] + 1.0
  Kernel 3: output3[i] = relu(output2[i])

  Problem: 3 global memory roundtrips (slow!)

WITH FUSION (1 kernel):
  output[i] = relu(input[i] * 2.0 + 1.0)

  Benefit: Single pass, no intermediate storage needed!
-/

namespace Hesper

open WGSL

-- Size of our test array
def N : Nat := 256

/-- ReLU activation function: max(0, x) -/
def relu (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  Exp.max x (Exp.litF32 0.0)

/-- Load operation: reads from global memory at given index -/
def loadVec (bufferName : String)
    : Kernel N 1 1 (Exp (.scalar .u32)) (Exp (.scalar .f32)) :=
  loadBuffer (n := N) bufferName

/-- Store operation: writes to global memory -/
def storeVec (bufferName : String)
    : Kernel N 1 1 (Exp (.scalar .u32) × Exp (.scalar .f32)) Unit :=
  storeBuffer bufferName

/-- Fused computation: multiply by 2, add 1, apply ReLU

This is the KEY demonstration of fusion:
Three operations composed into one with Kernel.comp -/
def calcLogic : Kernel N 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  let step1 := mapK (· * Exp.litF32 2.0)     -- Step 1: multiply by 2
  let step2 := mapK (· + Exp.litF32 1.0)     -- Step 2: add 1
  let step3 := mapK relu                     -- Step 3: apply ReLU
  Kernel.comp step3 (Kernel.comp step2 step1)

/-- Complete fused kernel

This builds the computation procedurally using Kernel composition.
The result is a SINGLE shader that performs all operations in one pass. -/
def fusedKernel (inputName outputName : String) (idx : Exp (.scalar .u32))
    : Kernel N 1 1 (Exp (.scalar .u32)) Unit :=
  let load := loadVec inputName              -- Load input[idx]
  let compute := calcLogic                    -- Apply: (* 2) >>> (+ 1) >>> relu
  let pair := pairWithIndex idx               -- Pair with index for storage
  let store := storeVec outputName            -- Store to output[idx]
  Kernel.comp store (Kernel.comp pair (Kernel.comp compute load))

/-- Generate complete WGSL shader with fusion -/
def generateFusedShader : ComputeShader :=
  -- Get global invocation ID
  let gid : Exp (.scalar .u32) := "gid.x"

  -- Execute the fused kernel and extract statements
  let kernel := fusedKernel "input" "output" gid
  let stmts := execKernel kernel gid

  {
    extensions := [],
    diagnostics := [],
    structs := [],
    buffers := [
      { group := 0, binding := 0, name := "input", elemType := .scalar .f32, readWrite := false },
      { group := 0, binding := 1, name := "output", elemType := .scalar .f32, readWrite := true }
    ],
    workgroupVars := [],
    workgroupSize := { x := N, y := 1, z := 1 },
    builtins := [
      { builtin := BuiltinBinding.globalInvocationId, name := "gid" }
    ],
    body := stmts
  }

end Hesper

def main : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   Kernel Fusion Example: Map Fusion                      ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""
  IO.println "Demonstrating fusion of: Load → Multiply → Add → ReLU → Store"
  IO.println ""

  -- Generate shader
  let shader := Hesper.generateFusedShader
  IO.println "Generated WGSL (fused single-pass shader):"
  IO.println "═══════════════════════════════════════════════════════════"
  IO.println shader.toWGSL
  IO.println ""

  IO.println "Notice: The shader contains a SINGLE function that performs:"
  IO.println "  1. Load from input buffer"
  IO.println "  2. Multiply by 2.0"
  IO.println "  3. Add 1.0"
  IO.println "  4. Apply ReLU (max with 0)"
  IO.println "  5. Store to output buffer"
  IO.println ""
  IO.println "All in ONE GPU kernel launch - no intermediate storage!"
  IO.println ""

  IO.println "BENEFITS OF KERNEL FUSION:"
  IO.println "════════════════════════════════════════════════════════"
  IO.println "• Single GPU kernel launch (not 3 separate launches)"
  IO.println "• No intermediate global memory storage needed"
  IO.println "• Better cache utilization"
  IO.println "• Reduced memory bandwidth usage"
  IO.println "• Composable with monadic operations (|>)"
  IO.println ""
  IO.println "This is the foundation for optimizing complex ML pipelines!"
  IO.println ""

  IO.println "✓ Kernel fusion implemented successfully!"
