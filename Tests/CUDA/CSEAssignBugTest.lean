import Hesper.Backend
import Hesper.Backend.CUDA

/-!
# CSE × assign bug regression test

The auto-CSE pass in expToPTX (`Hesper/CUDA/CodeGen.lean`) memoizes
pure-arithmetic Exp trees by `Exp.toWGSL` key.  But if a cseable
expression refers to a mutable `Exp.var` that is `assign`-ed between
two evaluations, the cached register holds the *old* value:

```
let x : Exp u32 := Exp.var "x"     -- assume x = 5
let r1 = Exp.add x (Exp.litU32 1)  -- evaluates to 6, cached
ShaderM.assign "x" (Exp.litU32 10) -- x ← 10
let r2 = Exp.add x (Exp.litU32 1)  -- *should* be 11
                                   -- with stale cache: returns r1 = 6 (BUG)
```

Fix: invalidate `expCache` at every `Stmt.assign`.  This test reproduces
the bug end-to-end on CUDA: write 6, 11 to output; check both bytes.

Run: `lake exe cuda-cse-assign-bug-test`
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)

private def packU32s (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun (acc : ByteArray) (n : UInt32) =>
    acc.push n.toUInt8 |>.push (n>>>8).toUInt8
       |>.push (n>>>16).toUInt8 |>.push (n>>>24).toUInt8

private def unpackU32 (ba : ByteArray) (i : Nat) : UInt32 :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

/-- Kernel:
    var x : u32 = 5
    output[0] = x + 1   ; should be 6
    x = 10              ; assign
    output[1] = x + 1   ; should be 11

    With the CSE×assign bug, the cseable `x + 1` cache hits and
    output[1] returns 6 instead of 11.
-/
def cseBugKernel : ShaderM Unit := do
  let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) 2)
  -- Need a single-thread kernel so we can sequence things.  Lean unrolls
  -- this entire body inside one warp anyway.
  ShaderM.varNamed "x" (.scalar .u32) (Exp.litU32 5)
  let xPlusOne : Exp (.scalar .u32) :=
    Exp.add (Exp.var "x" : Exp (.scalar .u32)) (Exp.litU32 1)
  ShaderM.writeBuffer (ty := .scalar .u32) "output" (Exp.litU32 0) xPlusOne
  ShaderM.assign "x" (Exp.litU32 10)
  let xPlusOne2 : Exp (.scalar .u32) :=
    Exp.add (Exp.var "x" : Exp (.scalar .u32)) (Exp.litU32 1)
  ShaderM.writeBuffer (ty := .scalar .u32) "output" (Exp.litU32 1) xPlusOne2

def runCseBugTest [GPUBackend β] (ctx : β) : IO Bool := do
  let outBuf ← GPUBackend.allocBuffer ctx 8
  GPUBackend.execute ctx cseBugKernel
    [("output", outBuf)]
    ({ workgroupSize := { x := 1, y := 1, z := 1 },
       numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
  let bytes ← GPUBackend.readBuffer ctx outBuf 8
  let out0 := unpackU32 bytes 0
  let out1 := unpackU32 bytes 1
  IO.println s!"  output[0] = {out0}  (expected 6)"
  IO.println s!"  output[1] = {out1}  (expected 11; with CSE bug: 6)"
  return out0 == 6 && out1 == 11

def main : IO UInt32 := do
  IO.println "═══ CSE × assign bug regression test ═══"
  let ctx ← CUDAContext.init
  let ok ← runCseBugTest ctx
  if ok then
    IO.println "✓ PASS — CSE invalidates cache at assign"
    return 0
  else
    IO.println "✗ FAIL — CSE×assign bug is back (or never fixed)"
    return 1
