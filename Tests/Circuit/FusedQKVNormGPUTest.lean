import Hesper.Backend.CUDA
import Hesper.Models.Gemma4
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
Validates `fusedPerHeadQKVNormKernel` by running it at a Gemma 4-like
shape and comparing all three output buffers (qOut, kOut, vOut) against
a CPU per-head RMSNorm reference.

Shapes tested: numHeads=8, numKVHeads=4, headDim=256 (mimicking one
attention layer's Q/K/V heads).
-/

open Hesper

abbrev β := Hesper.CUDAContext

def cpuPerHeadNormQKV
    (nQ nKV D : Nat) (eps : Float)
    (qIn qScale : Array Float) (kIn kScale : Array Float) (vIn : Array Float)
    : Array Float × Array Float × Array Float := Id.run do
  let norm := fun (input : Array Float) (scale : Option (Array Float)) (nHeads : Nat) =>
    Id.run do
      let total := nHeads * D
      let mut out : Array Float := Array.replicate total 0.0
      for h in [0:nHeads] do
        let base := h * D
        let mut sumSq : Float := 0.0
        for i in [0:D] do sumSq := sumSq + input[base + i]! * input[base + i]!
        let inv := 1.0 / Float.sqrt (sumSq / D.toFloat + eps)
        for i in [0:D] do
          let raw := input[base + i]! * inv
          let final := match scale with
            | some s => raw * s[i]!
            | none => raw
          out := out.set! (base + i) final
      return out
  let qO := norm qIn (some qScale) nQ
  let kO := norm kIn (some kScale) nKV
  let vO := norm vIn none nKV
  return (qO, kO, vO)

unsafe def main : IO Unit := do
  let nQ : Nat := 8
  let nKV : Nat := 4
  let D : Nat := 256
  let eps : Float := 1e-6
  let ctx ← Hesper.CUDAContext.init

  let mkArr := fun (n : Nat) (seed : Float) =>
    Array.ofFn (n := n) fun i => seed + (i.val.toFloat * 0.011) - 1.3
  let qIn     := mkArr (nQ * D)  0.2
  let kIn     := mkArr (nKV * D) 0.5
  let vIn     := mkArr (nKV * D) (-0.3)
  let qScale  := mkArr D 0.9
  let kScale  := mkArr D 1.1

  let upload := fun (arr : Array Float) (n : Nat) => do
    let bytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes arr
    let buf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
    GPUBackend.writeBuffer (β := β) ctx buf bytes
    pure buf

  let qInBuf    ← upload qIn     (nQ * D)
  let kInBuf    ← upload kIn     (nKV * D)
  let vInBuf    ← upload vIn     (nKV * D)
  let qScaleBuf ← upload qScale  D
  let kScaleBuf ← upload kScale  D
  let qOutBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * nQ * D).toUSize
  let kOutBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * nKV * D).toUSize
  let vOutBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * nKV * D).toUSize

  let shader := Hesper.Models.Gemma4.fusedPerHeadQKVNormKernel nQ nKV D eps
  let exec : Hesper.ExecConfig :=
    { numWorkgroups := (nQ, 3, 1),
      workgroupSize := { x := min D 256, y := 1, z := 1 } }
  GPUBackend.execute (β := β) ctx shader
    [("q_in", qInBuf), ("q_scale", qScaleBuf), ("q_out", qOutBuf),
     ("k_in", kInBuf), ("k_scale", kScaleBuf), ("k_out", kOutBuf),
     ("v_in", vInBuf),                          ("v_out", vOutBuf)]
    exec

  let (qRef, kRef, vRef) := cpuPerHeadNormQKV nQ nKV D eps qIn qScale kIn kScale vIn
  let checkBuf := fun (name : String) (buf : GPUBackend.Buf β) (ref : Array Float) (n : Nat) => do
    let bytes ← GPUBackend.readBuffer (β := β) ctx buf (4 * n).toUSize
    let mut maxErr : Float := 0.0
    let mut ok := true
    for i in [0:n] do
      let actual ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
      let err := (actual - ref[i]!).abs
      if err > maxErr then maxErr := err
      if err > 1e-4 then
        if ok then IO.println s!"  [{name}] first mismatch i={i}: actual={actual} expected={ref[i]!}"
        ok := false
    IO.println s!"  [{name}] max abs error = {maxErr}"
    pure ok

  let okQ ← checkBuf "qOut" qOutBuf qRef (nQ * D)
  let okK ← checkBuf "kOut" kOutBuf kRef (nKV * D)
  let okV ← checkBuf "vOut" vOutBuf vRef (nKV * D)

  if okQ && okK && okV then
    IO.println "✓ PASS: fusedPerHeadQKVNormKernel matches per-head CPU reference on all 3 outputs"
  else
    IO.Process.exit 1
