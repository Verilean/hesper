import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Basic
import Hesper.WGSL.Execute

/-! Batched cross-position attention parity (the genuinely-new kernel for the bidirectional forward).
    seqLen=N, region mask allowed(i,j) = (i≥P) || (j≤i) [canvas bidir / prompt causal].
    scale=1.0. Validates GPU kernel vs a Lean CPU reference. -/
open Hesper.WebGPU
open Hesper.WGSL
open Hesper.WGSL.Monad

/-- One workgroup per (head h = wid.x, query i = wid.y): scores over keys → masked softmax → weighted-V. -/
def batchAttnKernel (N P nHead hd nKV : Nat) (ws : Nat := 256) : Hesper.WGSL.Monad.ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let h := Exp.vec3X wid
  let i := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let qDim := nHead*hd; let kvDim := nKV*hd; let groupSize := nHead/nKV
  let _q ← ShaderM.declareReadOnlyBuffer "q" (.array (.scalar .f32) (N*qDim))
  let _k ← ShaderM.declareReadOnlyBuffer "k" (.array (.scalar .f32) (N*kvDim))
  let _v ← ShaderM.declareReadOnlyBuffer "v" (.array (.scalar .f32) (N*kvDim))
  let _o ← ShaderM.declareOutputBuffer "ctx" (.array (.scalar .f32) (N*qDim))
  ShaderM.sharedNamed "sScores" (.array (.scalar .f32) N)
  let kvhE := Exp.div h (Exp.litU32 groupSize)
  let qBase := Exp.add (Exp.mul i (Exp.litU32 qDim)) (Exp.mul h (Exp.litU32 hd))
  let kvBase0 := Exp.mul kvhE (Exp.litU32 hd)
  ShaderM.loop tid (Exp.litU32 N) (Exp.litU32 ws) fun j => do
    let allowed := Exp.or (Exp.ge i (Exp.litU32 P)) (Exp.ge i j)
    let kBase := Exp.add (Exp.mul j (Exp.litU32 kvDim)) kvBase0
    let sc ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let qv := Exp.index (Exp.var "q" : Exp (.array (.scalar .f32) (N*qDim))) (Exp.add qBase d)
      let kv := Exp.index (Exp.var "k" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add kBase d)
      ShaderM.assign sc (Exp.add (Exp.var sc) (Exp.mul qv kv))
    ShaderM.assignIndex "sScores" j (Exp.select allowed (Exp.var sc : Exp (.scalar .f32)) (Exp.litF32 (-1e30)))
  ShaderM.barrier
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let mx ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1e30))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assign mx (Exp.max (Exp.var mx) (Exp.index (Exp.var "sScores" : Exp (.array (.scalar .f32) N)) j))
    -- exp in-place, THEN sum (separate loops: the DSL inlines `let`, so summing in the
    -- same loop re-reads the overwritten sScores — the documented softmax bug, CLAUDE.md #5).
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assignIndex "sScores" j (Exp.exp (Exp.sub (Exp.index (Exp.var "sScores" : Exp (.array (.scalar .f32) N)) j) (Exp.var mx)))
    let sm ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      ShaderM.assign sm (Exp.add (Exp.var sm) (Exp.index (Exp.var "sScores" : Exp (.array (.scalar .f32) N)) j))
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      let w := Exp.index (Exp.var "sScores" : Exp (.array (.scalar .f32) N)) j
      ShaderM.assignIndex "sScores" j (Exp.div w (Exp.var sm))) (pure ())
  ShaderM.barrier
  ShaderM.loop tid (Exp.litU32 hd) (Exp.litU32 ws) fun d => do
    let acc ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 N) (Exp.litU32 1) fun j => do
      let w := Exp.index (Exp.var "sScores" : Exp (.array (.scalar .f32) N)) j
      let vv := Exp.index (Exp.var "v" : Exp (.array (.scalar .f32) (N*kvDim))) (Exp.add (Exp.add (Exp.mul j (Exp.litU32 kvDim)) kvBase0) d)
      ShaderM.assign acc (Exp.add (Exp.var acc) (Exp.mul w vv))
    ShaderM.writeBuffer (ty := .scalar .f32) "ctx" (Exp.add qBase d) (Exp.var acc : Exp (.scalar .f32))

def main : IO Unit := do
  let N := 6; let P := 2; let nHead := 4; let hd := 4; let nKV := 2
  let qDim := nHead*hd; let kvDim := nKV*hd; let groupSize := nHead/nKV
  IO.println "[dg-battn-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let qa := (List.range (N*qDim)).toArray.map (fun t => Float.sin (t.toFloat * 0.3) * 0.5)
  let ka := (List.range (N*kvDim)).toArray.map (fun t => Float.cos (t.toFloat * 0.2) * 0.5)
  let va := (List.range (N*kvDim)).toArray.map (fun t => Float.sin (t.toFloat * 0.15 + 1.0) * 0.5)
  let mk (a : Array Float) : IO Buffer := do
    let b ← createBuffer device { size := (a.size*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
    writeBuffer device b 0 (← Hesper.Basic.floatArrayToBytes a); return b
  let qB ← mk qa; let kB ← mk ka; let vB ← mk va
  let ctxB ← createBuffer device { size := (N*qDim*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let bufs := ("q", qB) :: ("k", kB) :: ("v", vB) :: ("ctx", ctxB) :: List.nil
  let cfg : Hesper.ExecConfig := { numWorkgroups := (nHead, N, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device (batchAttnKernel N P nHead hd nKV) bufs cfg
  let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device ctxB 0 (N*qDim*4).toUSize)
  unmapBuffer ctxB
  -- CPU reference
  let mut ref := Array.replicate (N*qDim) 0.0
  for i in [0:N] do
    for h in [0:nHead] do
      let kvh := h / groupSize
      let mut sc := Array.replicate N (-1e30)
      for j in [0:N] do
        if i ≥ P || j ≤ i then
          let mut s := 0.0
          for d in [0:hd] do s := s + qa[i*qDim+h*hd+d]! * ka[j*kvDim+kvh*hd+d]!
          sc := sc.set! j s
      let mx := sc.foldl (fun a b => if b > a then b else a) (-1e30)
      let mut sm := 0.0
      let mut ex := Array.replicate N 0.0
      for j in [0:N] do let e := Float.exp (sc[j]! - mx); ex := ex.set! j e; sm := sm + e
      for d in [0:hd] do
        let mut acc := 0.0
        for j in [0:N] do acc := acc + (ex[j]!/sm) * va[j*kvDim+kvh*hd+d]!
        ref := ref.set! (i*qDim+h*hd+d) acc
  let mut err := 0.0
  for t in [0:N*qDim] do
    let dd := (gpu[t]! - ref[t]!).abs
    if dd > err then err := dd
  IO.println s!"  N={N} P={P} nHead={nHead} hd={hd} nKV={nKV}  GPU-vs-CPU maxAbsErr={err}"
  IO.println s!"  gpu[0..4]={(gpu.extract 0 4).toList}"
  IO.println s!"  ref[0..4]={(ref.extract 0 4).toList}"
  if err < 1e-4 then
    IO.println "✓ batched cross-position attention (region mask, GQA, softmax, weighted-V) matches CPU reference"
  else
    IO.println "✗ FAIL"; throw (IO.userError "battn parity failed")
