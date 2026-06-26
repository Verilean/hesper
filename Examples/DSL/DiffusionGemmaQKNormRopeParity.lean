import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Basic

/-! Batched per-head qk-norm (RMS × weight) + RoPE over [N, nHead*hd]. Validates GPU vs CPU ref. -/
open Hesper.WebGPU
open Hesper.WGSL
open Hesper.WGSL.Monad

/-- One thread per (position p = t/nHead, head); RMS-norm the head ×weight, then RoPE(pos=p). -/
def qkNormRopeK (N nHead hd : Nat) (theta eps : Float) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← ShaderM.globalId
  let t := Exp.vec3X gid
  let qDim := nHead*hd
  let half := hd/2
  let coef := -2.0 * Float.log theta / hd.toFloat
  let _in ← ShaderM.declareReadOnlyBuffer "qin" (.array (.scalar .f32) (N*qDim))
  let _w ← ShaderM.declareReadOnlyBuffer "wnorm" (.array (.scalar .f32) hd)
  let _out ← ShaderM.declareOutputBuffer "qout" (.array (.scalar .f32) (N*qDim))
  ShaderM.if_ (Exp.lt t (Exp.litU32 (N*nHead))) (do
    let p := Exp.div t (Exp.litU32 nHead)
    let base := Exp.mul t (Exp.litU32 hd)
    let ss ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 hd) (Exp.litU32 1) fun d => do
      let v := Exp.index (Exp.var "qin" : Exp (.array (.scalar .f32) (N*qDim))) (Exp.add base d)
      ShaderM.assign ss (Exp.add (Exp.var ss) (Exp.mul v v))
    let inv := Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add (Exp.div (Exp.var ss) (Exp.litF32 hd.toFloat)) (Exp.litF32 eps)))
    let pf := Exp.toF32 p
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 half) (Exp.litU32 1) fun j => do
      let jh := Exp.add j (Exp.litU32 half)
      let qi : Exp (.array (.scalar .f32) (N*qDim)) := Exp.var "qin"
      let wn2 : Exp (.array (.scalar .f32) hd) := Exp.var "wnorm"
      let a := Exp.mul (Exp.mul (Exp.index qi (Exp.add base j)) inv) (Exp.index wn2 j)
      let b := Exp.mul (Exp.mul (Exp.index qi (Exp.add base jh)) inv) (Exp.index wn2 jh)
      let freq := Exp.exp (Exp.mul (Exp.litF32 coef) (Exp.toF32 j))
      let ang := Exp.mul pf freq
      let cosA := Exp.cos ang
      let sinA := Exp.sin ang
      ShaderM.assignIndex "qout" (Exp.add base j) (Exp.sub (Exp.mul a cosA) (Exp.mul b sinA))
      ShaderM.assignIndex "qout" (Exp.add base jh) (Exp.add (Exp.mul a sinA) (Exp.mul b cosA))) (pure ())

def main : IO Unit := do
  let N := 5; let nHead := 3; let hd := 8; let theta := 1000000.0; let eps := 1e-6
  let qDim := nHead*hd; let half := hd/2
  IO.println "[dg-qknormrope-parity] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let qa := (List.range (N*qDim)).toArray.map (fun t => Float.sin (t.toFloat * 0.21) * 0.7)
  let wn := (List.range hd).toArray.map (fun t => 0.5 + Float.cos (t.toFloat) * 0.2)
  let mk (a : Array Float) : IO Buffer := do
    let b ← createBuffer device { size := (a.size*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
    writeBuffer device b 0 (← Hesper.Basic.floatArrayToBytes a); return b
  let qB ← mk qa; let wB ← mk wn
  let oB ← createBuffer device { size := (N*qDim*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let bufs := ("qin", qB) :: ("wnorm", wB) :: ("qout", oB) :: List.nil
  let cfg : Hesper.ExecConfig := { numWorkgroups := ((N*nHead+255)/256, 1, 1), workgroupSize := { x := 256 } }
  Hesper.GPUBackend.execute device (qkNormRopeK N nHead hd theta eps) bufs cfg
  let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device oB 0 (N*qDim*4).toUSize)
  unmapBuffer oB
  -- CPU ref
  let logTheta := Float.log theta
  let mut ref := Array.replicate (N*qDim) 0.0
  for p in [0:N] do
    for h in [0:nHead] do
      let base := p*qDim + h*hd
      let mut ss := 0.0
      for d in [0:hd] do ss := ss + qa[base+d]! * qa[base+d]!
      let inv := 1.0 / Float.sqrt (ss / hd.toFloat + eps)
      let nrm := (List.range hd).toArray.map (fun d => qa[base+d]! * inv * wn[d]!)
      for j in [0:half] do
        let freq := Float.exp (-(2.0 * j.toFloat / hd.toFloat) * logTheta)
        let ang := p.toFloat * freq
        let a := nrm[j]!; let b := nrm[j+half]!
        ref := ref.set! (base+j) (a * Float.cos ang - b * Float.sin ang)
        ref := ref.set! (base+j+half) (a * Float.sin ang + b * Float.cos ang)
  let mut err := 0.0
  for t in [0:N*qDim] do
    let d := (gpu[t]! - ref[t]!).abs
    if d > err then err := d
  IO.println s!"  N={N} nHead={nHead} hd={hd}  GPU-vs-CPU maxAbsErr={err}"
  if err < 1e-4 then IO.println "✓ batched per-head qk-norm + RoPE matches CPU reference"
  else IO.println s!"✗ FAIL gpu[0..4]={(gpu.extract 0 4).toList} ref[0..4]={(ref.extract 0 4).toList}"; throw (IO.userError "fail")
