import Hesper.Backend.CUDA
import Hesper.Layers.RMSNorm
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
Validates `Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel` on the CUDA
backend, comparing GPU output against a CPU reference that does
RMSNorm + Q8_1 quantize end-to-end.

Q8_1 layout (per 32-element block):
  u32[0]   = bitcast<u32>(d : f32)        -- block scale; s field unused
  u32[1..8] = packed 4×int8 quants

Tolerance: f32 reduction is ~exact for our inputs; quants are int8
so they must match exactly.  We allow `d` to differ by 1 ULP.
-/

open Hesper

abbrev β := Hesper.CUDAContext

structure CpuRef where
  ds : Array Float                  -- per-block scale (f32)
  qs : Array Int                    -- per-element int8 quant

def cpuFusedNormQuantize (xArr scaleArr : Array Float) (eps : Float) : CpuRef := Id.run do
  let D := xArr.size
  let mut sumSq : Float := 0.0
  for i in [0:D] do sumSq := sumSq + xArr[i]! * xArr[i]!
  let invRms : Float := 1.0 / Float.sqrt (sumSq / D.toFloat + eps)
  let mut normed : Array Float := Array.replicate D 0.0
  for i in [0:D] do normed := normed.set! i (xArr[i]! * scaleArr[i]! * invRms)
  let nBlocks := D / 32
  let mut ds : Array Float := Array.replicate nBlocks 0.0
  let mut qs : Array Int := Array.replicate D 0
  for b in [0:nBlocks] do
    let mut amax : Float := 0.0
    for k in [0:32] do
      let v := normed[b*32 + k]!
      let av := if v < 0.0 then -v else v
      if av > amax then amax := av
    let d := amax / 127.0
    ds := ds.set! b d
    for k in [0:32] do
      let v := normed[b*32 + k]!
      let q := if d == 0.0 then 0 else (v / d).round.toInt32.toInt
      let qClamped := max (-128) (min 127 q)
      qs := qs.set! (b*32 + k) qClamped
  return { ds, qs }

unsafe def main : IO Unit := do
  let D : Nat := 256
  let eps : Float := 1e-6
  let ctx ← Hesper.CUDAContext.init

  let xArr : Array Float :=
    Array.ofFn (n := D) fun i => (i.val.toFloat * 0.013) - 1.5
  let scaleArr : Array Float :=
    Array.ofFn (n := D) fun i => 0.7 + 0.005 * i.val.toFloat

  let xBytes     ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let scaleBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes scaleArr
  let xBuf     ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  let scaleBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  GPUBackend.writeBuffer (β := β) ctx xBuf     xBytes
  GPUBackend.writeBuffer (β := β) ctx scaleBuf scaleBytes

  let nBlocks := D / 32
  let outU32 := nBlocks * 9
  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * outU32).toUSize

  let cfg : Hesper.Layers.RMSNorm.Config := { dim := D, eps }
  let shader := Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel cfg
  let exec : Hesper.ExecConfig :=
    { numWorkgroups := (1, 1, 1)
      workgroupSize := { x := 256, y := 1, z := 1 } }
  GPUBackend.execute (β := β) ctx shader
    [("input", xBuf), ("scale", scaleBuf), ("output", outBuf)] exec

  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * outU32).toUSize
  let cpu := cpuFusedNormQuantize xArr scaleArr eps

  let mut dErrMax : Float := 0.0
  let mut qMismatch : Nat := 0
  let mut firstMismatch : Option (Nat × Int × Int) := none
  for b in [0:nBlocks] do
    let dActual ← Hesper.Basic.bytesToFloat32 outBytes (b * 9 * 4)
    let dExpected := cpu.ds[b]!
    let derr := (dActual - dExpected).abs
    if derr > dErrMax then dErrMax := derr
    for u in [0:8] do
      let off := (b * 9 + 1 + u) * 4
      let bytes : Array UInt8 :=
        #[outBytes[off]!, outBytes[off + 1]!, outBytes[off + 2]!, outBytes[off + 3]!]
      for k in [0:4] do
        let qByte := bytes[k]!
        let qActual : Int :=
          if qByte ≥ 128 then (qByte.toNat : Int) - 256 else (qByte.toNat : Int)
        let qExpected := cpu.qs[b * 32 + u * 4 + k]!
        if qActual != qExpected then
          qMismatch := qMismatch + 1
          if firstMismatch.isNone then
            firstMismatch := some (b * 32 + u * 4 + k, qActual, qExpected)

  IO.println s!"Per-block scale max abs error: {dErrMax}"
  IO.println s!"Quant mismatches: {qMismatch} / {D}"
  if qMismatch == 0 && dErrMax < 1e-5 then
    IO.println s!"✓ PASS: fusedRMSNormQ8_1Kernel matches CPU reference"
  else
    match firstMismatch with
    | some (i, a, e) => IO.println s!"  first quant mismatch: i={i} actual={a} expected={e}"
    | none => pure ()
    IO.Process.exit 1
