import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.Layers.Linear

open Hesper.WebGPU

namespace Examples.Compute.GroupedDownTest

/-- Standalone correctness + determinism test for fusedQ8_0BatchExpertF32WarpGroupedKernel
    (the tiled grouped MoE down). Q8_0 weights with d=1.0 so dequant(W)=q (the int8). -/

-- weight pattern: q[expert,o,k] = (expert*7 + o*3 + k) % 11 - 5   (range [-5,5])
def qf (expert o k : Nat) : Int := ((expert*7 + o*3 + k) % 11 : Nat) - 5
-- input pattern: in[g,k] = (g*2 + k) % 7 - 3
def inf (g k : Nat) : Float := (((g*2 + k) % 7 : Nat).toFloat) - 3.0
def i2f (i : Int) : Float := if i >= 0 then i.toNat.toFloat else -(((-i).toNat).toFloat)

def main : IO Unit := do
  IO.println "=== tiled grouped down: standalone CPU validation ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  -- REAL inDim=704 (22 blocks → exercises the full wcache), + a SENTINEL tile (dummy padding)
  let inDim := 704; let outDim := 64; let nExpert := 2; let maxPadded := 96
  let bpr := inDim / 32                      -- blocks per row = 22
  let perExpertBytes := outDim * bpr * 34
  -- tile 0 → expert 0, tile 1 → expert 1, tile 2 → 5 (SENTINEL ≥ nExpert; kernel clamps to nExpert-1)
  let tileExpert : Array Nat := #[0, 1, 5]
  let expertOf (g : Nat) : Nat := let te := tileExpert.getD (g/32) 0; if te < nExpert then te else nExpert-1
  -- build Q8_0 weights: d=1.0 (f16 0x3C00) + 32 int8 per block
  let mut wBytes : ByteArray := ByteArray.empty
  for expert in [0:nExpert] do
    for o in [0:outDim] do
      for blk in [0:bpr] do
        wBytes := wBytes.push 0x00; wBytes := wBytes.push 0x3C   -- d = 1.0 (f16)
        for i in [0:32] do
          let q := qf expert o (blk*32 + i)
          let b : Nat := if q >= 0 then q.toNat else (q + 256).toNat
          wBytes := wBytes.push b.toUInt8
  -- build input f32 [maxPadded, inDim]
  let mut inA : Array Float := #[]
  for g in [0:maxPadded] do for k in [0:inDim] do inA := inA.push (inf g k)
  -- tileExpert buffer (u32)
  let mut teBytes : ByteArray := ByteArray.empty
  for t in tileExpert do
    teBytes := teBytes.push (t &&& 0xFF).toUInt8; teBytes := teBytes.push ((t>>>8)&&&0xFF).toUInt8
    teBytes := teBytes.push ((t>>>16)&&&0xFF).toUInt8; teBytes := teBytes.push ((t>>>24)&&&0xFF).toUInt8
  let mkBuf (bytesLen : Nat) : IO Buffer := createBuffer device {
    size := bytesLen.toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let wBuf ← mkBuf (nExpert*perExpertBytes)
  let inBuf ← mkBuf (maxPadded*inDim*4)
  let teBuf ← mkBuf (maxPadded/32*4)
  let outBuf ← mkBuf (maxPadded*outDim*4)
  writeBuffer device wBuf 0 wBytes
  writeBuffer device inBuf 0 (← Hesper.Basic.floatArrayToBytes inA)
  writeBuffer device teBuf 0 teBytes
  -- CPU reference: out[g,o] = Σ_k in[g,k] * qf(expert(g),o,k)
  let cpuRef : Array Float := Id.run do
    let mut r : Array Float := #[]
    for g in [0:maxPadded] do
      for o in [0:outDim] do
        let mut s := 0.0
        for k in [0:inDim] do s := s + (inf g k) * i2f (qf (expertOf g) o k)
        r := r.push s
    pure r
  let bufs : List (String × Buffer) := [("weights",wBuf),("input",inBuf),("tileExpert",teBuf),("output",outBuf)]
  let kern := Hesper.Layers.Linear.fusedQ8_0BatchExpertF32WarpGroupedKernel { inDim:=inDim, outDim:=outDim } nExpert maxPadded
  -- run ×5 (determinism), grid (outDim, maxPadded/32), workgroupSize 32
  for run in [0:5] do
    let r ← IO.mkRef none
    Hesper.GPUBackend.executeWithConfigCached device kern bufs
      { numWorkgroups := (outDim, maxPadded/32, 1), workgroupSize := {x:=32} } (run.toUInt64+777) r
    let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outBuf 0 (maxPadded*outDim*4).toUSize)
    let mut maxd := 0.0; let mut bad := 0
    for i in [0:maxPadded*outDim] do
      let d := (gpu.getD i 0.0 - cpuRef.getD i 0.0).abs
      if d > maxd then maxd := d
      if d > 0.5 then bad := bad+1
    let v := if bad==0 then "✅ matches CPU" else "❌ WRONG"
    IO.println s!"  [run {run}] maxDiff={maxd} bad={bad}/{maxPadded*outDim} → {v}; sample g0o0 gpu={gpu.getD 0 0.0} cpu={cpuRef.getD 0 0.0}; g40o0 gpu={gpu.getD (40*outDim) 0.0} cpu={cpuRef.getD (40*outDim) 0.0}"

end Examples.Compute.GroupedDownTest

def main : IO Unit := Examples.Compute.GroupedDownTest.main
