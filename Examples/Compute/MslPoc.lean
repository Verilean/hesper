import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.Quantization.Q4_K_M

open Hesper.WebGPU
open Hesper.WGSL (Exp)

namespace Examples.Compute.MslPoc

/-! MSL 1-kernel PoC (macOS DEBUG/REFERENCE, metal_replacer family — NOT production).

Runs the deployed MoE gate/up kernel (`q4kMatmulGroupedRegIndexedKernel`) and its hand-written
native-Metal port (`mslQ4kBench`, native/metal_replace.mm) on IDENTICAL Dawn buffers at the real
deployed shape + realistic ragged routing, and reports: WGSL ms | MSL ms | ratio | maxDiff.
Decides whether full MSL lowering is worth it: ratio ≥1.3× → codegen (Tint) is the gap;
~1.0× → the remaining per-step gap vs llama.cpp is ALGORITHM, not codegen. -/

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

/-- GPU fill: out[i] = (i * 2654435761) & mask (u32 Knuth hash; mask keeps f16 halves finite). -/
def fillHashU32 (n gridW : Nat) (mask : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId
  let flat := Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridW))
  let _o ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "out" (.array (.scalar .u32) n)
  Hesper.WGSL.Monad.ShaderM.if_ (Exp.lt flat (Exp.litU32 n)) (do
    let h := Exp.mul flat (Exp.litU32 2654435761)
    Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .u32) "out" flat (Exp.bitAnd h (Exp.litU32 mask))) (pure ())

/-- GPU fill: out[i] = f32 in ~[-1,1) from the hash. -/
def fillHashF32 (n gridW : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId
  let flat := Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridW))
  let _o ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) n)
  Hesper.WGSL.Monad.ShaderM.if_ (Exp.lt flat (Exp.litU32 n)) (do
    let h := Exp.bitAnd (Exp.mul flat (Exp.litU32 2654435761)) (Exp.litU32 0xFFFF)
    let v := Exp.sub (Exp.div (Exp.toF32 h) (Exp.litF32 32768.0)) (Exp.litF32 1.0)
    Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .f32) "out" flat v) (pure ())

def dispFill (device : Device) (kern : Hesper.WGSL.Monad.ShaderM Unit)
    (buf : Buffer) (n : Nat) : IO Unit := do
  let wg := (n + 255) / 256
  let nx := min wg 32768
  let ny := (wg + nx - 1) / nx
  let cfg : Hesper.ExecConfig := { numWorkgroups := (nx, ny, 1), workgroupSize := {x:=256}, extensions := [], diagnostics := [] }
  let r ← IO.mkRef none
  let bufs : List (String × Buffer) := [("out", buf)]
  Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r

def benchKernelMs (device : Device) (kern : Hesper.WGSL.Monad.ShaderM Unit)
    (bufs : List (String × Buffer)) (cfg : Hesper.ExecConfig) (iters : Nat := 50) : IO Float := do
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
  let t0 ← IO.monoMsNow
  Hesper.GPUBackend.beginBatch device
  for _ in [0:iters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
  Hesper.GPUBackend.endBatch device
  let t1 ← IO.monoMsNow
  pure ((t1-t0).toFloat / iters.toFloat)

def u32Bytes (a : Array Nat) : ByteArray := Id.run do
  let mut b := ByteArray.empty
  for v in a do
    b := b.push (UInt8.ofNat (v % 256))
    b := b.push (UInt8.ofNat ((v/256) % 256))
    b := b.push (UInt8.ofNat ((v/65536) % 256))
    b := b.push (UInt8.ofNat ((v/16777216) % 256))
  return b

def main : IO Unit := do
  IO.println "=== MSL 1-kernel PoC: q4kMatmulGroupedRegIndexedKernel WGSL(Tint) vs hand-MSL ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  -- deployed gate/up shape at the renoise default (N tokens = 277)
  let srcRows := 277
  let nExpert := 128
  let K := 2816              -- dim
  let N := 1408              -- 2*expFF
  let totalTok := srcRows * 8
  let maxPadded := (((totalTok + 32*nExpert) + 31) / 32) * 32
  let M := maxPadded
  IO.println s!"shape: M={M} N={N} K={K} nExpert={nExpert} srcRows={srcRows}"
  -- realistic ragged routing on CPU (LCG over totalTok assignments)
  let mut cnt : Array Nat := Array.replicate nExpert 0
  let mut seed : Nat := 42
  let mut assign : Array (Array Nat) := Array.replicate nExpert #[]
  for t in [0:totalTok] do
    seed := (seed * 1103515245 + 12345) % 2147483648
    let e := seed % nExpert
    cnt := cnt.set! e (cnt[e]! + 1)
    assign := assign.set! e (assign[e]!.push (t % srcRows))
  let nTiles := M / 32
  let mut te : Array Nat := Array.replicate nTiles nExpert   -- sentinel
  let mut tr : Array Nat := Array.replicate nTiles 0
  let mut idxA : Array Nat := Array.replicate M 0
  let mut tile := 0
  let mut activeRows := 0
  for e in [0:nExpert] do
    let c := cnt[e]!
    if c > 0 then
      let tls := (c + 31) / 32
      for tI in [0:tls] do
        te := te.set! tile e
        tr := tr.set! tile (min 32 (c - tI*32))
        for i in [0:min 32 (c - tI*32)] do
          idxA := idxA.set! (tile*32 + i) (assign[e]![tI*32 + i]!)
        tile := tile + 1
      activeRows := activeRows + c
  IO.println s!"routing: {tile} active tiles / {nTiles}, {activeRows} real rows"
  -- buffers
  let bU32 := nExpert * N * (K/256) * 36
  let src ← mkBuf device (srcRows*K)
  let idx ← mkBuf device M
  let b   ← mkBuf device bU32
  let c1  ← mkBuf device (M*N)
  let c2  ← mkBuf device (M*N)
  let teB ← mkBuf device nTiles
  let trB ← mkBuf device nTiles
  writeBuffer device idx 0 (u32Bytes idxA)
  writeBuffer device teB 0 (u32Bytes te)
  writeBuffer device trB 0 (u32Bytes tr)
  -- GPU fills: weights (finite-f16 mask), activations, and ZERO both outputs (also marks them
  -- Dawn-initialized — the lazy-clear gotcha for the MSL-written c2)
  dispFill device (fillHashU32 bU32 (32768*256) 0x3BFF3BFF) b bU32
  dispFill device (fillHashF32 (srcRows*K) (32768*256)) src (srcRows*K)
  dispFill device (fillHashU32 (M*N) (32768*256) 0x0) c1 (M*N)
  dispFill device (fillHashU32 (M*N) (32768*256) 0x0) c2 (M*N)
  -- WGSL (Tint) side
  let kern := Hesper.Quantization.Q4_K_M.q4kMatmulGroupedRegIndexedKernel M N K nExpert srcRows
  let cfg : Hesper.ExecConfig := { numWorkgroups := ((N+31)/32, (M+31)/32, 1), workgroupSize := {x:=128}, extensions := ["f16","chromium_experimental_subgroup_matrix"], diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let bufs : List (String × Buffer) := [("src",src),("idx",idx),("b",b),("c",c1),("tileExpert",teB),("tileRows",trB)]
  IO.println "[poc] wgsl bench..."; (← IO.getStdout).flush
  let wgslMs ← benchKernelMs device kern bufs cfg 50
  IO.println s!"[poc] wgsl done: {wgslMs} ms"; (← IO.getStdout).flush
  let c1Bytes ← mapBufferRead device c1 0 (M*N*4).toUSize   -- full sync + golden data
  -- MSL side (same buffers except output c2)
  IO.println "[poc] msl bench..."; (← IO.getStdout).flush
  let mslMsS ← mslQ4kBench device src idx b c2 teB trB M.toUInt32 N.toUInt32 K.toUInt32 nExpert.toUInt32 srcRows.toUInt32 50
  IO.println s!"[poc] msl done: {mslMsS} ms"; (← IO.getStdout).flush
  let mslMs := (match mslMsS.splitOn "." with
    | [a] => a.toNat!.toFloat
    | a :: b :: _ => a.toNat!.toFloat + b.toNat!.toFloat / (10.0 ^ b.length.toFloat)
    | _ => 0.0)
  let c2Bytes ← mapBufferRead device c2 0 (M*N*4).toUSize
  IO.println "[poc] comparing..."; (← IO.getStdout).flush
  -- golden: element-wise maxDiff over the whole buffer (unwritten rows are 0 on both sides)
  let f1 := Hesper.WebGPU.bytesToFloatArray c1Bytes
  let f2 := Hesper.WebGPU.bytesToFloatArray c2Bytes
  let mut maxd := 0.0
  let mut maxrel := 0.0
  let mut nbad := 0
  for i in [0:M*N] do
    let a := f1.getD i 0.0
    let bv := f2.getD i 0.0
    let d := (a - bv).abs
    let rel := d / (max 1.0 a.abs)
    if d > maxd then maxd := d
    if rel > maxrel then maxrel := rel
    if rel > 0.05 then nbad := nbad + 1
  IO.println s!"WGSL(Tint): {wgslMs} ms | MSL(native): {mslMs} ms | ratio: {wgslMs/mslMs}x | maxDiff: {maxd} maxRel: {maxrel} nBad(rel>0.05): {nbad}"
  if maxd.isNaN || nbad > 0 then
    IO.println "❌ GOLDEN FAILED — timing comparison void"
  else
    IO.println s!"✅ golden OK — native Metal is {wgslMs/mslMs}x vs WGSL/Tint on the identical algorithm+data"

end Examples.Compute.MslPoc

def main : IO Unit := Examples.Compute.MslPoc.main
