import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.WebGPU
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.FlashAttention
import Hesper

/-!
# FlashAttention Golden Value Tests

Compares GPU FlashAttention output against CPU reference implementation
(`flashAttentionSpec`) for all three kernel variants, using BitNet-scale
parameters (numHeads=20, numKVHeads=4, headDim=128, GQA).

Three-way comparison: CPU spec → WebGPU → CUDA.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

-- ═══ Helpers ═══

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def pf (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def uf (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -fv else fv

private def pu (arr : Array Nat) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun acc v =>
    acc.push (v % 256).toUInt8 |>.push ((v/256) % 256).toUInt8
      |>.push ((v/65536) % 256).toUInt8 |>.push ((v/16777216) % 256).toUInt8

private def ufs (ba : ByteArray) (n : Nat) : Array Float :=
  Array.range n |>.map (uf ba)

-- ═══ Deterministic pseudo-random ═══

private def lcg (seed : Nat) : Nat := (seed * 1103515245 + 12345) % (2^31)

private def randFloat (seed : Nat) : Float × Nat :=
  let s := lcg seed
  let f := (s.toFloat / (2^31).toFloat) * 2.0 - 1.0  -- range [-1, 1]
  (f, s)

/-- Snap to f32 precision: f64 → f32 bits → f64 -/
private def snapF32 (f : Float) : Float := uf (pf #[f]) 0

private def randArray (n : Nat) (seed : Nat) : Array Float × Nat := Id.run do
  let mut arr := Array.mkEmpty n
  let mut s := seed
  for _ in [:n] do
    let (v, s') := randFloat s
    arr := arr.push (snapF32 v)  -- snap to f32 precision for CPU↔GPU consistency
    s := s'
  (arr, s)

-- ═══ Config ═══

structure FAConfig where
  numHeads : Nat
  numKVHeads : Nat
  headDim : Nat
  maxSeqLen : Nat
  cacheLen : Nat

-- ═══ CPU reference ═══

/-- Run CPU flashAttentionSpec for one query head, returning headDim floats. -/
private def cpuFlashAttnHead (q : Array Float) (kCache vCache : Array (Array Float))
    (scale : Float) : Array Float :=
  Hesper.WGSL.FlashAttention.flashAttentionSpec q kCache vCache scale

/-- Run CPU reference for all heads. Returns flat array [numHeads * headDim]. -/
private def cpuFlashAttn (cfg : FAConfig) (qFlat : Array Float)
    (kFlat vFlat : Array Float) (scale : Float) : Array Float := Id.run do
  let mut out := Array.mkEmpty (cfg.numHeads * cfg.headDim)
  for h in [:cfg.numHeads] do
    -- Extract Q for this head
    let qHead := Array.range cfg.headDim |>.map (fun d => qFlat.getD (h * cfg.headDim + d) 0.0)
    -- KV head (GQA)
    let kvH := h / (cfg.numHeads / cfg.numKVHeads)
    -- Extract K/V cache as array of arrays [cacheLen][headDim]
    let kCache := Array.range cfg.cacheLen |>.map fun s =>
      Array.range cfg.headDim |>.map fun d =>
        kFlat.getD (kvH * cfg.maxSeqLen * cfg.headDim + s * cfg.headDim + d) 0.0
    let vCache := Array.range cfg.cacheLen |>.map fun s =>
      Array.range cfg.headDim |>.map fun d =>
        vFlat.getD (kvH * cfg.maxSeqLen * cfg.headDim + s * cfg.headDim + d) 0.0
    let headOut := cpuFlashAttnHead qHead kCache vCache scale
    out := out ++ headOut
  out

-- ═══ GPU runner ═══

private def runFlashAttnGPU [GPUBackend β] (ctx : β) (cfg : FAConfig)
    (qData kData vData : ByteArray) (scale : Float)
    (variant : String) : IO (Array Float) := do
  let qSize := (cfg.numHeads * cfg.headDim * 4).toUSize
  let kvSize := (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim * 4).toUSize
  let outSize := qSize

  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let outBuf ← GPUBackend.allocBuffer ctx outSize

  GPUBackend.writeBuffer ctx qBuf qData
  GPUBackend.writeBuffer ctx kBuf kData
  GPUBackend.writeBuffer ctx vBuf vData

  match variant with
  | "subgroup" =>
    -- flashAttentionSubgroupKernel: wgSize=32, 1 WG per head
    let kernel := Hesper.WGSL.FlashAttention.flashAttentionSubgroupKernel
      cfg.numHeads cfg.numKVHeads cfg.maxSeqLen cfg.headDim cfg.cacheLen scale
    let bufs : List (String × GPUBackend.Buf β) :=
      [("q", qBuf), ("k_cache", kBuf), ("v_cache", vBuf), ("output", outBuf)]
    GPUBackend.execute ctx kernel bufs
      { workgroupSize := { x := 32 }, numWorkgroups := (cfg.numHeads, 1, 1),
        extensions := ["subgroups"] }
    let result ← GPUBackend.readBuffer ctx outBuf outSize
    GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf
    GPUBackend.freeBuffer ctx vBuf; GPUBackend.freeBuffer ctx outBuf
    return ufs result (cfg.numHeads * cfg.headDim)

  | "swa" =>
    -- flashAttentionSWASubgroupKernel: sliding window
    let windowSize := 4
    let currentPos := cfg.cacheLen - 1
    let kernel := Hesper.WGSL.FlashAttention.flashAttentionSWASubgroupKernel
      cfg.numHeads cfg.numKVHeads cfg.maxSeqLen cfg.headDim
      cfg.cacheLen windowSize currentPos scale
    let bufs : List (String × GPUBackend.Buf β) :=
      [("q", qBuf), ("k_cache", kBuf), ("v_cache", vBuf), ("output", outBuf)]
    GPUBackend.execute ctx kernel bufs
      { workgroupSize := { x := 32 }, numWorkgroups := (cfg.numHeads, 1, 1),
        extensions := ["subgroups"] }
    let result ← GPUBackend.readBuffer ctx outBuf outSize
    GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf
    GPUBackend.freeBuffer ctx vBuf; GPUBackend.freeBuffer ctx outBuf
    return ufs result (cfg.numHeads * cfg.headDim)

  | "params" =>
    -- flashAttentionParamsKernel: params buffer with dynamic cacheLen
    let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
    GPUBackend.writeBuffer ctx paramsBuf (pu #[cfg.cacheLen - 1, cfg.cacheLen])
    -- Q is read and output goes to the same buffer (in-place)
    GPUBackend.writeBuffer ctx outBuf qData
    let wgSize := 256
    let kernel := Hesper.WGSL.FlashAttention.flashAttentionParamsKernel
      cfg.numHeads cfg.numKVHeads cfg.maxSeqLen cfg.headDim scale wgSize
    let bufs : List (String × GPUBackend.Buf β) :=
      [("q_output", outBuf), ("k_cache", kBuf), ("v_cache", vBuf), ("params", paramsBuf)]
    GPUBackend.execute ctx kernel bufs
      { workgroupSize := { x := wgSize }, numWorkgroups := (cfg.numHeads, 1, 1),
        extensions := ["subgroups"] }
    let result ← GPUBackend.readBuffer ctx outBuf outSize
    GPUBackend.freeBuffer ctx qBuf; GPUBackend.freeBuffer ctx kBuf
    GPUBackend.freeBuffer ctx vBuf; GPUBackend.freeBuffer ctx outBuf
    GPUBackend.freeBuffer ctx paramsBuf
    return ufs result (cfg.numHeads * cfg.headDim)

  | _ => throw (IO.userError s!"Unknown variant: {variant}")

-- ═══ Comparison ═══

private def compareArrays (name : String) (a b : Array Float) (tol : Float := 1e-3) : IO Bool := do
  let n := min a.size b.size
  let mut maxDiff : Float := 0.0
  let mut maxIdx := 0
  for i in [:n] do
    let d := (a[i]! - b[i]!).abs
    if d > maxDiff then
      maxDiff := d
      maxIdx := i
  if maxDiff < tol then
    IO.println s!"  ✓ {name}: max diff={maxDiff} at [{maxIdx}]"
    return true
  else
    IO.println s!"  ✗ {name}: max diff={maxDiff} at [{maxIdx}]"
    IO.println s!"    a[{maxIdx}]={a[maxIdx]!} b[{maxIdx}]={b[maxIdx]!}"
    -- Show first few diffs
    for i in [:min 4 n] do
      if (a[i]! - b[i]!).abs > tol * 0.1 then
        IO.println s!"    [{i}] a={a[i]!} b={b[i]!} diff={(a[i]! - b[i]!).abs}"
    return false

-- ═══ Main ═══

def main : IO Unit := do
  IO.println "═══ FlashAttention Golden Value Tests ═══\n"

  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let cuda ← CUDAContext.init

  let mut passed := 0
  let mut failed := 0

  -- ── Test 1: Small config (fast, easy to debug) ──
  IO.println "── Test 1: Small (heads=2, kvHeads=1, dim=32, seq=4) ──"
  do
    let cfg : FAConfig := { numHeads := 2, numKVHeads := 1, headDim := 32, maxSeqLen := 8, cacheLen := 4 }
    let scale := 1.0 / Float.sqrt cfg.headDim.toFloat

    let (qArr, s1) := randArray (cfg.numHeads * cfg.headDim) 42
    let (kArr, s2) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s1
    let (vArr, _) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s2

    let cpuOut := cpuFlashAttn cfg qArr kArr vArr scale
    IO.println s!"  CPU ref computed ({cpuOut.size} floats)"

    let qBytes := pf qArr; let kBytes := pf kArr; let vBytes := pf vArr

    let wOut ← runFlashAttnGPU device cfg qBytes kBytes vBytes scale "subgroup"
    let cOut ← runFlashAttnGPU cuda cfg qBytes kBytes vBytes scale "subgroup"

    if ← compareArrays "CPU↔WebGPU subgroup" cpuOut wOut then passed := passed + 1
    else failed := failed + 1
    if ← compareArrays "CPU↔CUDA subgroup" cpuOut cOut then passed := passed + 1
    else failed := failed + 1
    if ← compareArrays "WebGPU↔CUDA subgroup" wOut cOut then passed := passed + 1
    else failed := failed + 1

  -- ── Test 2: BitNet-scale (numHeads=20, numKVHeads=4, headDim=128) ──
  IO.println "\n── Test 2: BitNet-scale (heads=20, kvHeads=4, dim=128, seq=8) ──"
  do
    let cfg : FAConfig := { numHeads := 20, numKVHeads := 4, headDim := 128, maxSeqLen := 32, cacheLen := 8 }
    let scale := 1.0 / Float.sqrt cfg.headDim.toFloat

    let (qArr, s1) := randArray (cfg.numHeads * cfg.headDim) 123
    let (kArr, s2) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s1
    let (vArr, _) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s2

    let cpuOut := cpuFlashAttn cfg qArr kArr vArr scale
    let qBytes := pf qArr; let kBytes := pf kArr; let vBytes := pf vArr

    -- Subgroup variant
    let wOut ← runFlashAttnGPU device cfg qBytes kBytes vBytes scale "subgroup"
    let cOut ← runFlashAttnGPU cuda cfg qBytes kBytes vBytes scale "subgroup"
    if ← compareArrays "CPU↔WebGPU subgroup" cpuOut wOut then passed := passed + 1
    else failed := failed + 1
    if ← compareArrays "WebGPU↔CUDA subgroup" wOut cOut then passed := passed + 1
    else failed := failed + 1

    -- Params variant (production kernel)
    let wOutP ← runFlashAttnGPU device cfg qBytes kBytes vBytes scale "params"
    let cOutP ← runFlashAttnGPU cuda cfg qBytes kBytes vBytes scale "params"
    if ← compareArrays "CPU↔WebGPU params" cpuOut wOutP then passed := passed + 1
    else failed := failed + 1
    if ← compareArrays "WebGPU↔CUDA params" wOutP cOutP then passed := passed + 1
    else failed := failed + 1

  -- ── Test 3: SWA variant ──
  IO.println "\n── Test 3: SWA sliding window (heads=4, kvHeads=2, dim=64, seq=8, window=4) ──"
  do
    let cfg : FAConfig := { numHeads := 4, numKVHeads := 2, headDim := 64, maxSeqLen := 16, cacheLen := 8 }
    let scale := 1.0 / Float.sqrt cfg.headDim.toFloat

    let (qArr, s1) := randArray (cfg.numHeads * cfg.headDim) 777
    let (kArr, s2) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s1
    let (vArr, _) := randArray (cfg.numKVHeads * cfg.maxSeqLen * cfg.headDim) s2

    let qBytes := pf qArr; let kBytes := pf kArr; let vBytes := pf vArr

    let wOut ← runFlashAttnGPU device cfg qBytes kBytes vBytes scale "swa"
    let cOut ← runFlashAttnGPU cuda cfg qBytes kBytes vBytes scale "swa"
    if ← compareArrays "WebGPU↔CUDA SWA" wOut cOut then passed := passed + 1
    else failed := failed + 1

  -- ═══ Summary ═══
  IO.println s!"\n═══ {passed} passed, {failed} failed ═══"
  if failed > 0 then IO.println "✗ SOME TESTS FAILED"
  else IO.println "✓ ALL TESTS PASSED"
