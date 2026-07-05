import Hesper.WebGPU.Device
import Hesper.WGSL.Types
import Std.Data.HashMap

/-!
# Native Metal replay capture (Experiment 2 Phase A)

Falsification test of the post-mortem's C2 ("the E2B gap is the dispatch layer, not the
kernels"): capture every dispatch of ONE decode token — Tint-CLI-generated MSL, the Dawn
buffers in MSL binding order, grid/workgroup dims, threadgroup bytes — then replay the
whole token natively (`replayRun`) as Serial (sanity: should match Dawn's GPU
time) vs `MTLDispatchTypeConcurrent` (no-barrier upper bound / per-layer-barrier
realistic bound).

Mechanics: `Execute.executeShaderNamed` calls `recordDispatch` when `capturingRef` is
set. MSL comes from running the tint CLI (`HESPER_TINT_BIN`, same Tint version as the
pinned Dawn) on the exact WGSL we hand Dawn; the CLI assigns its own `[[buffer(i)]]`
indices, so we parse the entry-point signature and order the buffers to match. Replay is
TIMING-ONLY: it reruns on the live buffers after decode finishes (values become garbage;
all kernels are fixed-trip-count so timing is value-independent).
-/

namespace Hesper.WGSL.NativeReplay

open Hesper.WebGPU

initialize capturingRef : IO.Ref Bool ← IO.mkRef false
initialize recordedRef : IO.Ref Nat ← IO.mkRef 0
initialize missesRef : IO.Ref (List String) ← IO.mkRef []
/-- wgsl-hash → (msl, actual MSL entry name — Tint may rename, buffer names in MSL
    binding order, msl has a threadgroup(0) param) -/
initialize mslCacheRef : IO.Ref (Std.HashMap UInt64 (String × String × Array String × Bool)) ←
  IO.mkRef {}

def isCapturing : IO Bool := capturingRef.get

def startCapture : IO Unit := do
  replayReset
  recordedRef.set 0
  missesRef.set []
  capturingRef.set true

def stopCapture : IO (Nat × List String) := do
  capturingRef.set false
  return (← recordedRef.get, ← missesRef.get)

/-- Layer-boundary marker (honored by replayRun mode=2). No-op unless capturing. -/
def layerBarrier : IO Unit := do
  if ← capturingRef.get then replayBarrier

private def scalarBytes : ScalarType → Nat
  | .f16 => 2
  | .u64 => 8
  | _ => 4

/-- Upper-bound byte size of a workgroup-shared value (vec3 padded like vec4). -/
private partial def typeBytes : WGSLType → Nat
  | .scalar s => scalarBytes s
  | .vec2 s => 2 * scalarBytes s
  | .vec3 s | .vec4 s => 4 * scalarBytes s
  | .mat2x2 s => 4 * scalarBytes s
  | .mat3x3 s | .mat4x4 s => 16 * scalarBytes s
  | .array t n => n * typeBytes t
  | _ => 64  -- unknown shared type: generous fallback

/-- Total threadgroup-memory upper bound: Tint packs all shared vars into one struct at
    [[threadgroup(0)]]; over-allocating is harmless, so pad each member to 16. -/
def sharedBytes (vars : List (String × WGSLType)) : Nat :=
  vars.foldl (fun acc (_, t) => acc + ((typeBytes t + 15) / 16) * 16) 0

private def tmpDir : IO String := do
  return (← IO.getEnv "HESPER_REPLAY_TMP").getD "/tmp/hesper-replay"

/-- WGSL → MSL via the tint CLI (HESPER_TINT_BIN). tint infers formats from extensions,
    so the input must land in a `.wgsl` file. -/
private def wgslToMsl (wgsl : String) (h : UInt64) : IO String := do
  let some tint ← IO.getEnv "HESPER_TINT_BIN"
    | throw (IO.userError "HESPER_NATIVE_REPLAY: set HESPER_TINT_BIN to a tint CLI binary")
  let dir ← tmpDir
  IO.FS.createDirAll dir
  let path := s!"{dir}/k{h}.wgsl"
  IO.FS.writeFile path wgsl
  let out ← IO.Process.output { cmd := tint, args := #["--format", "msl", path] }
  if out.exitCode != 0 then
    throw (IO.userError s!"tint failed on {path}: {out.stderr}")
  return out.stdout

private def isIdentChar (c : Char) : Bool :=
  c.isAlphanum || c == '_'

/-- Parse the MSL entry point: Tint may RENAME the WGSL entry (e.g. `main` → `v`,
    reserved-word avoidance), so we locate the sole `kernel void NAME(` and return the
    ACTUAL name plus the buffer names indexed by their `[[buffer(K)]]` binding. Robust
    to commas inside `tint_array<T, N>` types (we never split on commas). -/
private def parseEntryAndBindings (msl : String) : String × Array String := Id.run do
  let marker := "kernel void "
  let parts := msl.splitOn marker
  if parts.length < 2 then return ("", #[])
  let rest := parts[1]!
  let entry := (rest.takeWhile isIdentChar).toString
  -- signature region: up to the opening brace of the function body
  let sig := (rest.splitOn "{")[0]!
  let pieces := sig.splitOn "[[buffer("
  if pieces.length < 2 then return (entry, #[])
  let mut pairs : Array (Nat × String) := #[]
  for i in [1:pieces.length] do
    let prev := pieces[i-1]!
    let cur := pieces[i]!
    -- binding index: leading digits of `cur`
    let idxStr := cur.takeWhile Char.isDigit
    -- buffer name: trailing identifier of `prev` (skip trailing spaces)
    let name := String.mk (((prev.trimRight).toList.reverse.takeWhile isIdentChar).reverse)
    match idxStr.toNat? with
    | some idx => pairs := pairs.push (idx, name)
    | none => pure ()
  let sorted := pairs.qsort (fun a b => a.1 < b.1)
  return (entry, sorted.map (·.2))

/-- Record one dispatch into the native replay sequence. Called from
    `Execute.executeShaderNamed` while capturing. Coverage misses (unparseable entry,
    unmatched buffer name) are logged, not fatal — the coverage report surfaces them. -/
def recordDispatch (device : Device) (wgsl : String) (entry : String)
    (namedBuffers : List (String × Buffer)) (tgBytes : Nat)
    (numWorkgroups : Nat × Nat × Nat) (wgX wgY wgZ : Nat) : IO Unit := do
  let h := hash wgsl
  let (msl, mslEntry, order, hasTg) ← do
    match (← mslCacheRef.get).get? h with
    | some x => pure x
    | none =>
      let msl ← wgslToMsl wgsl h
      let (mslEntry, order) := parseEntryAndBindings msl
      let hasTg : Bool := decide ((msl.splitOn "[[threadgroup(0)]]").length > 1)
      mslCacheRef.modify (·.insert h (msl, mslEntry, order, hasTg))
      pure (msl, mslEntry, order, hasTg)
  if mslEntry.isEmpty || (order.isEmpty && !namedBuffers.isEmpty) then
    missesRef.modify (s!"{entry}: no MSL entry/[[buffer]] params parsed" :: ·)
    return
  let mut bufs : Array Buffer := #[]
  for name in order do
    match namedBuffers.find? (·.1 == name) with
    | some (_, b) => bufs := bufs.push b
    | none =>
      missesRef.modify (s!"{entry}: buffer '{name}' not in namedBuffers" :: ·)
      return
  let (gx, gy, gz) := numWorkgroups
  replayRecord device msl mslEntry bufs
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    wgX.toUInt32 wgY.toUInt32 wgZ.toUInt32
    (if hasTg then tgBytes.toUInt32 else 0)
  recordedRef.modify (· + 1)

/-- Run the captured token natively in all three modes and return a report.
    Device-free: the MTLDevice was stashed at record time. -/
def runAll (iters : UInt32 := 20) : IO String := do
  let serial ← replayRun 0 iters
  let concNB ← replayRun 1 iters
  let concB ← replayRun 2 iters
  return s!"serial: {serial}\nconcurrent(no-barrier): {concNB}\nconcurrent(layer-barriers): {concB}"

end Hesper.WGSL.NativeReplay
