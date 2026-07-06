import Hesper.WebGPU.Device
import Hesper.WGSL.Types
import Std.Data.HashMap

/-!
# Native Metal replay (Experiment 2 Phase A/B)

Falsification test of the post-mortem's C2 ("the E2B gap is the dispatch layer, not the
kernels"), in two phases:

* **Phase A (timing)**: capture every dispatch of ONE decode token — Tint-CLI MSL, the
  Dawn buffers in MSL binding order, grid/workgroup dims, threadgroup bytes — then replay
  the token natively (`replayRun`): Serial 6.78 ms ≈ Dawn's 7.7 ms, Concurrent
  no-barrier 3.43 ms, per-layer barriers 3.92 ms (M4 Max, identical kernels).

* **Phase B (real execution)**: `mode := .nativeExec` makes the `executeShaderNamed`
  hook SKIP the Dawn dispatch entirely; the token's ops are recorded and then executed
  ONCE by `commitToken` on Dawn's own MTLCommandQueue (single-queue FIFO keeps ordering
  with Dawn staging writes and readbacks) with mode-3 automatic hazard barriers
  (llama.cpp `ggml_mem_ranges`-style, whole-buffer granularity). Correctness is gated on
  the generated text.

MSL comes from running the tint CLI (`HESPER_TINT_BIN`, same Tint version as the pinned
Dawn) on the exact WGSL we hand Dawn; the CLI assigns its own `[[buffer(i)]]` indices
and may RENAME the entry point (`main` → `v`), so both are parsed from the MSL. Steady
state is served from a cache keyed by the caller's authoritative cacheKey, so the
per-dispatch host cost is a HashMap probe + buffer-pointer pushes — no WGSL regeneration.
-/

namespace Hesper.WGSL.NativeReplay

open Hesper.WebGPU

inductive Mode where
  | off
  /-- Record ops AND still execute them through Dawn (Phase A timing capture). -/
  | capture
  /-- Record ops and SKIP the Dawn dispatch; `commitToken` executes them natively. -/
  | nativeExec
  deriving BEq, Inhabited

initialize modeRef : IO.Ref Mode ← IO.mkRef .off
initialize recordedRef : IO.Ref Nat ← IO.mkRef 0
initialize missesRef : IO.Ref (List String) ← IO.mkRef []

structure CacheEntry where
  msl : String
  /-- Actual MSL entry name (Tint may rename, e.g. `main` → `v`). Empty = unusable. -/
  mslEntry : String
  /-- Buffer names in MSL `[[buffer(i)]]` order. -/
  order : Array String
  /-- Threadgroup bytes to bind at index 0 (0 = kernel has no threadgroup param). -/
  tgBytes : UInt32
  /-- Bit i set ⇒ order[i] is a read_write binding (drives mode-3 hazard barriers). -/
  writeMask : UInt32
  /-- perm[i] = position of order[i] in the call site's namedBuffers list (captured at
      build time). Steady-state emit indexes instead of string-searching; a name
      mismatch (different call-site ordering under the same key) falls back to find. -/
  perm : Array Nat

/-- Keyed by the caller's authoritative cacheKey (or hash of the WGSL when unkeyed). -/
initialize cacheRef : IO.Ref (Std.HashMap UInt64 CacheEntry) ← IO.mkRef {}

def currentMode : IO Mode := modeRef.get

def isActive : IO Bool := do return (← modeRef.get) != .off

def startCapture : IO Unit := do
  replayReset
  recordedRef.set 0
  missesRef.set []
  modeRef.set .capture

def stopCapture : IO (Nat × List String) := do
  modeRef.set .off
  return (← recordedRef.get, ← missesRef.get)

/-- Phase B: begin recording a token that will NOT be dispatched through Dawn. -/
def beginNativeToken : IO Unit := do
  replayReset
  recordedRef.set 0
  missesRef.set []
  modeRef.set .nativeExec

/-- Phase B: execute the recorded token natively (mode 3 = concurrent + hazard barriers
    by default) and drop back to normal Dawn dispatching. Throws if any dispatch was
    missed (a partial token would silently corrupt the decode). -/
def commitToken (mode : UInt32 := 3) : IO String := do
  modeRef.set .off
  let misses ← missesRef.get
  if !misses.isEmpty then
    throw (IO.userError s!"nativeExec token had {misses.length} missed dispatches, first: {misses.head!}")
  replayExec mode

/-- Layer-boundary marker (honored by replayRun mode=2). No-op unless active. -/
def layerBarrier : IO Unit := do
  if (← modeRef.get) != .off then replayBarrier

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

/-- Order `namedBuffers` per the cache entry and push the dispatch to the C++ side.
    Steady state uses the cached permutation (index + name check); string search only
    on a permutation mismatch. -/
private def emit (device : Device) (key : UInt64) (e : CacheEntry) (dbgName : String)
    (namedBuffers : List (String × Buffer))
    (numWorkgroups : Nat × Nat × Nat) (wgX wgY wgZ : Nat) : IO Unit := do
  if e.mslEntry.isEmpty || (e.order.isEmpty && !namedBuffers.isEmpty) then
    missesRef.modify (s!"{dbgName}: no MSL entry/[[buffer]] params parsed" :: ·)
    return
  let nb := namedBuffers.toArray
  let mut bufs : Array Buffer := #[]
  for i in [0:e.order.size] do
    let name := e.order[i]!
    let viaPerm : Option Buffer := do
      let j ← e.perm[i]?
      let (n, b) ← nb[j]?
      if n == name then some b else none
    match viaPerm <|> (nb.find? (·.1 == name)).map (·.2) with
    | some b => bufs := bufs.push b
    | none =>
      missesRef.modify (s!"{dbgName}: buffer '{name}' not in namedBuffers" :: ·)
      return
  let (gx, gy, gz) := numWorkgroups
  replayRecord device e.msl e.mslEntry bufs
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    wgX.toUInt32 wgY.toUInt32 wgZ.toUInt32
    e.tgBytes e.writeMask key
  recordedRef.modify (· + 1)

/-- Steady-state record: cache hit on the caller's authoritative key ⇒ record with no
    WGSL/tint work at all. Returns false on miss (caller falls back to `recordSlow`). -/
def tryRecordFast (device : Device) (key : UInt64)
    (namedBuffers : List (String × Buffer))
    (numWorkgroups : Nat × Nat × Nat) (wgX wgY wgZ : Nat) : IO Bool := do
  if key == 0 then return false
  match (← cacheRef.get).get? key with
  | some e => emit device key e s!"key{key}" namedBuffers numWorkgroups wgX wgY wgZ; return true
  | none => return false

/-- Cold-path record: run tint, parse the entry/bindings, compute the write mask from
    the declared read_write buffer names, cache under `key`, and record. -/
def recordSlow (device : Device) (key : UInt64) (wgsl : String) (dbgName : String)
    (namedBuffers : List (String × Buffer)) (writableNames : List String) (tgBytes : Nat)
    (numWorkgroups : Nat × Nat × Nat) (wgX wgY wgZ : Nat) : IO Unit := do
  let msl ← wgslToMsl wgsl (hash wgsl)
  let (mslEntry, order) := parseEntryAndBindings msl
  let hasTg : Bool := decide ((msl.splitOn "[[threadgroup(0)]]").length > 1)
  let mut writeMask : UInt32 := 0
  for i in [0:order.size] do
    if writableNames.contains order[i]! then
      writeMask := writeMask ||| (1 <<< (UInt32.ofNat i))
  let names := (namedBuffers.map (·.1)).toArray
  let perm := order.map (fun n => (names.findIdx? (· == n)).getD names.size)
  let e : CacheEntry := {
    msl, mslEntry, order
    tgBytes := if hasTg then tgBytes.toUInt32 else 0
    writeMask, perm }
  cacheRef.modify (·.insert key e)
  emit device key e dbgName namedBuffers numWorkgroups wgX wgY wgZ

/-- Run the captured token natively in all four timing modes and return a report.
    Device-free: the MTLDevice was stashed at record time. TIMING ONLY. -/
def runAll (iters : UInt32 := 20) : IO String := do
  let serial ← replayRun 0 iters
  let concNB ← replayRun 1 iters
  let concB ← replayRun 2 iters
  let concH ← replayRun 3 iters
  return s!"serial: {serial}\nconcurrent(no-barrier): {concNB}\nconcurrent(layer-barriers): {concB}\nconcurrent(hazard-barriers): {concH}"

end Hesper.WGSL.NativeReplay
