import Std.Data.HashMap

/-!
# JS-replay trace (`DG_TRACE_JS=<outdir>`)

Dumps everything a thin JS+WebGPU replayer needs to re-execute a decode step
of the DiffusionGemma engine (the Campaign-3 trace-replay pattern, aimed at
Chrome instead of Metal):

* `k<hash>.wgsl` — every template-expanded kernel, written once at first
  pipeline compile (per-process pipeline cache guarantees the source is in
  hand exactly then).
* `ops.jsonl` — the event stream while ARMED:
    {"t":"d","k":H,"n":name,"g":[x,y,z],"b":[[bufName,uid],…]}   dispatch
    {"t":"w","u":uid,"o":off,"s":size,"hex":"…"?}                writeBuffer
    {"t":"r","u":uid,"o":off,"s":size}                           readback
    {"t":"f"}                                                    queue flush
    {"t":"m","tag":str}                                          marker
* `buffers.json` — uid → byte size for every buffer created while enabled.
* `tensors.json` — uid → GGUF tensor name (weight provenance: the JS side
  loads the GGUF itself and uploads each tensor into its replay slot).

Buffer identity = `getBufferId` (the raw WGPUBuffer handle), already used by
the bind-group cache. Write contents are recorded as hex for writes ≤ 64 KB
(params, canvases, norms); larger writes (weights) are size-only — their
provenance comes from `tensors.json`.
-/

namespace Hesper.WGSL.JSTrace

initialize dirRef : IO.Ref (Option String) ← IO.mkRef none
initialize armedRef : IO.Ref Bool ← IO.mkRef false
initialize seenKernelsRef : IO.Ref (Std.HashMap UInt64 Bool) ← IO.mkRef {}
initialize opsRef : IO.Ref (Array String) ← IO.mkRef #[]
initialize bufSizesRef : IO.Ref (Array (UInt64 × Nat)) ← IO.mkRef #[]
initialize tensorsRef : IO.Ref (Array (UInt64 × String)) ← IO.mkRef #[]
initialize untracedRef : IO.Ref Nat ← IO.mkRef 0
initialize refUidsRef : IO.Ref (Std.HashMap UInt64 Bool) ← IO.mkRef {}
initialize hexUidsRef : IO.Ref (Std.HashMap UInt64 Bool) ← IO.mkRef {}

/-- one-time env read: DG_TRACE_JS=<outdir> enables tracing. -/
initialize do
  match (← IO.getEnv "DG_TRACE_JS") with
  | some d =>
    IO.FS.createDirAll d
    dirRef.set (some d)
  | none => pure ()

@[inline] def enabled : IO Bool := do return (← dirRef.get).isSome

def arm : IO Unit := do
  if ← enabled then armedRef.set true

def disarm : IO Unit := armedRef.set false

@[inline] def armed : IO Bool := armedRef.get

private def hexdigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat ('0'.toNat + n) else Char.ofNat ('a'.toNat + n - 10)

def toHex (b : ByteArray) : String := Id.run do
  let mut s := ""
  for byte in b do
    s := s.push (hexdigit (byte.toNat / 16)) |>.push (hexdigit (byte.toNat % 16))
  return s

def emit (line : String) : IO Unit := do
  if ← armed then opsRef.modify (·.push line)

/-- record a kernel's WGSL at first compile (independent of arming: the
pipeline cache means compile happens once, possibly before the traced step). -/
def kernel (sourceHash : UInt64) (wgsl : String) : IO Unit := do
  let some d ← dirRef.get | return
  let seen ← seenKernelsRef.get
  if seen.contains sourceHash then return
  seenKernelsRef.modify (·.insert sourceHash true)
  IO.FS.writeFile s!"{d}/k{sourceHash}.wgsl" wgsl

def dispatch (sourceHash : UInt64) (name : String) (grid : Nat × Nat × Nat)
    (bufs : Array (String × UInt64)) : IO Unit := do
  if ← armed then
    for (_, u) in bufs do refUidsRef.modify (·.insert u true)
  let (x, y, z) := grid
  let bs := ",".intercalate (bufs.toList.map fun (n, u) => s!"[\"{n}\",{u}]")
  emit s!"\{\"t\":\"d\",\"k\":\"{sourceHash}\",\"n\":\"{name}\",\"g\":[{x},{y},{z}],\"b\":[{bs}]}"

def write (uid : UInt64) (offset : Nat) (data : ByteArray) : IO Unit := do
  if ¬(← armed) then return
  if data.size ≤ 65536 then
    hexUidsRef.modify (·.insert uid true)
    emit s!"\{\"t\":\"w\",\"u\":{uid},\"o\":{offset},\"s\":{data.size},\"hex\":\"{toHex data}\"}"
  else
    emit s!"\{\"t\":\"w\",\"u\":{uid},\"o\":{offset},\"s\":{data.size}}"

/-- readback event; contents ≤ 2MB are recorded so a replayer can gate on
bit-equality without porting the CPU logic that consumes them. -/
def read (uid : UInt64) (offset size : Nat) (data : ByteArray) : IO Unit := do
  if data.size ≤ 2097152 then
    emit s!"\{\"t\":\"r\",\"u\":{uid},\"o\":{offset},\"s\":{size},\"hex\":\"{toHex data}\"}"
  else
    emit s!"\{\"t\":\"r\",\"u\":{uid},\"o\":{offset},\"s\":{size}}"

def flush : IO Unit := do
  emit "{\"t\":\"f\"}"

def mark (tag : String) : IO Unit := do
  emit s!"\{\"t\":\"m\",\"tag\":\"{tag}\"}"

def bufCreated (uid : UInt64) (size : Nat) : IO Unit := do
  if ← enabled then bufSizesRef.modify (·.push (uid, size))

def tensor (uid : UInt64) (name : String) : IO Unit := do
  if ← enabled then tensorsRef.modify (·.push (uid, name))

initialize cksumRef : IO.Ref Bool ← IO.mkRef false
initialize dispCntRef : IO.Ref Nat ← IO.mkRef 0

initialize do
  if (← IO.getEnv "DG_TRACE_JS_CKSUM").isSome then cksumRef.set true

@[inline] def cksumOn : IO Bool := do
  return (← cksumRef.get) ∧ (← armedRef.get)

/-- FNV-1a 32-bit over the first `n` bytes (mirrored by the JS replayer). -/
def fnv32 (b : ByteArray) (n : Nat := 4096) : UInt32 := Id.run do
  let mut h : UInt32 := 0x811c9dc5
  for i in [0:min n b.size] do
    h := (h ^^^ (b[i]!).toUInt32) * 16777619
  return h

def cksum (hs : Array UInt32) : IO Unit := do
  let n ← dispCntRef.get
  dispCntRef.modify (· + 1)
  let l := ",".intercalate (hs.toList.map toString)
  emit s!"\{\"t\":\"c\",\"n\":{n},\"hs\":[{l}]}"

def outDirGet : IO (Option String) := dirRef.get

/-- drop accumulated ops (used between the ref-collection step and the
recorded replay step). -/
def clearOps : IO Unit := opsRef.set #[]

/-- uids referenced by traced dispatches that have NEITHER GGUF tensor
provenance NOR recorded write contents — the derived buffers a replayer
must load from .bin dumps. -/
def missingUids : IO (Array UInt64) := do
  -- dump EVERYTHING referenced (incl. GGUF-provenance buffers): the replayer
  -- prefers .bin dumps and uses GGUF ranges only as fallback — removes the
  -- range-fetch/repack uncertainty from the parity equation entirely
  let refs ← refUidsRef.get
  return refs.toList.toArray.map (·.1)

def untraced : IO Unit := do
  if ← armed then untracedRef.modify (· + 1)

/-- write ops.jsonl / buffers.json / tensors.json and report. -/
def save : IO Unit := do
  let some d ← dirRef.get | return
  let ops ← opsRef.get
  IO.FS.writeFile s!"{d}/ops.jsonl" ("\n".intercalate ops.toList)
  let sizes ← bufSizesRef.get
  IO.FS.writeFile s!"{d}/buffers.json"
    ("{" ++ ",".intercalate (sizes.toList.map fun (u, s) => s!"\"{u}\":{s}") ++ "}")
  let tens ← tensorsRef.get
  IO.FS.writeFile s!"{d}/tensors.json"
    ("{" ++ ",".intercalate (tens.toList.map fun (u, n) => s!"\"{u}\":\"{n}\"") ++ "}")
  let miss ← untracedRef.get
  IO.println (s!"[JSTrace] saved {ops.size} events, {sizes.size} buffers, \
{tens.size} tensors to {d}"
    ++ (if miss > 0 then s!" — WARNING: {miss} UNTRACED dispatches" else ""))

end Hesper.WGSL.JSTrace
