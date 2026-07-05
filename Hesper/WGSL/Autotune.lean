import Hesper.Backend
import Hesper.Backend.WebGPU
import Hesper.WGSL.Execute
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Autotune — the measured-JIT core (DEVPLAN M2)

The Triton-`@autotune` contract, for ShaderM kernels. A kernel FAMILY declares its tunable
parameter space and how to generate/validate itself; this engine provides everything else,
SHARED across families:

  sweep      resumable measured search: static feasibility → golden gate (small shape) →
             batched timing → occupancy probe (mandatory resource column) → tune log CSV
  winners    per (family, shape) best params persisted to tune/winners.csv
  best       RUNTIME lookup — consumers (the decode) fetch winning params by (family, shape)
             and drive the generator with them; nobody hand-copies numbers

Params are plain named Nats, so any `Nat → … → ShaderM Unit` generator can be driven at
runtime with zero Lean rebuild (the whole point: parameter iteration is in-process).

Adding a new tunable kernel = declaring a `Family` (~40-60 lines: space, feasible, gen, cfg,
golden). The engine, log, cache, and lookup are reused as-is. This is what makes it a
framework rather than a hand-written tuning loop per kernel.
-/

namespace Hesper.WGSL.Autotune

open Hesper.WebGPU

/-- A tunable parameter assignment: named Nat knobs. Plain data → serializable, runtime-drivable. -/
abbrev Params := List (String × Nat)

def Params.get! (p : Params) (k : String) : Nat :=
  match p.find? (·.1 == k) with
  | some (_, v) => v
  | none => panic! s!"Autotune.Params: missing knob '{k}'"

/-- Canonical serialized form, e.g. "BK=16;TM=2;TN=1;sgC=4;sgR=1" (sorted, order-stable). -/
def Params.key (p : Params) : String :=
  String.intercalate ";" ((p.map (fun (k,v) => s!"{k}={v}")).toArray.qsort (· < ·)).toList

def Params.parse (s : String) : Params :=
  s.splitOn ";" |>.filterMap fun kv =>
    match kv.splitOn "=" with
    | [k, v] => v.toNat?.map fun n => (k, n)
    | _ => none

structure Shape where
  M : Nat
  N : Nat
  K : Nat
  deriving BEq, Repr

def Shape.key (s : Shape) : String := s!"{s.M}x{s.N}x{s.K}"

/-- What a kernel family declares to become tunable (the whole per-kernel surface). -/
structure Family where
  name : String
  /-- the raw search space (the engine applies `feasible` + resume filtering) -/
  space : List Params
  /-- static feasibility: device limits, divisibility, shared-memory budget — reject BEFORE generation -/
  feasible : Params → Shape → Bool
  /-- the parameterized generator (runtime WGSL gen — no Lean rebuild per variant) -/
  gen : Params → Shape → Hesper.WGSL.Monad.ShaderM Unit
  /-- dispatch config (grid, workgroupSize, extensions) for params at a shape -/
  cfg : Params → Shape → Hesper.ExecConfig
  /-- timing buffers: (bindingName, f32-word count) at a (possibly padded) shape -/
  benchBuffers : Shape → List (String × Nat)
  /-- golden gate — MUST validate the variant (typically at a params-scaled small shape;
      real-shape CPU references are too slow). none = pass, some err = auto-disqualify. -/
  golden : Params → Device → IO (Option String)
  /-- shape padding for timing (e.g. M rounded up for WMMA-tail contracts) -/
  padShape : Params → Shape → Shape := fun _ s => s

def tuneDir : System.FilePath := "tune"
def logPath : System.FilePath := "tune/tune_log.csv"
def winnersPath : System.FilePath := "tune/winners.csv"

private def parseMs (s : String) : Option Float :=
  match s.splitOn "." with
  | [a] => a.toNat?.map (·.toFloat)
  | [a, b] =>
    match a.toNat?, b.toNat? with
    | some ai, some bi => some (ai.toFloat + bi.toFloat / (10.0 ^ b.length.toFloat))
    | _, _ => none
  | _ => none

private def readDoneKeys (family shape : String) : IO (List String) := do
  if ← winnersPath.pathExists then pure () -- winners existence irrelevant for resume
  if ← logPath.pathExists then
    let content ← IO.FS.readFile logPath
    pure <| content.splitOn "\n" |>.filterMap fun line =>
      match line.splitOn "," with
      | f :: s :: v :: _ => if f == family && s == shape then some v else none
      | _ => none
  else pure []

/-- Occupancy probe wrapper (mandatory resource column — DEVPLAN principle 2). Requires the
    process to run with HESPER_DUMP_MSL=1 (+_QUIET=1); returns "occ=n/a" otherwise. -/
private def probeOccupancy (device : Device) : IO String := do
  (do
    let msl ← Hesper.WebGPU.lastDumpedMsl
    if msl.isEmpty then pure "occ=n/a"
    else Hesper.WebGPU.mslOccupancyProbe device msl) <|> pure "occ=err"

/-- Rewrite tune/winners.csv with one (family, shape) row replaced. -/
private def writeWinnerRow (family : String) (shape : Shape) (params : String) (ms : Float)
    (tag : String := "") : IO Unit := do
  let old ← (do
    if ← winnersPath.pathExists then
      let c ← IO.FS.readFile winnersPath
      pure <| c.splitOn "\n" |>.filter (fun l =>
        !l.startsWith s!"{family},{shape.key}," && l != "")
    else pure ["family,shape,params,ms"])
  let h ← IO.FS.Handle.mk winnersPath .write
  for l in old do h.putStrLn l
  h.putStrLn s!"{family},{shape.key},{params},{ms}{tag}"
  h.flush

private def readWinnerMs (family : String) (shape : Shape) : IO Float := do
  if ← winnersPath.pathExists then
    let content ← IO.FS.readFile winnersPath
    for line in content.splitOn "\n" do
      match line.splitOn "," with
      | f :: s :: _v :: msS :: _ =>
        if f == family && s == shape.key then
          if let some ms := parseMs msS then return ms
      | _ => pure ()
  pure 1e18

/-- Resumable measured sweep of `fam` at `shape`. Appends every measurement to the tune log
    (negative results included) and updates tune/winners.csv with the best PASS row.
    `limit` caps variants per process (resource-lifetime workaround — re-run to continue).

    PROBE-PRUNE (standard autotune practice): each variant first gets a short `probeIters`
    measurement; if it is already `pruneFactor`× slower than the best seen so far it is logged
    as PRUNED and skipped — sweep time goes to the contenders, not the obviously-slow tail. -/
def sweep (device : Device) (fam : Family) (shape : Shape)
    (limit : Nat := 120) (iters : Nat := 30)
    (probeIters : Nat := 3) (pruneFactor : Float := 2.0) : IO Nat := do
  let stdout ← IO.getStdout
  IO.FS.createDirAll tuneDir
  let logExists ← logPath.pathExists
  let done ← readDoneKeys fam.name shape.key
  let h ← IO.FS.Handle.mk logPath (if logExists then .append else .write)
  if !logExists then
    h.putStrLn "family,shape,params,ms,gflops,occ,wg,golden"
  let feasibleAll := fam.space.filter (fam.feasible · shape)
  let todo := feasibleAll.filter (fun p => !(done.contains p.key))
  IO.println s!"[autotune] {fam.name} @ {shape.key}: {todo.length} remaining of {feasibleAll.length} feasible ({fam.space.length} space)"
  stdout.flush
  let flops := 2.0 * shape.M.toFloat * shape.N.toFloat * shape.K.toFloat
  -- best-so-far seeds the pruning threshold (resumes tight across processes)
  let mut bestSoFar ← readWinnerMs fam.name shape
  let mut ran := 0
  let mut pruned := 0
  for p in todo do
    if ran ≥ limit then break
    ran := ran + 1
    match ← fam.golden p device with
    | some err =>
      IO.println s!"  {p.key}: ❌ GOLDEN FAIL {err}"; stdout.flush
      h.putStrLn s!"{fam.name},{shape.key},{p.key},,,,,FAIL"
    | none =>
      let padded := fam.padShape p shape
      let kern := fam.gen p padded
      let cfg := fam.cfg p shape
      let bufSpecs := fam.benchBuffers padded
      let mut bufs : List (String × Buffer) := []
      for (nm, words) in bufSpecs do
        let b ← createBuffer device { size := (words*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
        bufs := bufs ++ [(nm, b)]
      let r ← IO.mkRef none
      Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
      let occ ← probeOccupancy device
      -- stage 1: short probe; prune the obviously-slow tail
      let tp0 ← IO.monoMsNow
      Hesper.GPUBackend.beginBatch device
      for _ in [0:probeIters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
      Hesper.GPUBackend.endBatch device
      let tp1 ← IO.monoMsNow
      let msProbe := (tp1-tp0).toFloat / probeIters.toFloat
      if msProbe > pruneFactor * bestSoFar then
        pruned := pruned + 1
        h.putStrLn s!"{fam.name},{shape.key},{p.key},{msProbe},,{occ},{cfg.workgroupSize.x},PRUNED"
      else
        -- stage 2: full measurement (contenders only)
        let t0 ← IO.monoMsNow
        Hesper.GPUBackend.beginBatch device
        for _ in [0:iters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
        Hesper.GPUBackend.endBatch device
        let t1 ← IO.monoMsNow
        let ms := (t1-t0).toFloat / iters.toFloat
        let gflops := flops / (ms/1000.0) / 1.0e9
        if ms < bestSoFar then bestSoFar := ms
        h.putStrLn s!"{fam.name},{shape.key},{p.key},{ms},{gflops},{occ},{cfg.workgroupSize.x},PASS"
      if ran % 20 == 0 then
        IO.println s!"  ... {ran}/{todo.length} this process ({pruned} pruned)"; stdout.flush
  if pruned > 0 then
    IO.println s!"  [prune] {pruned}/{ran} variants pruned at probe stage (> {pruneFactor}× best)"
  h.flush
  -- recompute + persist the winner for this (family, shape) from the full log
  updateWinner fam.name shape
  pure (todo.length - ran)   -- remaining after this process

where
  updateWinner (family : String) (shape : Shape) : IO Unit := do
    let content ← IO.FS.readFile logPath
    let mut bestMs : Float := 1e18
    let mut bestParams := ""
    for line in content.splitOn "\n" do
      match line.splitOn "," with
      | f :: s :: v :: msS :: _rest =>
        if f == family && s == shape.key && line.endsWith "PASS" then
          match parseMs msS with
          | some ms => if ms > 0 && ms < bestMs then bestMs := ms; bestParams := v
          | none => pure ()
      | _ => pure ()
    if bestParams != "" then
      writeWinnerRow family shape bestParams bestMs
      IO.println s!"[autotune] provisional winner {family}@{shape.key} = {bestParams} ({bestMs}ms) — run refine to confirm"

/-- REFINE stage (run after the sweep completes): single sweep measurements carry transient
    outliers up to ~3× (measured: the same config scored 0.267ms and 0.767ms in two runs).
    Re-measure the TOP-`k` distinct configs with `iters`×`reps` and crown the winner by
    min-of-reps — the sweep RANKS cheaply, the refine stage DECIDES reliably. -/
def refineTopK (device : Device) (fam : Family) (shape : Shape)
    (k : Nat := 10) (iters : Nat := 300) (reps : Nat := 3) : IO Unit := do
  let stdout ← IO.getStdout
  let content ← IO.FS.readFile logPath
  let mut rows : Array (Float × String) := #[]
  for line in content.splitOn "\n" do
    match line.splitOn "," with
    | f :: s :: v :: msS :: _ =>
      if f == fam.name && s == shape.key && line.endsWith "PASS" then
        if let some ms := parseMs msS then rows := rows.push (ms, v)
    | _ => pure ()
  let top := (((rows.qsort (fun a b => a.1 < b.1)).toList.map (·.2)).eraseDups).take k
  IO.println s!"[refine] {fam.name}@{shape.key}: top {top.length} configs × {reps} reps @ {iters} iters"
  stdout.flush
  let mut best : Float := 1e18
  let mut bestP := ""
  for v in top do
    let p := Params.parse v
    let padded := fam.padShape p shape
    let kern := fam.gen p padded
    let cfg := fam.cfg p shape
    let mut bufs : List (String × Buffer) := []
    for (nm, words) in fam.benchBuffers padded do
      let b ← createBuffer device { size := (words*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
      bufs := bufs ++ [(nm, b)]
    let r ← IO.mkRef none
    Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
    let mut mn : Float := 1e18
    let mut ts : List Float := []
    for _ in [0:reps] do
      let t0 ← IO.monoMsNow
      Hesper.GPUBackend.beginBatch device
      for _ in [0:iters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
      Hesper.GPUBackend.endBatch device
      let t1 ← IO.monoMsNow
      let ms := (t1-t0).toFloat / iters.toFloat
      ts := ts ++ [ms]
      if ms < mn then mn := ms
    IO.println s!"  {v}: reps={ts.map (fun t => (t*1000).round/1000)} min={((mn*1000).round)/1000}ms"
    stdout.flush
    if mn < best then best := mn; bestP := v
  if bestP != "" then
    writeWinnerRow fam.name shape bestP best
    IO.println s!"[refine] WINNER {fam.name}@{shape.key} = {bestP} ({best}ms, min of {reps}×{iters})"

/-- RUNTIME winner lookup: consumers drive their generator with these params. Falls back to
    `default` when the shape was never tuned (and logs that fact — silent fallbacks hide
    un-tuned deployments). -/
def best (family : String) (shape : Shape) (default : Params) : IO Params := do
  if ← winnersPath.pathExists then
    let content ← IO.FS.readFile winnersPath
    for line in content.splitOn "\n" do
      match line.splitOn "," with
      | f :: s :: v :: _ =>
        if f == family && s == shape.key then
          return Params.parse v
      | _ => pure ()
  IO.eprintln s!"[autotune] NO WINNER for {family}@{shape.key} — using default (run the sweep!)"
  pure default

end Hesper.WGSL.Autotune
