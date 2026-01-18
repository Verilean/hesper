namespace Hesper

/-! GPU profiling infrastructure for performance analysis.

Provides types and utilities for measuring GPU kernel execution times
and exporting profiling data for visualization.
-/

/-- A single profiling event with name, timestamps, and duration -/
structure ProfileEvent where
  eventName : String        -- Event name
  startTime : UInt64        -- Start timestamp in nanoseconds
  endTime : UInt64          -- End timestamp in nanoseconds
  durationMs : Float        -- Duration in milliseconds
  deriving Repr, BEq

namespace ProfileEvent

/-- Create a ProfileEvent from start and end timestamps (in nanoseconds) -/
def fromTimestamps (name : String) (start : UInt64) (end_ : UInt64) : ProfileEvent :=
  let durationNs := end_.toNat - start.toNat
  let durationMs := durationNs.toFloat / 1_000_000.0
  { eventName := name
    startTime := start
    endTime := end_
    durationMs := durationMs }

/-- Get duration in nanoseconds -/
def durationNs (e : ProfileEvent) : Nat :=
  e.endTime.toNat - e.startTime.toNat

/-- Get duration in microseconds -/
def durationUs (e : ProfileEvent) : Float :=
  e.durationNs.toFloat / 1_000.0

instance : ToString ProfileEvent where
  toString e := s!"ProfileEvent(name={e.eventName}, start={e.startTime}ns, end={e.endTime}ns, duration={e.durationMs}ms)"

/-- Ordering by start time for sorting events -/
instance : Ord ProfileEvent where
  compare a b := compare a.startTime b.startTime

end ProfileEvent

/-- Summary statistics for a collection of profiling events -/
structure ProfileSummary where
  totalEvents : Nat
  totalDurationMs : Float
  minDurationMs : Float
  maxDurationMs : Float
  avgDurationMs : Float
  deriving Repr

namespace ProfileSummary

/-- Compute summary statistics from a list of events -/
def fromEvents (events : List ProfileEvent) : ProfileSummary :=
  if events.isEmpty then
    { totalEvents := 0
      totalDurationMs := 0.0
      minDurationMs := 0.0
      maxDurationMs := 0.0
      avgDurationMs := 0.0 }
  else
    let durations := events.map (·.durationMs)
    let total := durations.foldl (· + ·) 0.0
    let min := durations.foldl (fun a b => if a < b then a else b) (durations.head!)
    let max := durations.foldl (fun a b => if a > b then a else b) (durations.head!)
    let avg := total / events.length.toFloat
    { totalEvents := events.length
      totalDurationMs := total
      minDurationMs := min
      maxDurationMs := max
      avgDurationMs := avg }

instance : ToString ProfileSummary where
  toString s :=
    s!"ProfileSummary:\n" ++
    s!"  Total events: {s.totalEvents}\n" ++
    s!"  Total time: {s.totalDurationMs} ms\n" ++
    s!"  Min duration: {s.minDurationMs} ms\n" ++
    s!"  Max duration: {s.maxDurationMs} ms\n" ++
    s!"  Avg duration: {s.avgDurationMs} ms"

end ProfileSummary

/-- Group events by name and compute per-kernel statistics -/
def groupByKernel (events : List ProfileEvent) : List (String × List ProfileEvent) :=
  let grouped := events.foldl (fun acc e =>
    match acc.find? (fun (name, _) => name == e.eventName) with
    | some (name, evts) =>
        acc.filter (fun (n, _) => n != name) ++ [(name, e :: evts)]
    | none =>
        acc ++ [(e.eventName, [e])]
  ) []
  grouped

/-- Print a profiling report to the console -/
def printReport (events : List ProfileEvent) : IO Unit := do
  if events.isEmpty then
    IO.println "No profiling events recorded."
    return

  IO.println "═══════════════════════════════════════════════"
  IO.println "GPU Profiling Report"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Overall summary
  let summary := ProfileSummary.fromEvents events
  IO.println (toString summary)
  IO.println ""

  -- Per-kernel breakdown
  IO.println "Per-Kernel Breakdown:"
  IO.println "─────────────────────────────────────────────"
  let grouped := groupByKernel events
  for (kernelName, kernelEvents) in grouped do
    let kernelSummary := ProfileSummary.fromEvents kernelEvents
    IO.println s!"  {kernelName}:"
    IO.println s!"    Invocations: {kernelSummary.totalEvents}"
    IO.println s!"    Total time: {kernelSummary.totalDurationMs} ms"
    IO.println s!"    Avg time: {kernelSummary.avgDurationMs} ms"
    IO.println ""

end Hesper
