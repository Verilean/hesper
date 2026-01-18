import Hesper.Profile

namespace Hesper.Profile.Trace

/-! Chrome Tracing format export for GPU profiling.

Export GPU profiling data in Chrome Tracing JSON format for visualization.

Usage Pattern:
1. Run profiled kernels and collect ProfileEvent list
2. Convert to Chrome Tracing JSON
3. Write to file
4. Open in chrome://tracing

Chrome Tracing Format:
{
  "traceEvents": [
    {
      "name": "kernel_name",
      "cat": "gpu",
      "ph": "X",
      "ts": 1234567.890,
      "dur": 123.456,
      "pid": 1,
      "tid": 1
    }
  ],
  "displayTimeUnit": "ns"
}

Where:
- name: Event name
- cat: Category (always "gpu")
- ph: Phase ("X" for complete events with duration)
- ts: Timestamp in microseconds (start time)
- dur: Duration in microseconds
- pid: Process ID (always 1)
- tid: Thread ID (always 1 for sequential GPU operations)
-/

/-- A single event in Chrome Tracing format -/
structure ChromeTraceEvent where
  eventName : String        -- Event name
  eventCategory : String    -- Category (e.g., "gpu")
  eventPhase : String       -- Phase type ("X" for complete events)
  eventTimestamp : Float    -- Start timestamp in microseconds
  eventDuration : Float     -- Duration in microseconds
  eventPid : Nat            -- Process ID
  eventTid : Nat            -- Thread ID
  deriving Repr

namespace ChromeTraceEvent

/-- Convert to JSON object string -/
def toJSON (e : ChromeTraceEvent) : String :=
  "{" ++
  s!"\"name\": \"{e.eventName}\", " ++
  s!"\"cat\": \"{e.eventCategory}\", " ++
  s!"\"ph\": \"{e.eventPhase}\", " ++
  s!"\"ts\": {e.eventTimestamp}, " ++
  s!"\"dur\": {e.eventDuration}, " ++
  s!"\"pid\": {e.eventPid}, " ++
  s!"\"tid\": {e.eventTid}" ++
  "}"

end ChromeTraceEvent

/-- Chrome Tracing document containing multiple events -/
structure ChromeTrace where
  traceEvents : List ChromeTraceEvent
  displayTimeUnit : String
  deriving Repr

namespace ChromeTrace

/-- Convert to JSON string -/
def toJSON (t : ChromeTrace) : String :=
  let eventsJSON := t.traceEvents.map ChromeTraceEvent.toJSON
  let eventsStr := String.intercalate ",\n    " eventsJSON
  "{" ++
  "\"traceEvents\": [\n    " ++
  eventsStr ++
  "\n  ],\n" ++
  s!"  \"displayTimeUnit\": \"{t.displayTimeUnit}\"\n" ++
  "}"

end ChromeTrace

/-- Convert a ProfileEvent to ChromeTraceEvent.

GPU timestamps are in nanoseconds, Chrome Tracing expects microseconds -/
def profileEventToTraceEvent (event : ProfileEvent) : ChromeTraceEvent :=
  { eventName := event.eventName
    eventCategory := "gpu"
    eventPhase := "X"  -- Complete event (has duration)
    eventTimestamp := event.startTime.toNat.toFloat / 1000.0  -- ns -> μs
    eventDuration := (event.endTime.toNat - event.startTime.toNat).toFloat / 1000.0  -- ns -> μs
    eventPid := 1
    eventTid := 1 }

/-- Convert list of ProfileEvents to ChromeTrace.

Events are sorted by start time for better visualization -/
def profileEventsToTrace (events : List ProfileEvent) : ChromeTrace :=
  let sortedEvents : List ProfileEvent := events.mergeSort (fun a b => a.startTime < b.startTime)
  let traceEvents : List ChromeTraceEvent := List.map profileEventToTraceEvent sortedEvents
  { traceEvents := traceEvents
    displayTimeUnit := "ns" }

/-- Generate Chrome Tracing JSON from ProfileEvents -/
def chromeTraceJSON (events : List ProfileEvent) : String :=
  (profileEventsToTrace events).toJSON

/-- Write Chrome Tracing JSON to file.

The resulting file can be loaded in chrome://tracing for visualization.

Example:
  events <- runProfiledKernels
  writeChromeTrace "gpu_profile.json" events
  -- Then open chrome://tracing and load gpu_profile.json
-/
def writeChromeTrace (filepath : String) (events : List ProfileEvent) : IO Unit := do
  let json := chromeTraceJSON events
  IO.FS.writeFile filepath json
  IO.println s!"Chrome trace written to {filepath}"
  IO.println "Open chrome://tracing and load the file to visualize"

end Hesper.Profile.Trace
