import Hesper.Async

/-!
# Async GPU Operations Demo

Demonstrates asynchronous GPU compute using Lean 4's Task system.
-/

namespace Examples.AsyncDemo

open Hesper.Async

/-- Demo 1: Basic async task -/
def demo1_basic : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 1: Basic Async Task"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Launching async shader execution..."
  let task ← asyncShaderExec "shader_code" (256, 1, 1)

  IO.println "Task launched! Doing other work..."
  IO.sleep 5

  IO.println "Waiting for task to complete..."
  let result ← IO.wait task

  IO.println s!"✓ Task completed in {result.executionTimeMs}ms"
  IO.println ""

/-- Demo 2: Parallel tasks -/
def demo2_parallel : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 2: Parallel GPU Tasks"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Launching 4 parallel shader tasks..."
  let tasks := #[
    asyncShaderExec "shader1" (128, 1, 1),
    asyncShaderExec "shader2" (128, 1, 1),
    asyncShaderExec "shader3" (128, 1, 1),
    asyncShaderExec "shader4" (128, 1, 1)
  ]

  let taskArray ← parallel tasks
  IO.println s!"✓ {taskArray.size} tasks launched"

  IO.println "Waiting for all tasks..."
  let results ← awaitAll taskArray

  IO.println s!"✓ All {results.size} tasks completed:"
  for i in [:results.size] do
    IO.println s!"  Task {i+1}: {results[i]!.executionTimeMs}ms"
  IO.println ""

/-- Demo 3: Pipeline -/
def demo3_pipeline : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 3: Async Pipeline"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Building async pipeline:"
  IO.println "  Stage 1: Shader execution"
  IO.println "  Stage 2: Buffer read"
  IO.println "  Stage 3: Transform result"

  let pipeline := AsyncPipeline.fromTask (asyncShaderExec "compute" (64, 64, 1))
    |>.andThen (fun _ => do
        IO.println "  [Stage 1 complete, starting Stage 2...]"
        -- Dummy buffer for demo
        let dummyBuffer : Hesper.WebGPU.Buffer := default
        asyncBufferRead dummyBuffer 1024
      )
    |>.map (fun data => data.size)

  IO.println "Executing pipeline..."
  let result ← pipeline.execute

  IO.println s!"✓ Pipeline completed in {result.executionTimeMs}ms"
  IO.println s!"  Result: {result.value} elements"
  IO.println ""

/-- Demo 4: Batched execution -/
def demo4_batch : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 4: Batched Execution (Concurrency Control)"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Create 10 operations
  let ops := Array.range 10 |>.map fun i =>
    asyncShaderExec s!"shader_{i}" (128, 1, 1)

  IO.println s!"Executing {ops.size} operations with max concurrency = 3"

  let config : BatchConfig Unit := {
    operations := ops
    maxConcurrency := 3
  }

  let results ← executeBatch config

  IO.println s!"✓ All {results.size} operations completed"
  let totalTime := results.foldl (fun acc r => acc + r.executionTimeMs) 0.0
  IO.println s!"  Total time: {totalTime}ms"
  IO.println s!"  Average: {totalTime / results.size.toFloat}ms per operation"
  IO.println ""

/-- Demo 5: Concurrent tasks -/
def demo5_concurrent : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 5: Concurrent Task Execution"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Launching 3 tasks concurrently..."

  let task1 ← IO.asTask do
    IO.sleep 15
    return "Task 1 complete"

  let task2 ← IO.asTask do
    IO.sleep 10
    return "Task 2 complete"

  let task3 ← IO.asTask do
    IO.sleep 20
    return "Task 3 complete"

  IO.println "✓ All 3 tasks launched"

  let result1 ← IO.wait task1
  let result2 ← IO.wait task2
  let result3 ← IO.wait task3

  IO.println s!"  {result1}"
  IO.println s!"  {result2}"
  IO.println s!"  {result3}"
  IO.println ""

/-- Demo 6: Map and transform -/
def demo6_transform : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 6: Async Transformations"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Launch compute task..."
  let dummyBuffer : Hesper.WebGPU.Buffer := default
  let task ← asyncBufferRead dummyBuffer 100

  IO.println "Transform result asynchronously..."
  let transformed := mapTask (fun data => data.size * 2) task

  let result ← IO.wait transformed
  IO.println s!"✓ Transformed result: {result.value}"
  IO.println s!"  Execution time: {result.executionTimeMs}ms"
  IO.println ""

/-- Demo 7: Full async compute -/
def demo7_full_compute : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 7: Full Async Compute Pipeline"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  let config : AsyncComputeConfig := {
    shader := "compute_shader_code"
    workgroups := (64, 64, 1)
    inputSizes := #[1024, 1024]
    outputSize := 1024
  }

  let inputs := #[
    Array.mk (List.replicate 1024 1.0),
    Array.mk (List.replicate 1024 2.0)
  ]

  IO.println "Launching async compute..."
  IO.println s!"  Workgroups: {config.workgroups}"
  IO.println s!"  Input sizes: {config.inputSizes}"
  IO.println s!"  Output size: {config.outputSize}"

  let task ← asyncCompute config inputs

  IO.println "Computing... (doing other work in parallel)"
  IO.sleep 10

  IO.println "Waiting for result..."
  let result ← IO.wait task

  IO.println s!"✓ Compute completed in {result.executionTimeMs}ms"
  IO.println s!"  Output size: {result.value.size} elements"
  IO.println s!"  First 5 values: {result.value.toSubarray 0 (min 5 result.value.size) |>.toArray}"
  IO.println ""

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Asynchronous GPU Operations Demo            ║"
  IO.println "║   Using Lean 4 Task System                    ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  demo1_basic
  demo2_parallel
  demo3_pipeline
  demo4_batch
  demo5_concurrent
  demo6_transform
  demo7_full_compute

  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   All async demos complete!                   ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "Summary:"
  IO.println "  ✓ Basic async tasks with IO.asTask"
  IO.println "  ✓ Parallel execution with task arrays"
  IO.println "  ✓ Pipeline composition with andThen/map"
  IO.println "  ✓ Batched execution with concurrency control"
  IO.println "  ✓ Concurrent task execution"
  IO.println "  ✓ Async transformations with mapTask"
  IO.println "  ✓ Full async compute pipeline"
  IO.println ""

end Examples.AsyncDemo

def main : IO Unit := Examples.AsyncDemo.main
