/-!
# Learning Rate Scheduler

Standard learning rate schedules for training.
Pure CPU computation — no GPU kernels needed.

## Schedules

- **Linear Warmup**: lr ramps from 0 to baseLR over warmupSteps
- **Cosine Decay**: lr decays from baseLR to minLR following cosine curve
- **Constant**: fixed lr (for debugging)

## Standard Usage (matches HuggingFace Trainer defaults)

```
warmupSteps = totalSteps * 0.1   (10% warmup)
baseLR = 2e-4                     (standard for LoRA)
minLR = 0.0
```
-/

namespace Hesper.Training.LRScheduler

inductive ScheduleType where
  | constant
  | linearWarmupCosineDecay
  | linearWarmupLinearDecay
  deriving Repr

structure Config where
  baseLR : Float := 2e-4
  warmupSteps : Nat := 0
  totalSteps : Nat := 1000
  minLR : Float := 0.0
  scheduleType : ScheduleType := .linearWarmupCosineDecay
  deriving Repr

/-- Compute learning rate at given step -/
def getLR (config : Config) (step : Nat) : Float :=
  match config.scheduleType with
  | .constant => config.baseLR
  | .linearWarmupCosineDecay =>
    if step < config.warmupSteps then
      -- Linear warmup: lr = baseLR * step / warmupSteps
      if config.warmupSteps == 0 then config.baseLR
      else config.baseLR * step.toFloat / config.warmupSteps.toFloat
    else
      -- Cosine decay
      let decaySteps := config.totalSteps - config.warmupSteps
      if decaySteps == 0 then config.baseLR
      else
        let progress := (step - config.warmupSteps).toFloat / decaySteps.toFloat
        let progress := if progress > 1.0 then 1.0 else progress
        let cosineDecay := 0.5 * (1.0 + Float.cos (progress * 3.14159265358979323846))
        config.minLR + (config.baseLR - config.minLR) * cosineDecay
  | .linearWarmupLinearDecay =>
    if step < config.warmupSteps then
      if config.warmupSteps == 0 then config.baseLR
      else config.baseLR * step.toFloat / config.warmupSteps.toFloat
    else
      let decaySteps := config.totalSteps - config.warmupSteps
      if decaySteps == 0 then config.baseLR
      else
        let progress := (step - config.warmupSteps).toFloat / decaySteps.toFloat
        let progress := if progress > 1.0 then 1.0 else progress
        config.minLR + (config.baseLR - config.minLR) * (1.0 - progress)

/-- Create scheduler from training parameters -/
def create (baseLR : Float) (numExamples epochs : Nat) (warmupRatio : Float := 0.1) : Config :=
  let totalSteps := numExamples * epochs
  let warmupSteps := (totalSteps.toFloat * warmupRatio).toUInt64.toNat
  { baseLR, warmupSteps, totalSteps, scheduleType := .linearWarmupCosineDecay }

end Hesper.Training.LRScheduler
