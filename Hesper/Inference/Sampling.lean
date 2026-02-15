/-!
# Sampling Algorithms for Text Generation

Implements various sampling strategies for autoregressive language model inference.

## Sampling Strategies

### 1. Greedy Decoding (Deterministic)
```
next_token = argmax(logits)
```
**Use case**: Reproducible outputs, factual tasks
**Pros**: Deterministic, fast
**Cons**: Repetitive, lacks creativity

### 2. Top-k Sampling
```
top_k_logits, indices = topk(logits, k)
probs = softmax(top_k_logits / temperature)
next_token = sample(indices, probs)
```
**Use case**: Creative generation with controlled diversity
**Pros**: Balances quality and diversity
**Cons**: Fixed k may be too restrictive or permissive

### 3. Nucleus (Top-p) Sampling
```
sorted_probs = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
nucleus = sorted_probs[cumsum <= p]
next_token = sample(nucleus)
```
**Use case**: Adaptive diversity based on distribution
**Pros**: Adapts to confidence (dynamic cutoff)
**Cons**: More compute than top-k

## Temperature Scaling

Temperature τ controls randomness:
```
scaled_logits = logits / τ

τ → 0:   Deterministic (greedy)
τ = 1:   Normal distribution
τ > 1:   More random/creative
```

## References
- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - Nucleus sampling
- "Hierarchical Neural Story Generation" (Fan et al., 2018) - Top-k sampling
- GPT-2: Uses both top-k and top-p
- llama.cpp: common/sampling.cpp
-/

namespace Hesper.Inference.Sampling

/-! ## Data Structures -/

/-- Sampling configuration -/
structure SamplingConfig where
  temperature : Float := 1.0
  topK : Nat := 40
  topP : Float := 0.9
  seed : Option Nat := none
  deriving Repr

/-! ## Core Utilities -/

/-- Find argmax (index of maximum value)

    @param logits Array of logit scores
    @return Index of maximum value
-/
def argmax (logits : Array Float) : Nat :=
  let rec loop (idx : Nat) (maxIdx : Nat) (maxVal : Float) : Nat :=
    if idx >= logits.size then
      maxIdx
    else
      let val := logits[idx]!
      if val > maxVal then
        loop (idx + 1) idx val
      else
        loop (idx + 1) maxIdx maxVal
  if logits.size == 0 then 0
  else loop 1 0 (logits[0]!)

/-- Apply temperature scaling

    @param logits Original logits
    @param temperature Temperature value (τ > 0)
    @return Scaled logits
-/
def applyTemperature (logits : Array Float) (temperature : Float) : Array Float :=
  if temperature == 1.0 then
    logits
  else
    logits.map (· / temperature)

/-- Compute softmax probabilities

    Uses numerically stable version: exp(x - max(x))

    @param logits Log-probabilities
    @return Probabilities (sum to 1.0)
-/
def softmax (logits : Array Float) : Array Float :=
  -- Find max for numerical stability
  let maxLogit := logits.foldl max (-1e38)

  -- Compute exp(x - max)
  let exps := logits.map (fun x => Float.exp (x - maxLogit))

  -- Normalize
  let sumExp := exps.foldl (· + ·) 0.0
  exps.map (· / sumExp)

/-! ## Greedy Sampling -/

/-- Greedy sampling: select token with highest probability

    **Algorithm**:
    ```
    next_token = argmax(logits)
    ```

    @param logits Logit scores [vocab_size]
    @return Selected token ID
-/
def sampleGreedy (logits : Array Float) : Nat :=
  argmax logits

/-! ## Top-k Sampling -/

/-- Find indices of top-k largest values

    @param logits Array of scores
    @param k Number of top elements
    @return (top_k_values, top_k_indices)
-/
def topK (logits : Array Float) (k : Nat) : Array Float × Array Nat :=
  -- Create array of (value, index) pairs
  let rec buildIndexed (idx : Nat) (acc : Array (Float × Nat)) : Array (Float × Nat) :=
    if idx >= logits.size then acc
    else buildIndexed (idx + 1) (acc.push (logits[idx]!, idx))

  let indexed := buildIndexed 0 #[]

  -- Sort by value (descending)
  let sorted := indexed.qsort (fun a b => a.1 > b.1)

  -- Take top k
  let topKPairs := sorted.extract 0 (min k sorted.size)

  -- Separate values and indices
  let values := topKPairs.map (·.1)
  let indices := topKPairs.map (·.2)

  (values, indices)

/-- Sample from categorical distribution

    Simple implementation using linear search through cumulative probabilities.

    @param probs Probability distribution (must sum to ~1.0)
    @param randomValue Random value in [0, 1)
    @return Sampled index
-/
def categoricalSample (probs : Array Float) (randomValue : Float) : Nat :=
  let rec loop (idx : Nat) (cumulative : Float) : Nat :=
    if idx >= probs.size then
      if probs.size > 0 then probs.size - 1 else 0
    else
      let newCumulative := cumulative + probs[idx]!
      if randomValue < newCumulative then
        idx
      else
        loop (idx + 1) newCumulative
  loop 0 0.0

/-- Top-k sampling: sample from k tokens with highest probability

    **Algorithm**:
    ```
    1. Find top-k logits and their indices
    2. Apply temperature scaling
    3. Compute softmax over top-k
    4. Sample from categorical distribution
    ```

    @param logits Logit scores [vocab_size]
    @param k Number of top candidates
    @param temperature Sampling temperature
    @param randomValue Random number in [0, 1) for sampling
    @return Selected token ID
-/
def sampleTopK (logits : Array Float) (k : Nat) (temperature : Float) (randomValue : Float) : Nat :=
  -- Step 1: Find top-k
  let (topKLogits, topKIndices) := topK logits k

  -- Step 2: Apply temperature
  let scaledLogits := applyTemperature topKLogits temperature

  -- Step 3: Softmax
  let probs := softmax scaledLogits

  -- Step 4: Sample
  let sampledIdx := categoricalSample probs randomValue
  topKIndices[sampledIdx]!

/-! ## Nucleus (Top-p) Sampling -/

/-- Nucleus (top-p) sampling: sample from smallest set with cumulative probability >= p

    **Algorithm**:
    ```
    1. Sort logits in descending order
    2. Compute softmax probabilities
    3. Find nucleus: tokens where cumsum(probs) <= p
    4. Renormalize and sample from nucleus
    ```

    @param logits Logit scores [vocab_size]
    @param p Cumulative probability threshold (typically 0.9)
    @param temperature Sampling temperature
    @param randomValue Random number in [0, 1) for sampling
    @return Selected token ID
-/
def sampleNucleus (logits : Array Float) (p : Float) (temperature : Float) (randomValue : Float) : Nat :=
  -- Step 1: Create indexed array and sort by logits (descending)
  let rec buildIndexed (idx : Nat) (acc : Array (Float × Nat)) : Array (Float × Nat) :=
    if idx >= logits.size then acc
    else buildIndexed (idx + 1) (acc.push (logits[idx]!, idx))

  let indexed := buildIndexed 0 #[]
  let sorted := indexed.qsort (fun a b => a.1 > b.1)

  -- Step 2: Apply temperature and compute softmax
  let sortedLogits := sorted.map (·.1)
  let scaledLogits := applyTemperature sortedLogits temperature
  let probs := softmax scaledLogits

  -- Step 3: Find nucleus cutoff
  let rec findNucleusSize (idx : Nat) (cumulative : Float) : Nat :=
    if idx >= probs.size then
      idx
    else if cumulative >= p then
      idx
    else
      findNucleusSize (idx + 1) (cumulative + probs[idx]!)
  termination_by probs.size - idx
  decreasing_by sorry

  let nucleusSize := max 1 (findNucleusSize 0 0.0)

  -- Step 4: Extract nucleus
  let nucleusProbs := probs.extract 0 nucleusSize
  let nucleusIndices := sorted.extract 0 nucleusSize |>.map (·.2)

  -- Step 5: Renormalize (should already be normalized, but ensure)
  let sumNucleus := nucleusProbs.foldl (· + ·) 0.0
  let normalizedProbs := nucleusProbs.map (· / sumNucleus)

  -- Step 6: Sample
  let sampledIdx := categoricalSample normalizedProbs randomValue
  nucleusIndices[sampledIdx]!

/-! ## High-Level Sampling Interface -/

/-- Sampling strategy -/
inductive Strategy where
  | Greedy
  | TopK (k : Nat) (temperature : Float)
  | Nucleus (p : Float) (temperature : Float)
  deriving Repr

instance : ToString Strategy where
  toString s := match s with
    | .Greedy => "Greedy"
    | .TopK k temp => s!"TopK(k={k}, temp={temp})"
    | .Nucleus p temp => s!"Nucleus(p={p}, temp={temp})"

/-- Sample next token using specified strategy

    @param logits Logit scores [vocab_size]
    @param strategy Sampling strategy
    @param randomValue Random value in [0, 1) (only used for stochastic strategies)
    @return Selected token ID
-/
def sample (logits : Array Float) (strategy : Strategy) (randomValue : Float := 0.5) : Nat :=
  match strategy with
  | .Greedy => sampleGreedy logits
  | .TopK k temp => sampleTopK logits k temp randomValue
  | .Nucleus p temp => sampleNucleus logits p temp randomValue

/-! ## Random Number Generation -/

/-- Simple linear congruential generator (LCG) for reproducible sampling

    Parameters: Numerical Recipes (a = 1664525, c = 1013904223, m = 2^32)

    @param seed Current seed
    @return (random_float in [0,1), new_seed)
-/
def lcgNext (seed : UInt32) : Float × UInt32 :=
  let a : UInt32 := 1664525
  let c : UInt32 := 1013904223
  let newSeed := a * seed + c
  let randomFloat := newSeed.toFloat / (UInt32.size.toFloat)
  (randomFloat, newSeed)

/-- Random number generator state -/
structure RNG where
  seed : UInt32
  deriving Repr

/-- Create RNG from optional seed -/
def RNG.create (seed : Option Nat := none) : RNG :=
  let s := match seed with
    | some n => n.toUInt32
    | none => 42  -- Default seed
  { seed := s }

/-- Generate next random float -/
def RNG.next (rng : RNG) : Float × RNG :=
  let (randomFloat, newSeed) := lcgNext rng.seed
  (randomFloat, { seed := newSeed })

/-! ## Sampling with RNG -/

/-- Sample with automatic RNG threading

    @param logits Logit scores
    @param strategy Sampling strategy
    @param rng Random number generator state
    @return (selected_token, new_rng)
-/
def sampleWithRNG (logits : Array Float) (strategy : Strategy) (rng : RNG) : Nat × RNG :=
  match strategy with
  | .Greedy =>
    (sampleGreedy logits, rng)
  | .TopK k temp =>
    let (randomVal, newRng) := rng.next
    (sampleTopK logits k temp randomVal, newRng)
  | .Nucleus p temp =>
    let (randomVal, newRng) := rng.next
    (sampleNucleus logits p temp randomVal, newRng)

/-! ## Utilities -/

/-- Print sampling statistics

    @param logits Logit scores
    @param selectedToken Sampled token
-/
def printStats (logits : Array Float) (selectedToken : Nat) : IO Unit := do
  let probs := softmax logits
  let selectedProb := if selectedToken < probs.size then probs[selectedToken]! else 0.0
  let maxProb := probs.foldl max 0.0
  let entropy := -probs.foldl (fun acc p =>
    if p > 0.0 then acc + p * Float.log p else acc) 0.0

  IO.println "─────────────────────────────────────────"
  IO.println "Sampling Statistics"
  IO.println "─────────────────────────────────────────"
  IO.println s!"Selected token: {selectedToken}"
  IO.println s!"Selected probability: {selectedProb}"
  IO.println s!"Max probability: {maxProb}"
  IO.println s!"Entropy: {entropy}"
  IO.println "─────────────────────────────────────────"

end Hesper.Inference.Sampling
