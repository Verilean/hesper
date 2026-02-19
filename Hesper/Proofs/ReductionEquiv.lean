/-!
# Reduction Equivalence Proof

Proves that tree reduction (shared-memory fallback) produces the same result
as linear summation (subgroupAdd) for associative integer addition.

## Key Theorem

`treeReduce_eq_rangeSum`: For any k and function f, tree reduction of 2^k
elements equals the linear sum of those elements.

## Corollary

`treeReduce32_eq_sum`: Specialization for 32 elements (5 tree reduction steps),
matching the GPU workgroup size used in the shared-memory fallback kernel.

## Note on Floating Point

IEEE 754 floating-point addition is NOT associative, so this proof applies
to the abstract algebraic operation (Int). The LSpec GPU tests in
`Tests/SubgroupFallbackTests.lean` provide empirical validation that the
floating-point results match between both kernel implementations.
-/

namespace Hesper.Proofs.ReductionEquiv

/-- Linear sum: f(0) + f(1) + ... + f(n-1).
    Models `subgroupAdd` which sums all lane values. -/
def rangeSum : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => rangeSum n f + f n

/-- Tree reduction for 2^k elements.
    Models the shared-memory reduction algorithm used in the fallback kernel:

    ```
    shared[tid] = value
    workgroupBarrier()
    for stride in [2^(k-1), ..., 2, 1]:
      if tid < stride:
        shared[tid] += shared[tid + stride]
      workgroupBarrier()
    result = shared[0]
    ```

    Each step combines elements at distance `stride`, halving the active range.
    After k steps, `shared[0]` contains the sum of all 2^k elements. -/
def treeReduce : Nat → (Nat → Int) → Int
  | 0, f => f 0
  | k + 1, f => treeReduce k (fun i => f i + f (i + 2 ^ k))

/-! ## Lemmas -/

/-- `rangeSum` distributes over pointwise addition of functions. -/
theorem rangeSum_add (n : Nat) (f g : Nat → Int) :
    rangeSum n (fun i => f i + g i) = rangeSum n f + rangeSum n g := by
  induction n with
  | zero => simp [rangeSum]
  | succ n ih =>
    show rangeSum n (fun i => f i + g i) + (f n + g n) =
         (rangeSum n f + f n) + (rangeSum n g + g n)
    rw [ih]
    omega

/-- `rangeSum` can be split at a boundary:
    sum over [0, a+b) = sum over [0, a) + sum over [a, a+b). -/
theorem rangeSum_append (a b : Nat) (f : Nat → Int) :
    rangeSum (a + b) f = rangeSum a f + rangeSum b (fun i => f (i + a)) := by
  induction b with
  | zero => simp [rangeSum]
  | succ b ih =>
    show rangeSum (a + b) f + f (a + b) =
         rangeSum a f + (rangeSum b (fun i => f (i + a)) + f (b + a))
    rw [ih]
    have hab : a + b = b + a := by omega
    rw [hab]
    omega

/-- 2^(k+1) = 2^k + 2^k -/
theorem two_pow_succ (k : Nat) : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by
  rw [Nat.pow_succ]
  omega

/-! ## Main Theorem -/

/-- **Tree reduction equals linear summation** for 2^k elements.

    This is the core correctness theorem: the shared-memory tree reduction
    algorithm computes exactly the same sum as a linear scan (subgroupAdd).

    The proof proceeds by induction on k:
    - Base (k=0): both are f(0)
    - Step (k→k+1): one tree reduction step pairs elements at distance 2^k,
      creating an array of sums. By IH, tree-reducing this half-sized array
      equals its linear sum. By linearity of rangeSum, this equals the sum of
      both halves, which equals the full linear sum. -/
theorem treeReduce_eq_rangeSum (k : Nat) (f : Nat → Int) :
    treeReduce k f = rangeSum (2 ^ k) f := by
  induction k generalizing f with
  | zero => simp [treeReduce, rangeSum]
  | succ k ih =>
    simp only [treeReduce]
    simp only [ih]
    simp only [rangeSum_add]
    rw [two_pow_succ k, rangeSum_append]

/-- Specialization for 32 elements (workgroup size = 32, k = 5).
    This is the exact configuration used in the GPU fallback kernel. -/
theorem treeReduce32_eq_sum (f : Nat → Int) :
    treeReduce 5 f = rangeSum 32 f :=
  treeReduce_eq_rangeSum 5 f

end Hesper.Proofs.ReductionEquiv
