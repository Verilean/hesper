import Hesper

/-!
# Test Data Generation and Golden Outputs

Utilities for generating test data and golden outputs:
- Deterministic random number generation
- Matrix/vector test datasets
- Edge case datasets
- Golden output computation
-/

namespace Hesper.Tests.Integration.TestData

/-- Simple linear congruential generator for reproducible "random" numbers
    Parameters from Numerical Recipes -/
structure LCG where
  state : UInt64
  deriving Repr

def LCG.init (seed : UInt64 := 12345) : LCG :=
  { state := seed }

def LCG.next (lcg : LCG) : LCG × UInt64 :=
  let a : UInt64 := 1664525
  let c : UInt64 := 1013904223
  let newState := a * lcg.state + c
  ({ state := newState }, newState)

def LCG.nextFloat (lcg : LCG) : LCG × Float :=
  let (newLcg, val) := lcg.next
  let normalized := val.toFloat / UInt64.size.toFloat  -- 0.0 to 1.0
  (newLcg, normalized)

def LCG.nextFloatRange (lcg : LCG) (min max : Float) : LCG × Float :=
  let (newLcg, val) := lcg.nextFloat
  let scaled := min + val * (max - min)
  (newLcg, scaled)

/-- Generate array of random floats in range [min, max) -/
def randomFloatArray (size : Nat) (min max : Float) (seed : UInt64 := 12345) : Array Float :=
  let rec loop (n : Nat) (lcg : LCG) (acc : Array Float) : Array Float :=
    match n with
    | 0 => acc
    | n' + 1 =>
      let (newLcg, val) := lcg.nextFloatRange min max
      loop n' newLcg (acc.push val)
  loop size (LCG.init seed) #[]

/-- Generate incrementing array: [0.0, 1.0, 2.0, ..., (size-1).0] -/
def sequentialArray (size : Nat) : Array Float :=
  Array.range size |>.map Float.ofNat

/-- Generate constant array: [val, val, val, ...] -/
def constantArray (size : Nat) (val : Float) : Array Float :=
  let rec loop (n : Nat) (acc : Array Float) : Array Float :=
    match n with
    | 0 => acc
    | n' + 1 => loop n' (acc.push val)
  loop size #[]

/-- Generate identity matrix (flattened, row-major) -/
def identityMatrix (size : Nat) : Array Float :=
  let rec loop (i j : Nat) (acc : Array Float) : Array Float :=
    if i >= size then acc
    else if j >= size then loop (i + 1) 0 acc
    else
      let val := if i == j then 1.0 else 0.0
      loop i (j + 1) (acc.push val)
  loop 0 0 #[]

/-- Generate zero matrix -/
def zeroMatrix (rows cols : Nat) : Array Float :=
  constantArray (rows * cols) 0.0

/-- Edge case datasets -/
def edgeCaseArrays : List (String × Array Float) := [
  ("zeros", #[0.0, 0.0, 0.0, 0.0]),
  ("ones", #[1.0, 1.0, 1.0, 1.0]),
  ("negative", #[-1.0, -2.0, -3.0, -4.0]),
  ("mixed", #[-1.0, 0.0, 1.0, 2.0]),
  ("small", #[1e-10, 1e-9, 1e-8, 1e-7]),
  ("large", #[1e7, 1e8, 1e9, 1e10])
]

/-- Golden output: Increment by 1 -/
def goldenIncrement (input : Array Float) : Array Float :=
  input.map (· + 1.0)

/-- Golden output: Vector addition -/
def goldenVectorAdd (a b : Array Float) : Array Float :=
  a.zipWith (· + ·) b

/-- Golden output: Matrix-vector multiplication (row-major)
    mat: rows×cols matrix (flattened)
    vec: cols-element vector
    result: rows-element vector -/
def goldenMatVec (mat : Array Float) (vec : Array Float) (rows cols : Nat) : Array Float :=
  let rec computeRow (rowIdx : Nat) (acc : Array Float) : Array Float :=
    if rowIdx >= rows then acc
    else
      let rowStart := rowIdx * cols
      let dotProduct := (Array.range cols).foldl (init := 0.0) fun sum colIdx =>
        sum + mat[rowStart + colIdx]! * vec[colIdx]!
      computeRow (rowIdx + 1) (acc.push dotProduct)
  computeRow 0 #[]

/-- Golden output: Small matrix multiplication (4x4 × 4x4)
    A, B: 4x4 matrices (row-major)
    result: 4x4 matrix -/
def goldenMatMul4x4 (a b : Array Float) : Array Float :=
  let n := 4
  let rec compute (i j : Nat) (acc : Array Float) : Array Float :=
    if i >= n then acc
    else if j >= n then compute (i + 1) 0 acc
    else
      let dotProduct := (Array.range n).foldl (init := 0.0) fun sum k =>
        sum + a[i * n + k]! * b[k * n + j]!
      compute i (j + 1) (acc.push dotProduct)
  compute 0 0 #[]

/-- Golden output: Reduction sum -/
def goldenSum (input : Array Float) : Float :=
  input.foldl (· + ·) 0.0

/-- Golden output: Prefix sum (inclusive scan) -/
def goldenPrefixSum (input : Array Float) : Array Float :=
  let (result, _) := input.foldl (init := (#[], 0.0)) fun (acc, sum) x =>
    let newSum := sum + x
    (acc.push newSum, newSum)
  result

/-- Test matrices for MatMul -/
def testMatrix4x4_A : Array Float :=
  #[1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0]

def testMatrix4x4_B : Array Float :=
  #[2.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 0.0,
    0.0, 0.0, 0.0, 2.0]

/-- Expected result for testMatrix4x4_A × testMatrix4x4_B
    (A × 2I should be 2A) -/
def testMatrix4x4_Result : Array Float :=
  #[2.0, 4.0, 6.0, 8.0,
    10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 22.0, 24.0,
    26.0, 28.0, 30.0, 32.0]

end Hesper.Tests.Integration.TestData
