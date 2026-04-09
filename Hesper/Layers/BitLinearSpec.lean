import Hesper.Basic

/-!
# BitLinear CPU Spec

A pure CPU reference implementation of BitLinear forward (and VJP backward)
that exactly mirrors the math of the GPU kernels. This is the **golden spec**
against which every GPU kernel variant (shared-mem fallback, subgroupAdd,
subgroup matrix, ...) is checked for numerical equivalence.

The spec is intentionally small and obvious:

```
y[row, i] = scale * Σ_j  W[i, j] * x[row, j]        (W is ternary)
```

Weights are stored in i2_s packing — 2 bits per element, grouped in blocks
of 128 elements per 32 bytes. The element layout inside a 32-byte group is:

  elements [ 0.. 31] = bytes[0..31] >> 6 & 3
  elements [32.. 63] = bytes[0..31] >> 4 & 3
  elements [64.. 95] = bytes[0..31] >> 2 & 3
  elements [96..127] = bytes[0..31] >> 0 & 3

Decoding: `ternary = code - 1` so codes (0, 1, 2) → (-1, 0, +1).

Row `r` of shape `inDim` starts at packed byte offset
`r * inDim / 4`  (one byte = 4 packed elements).

This file has **zero GPU dependencies** so it can be called from pure tests,
from the `Verified AD` framework, and anywhere else we want a known-good
oracle for BitLinear.
-/

namespace Hesper.Layers.BitLinearSpec

/-- Decode a single ternary element from a packed i2_s byte array.
    `rowStartByte` is the byte offset where this output row starts
    (typically `outRow * inDim / 4`). `col` is the input-dim index
    within that row.

    Returns a `Float` in `{-1.0, 0.0, 1.0}`. -/
def decodeI2S (packed : ByteArray) (rowStartByte : Nat) (col : Nat) : Float :=
  -- Each group-of-128 spans 32 packed bytes.
  let group128 := col / 128
  let colInGroup := col % 128      -- 0..127
  let bytePos := colInGroup % 32   -- which of the 32 bytes
  let shiftIdx := colInGroup / 32  -- 0..3, where higher shiftIdx ↔ lower bits
  let byteOffset := rowStartByte + group128 * 32 + bytePos
  if byteOffset < packed.size then
    let b := packed.get! byteOffset
    -- shiftIdx = 0 → bits 7..6   (shift 6)
    -- shiftIdx = 1 → bits 5..4   (shift 4)
    -- shiftIdx = 2 → bits 3..2   (shift 2)
    -- shiftIdx = 3 → bits 1..0   (shift 0)
    let shift : UInt8 := (6 - UInt8.ofNat (shiftIdx * 2))
    let code := (b >>> shift) &&& 0x03
    (code.toNat.toFloat) - 1.0
  else
    0.0

/-- Compute BitLinear forward for a single row.
    `input` has length `inDim`; the output has length `outDim`.
    `scale * ⟨Wᵢ, x⟩` for each output row `i`. -/
def forwardRow (packed : ByteArray) (scale : Float) (inDim outDim : Nat)
    (input : Array Float) : Array Float := Id.run do
  -- inDim must be a multiple of 4 (one packed byte = 4 elements).
  let bytesPerRow := inDim / 4
  let mut out := Array.replicate outDim 0.0
  for i in [:outDim] do
    let rowStartByte := i * bytesPerRow
    let mut acc : Float := 0.0
    for j in [:inDim] do
      let w := decodeI2S packed rowStartByte j
      acc := acc + w * input.getD j 0.0
    out := out.set! i (scale * acc)
  pure out

/-- Compute BitLinear forward for a batch of rows.
    `input` is the concatenation of `numRows` rows, each of length `inDim`.
    Returns the concatenation of `numRows` output rows, each of length `outDim`. -/
def forward (packed : ByteArray) (scale : Float) (inDim outDim : Nat)
    (numRows : Nat) (input : Array Float) : Array Float := Id.run do
  let mut out := Array.replicate (numRows * outDim) 0.0
  for r in [:numRows] do
    let rowInput : Array Float :=
      (Array.range inDim).map (fun j => input.getD (r * inDim + j) 0.0)
    let rowOut := forwardRow packed scale inDim outDim rowInput
    for i in [:outDim] do
      out := out.set! (r * outDim + i) (rowOut.getD i 0.0)
  pure out

/-- VJP (backward) for BitLinear: `dx = scale * Wᵀ · dy`.
    Uses the same weight decoding as `forward`. -/
def backward (packed : ByteArray) (scale : Float) (inDim outDim : Nat)
    (numRows : Nat) (_input : Array Float) (gradOut : Array Float) : Array Float := Id.run do
  let bytesPerRow := inDim / 4
  let mut dx := Array.replicate (numRows * inDim) 0.0
  for r in [:numRows] do
    for i in [:outDim] do
      let rowStartByte := i * bytesPerRow
      let dy_i := gradOut.getD (r * outDim + i) 0.0
      if dy_i != 0.0 then
        let coeff := scale * dy_i
        for j in [:inDim] do
          let w := decodeI2S packed rowStartByte j
          if w != 0.0 then
            let cur := dx.getD (r * inDim + j) 0.0
            dx := dx.set! (r * inDim + j) (cur + coeff * w)
  pure dx

/-! ## Utilities for constructing test inputs -/

/-- Build a packed i2_s byte array from an `Array Int` of ternary values
    (`-1`, `0`, or `1`), laid out in row-major `[outDim × inDim]` order.

    This is the inverse of `decodeI2S` and is used by tests to construct
    deterministic weight matrices that match what the GPU kernel would see
    if given the same byte array. `inDim` must be a multiple of 128. -/
def packI2S (ternary : Array Int) (inDim outDim : Nat) : ByteArray := Id.run do
  let bytesPerRow := inDim / 4
  let totalBytes := outDim * bytesPerRow
  let mut bytes : ByteArray := ByteArray.mk (Array.replicate totalBytes 0)
  for i in [:outDim] do
    let rowStartByte := i * bytesPerRow
    for j in [:inDim] do
      let t := ternary.getD (i * inDim + j) 0
      let code : UInt8 :=
        if t == (-1 : Int) then 0
        else if t == 0 then 1
        else 2
      let group128 := j / 128
      let colInGroup := j % 128
      let bytePos := colInGroup % 32
      let shiftIdx := colInGroup / 32
      let byteOffset := rowStartByte + group128 * 32 + bytePos
      let shift : UInt8 := (6 - UInt8.ofNat (shiftIdx * 2))
      let old := bytes.get! byteOffset
      let mask : UInt8 := (0x03 : UInt8) <<< shift
      let cleared : UInt8 := old &&& ((0xFF : UInt8) ^^^ mask)
      let merged : UInt8 := cleared ||| (code <<< shift)
      bytes := bytes.set! byteOffset merged
  pure bytes

end Hesper.Layers.BitLinearSpec
