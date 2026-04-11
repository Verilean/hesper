import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.TTT.Kernels

set_option maxRecDepth 2048

/-!
# CUDA Kernel Tests — reusing existing WebGPU-validated kernels

Runs `vecAddKernel` and `matVecKernel` from `Hesper.TTT.Kernels`
(already validated on WebGPU) through the PTX DSL → JIT → GPU pipeline.
-/

open Hesper.CUDA
open Hesper.CUDA.CodeGen
open Hesper.TTT.Kernels
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM

-- Float helpers (f64 ↔ f32 IEEE 754)
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def f32BitsToF64 (bits : UInt32) : Float :=
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def unpackFloat (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  f32BitsToF64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

/-- Naive matmul: one thread per output element, no tiling. -/
def matMulKernel (M N K : Nat) : ShaderM Unit := do
  let gid ← globalId
  let col := Exp.vec3X gid  -- x = column
  let row := Exp.vec3Y gid  -- y = row

  let _a ← declareInputBuffer "A" (.array (.scalar .f32) (M * K))
  let _b ← declareInputBuffer "B" (.array (.scalar .f32) (K * N))
  let _c ← declareOutputBuffer "C" (.array (.scalar .f32) (M * N))

  if_ (Exp.and (Exp.lt row (Exp.litU32 M)) (Exp.lt col (Exp.litU32 N))) (do
    let (accName, acc) ← varRef (.scalar .f32) (Exp.litF32 0.0)
    loop (Exp.litU32 0) (Exp.litU32 K) (Exp.litU32 1) fun k => do
      let aIdx := Exp.add (Exp.mul row (Exp.litU32 K)) k
      let aVal ← readBuffer (ty := .scalar .f32) (n := M * K) "A" aIdx
      let bIdx := Exp.add (Exp.mul k (Exp.litU32 N)) col
      let bVal ← readBuffer (ty := .scalar .f32) (n := K * N) "B" bIdx
      assign accName (Exp.add acc (Exp.mul aVal bVal))
    let cIdx := Exp.add (Exp.mul row (Exp.litU32 N)) col
    writeBuffer (ty := .scalar .f32) "C" cIdx acc
  ) (pure ())

/-- Tiled matmul with shared memory.
    Block = (TILE, TILE) threads. Each block computes a TILE×TILE output tile.
    Cooperative load: all threads in a block load A/B tiles into shared memory.
    Uses barrier synchronization between load and compute phases. -/
def tiledMatMulKernel (M N K TILE : Nat) : ShaderM Unit := do
  let gid ← globalId
  let lid ← localId
  let col := Exp.vec3X gid   -- global column
  let row := Exp.vec3Y gid   -- global row
  let tx := Exp.vec3X lid    -- local col within tile
  let ty := Exp.vec3Y lid    -- local row within tile

  let _a ← declareInputBuffer "A" (.array (.scalar .f32) (M * K))
  let _b ← declareInputBuffer "B" (.array (.scalar .f32) (K * N))
  let _c ← declareOutputBuffer "C" (.array (.scalar .f32) (M * N))

  -- Shared memory for A and B tiles
  sharedNamed "tileA" (.array (.scalar .f32) (TILE * TILE))
  sharedNamed "tileB" (.array (.scalar .f32) (TILE * TILE))

  let (accName, acc) ← varRef (.scalar .f32) (Exp.litF32 0.0)

  -- Loop over tiles along K dimension
  let numTiles := (K + TILE - 1) / TILE
  loop (Exp.litU32 0) (Exp.litU32 numTiles) (Exp.litU32 1) fun t => do
    let tileBase := Exp.mul t (Exp.litU32 TILE)

    -- Cooperative load: tileA[ty][tx] = A[row, tileBase + tx]
    let aCol := Exp.add tileBase tx
    let aInBounds := Exp.and (Exp.lt row (Exp.litU32 M)) (Exp.lt aCol (Exp.litU32 K))
    let aIdx := Exp.add (Exp.mul row (Exp.litU32 K)) aCol
    let aVal ← readBuffer (ty := .scalar .f32) (n := M * K) "A" aIdx
    let sharedIdx := Exp.add (Exp.mul ty (Exp.litU32 TILE)) tx
    writeWorkgroup (ty := .scalar .f32) "tileA" sharedIdx
      (Exp.select aInBounds aVal (Exp.litF32 0.0))

    -- Cooperative load: tileB[ty][tx] = B[tileBase + ty, col]
    let bRow := Exp.add tileBase ty
    let bInBounds := Exp.and (Exp.lt bRow (Exp.litU32 K)) (Exp.lt col (Exp.litU32 N))
    let bIdx := Exp.add (Exp.mul bRow (Exp.litU32 N)) col
    let bVal ← readBuffer (ty := .scalar .f32) (n := K * N) "B" bIdx
    writeWorkgroup (ty := .scalar .f32) "tileB" sharedIdx
      (Exp.select bInBounds bVal (Exp.litF32 0.0))

    -- Synchronize: all threads must finish loading before compute
    barrier

    -- Compute partial dot product from this tile
    loop (Exp.litU32 0) (Exp.litU32 TILE) (Exp.litU32 1) fun k => do
      let aShIdx := Exp.add (Exp.mul ty (Exp.litU32 TILE)) k
      let bShIdx := Exp.add (Exp.mul k (Exp.litU32 TILE)) tx
      let a ← readWorkgroup (ty := .scalar .f32) (n := TILE * TILE) "tileA" aShIdx
      let b ← readWorkgroup (ty := .scalar .f32) (n := TILE * TILE) "tileB" bShIdx
      assign accName (Exp.add acc (Exp.mul a b))

    -- Synchronize: ensure compute is done before next tile overwrites shared mem
    barrier

  -- Write result
  if_ (Exp.and (Exp.lt row (Exp.litU32 M)) (Exp.lt col (Exp.litU32 N))) (do
    let cIdx := Exp.add (Exp.mul row (Exp.litU32 N)) col
    writeBuffer (ty := .scalar .f32) "C" cIdx acc
  ) (pure ())

def main : IO Unit := do
  IO.println "═══ CUDA Kernel Tests (reusing WebGPU-validated kernels) ═══"
  let (_dev, _ctx) ← initCUDA

  -- ═══ Test 1: vecAddKernel ═══
  IO.println ""
  IO.println "Test 1: vecAddKernel (output = a + b)"
  IO.println "──────────────────────────────────────"
  let n := 16
  let ptx1 := generatePTX "vecAdd" {x := 256} (vecAddKernel n)
  IO.println s!"  PTX: {ptx1.length} chars"
  let mod1 ← cuModuleLoadData ptx1
  let func1 ← cuModuleGetFunction mod1 "vecAdd"
  IO.println "  JIT: OK"

  let aBuf ← createCUDABuffer (n * 4).toUSize
  let bBuf ← createCUDABuffer (n * 4).toUSize
  let outBuf ← createCUDABuffer (n * 4).toUSize

  -- a = [1..16], b = [100,200,...,1600]
  let aArr := Array.range n |>.map (fun i => Float.ofNat (i + 1))
  let bArr := Array.range n |>.map (fun i => Float.ofNat ((i + 1) * 100))
  writeCUDABuffer aBuf (packFloats aArr)
  writeCUDABuffer bBuf (packFloats bArr)

  cuLaunchKernel func1 1 1 1 n.toUInt32 1 1 0 #[aBuf.ptr, bBuf.ptr, outBuf.ptr]
  let res1 ← readCUDABufferFull outBuf

  let mut ok1 := true
  for i in [0, 1, 7, 15] do
    let got := unpackFloat res1 i
    let exp := Float.ofNat ((i + 1) + (i + 1) * 100)
    let pass := (got - exp).abs < 0.5
    IO.println s!"  out[{i}] = {got} (expect {exp}) {if pass then "✓" else "✗"}"
    if !pass then ok1 := false

  freeCUDABuffer aBuf; freeCUDABuffer bBuf; freeCUDABuffer outBuf
  IO.println s!"  → {if ok1 then "PASSED" else "FAILED"}"

  -- ═══ Test 2: matVecKernel ═══
  IO.println ""
  IO.println "Test 2: matVecKernel (output = W @ x)"
  IO.println "──────────────────────────────────────"
  let outDim := 4; let inDim := 4
  let ptx2 := generatePTX "matVec" {x := 256} (matVecKernel outDim inDim)
  IO.println s!"  PTX: {ptx2.length} chars"
  IO.println ptx2
  let mod2 ← cuModuleLoadData ptx2
  let func2 ← cuModuleGetFunction mod2 "matVec"
  IO.println "  JIT: OK"

  let wBuf ← createCUDABuffer (outDim * inDim * 4).toUSize
  let xBuf ← createCUDABuffer (inDim * 4).toUSize
  let yBuf ← createCUDABuffer (outDim * 4).toUSize

  -- W = identity, x = [1,2,3,4] → y = [1,2,3,4]
  let wArr : Array Float := #[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
  let xArr : Array Float := #[1,2,3,4]
  writeCUDABuffer wBuf (packFloats wArr)
  writeCUDABuffer xBuf (packFloats xArr)

  cuLaunchKernel func2 1 1 1 outDim.toUInt32 1 1 0 #[wBuf.ptr, xBuf.ptr, yBuf.ptr]
  let res2 ← readCUDABufferFull yBuf

  IO.println "  W = I, x = [1,2,3,4]:"
  let mut ok2 := true
  for i in List.range outDim do
    let got := unpackFloat res2 i
    let exp := xArr.getD i 0.0
    let pass := (got - exp).abs < 0.01
    IO.println s!"    y[{i}] = {got} (expect {exp}) {if pass then "✓" else "✗"}"
    if !pass then ok2 := false

  -- W = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], x = [1,1,1,1] → y = [10,26,42,58]
  let wArr2 : Array Float := #[1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]
  let xArr2 : Array Float := #[1,1,1,1]
  let expected2 : Array Float := #[10, 26, 42, 58]
  writeCUDABuffer wBuf (packFloats wArr2)
  writeCUDABuffer xBuf (packFloats xArr2)

  cuLaunchKernel func2 1 1 1 outDim.toUInt32 1 1 0 #[wBuf.ptr, xBuf.ptr, yBuf.ptr]
  let res2b ← readCUDABufferFull yBuf

  IO.println "  W = [[1..4],[5..8],[9..12],[13..16]], x = [1,1,1,1]:"
  for i in List.range outDim do
    let got := unpackFloat res2b i
    let exp := expected2.getD i 0.0
    let pass := (got - exp).abs < 0.01
    IO.println s!"    y[{i}] = {got} (expect {exp}) {if pass then "✓" else "✗"}"
    if !pass then ok2 := false

  freeCUDABuffer wBuf; freeCUDABuffer xBuf; freeCUDABuffer yBuf
  IO.println s!"  → {if ok2 then "PASSED" else "FAILED"}"

  -- ═══ Test 3: matMulKernel ═══
  IO.println ""
  IO.println "Test 3: matMulKernel (C = A × B)"
  IO.println "─────────────────────────────────"
  let mM := 4; let mN := 4; let mK := 4
  let ptx3 := generatePTX "matMul" {x := mN, y := mM} (matMulKernel mM mN mK)
  IO.println s!"  PTX: {ptx3.length} chars"
  let mod3 ← cuModuleLoadData ptx3
  let func3 ← cuModuleGetFunction mod3 "matMul"
  IO.println "  JIT: OK"

  let maBuf ← createCUDABuffer (mM * mK * 4).toUSize
  let mbBuf ← createCUDABuffer (mK * mN * 4).toUSize
  let mcBuf ← createCUDABuffer (mM * mN * 4).toUSize

  -- A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
  -- B = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity)
  -- C = A
  let maArr : Array Float := #[1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]
  let mbArr : Array Float := #[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
  writeCUDABuffer maBuf (packFloats maArr)
  writeCUDABuffer mbBuf (packFloats mbArr)

  -- Launch: grid=(1,1), block=(N,M)
  cuLaunchKernel func3 1 1 1 mN.toUInt32 mM.toUInt32 1 0 #[maBuf.ptr, mbBuf.ptr, mcBuf.ptr]
  let res3 ← readCUDABufferFull mcBuf

  IO.println "  C = A × I (should equal A):"
  let mut ok3 := true
  for row in List.range mM do
    let mut line := s!"    row {row}: "
    for col in List.range mN do
      let idx := row * mN + col
      let got := unpackFloat res3 idx
      let exp := maArr.getD idx 0.0
      line := line ++ s!"{got} "
      if (got - exp).abs > 0.01 then ok3 := false
    IO.println line

  -- B = [[2,0,1,0],[0,1,0,0],[1,0,2,0],[0,0,0,3]]
  -- C = A × B by hand:
  -- row0: [1,2,3,4]*B = [1*2+3*1, 2*1, 1*1+3*2, 4*3] = [5,2,7,12]
  -- row1: [5,6,7,8]*B = [5*2+7*1, 6*1, 5*1+7*2, 8*3] = [17,6,19,24]
  -- row2: [9,10,11,12]*B = [9*2+11*1, 10*1, 9*1+11*2, 12*3] = [29,10,31,36]
  -- row3: [13,14,15,16]*B = [13*2+15*1, 14*1, 13*1+15*2, 16*3] = [41,14,43,48]
  let mbArr2 : Array Float := #[2,0,1,0, 0,1,0,0, 1,0,2,0, 0,0,0,3]
  let expected3 : Array Float := #[5,2,7,12, 17,6,19,24, 29,10,31,36, 41,14,43,48]
  writeCUDABuffer mbBuf (packFloats mbArr2)

  cuLaunchKernel func3 1 1 1 mN.toUInt32 mM.toUInt32 1 0 #[maBuf.ptr, mbBuf.ptr, mcBuf.ptr]
  let res3b ← readCUDABufferFull mcBuf

  IO.println "  C = A × B (general):"
  for row in List.range mM do
    let mut line := s!"    row {row}: "
    for col in List.range mN do
      let idx := row * mN + col
      let got := unpackFloat res3b idx
      let exp := expected3.getD idx 0.0
      line := line ++ s!"{got} "
      if (got - exp).abs > 0.01 then ok3 := false
    IO.println line

  freeCUDABuffer maBuf; freeCUDABuffer mbBuf; freeCUDABuffer mcBuf
  IO.println s!"  → {if ok3 then "PASSED" else "FAILED"}"

  -- ═══ Test 4: tiledMatMulKernel (shared memory + barrier) ═══
  IO.println ""
  IO.println "Test 4: tiledMatMulKernel (shared memory + cooperative load)"
  IO.println "─────────────────────────────────────────────────────────────"
  let tM := 8; let tN := 8; let tK := 8; let TILE := 4
  let ptx4 := generatePTX "tiledMatMul" {x := TILE, y := TILE} (tiledMatMulKernel tM tN tK TILE)
  IO.println s!"  PTX: {ptx4.length} chars"
  IO.FS.writeFile "debug_tiled_matmul.ptx" ptx4
  let mod4 ← cuModuleLoadData ptx4
  let func4 ← cuModuleGetFunction mod4 "tiledMatMul"
  IO.println "  JIT: OK"

  let taBuf ← createCUDABuffer (tM * tK * 4).toUSize
  let tbBuf ← createCUDABuffer (tK * tN * 4).toUSize
  let tcBuf ← createCUDABuffer (tM * tN * 4).toUSize

  -- A = 8×8 sequential, B = identity
  let taArr := Array.range (tM * tK) |>.map (fun i => Float.ofNat (i + 1))
  let mut tbArr : Array Float := Array.replicate (tK * tN) 0.0
  for i in List.range (min tK tN) do
    tbArr := tbArr.set! (i * tN + i) 1.0

  writeCUDABuffer taBuf (packFloats taArr)
  writeCUDABuffer tbBuf (packFloats tbArr)

  -- grid = (N/TILE, M/TILE), block = (TILE, TILE)
  let gridX := tN / TILE; let gridY := tM / TILE
  cuLaunchKernel func4 gridX.toUInt32 gridY.toUInt32 1 TILE.toUInt32 TILE.toUInt32 1 0
    #[taBuf.ptr, tbBuf.ptr, tcBuf.ptr]
  let res4 ← readCUDABufferFull tcBuf

  IO.println "  C = A × I (8×8, TILE=4, should equal A):"
  let mut ok4 := true
  for row in List.range tM do
    let mut line := s!"    row {row}: "
    for col in List.range tN do
      let idx := row * tN + col
      let got := unpackFloat res4 idx
      let exp := taArr.getD idx 0.0
      line := line ++ s!"{got} "
      if (got - exp).abs > 0.01 then ok4 := false
    IO.println line

  -- Test with non-trivial B
  -- B = diag(2,3,4,5,6,7,8,9)
  let mut tbArr2 : Array Float := Array.replicate (tK * tN) 0.0
  for i in List.range (min tK tN) do
    tbArr2 := tbArr2.set! (i * tN + i) (Float.ofNat (i + 2))
  writeCUDABuffer tbBuf (packFloats tbArr2)

  cuLaunchKernel func4 gridX.toUInt32 gridY.toUInt32 1 TILE.toUInt32 TILE.toUInt32 1 0
    #[taBuf.ptr, tbBuf.ptr, tcBuf.ptr]
  let res4b ← readCUDABufferFull tcBuf

  IO.println "  C = A × diag(2..9) (8×8, TILE=4):"
  for row in List.range tM do
    let mut line := s!"    row {row}: "
    for col in List.range tN do
      let idx := row * tN + col
      let got := unpackFloat res4b idx
      let aVal : Float := (row * tK + col + 1).toFloat
      let exp := aVal * (col + 2).toFloat
      line := line ++ s!"{got} "
      if (got - exp).abs > 0.1 then ok4 := false
    IO.println line

  freeCUDABuffer taBuf; freeCUDABuffer tbBuf; freeCUDABuffer tcBuf
  IO.println s!"  → {if ok4 then "PASSED" else "FAILED"}"

  -- ═══ Test 5: outerProductKernel ═══
  IO.println ""
  IO.println "Test 5: outerProductKernel (R[i,j] = a[i] * b[j])"
  IO.println "───────────────────────────────────────────────────"
  let opM := 4; let opN := 4
  let ptx5 := generatePTX "outerProd" {x := 256} (outerProductKernel opM opN)
  IO.println s!"  PTX: {ptx5.length} chars"
  let mod5 ← cuModuleLoadData ptx5
  let func5 ← cuModuleGetFunction mod5 "outerProd"
  IO.println "  JIT: OK"

  let opABuf ← createCUDABuffer (opM * 4).toUSize
  let opBBuf ← createCUDABuffer (opN * 4).toUSize
  let opRBuf ← createCUDABuffer (opM * opN * 4).toUSize
  -- a = [1,2,3,4], b = [10,20,30,40]
  writeCUDABuffer opABuf (packFloats #[1,2,3,4])
  writeCUDABuffer opBBuf (packFloats #[10,20,30,40])

  cuLaunchKernel func5 1 1 1 (opM * opN).toUInt32 1 1 0 #[opABuf.ptr, opBBuf.ptr, opRBuf.ptr]
  let res5 ← readCUDABufferFull opRBuf

  let mut ok5 := true
  for i in List.range opM do
    let mut line := s!"    row {i}: "
    for j in List.range opN do
      let got := unpackFloat res5 (i * opN + j)
      let exp := (i + 1).toFloat * ((j + 1) * 10).toFloat
      line := line ++ s!"{got} "
      if (got - exp).abs > 0.01 then ok5 := false
    IO.println line
  freeCUDABuffer opABuf; freeCUDABuffer opBBuf; freeCUDABuffer opRBuf
  IO.println s!"  → {if ok5 then "PASSED" else "FAILED"}"

  -- ═══ Test 6: sgdUpdateKernel ═══
  IO.println ""
  IO.println "Test 6: sgdUpdateKernel (param -= lr * grad)"
  IO.println "─────────────────────────────────────────────"
  let sgdN := 8; let lr := 0.1
  let ptx6 := generatePTX "sgdUpdate" {x := 256} (sgdUpdateKernel sgdN lr)
  IO.println s!"  PTX: {ptx6.length} chars"
  let mod6 ← cuModuleLoadData ptx6
  let func6 ← cuModuleGetFunction mod6 "sgdUpdate"
  IO.println "  JIT: OK"

  let gradBuf ← createCUDABuffer (sgdN * 4).toUSize
  let paramBuf ← createCUDABuffer (sgdN * 4).toUSize
  -- param = [10..17], grad = [1..8]
  let paramArr := Array.range sgdN |>.map (fun i => (i + 10).toFloat)
  let gradArr := Array.range sgdN |>.map (fun i => (i + 1).toFloat)
  writeCUDABuffer gradBuf (packFloats gradArr)
  writeCUDABuffer paramBuf (packFloats paramArr)

  cuLaunchKernel func6 1 1 1 sgdN.toUInt32 1 1 0 #[gradBuf.ptr, paramBuf.ptr]
  let res6 ← readCUDABufferFull paramBuf

  let mut ok6 := true
  let mut line6 := "    param: "
  for i in List.range sgdN do
    let got := unpackFloat res6 i
    -- param[i] = (i+10) - 0.1 * (i+1)
    let exp := (i + 10).toFloat - lr * (i + 1).toFloat
    line6 := line6 ++ s!"{got} "
    if (got - exp).abs > 0.01 then ok6 := false
  IO.println line6
  freeCUDABuffer gradBuf; freeCUDABuffer paramBuf
  IO.println s!"  → {if ok6 then "PASSED" else "FAILED"}"

  -- ═══ Test 7: copyBufferKernel ═══
  IO.println ""
  IO.println "Test 7: copyBufferKernel (dst = src)"
  IO.println "─────────────────────────────────────"
  let cpN := 16
  let ptx7 := generatePTX "copyBuf" {x := 256} (copyBufferKernel cpN)
  IO.println s!"  PTX: {ptx7.length} chars"
  let mod7 ← cuModuleLoadData ptx7
  let func7 ← cuModuleGetFunction mod7 "copyBuf"
  IO.println "  JIT: OK"

  let srcBuf ← createCUDABuffer (cpN * 4).toUSize
  let dstBuf ← createCUDABuffer (cpN * 4).toUSize
  let srcArr := Array.range cpN |>.map (fun i => (i * 7 + 3).toFloat)
  writeCUDABuffer srcBuf (packFloats srcArr)

  cuLaunchKernel func7 1 1 1 cpN.toUInt32 1 1 0 #[srcBuf.ptr, dstBuf.ptr]
  let res7 ← readCUDABufferFull dstBuf

  let mut ok7 := true
  for i in [0, 1, 7, 15] do
    let got := unpackFloat res7 i
    let exp := srcArr.getD i 0.0
    let pass := (got - exp).abs < 0.01
    IO.println s!"    dst[{i}] = {got} (expect {exp}) {if pass then "✓" else "✗"}"
    if !pass then ok7 := false
  freeCUDABuffer srcBuf; freeCUDABuffer dstBuf
  IO.println s!"  → {if ok7 then "PASSED" else "FAILED"}"

  -- ═══ Test 8: Large matmul benchmark (256×256) ═══
  IO.println ""
  IO.println "Test 8: Large matMul (256×256, naive vs tiled)"
  IO.println "───────────────────────────────────────────────"
  let bM := 256; let bN := 256; let bK := 256; let bTILE := 16

  -- Generate random-ish data: A[i] = sin(i), B[i] = cos(i)
  let bAArr := Array.range (bM * bK) |>.map (fun i => Float.sin (i.toFloat * 0.01))
  let bBArr := Array.range (bK * bN) |>.map (fun i => Float.cos (i.toFloat * 0.007))

  let bABuf ← createCUDABuffer (bM * bK * 4).toUSize
  let bBBuf ← createCUDABuffer (bK * bN * 4).toUSize
  let bCBuf1 ← createCUDABuffer (bM * bN * 4).toUSize
  let bCBuf2 ← createCUDABuffer (bM * bN * 4).toUSize
  writeCUDABuffer bABuf (packFloats bAArr)
  writeCUDABuffer bBBuf (packFloats bBArr)

  -- Naive matmul
  let ptxNaive := generatePTX "matMulNaive" {x := 16, y := 16} (matMulKernel bM bN bK)
  let modN ← cuModuleLoadData ptxNaive
  let funcN ← cuModuleGetFunction modN "matMulNaive"
  let gridN := (bN + 15) / 16; let gridMN := (bM + 15) / 16

  -- Warmup
  cuLaunchKernel funcN gridN.toUInt32 gridMN.toUInt32 1 16 16 1 0 #[bABuf.ptr, bBBuf.ptr, bCBuf1.ptr]

  let t0 ← IO.monoNanosNow
  for _ in List.range 10 do
    cuLaunchKernel funcN gridN.toUInt32 gridMN.toUInt32 1 16 16 1 0 #[bABuf.ptr, bBBuf.ptr, bCBuf1.ptr]
  let t1 ← IO.monoNanosNow
  let naiveUs := (t1 - t0).toFloat / 10000.0  -- avg per run in μs

  -- Tiled matmul
  let ptxTiled := generatePTX "matMulTiled" {x := bTILE, y := bTILE} (tiledMatMulKernel bM bN bK bTILE)
  let modT ← cuModuleLoadData ptxTiled
  let funcT ← cuModuleGetFunction modT "matMulTiled"
  let gridTX := bN / bTILE; let gridTY := bM / bTILE

  cuLaunchKernel funcT gridTX.toUInt32 gridTY.toUInt32 1 bTILE.toUInt32 bTILE.toUInt32 1 0
    #[bABuf.ptr, bBBuf.ptr, bCBuf2.ptr]

  let t2 ← IO.monoNanosNow
  for _ in List.range 10 do
    cuLaunchKernel funcT gridTX.toUInt32 gridTY.toUInt32 1 bTILE.toUInt32 bTILE.toUInt32 1 0
      #[bABuf.ptr, bBBuf.ptr, bCBuf2.ptr]
  let t3 ← IO.monoNanosNow
  let tiledUs := (t3 - t2).toFloat / 10000.0

  -- Verify both produce same results
  let resN ← readCUDABufferFull bCBuf1
  let resT ← readCUDABufferFull bCBuf2
  let mut maxDiff : Float := 0.0
  for i in List.range (bM * bN) do
    let vn := unpackFloat resN i
    let vt := unpackFloat resT i
    let diff := (vn - vt).abs
    if diff > maxDiff then maxDiff := diff

  IO.println s!"  Naive:  {naiveUs} μs/run"
  IO.println s!"  Tiled:  {tiledUs} μs/run"
  IO.println s!"  Speedup: {naiveUs / tiledUs}×"
  IO.println s!"  Max diff (naive vs tiled): {maxDiff}"
  let ok8 := maxDiff < 0.1
  IO.println s!"  → {if ok8 then "PASSED" else "FAILED"}"

  freeCUDABuffer bABuf; freeCUDABuffer bBBuf
  freeCUDABuffer bCBuf1; freeCUDABuffer bCBuf2

  -- ═══ Test 9: mseResidualLossAndGradKernel ═══
  IO.println ""
  IO.println "Test 9: mseResidualLossAndGrad (shared mem reduction)"
  IO.println "──────────────────────────────────────────────────────"
  let mseDim := 32; let mseWG := 32
  let ptx9 := generatePTX "mseLoss" {x := mseWG} (mseResidualLossAndGradKernel mseDim mseWG)
  IO.println s!"  PTX: {ptx9.length} chars"
  let mod9 ← cuModuleLoadData ptx9
  let func9 ← cuModuleGetFunction mod9 "mseLoss"
  IO.println "  JIT: OK"

  let tttOutBuf ← createCUDABuffer (mseDim * 4).toUSize
  let hiddenTBuf ← createCUDABuffer (mseDim * 4).toUSize
  let hiddenT1Buf ← createCUDABuffer (mseDim * 4).toUSize
  let mseGradBuf ← createCUDABuffer (mseDim * 4).toUSize
  let mseLossBuf ← createCUDABuffer 4

  -- ttt_output = [0..31]/10, hidden_t = [0..31]*0, hidden_t1 = [0..31]/10
  -- target_residual = hidden_t1 - hidden_t = [0..31]/10
  -- diff = ttt_output - target_residual = 0 for all
  -- → MSE loss = 0
  let tttOut := Array.range mseDim |>.map (fun i => i.toFloat / 10.0)
  let hiddenT := Array.replicate mseDim 0.0
  let hiddenT1 := Array.range mseDim |>.map (fun i => i.toFloat / 10.0)
  writeCUDABuffer tttOutBuf (packFloats tttOut)
  writeCUDABuffer hiddenTBuf (packFloats hiddenT)
  writeCUDABuffer hiddenT1Buf (packFloats hiddenT1)

  cuLaunchKernel func9 1 1 1 mseWG.toUInt32 1 1 0
    #[tttOutBuf.ptr, hiddenTBuf.ptr, hiddenT1Buf.ptr, mseGradBuf.ptr, mseLossBuf.ptr]
  let lossBytes ← readCUDABufferFull mseLossBuf
  let loss := unpackFloat lossBytes 0
  IO.println s!"  Loss (expect 0.0): {loss}"
  let mut ok9 : Bool := loss.abs < 0.001

  -- Now with non-zero error: ttt_output = 1.0 for all, target = 0
  -- diff = 1.0, sq = 1.0, sum = 32, MSE = 32/32 = 1.0
  let tttOut2 := Array.replicate mseDim 1.0
  let hiddenT2 := Array.replicate mseDim 0.0
  let hiddenT12 := Array.replicate mseDim 0.0
  writeCUDABuffer tttOutBuf (packFloats tttOut2)
  writeCUDABuffer hiddenTBuf (packFloats hiddenT2)
  writeCUDABuffer hiddenT1Buf (packFloats hiddenT12)

  cuLaunchKernel func9 1 1 1 mseWG.toUInt32 1 1 0
    #[tttOutBuf.ptr, hiddenTBuf.ptr, hiddenT1Buf.ptr, mseGradBuf.ptr, mseLossBuf.ptr]
  let lossBytes2 ← readCUDABufferFull mseLossBuf
  let loss2 := unpackFloat lossBytes2 0
  IO.println s!"  Loss (expect 1.0): {loss2}"
  if (loss2 - 1.0).abs > 0.01 then ok9 := false

  -- Check gradients: grad[i] = 2 * 1.0 / 32 = 0.0625
  let gradBytes ← readCUDABufferFull mseGradBuf
  let grad0 := unpackFloat gradBytes 0
  let grad15 := unpackFloat gradBytes 15
  IO.println s!"  grad[0] (expect 0.0625): {grad0}"
  IO.println s!"  grad[15] (expect 0.0625): {grad15}"
  if (grad0 - 0.0625).abs > 0.001 then ok9 := false
  if (grad15 - 0.0625).abs > 0.001 then ok9 := false

  freeCUDABuffer tttOutBuf; freeCUDABuffer hiddenTBuf
  freeCUDABuffer hiddenT1Buf; freeCUDABuffer mseGradBuf; freeCUDABuffer mseLossBuf
  let ok9f : Bool := ok9
  IO.println s!"  → {if ok9f then "PASSED" else "FAILED"}"

  IO.println ""
  let ok5f : Bool := ok5; let ok6f : Bool := ok6; let ok7f : Bool := ok7
  let allOk := ok1 && ok2 && ok3 && ok4 && ok5f && ok6f && ok7f && ok8 && ok9f
  if allOk then
    IO.println "✓ ALL 9 CUDA KERNEL TESTS PASSED"
  else
    IO.println "✗ SOME TESTS FAILED"
    IO.Process.exit 1
