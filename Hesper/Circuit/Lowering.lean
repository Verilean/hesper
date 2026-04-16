import Hesper.Circuit.IR
import Hesper.Backend
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

/-!
# Circuit Lowering — MVP

Takes a `CircuitState` + a resolver from `TensorRef → Buf` and
dispatches each Op through the existing `Linear.LinearLayer.forward`
machinery.  This is the zero-fusion baseline: one Op per dispatch,
same kernels/token as hand-written code — we want to prove the
framework is overhead-free before adding passes.

Subsequent stages will add passes (constFold, mergeSameDispatch,
inlineProducer, …) that rewrite the `ops` list before emit.
-/

namespace Hesper.Circuit

open Hesper
open Hesper.WGSL
open Hesper.WGSL.Monad

/-! ## Generic pointwise lowering

`lowerScalarExp` folds a `ScalarExp` tree into a typed `Exp` over the
`f32` tensor element at the current lane.  `lowerPointwise` builds the
entire `ShaderM Unit` from the body + input-slot names.

Key property: the lowering is a **function of the IR alone**.  There is
no pattern registry that maps `Prim.pointwise scaleBody → scaleKernel`;
instead, any `ScalarExp` body — whether hand-written, fusion-produced,
or constant-folded — flows through the same lowering.  That's what
makes this a compiler pass instead of a dispatcher. -/

/-- Per-input metadata used by `ScalarExp.indexed` for gather reads.
    `name` is the buffer binding; `len` is its element count (needed by
    `ShaderM.readBuffer`'s size parameter). -/
structure InputDecl where
  name : String
  len  : Nat
  deriving Inhabited

/-- Lower a `ScalarExp` to a typed WGSL `Exp` of f32.  Per-element
    reads from the input slots are pre-loaded into `slot : Array (Exp f32)`
    so the same `input i` reference inside `body` compiles to a single
    shared `var` read, not N redundant buffer loads.  `.indexed` nodes
    perform a fresh buffer read at a computed address using `decls`. -/
def lowerScalarExp (slot : Array (Exp (.scalar .f32)))
    (laneIdxExp : Exp (.scalar .f32) := Exp.litF32 0.0)
    (decls : Array InputDecl := #[])
    : ScalarExp → ShaderM (Exp (.scalar .f32))
  | .input i      =>
    match slot[i]? with
    | some e => pure e
    | none   => pure (Exp.litF32 0.0)
  | .const v      => pure (Exp.litF32 v)
  | .laneIdx      => pure laneIdxExp
  | .indexed i addr => do
    let ae ← lowerScalarExp slot laneIdxExp decls addr
    let addrU32 := Exp.toU32 ae
    match decls[i]? with
    | some decl =>
      ShaderM.readBuffer (ty := .scalar .f32) (n := decl.len) decl.name addrU32
    | none =>
      -- Caller didn't supply a buffer decl for this slot: fall back to 0.
      pure (Exp.litF32 0.0)
  | .add a b      => do
    return Exp.add (← lowerScalarExp slot laneIdxExp decls a) (← lowerScalarExp slot laneIdxExp decls b)
  | .sub a b      => do
    return Exp.sub (← lowerScalarExp slot laneIdxExp decls a) (← lowerScalarExp slot laneIdxExp decls b)
  | .mul a b      => do
    return Exp.mul (← lowerScalarExp slot laneIdxExp decls a) (← lowerScalarExp slot laneIdxExp decls b)
  | .div a b      => do
    return Exp.div (← lowerScalarExp slot laneIdxExp decls a) (← lowerScalarExp slot laneIdxExp decls b)
  | .neg a        => do return Exp.neg (← lowerScalarExp slot laneIdxExp decls a)
  | .rsqrt a      => do return Exp.inverseSqrt (← lowerScalarExp slot laneIdxExp decls a)
  | .exp a        => do return Exp.exp (← lowerScalarExp slot laneIdxExp decls a)
  | .tanh a       => do return Exp.tanh (← lowerScalarExp slot laneIdxExp decls a)
  | .gelu a       => do
    let x ← lowerScalarExp slot laneIdxExp decls a
    let c : Float := 0.7978845608028654
    let k : Float := 0.044715
    let x3 := Exp.mul (Exp.mul x x) x
    let inner := Exp.mul (Exp.litF32 c) (Exp.add x (Exp.mul (Exp.litF32 k) x3))
    return Exp.mul (Exp.mul (Exp.litF32 0.5) x) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
  | .silu a       => do
    let x ← lowerScalarExp slot laneIdxExp decls a
    return Exp.div x (Exp.add (Exp.litF32 1.0) (Exp.exp (Exp.neg x)))
  | .cos a        => do return Exp.cos (← lowerScalarExp slot laneIdxExp decls a)
  | .sin a        => do return Exp.sin (← lowerScalarExp slot laneIdxExp decls a)
  | .pow a b      => do
    return Exp.pow (← lowerScalarExp slot laneIdxExp decls a) (← lowerScalarExp slot laneIdxExp decls b)
  | .lt a b       => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    let vb ← lowerScalarExp slot laneIdxExp decls b
    return Exp.select (Exp.lt va vb) (Exp.litF32 1.0) (Exp.litF32 0.0)
  | .select c t f => do
    let vc ← lowerScalarExp slot laneIdxExp decls c
    let vt ← lowerScalarExp slot laneIdxExp decls t
    let vf ← lowerScalarExp slot laneIdxExp decls f
    return Exp.select (Exp.lt (Exp.litF32 0.5) vc) vt vf
  | .mod a b      => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    let vb ← lowerScalarExp slot laneIdxExp decls b
    let quotient := Exp.toF32 (Exp.toU32 (Exp.div va vb))
    return Exp.sub va (Exp.mul quotient vb)
  | .idiv a b     => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    let vb ← lowerScalarExp slot laneIdxExp decls b
    return Exp.toF32 (Exp.toU32 (Exp.div va vb))
  | .fastdiv n mp L d => do
    -- Constant-folding: d=1 is the identity.  This catches llama.cpp's
    -- nchannels_y = (0,0,1) / channel_ratio = (0,0,1) pattern where
    -- the fastdiv reduces to "read n" at zero PTX cost.
    if d == 1 then
      lowerScalarExp slot laneIdxExp decls n
    else
      let vn ← lowerScalarExp slot laneIdxExp decls n
      let nU32 := Exp.toU32 vn
      let mpU32 : Exp (.scalar .u32) := Exp.litU32 mp
      let hi : Exp (.scalar .u32) := Exp.mulhiU32 nU32 mpU32
      let sum : Exp (.scalar .u32) := Exp.add hi nU32
      let quot : Exp (.scalar .u32) := Exp.shiftRight sum (Exp.litU32 L)
      return Exp.toF32 quot
  | .toFloat a    => lowerScalarExp slot laneIdxExp decls a
  | .warpSum a    => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    pure (Exp.subgroupAdd va)
  | .warpBroadcast a => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    pure (Exp.subgroupBroadcastFirst va)
  | .warpShuffleXor a mask => do
    let va ← lowerScalarExp slot laneIdxExp decls a
    pure (Exp.subgroupShuffleXor va (Exp.litU32 mask))
termination_by e => sizeOf e

/-- Build a single-WG shared-memory reduction `ShaderM` for
    `Prim.reduceLastAxis`.  Caller passes the element count `D` and
    the `ReduceOp`.  Dispatch shape is `numWorkgroups = 1`,
    `workgroupSize = min D 256`.

    Algorithm:
      1. Each lane accumulates a partial using strided pass over the
         input, materialising `x*x` (for sumOfSquares) in-register.
      2. Partial written to shared memory, barrier.
      3. Classic power-of-two tree reduction with barriers.
      4. Lane 0 writes shared[0] to out[0].

    Limitation: `D ≤ 256` fully parallel, or strided for larger D.
    Multi-WG split-reduction is a Stage 2b follow-up. -/
def lowerReduceLastAxis
    (op : ReduceOp) (inputName outputName : String)
    (D : Nat) (workgroupSize : Nat) : ShaderM Unit := do
  ShaderM.sharedNamed "scratch" (.array (.scalar .f32) workgroupSize)
  let _ ← ShaderM.declareInputBuffer inputName (.array (.scalar .f32) D)
  let _ ← ShaderM.declareOutputBuffer outputName (.array (.scalar .f32) 1)
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  ShaderM.varNamed "accum" (.scalar .f32) (Exp.litF32 0.0)
  let accumE : Exp (.scalar .f32) := Exp.var "accum"
  -- Strided accumulation.  `contrib` is per-lane-per-iteration
  -- contribution — `x*x` for sumOfSquares, `x` for plain sum.
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) inputName loopIdx
    let contrib := match op with
      | .sum          => v
      | .sumOfSquares => Exp.mul v v
    ShaderM.assign "accum" (Exp.add accumE contrib)
  ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx accumE
  ShaderM.barrier
  -- Tree reduction.
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch"
                (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  -- Lane 0 writes the result.
  ShaderM.if_ (Exp.eq localIdx (Exp.litU32 0)) (do
    let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) outputName (Exp.litU32 0) total
  ) (pure ())

/-- Dispatch a `Prim.reduceLastAxis` through `executeWithConfigCached`.
    Single-WG reduction; caller provides the input + output buffers. -/
def runReduceOp [GPUBackend β]
    (ctx : β) (op : ReduceOp) (D : Nat)
    (inputBuf outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let wgSize := min D 256
  let inputName := "in0"
  let outputName := "out"
  let shader : ShaderM Unit :=
    lowerReduceLastAxis op inputName outputName D wgSize
  let namedBufs : List (String × GPUBackend.Buf β) :=
    [(outputName, outputBuf), (inputName, inputBuf)]
  let config : ExecConfig :=
    { numWorkgroups := (1, 1, 1)
      workgroupSize := { x := wgSize, y := 1, z := 1 } }
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Reduce + per-lane epilogue, fused into one kernel.  Phase 1 is the
    same strided-accum + tree reduce as `lowerReduceLastAxis`; phase 2
    reads the scalar back from shared memory and every lane writes
    `body[input 0 := scratch[0], input (k+1) := epilogueInput[k][lane]]`
    to its slot of the output.  Used to collapse the canonical
    "reduce → use-result-broadcast-in-pointwise" pattern (RMSNorm,
    softmax-pre-norm, etc.) into a single dispatch. -/
def lowerReduceLastAxisWithEpilogue
    (op : ReduceOp)
    (reduceInputName : String)
    (epilogueInputNames : Array String)
    (outputName : String)
    (D : Nat) (workgroupSize : Nat)
    (body : ScalarExp) : ShaderM Unit := do
  ShaderM.sharedNamed "scratch" (.array (.scalar .f32) workgroupSize)
  let _ ← ShaderM.declareInputBuffer reduceInputName (.array (.scalar .f32) D)
  for name in epilogueInputNames do
    let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) D)
  let _ ← ShaderM.declareOutputBuffer outputName (.array (.scalar .f32) D)
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  -- Phase 1: strided accumulate.
  ShaderM.varNamed "accum" (.scalar .f32) (Exp.litF32 0.0)
  let accumE : Exp (.scalar .f32) := Exp.var "accum"
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) reduceInputName loopIdx
    let contrib := match op with
      | .sum          => v
      | .sumOfSquares => Exp.mul v v
    ShaderM.assign "accum" (Exp.add accumE contrib)
  ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx accumE
  ShaderM.barrier
  -- Tree reduction.
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch"
                (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  -- Phase 2: every lane writes its slot of the output.  Strided over D
  -- so workgroupSize lanes cover all D elements.
  let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" (Exp.litU32 0)
  let totalName ← ShaderM.var (.scalar .f32) total
  let totalRef : Exp (.scalar .f32) := Exp.var totalName
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    -- Build slot array: input 0 = scalar reduction; input (k+1) = epilogueInputs[k][loopIdx]
    let mut slots : Array (Exp (.scalar .f32)) := #[totalRef]
    let mut decls : Array InputDecl := #[{ name := "__reduced__", len := 1 }]
    for name in epilogueInputNames do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) name loopIdx
      let vName ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
      decls := decls.push { name := name, len := D }
    let result ← lowerScalarExp slots (Exp.litF32 0.0) decls body
    ShaderM.writeBuffer (ty := .scalar .f32) outputName loopIdx result

/-- **Level 3**: block-cooperative reduce + dynamic-address scatter.
    Same shape as `lowerReduceLastAxisWithEpilogue` but the epilogue
    writes to `outputName[addrExpr]` (with `addrExpr` evaluated per
    lane) instead of `outputName[loopIdx]`, and `outputName` has
    `dstSize` elements. -/
def lowerReduceScatterEpilogue
    (op : ReduceOp)
    (reduceInputName : String)
    (epilogueInputNames : Array String)
    (outputName : String)
    (D : Nat) (dstSize : Nat) (workgroupSize : Nat)
    (valueExpr addrExpr : ScalarExp) : ShaderM Unit := do
  ShaderM.sharedNamed "scratch" (.array (.scalar .f32) workgroupSize)
  let _ ← ShaderM.declareInputBuffer reduceInputName (.array (.scalar .f32) D)
  for name in epilogueInputNames do
    let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) D)
  let _ ← ShaderM.declareOutputBuffer outputName (.array (.scalar .f32) dstSize)
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  -- Phase 1: strided accumulate.
  ShaderM.varNamed "accum" (.scalar .f32) (Exp.litF32 0.0)
  let accumE : Exp (.scalar .f32) := Exp.var "accum"
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) reduceInputName loopIdx
    let contrib := match op with
      | .sum          => v
      | .sumOfSquares => Exp.mul v v
    ShaderM.assign "accum" (Exp.add accumE contrib)
  ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx accumE
  ShaderM.barrier
  -- Tree reduction.
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch"
                (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  -- Phase 2: every lane evaluates value and addr, writes dst[addr] = value.
  -- Strided over D so workgroupSize lanes cover all D elements.
  let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch" (Exp.litU32 0)
  let totalName ← ShaderM.var (.scalar .f32) total
  let totalRef : Exp (.scalar .f32) := Exp.var totalName
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    -- slots[0] = scalar reduction; slots[k+1] = epilogueInputs[k][loopIdx]
    let mut slots : Array (Exp (.scalar .f32)) := #[totalRef]
    let mut decls : Array InputDecl := #[{ name := "__reduced__", len := 1 }]
    for name in epilogueInputNames do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) name loopIdx
      let vName ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
      decls := decls.push { name := name, len := D }
    let laneIdxF32 := Exp.toF32 loopIdx
    let value ← lowerScalarExp slots laneIdxF32 decls valueExpr
    let addrF32 ← lowerScalarExp slots laneIdxF32 decls addrExpr
    let addrU32 := Exp.toU32 addrF32
    ShaderM.writeBuffer (ty := .scalar .f32) outputName addrU32 value

def runReduceScatterEpilogueOp [GPUBackend β]
    (ctx : β) (op : ReduceOp) (D : Nat) (dstSize : Nat)
    (valueExpr addrExpr : ScalarExp)
    (reduceInputBuf : GPUBackend.Buf β)
    (epilogueInputBufs : Array (GPUBackend.Buf β))
    (outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let wgSize := min D 256
  let reduceInputName := "in0"
  let epilogueInputNames : Array String :=
    Array.ofFn (fun (i : Fin epilogueInputBufs.size) => s!"in{i.val + 1}")
  let outputName := "out"
  let shader : ShaderM Unit :=
    lowerReduceScatterEpilogue op reduceInputName epilogueInputNames outputName
      D dstSize wgSize valueExpr addrExpr
  let mut namedBufs : List (String × GPUBackend.Buf β) := [(outputName, outputBuf)]
  let n := epilogueInputBufs.size
  for i in [0:n] do
    match epilogueInputBufs[i]? with
    | some buf => namedBufs := (s!"in{i+1}", buf) :: namedBufs
    | none     => pure ()
  namedBufs := (reduceInputName, reduceInputBuf) :: namedBufs
  let config : ExecConfig :=
    { numWorkgroups := (1, 1, 1)
      workgroupSize := { x := wgSize, y := 1, z := 1 } }
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Dispatch a `Prim.reduceLastAxisWithEpilogue` through cached exec. -/
def runReduceWithEpilogueOp [GPUBackend β]
    (ctx : β) (op : ReduceOp) (D : Nat) (body : ScalarExp)
    (reduceInputBuf : GPUBackend.Buf β)
    (epilogueInputBufs : Array (GPUBackend.Buf β))
    (outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let wgSize := min D 256
  let reduceInputName := "in0"
  let epilogueInputNames : Array String :=
    Array.ofFn (fun (i : Fin epilogueInputBufs.size) => s!"in{i.val + 1}")
  let outputName := "out"
  let shader : ShaderM Unit :=
    lowerReduceLastAxisWithEpilogue op reduceInputName epilogueInputNames
      outputName D wgSize body
  let mut namedBufs : List (String × GPUBackend.Buf β) :=
    [(outputName, outputBuf), (reduceInputName, reduceInputBuf)]
  let n := epilogueInputBufs.size
  for i in [0:n] do
    match epilogueInputBufs[i]? with
    | some buf => namedBufs := (s!"in{i + 1}", buf) :: namedBufs
    | none     => pure ()
  let config : ExecConfig :=
    { numWorkgroups := (1, 1, 1)
      workgroupSize := { x := wgSize, y := 1, z := 1 } }
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Q4_K matmul (dp4a) with a lane-local pointwise epilogue.

    The matmul body is shared with `fusedQ4KMLinearDP4AKernel` via
    `Hesper.Layers.Linear.emitQ4KMLinearDP4ABody`, which returns the
    per-warp `total` scalar and the `(outIdx, tid, inBounds)` triple.

    On lane 0 (`tid == 0`), we evaluate the caller's epilogue
    `ScalarExp` against the slot array:
      slot[0]   = `total` (the matmul dot product)
      slot[k+1] = `epiInputs[k][outIdx + epiReadOffsets[k]]`
    and write the result to `output`.

    Other lanes of the warp do nothing in the epilogue (matches the
    base kernel's `if tid == 0` guard).

    Buffer binding naming:
      "weights", "input_q8"   — owned by the matmul body
      "epi0", "epi1", …       — the k-th epilogue side input
      "output"                — result
-/
def lowerMatmulQ4KWithEpilogueKernel
    (config : Hesper.Layers.Linear.Config)
    (epiInputNames : Array String) (epiBufferSizes : Array Nat)
    (epiReadOffsets : Array Nat) (epiBody : ScalarExp) : ShaderM Unit := do
  let (outIdx, tid, inBounds, total) ←
    Hesper.Layers.Linear.emitQ4KMLinearDP4ABody config
  -- Declare epilogue side inputs + output.
  for i in [0 : epiInputNames.size] do
    let name := epiInputNames[i]!
    let sz   := epiBufferSizes[i]?.getD config.outDim
    let _ ← ShaderM.declareReadOnlyBuffer name (.array (.scalar .f32) sz)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    -- Bind `total` to a named var so the body can reference it via slot 0.
    let totalName ← ShaderM.var (.scalar .f32) total
    let mut slots : Array (Exp (.scalar .f32)) := #[Exp.var totalName]
    let mut decls : Array InputDecl := #[{ name := "__total__", len := 1 }]
    for i in [0 : epiInputNames.size] do
      let name   := epiInputNames[i]!
      let offset := epiReadOffsets[i]?.getD 0
      let sz     := epiBufferSizes[i]?.getD config.outDim
      let idx    := Exp.add outIdx (Exp.litU32 offset)
      let v      ← ShaderM.readBuffer (ty := .scalar .f32) (n := sz) name idx
      let vName  ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
      decls := decls.push { name := name, len := sz }
    let result ← lowerScalarExp slots (Exp.litF32 0.0) decls epiBody
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-- Dispatch a `Prim.matmulQ4KWithEpilogue` through cached exec.

    Emits TWO dispatches:
      1. Q8_1 quantize of the f32 `inputBuf` into the layer's own
         `dp4aQ8Buf`.  Cached on `layer.dp4aQuantizePrepared`, so
         repeat calls to the same layer reuse the prepared kernel.
      2. The fused matmul + lane-local epilogue.

    The caller hands us `cacheRef` for the second (matmul+epilogue)
    kernel; we manage the quantize cache via the layer. -/
def runMatmulQ4KWithEpilogueOp [GPUBackend β]
    (ctx : β) (layer : Hesper.Layers.Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf : GPUBackend.Buf β)
    (epiInputBufs : Array (GPUBackend.Buf β))
    (epiBufferSizes : Array Nat) (epiReadOffsets : Array Nat)
    (epiBody : ScalarExp)
    (outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  -- Step 1: ensure the Q8_1 scratch buffer exists, then quantize the input.
  let nQ8Blocks := layer.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← layer.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      layer.dp4aQ8Buf.set (some b)
      pure b
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.Linear.quantizeQ8_1Kernel layer.config.inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q8_1-quantize", layer.config.inDim))
    layer.dp4aQuantizePrepared
  -- Step 2: the fused matmul + epilogue.
  let epiInputNames : Array String :=
    Array.ofFn (fun (i : Fin epiInputBufs.size) => s!"epi{i.val}")
  let shader : ShaderM Unit :=
    lowerMatmulQ4KWithEpilogueKernel layer.config
      epiInputNames epiBufferSizes epiReadOffsets epiBody
  let mut namedBufs : List (String × GPUBackend.Buf β) :=
    [("output", outputBuf), ("weights", layer.weightBuf), ("input_q8", q8Buf)]
  let n := epiInputBufs.size
  for i in [0:n] do
    match epiInputBufs[i]? with
    | some buf => namedBufs := (s!"epi{i}", buf) :: namedBufs
    | none     => pure ()
  let config : ExecConfig :=
    { numWorkgroups := (layer.config.outDim, 1, 1),
      workgroupSize := { x := 32, y := 1, z := 1 } }
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Build a complete `ShaderM Unit` that implements a `Prim.scatter`
    op.

    - `inputNames[k]` / `inputBroadcast[k]` describe inputs[k]
      (broadcast ⇒ [1]-shape, else lane-local).
    - `outputName` is the destination buffer binding.
    - `numel` is the dispatch grid (= outShape.numel).
    - `dstNumel` is the destination buffer total size.
    - `valueExpr` / `addrExpr` are both evaluated per lane with:
        `.input k`  → slots[k]
        `.laneIdx`  → thread's f32-cast global id
      `addrExpr` is cast to u32 at the end (truncation).

    `dst[addr] = valueExpr`.  Map = `addrExpr = .laneIdx`, `dstNumel = numel`. -/
def lowerScatter
    (inputNames : Array String) (inputBroadcast : Array Bool) (outputName : String)
    (numel : Nat) (dstNumel : Nat)
    (valueExpr addrExpr : ScalarExp) : ShaderM Unit := do
  for i in [0:inputNames.size] do
    let name := inputNames[i]!
    let bc := inputBroadcast[i]?.getD false
    let len := if bc then 1 else numel
    let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) len)
  let _ ← ShaderM.declareOutputBuffer outputName (.array (.scalar .f32) dstNumel)
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  ShaderM.if_ (Exp.lt idx (Exp.litU32 numel)) (do
    let mut slots : Array (Exp (.scalar .f32)) := #[]
    let mut decls : Array InputDecl := #[]
    for i in [0:inputNames.size] do
      let name := inputNames[i]!
      let bc := inputBroadcast[i]?.getD false
      let len := if bc then 1 else numel
      let readIdx := if bc then Exp.litU32 0 else idx
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := len) name readIdx
      let vName ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
      decls := decls.push { name := name, len := len }
    let idxF32 := Exp.toF32 idx
    let value ← lowerScalarExp slots idxF32 decls valueExpr
    let addrF32 ← lowerScalarExp slots idxF32 decls addrExpr
    let addrU32 := Exp.toU32 addrF32
    ShaderM.writeBuffer (ty := .scalar .f32) outputName addrU32 value
  ) (pure ())

/-- Multi-output scatter: shared dispatch grid + shared input slots,
    but writes to N independent destination buffers.  `outputNames[k]`
    is the binding for destination k; `dstNumels[k]` its element count;
    `bodies[k] = (valueExpr_k, addrExpr_k)` its compute. -/
def lowerScatterMulti
    (inputNames : Array String) (inputBroadcast : Array Bool)
    (outputNames : Array String) (dstNumels : Array Nat)
    (numel : Nat) (bodies : Array (ScalarExp × ScalarExp))
    : ShaderM Unit := do
  for i in [0:inputNames.size] do
    let name := inputNames[i]!
    let bc := inputBroadcast[i]?.getD false
    let len := if bc then 1 else numel
    let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) len)
  for i in [0:outputNames.size] do
    let name := outputNames[i]!
    let dstN := dstNumels[i]?.getD 0
    let _ ← ShaderM.declareOutputBuffer name (.array (.scalar .f32) dstN)
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  ShaderM.if_ (Exp.lt idx (Exp.litU32 numel)) (do
    let mut slots : Array (Exp (.scalar .f32)) := #[]
    let mut decls : Array InputDecl := #[]
    for i in [0:inputNames.size] do
      let name := inputNames[i]!
      let bc := inputBroadcast[i]?.getD false
      let len := if bc then 1 else numel
      let readIdx := if bc then Exp.litU32 0 else idx
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := len) name readIdx
      let vName ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
      decls := decls.push { name := name, len := len }
    let idxF32 := Exp.toF32 idx
    for k in [0 : bodies.size] do
      let (valueExpr, addrExpr) := bodies[k]!
      let value ← lowerScalarExp slots idxF32 decls valueExpr
      let addrF32 ← lowerScalarExp slots idxF32 decls addrExpr
      let addrU32 := Exp.toU32 addrF32
      let outName := outputNames[k]!
      ShaderM.writeBuffer (ty := .scalar .f32) outName addrU32 value
  ) (pure ())

/-- A mapping from TensorRef id to the concrete device buffer.
    Represented as a simple association list; linear lookup is fine
    for MVP circuit sizes (<100 tensors per compile). -/
abbrev BufferMap (BufT : Type) := List (Nat × BufT)

/-- Dispatch a `Prim.scatter` op.  Handles Map / Slice / dynamic-scatter
    uniformly; the `lowerScatter` body branches only at lowering time
    on the `ScalarExp` structure. -/
def runScatterOp [GPUBackend β]
    (ctx : β) (numel : Nat) (dstNumel : Nat)
    (inShapes : Array Shape) (valueExpr addrExpr : ScalarExp)
    (inputBufs : Array (GPUBackend.Buf β))
    (outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let inputNames : Array String :=
    Array.ofFn (fun (i : Fin inputBufs.size) => s!"in{i.val}")
  let inputBroadcast : Array Bool := inShapes.map (fun s => s == #[1])
  let outputName := "out"
  let shader : ShaderM Unit :=
    lowerScatter inputNames inputBroadcast outputName numel dstNumel valueExpr addrExpr
  let mut namedBufs : List (String × GPUBackend.Buf β) := [(outputName, outputBuf)]
  let n := inputBufs.size
  for i in [0:n] do
    match inputBufs[i]? with
    | some buf => namedBufs := (s!"in{i}", buf) :: namedBufs
    | none     => pure ()
  let config : ExecConfig := ExecConfig.dispatch1D numel
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Dispatch a `Prim.scatterMulti` op.  Same shape as `runScatterOp`
    but with N output buffers.  `dstNumels[k]` and `bodies[k] =
    (valueExpr, addrExpr)` describe destination k. -/
def runScatterMultiOp [GPUBackend β]
    (ctx : β) (numel : Nat)
    (inShapes : Array Shape)
    (bodies : Array (ScalarExp × ScalarExp))
    (dstNumels : Array Nat)
    (inputBufs : Array (GPUBackend.Buf β))
    (outputBufs : Array (GPUBackend.Buf β))
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let inputNames : Array String :=
    Array.ofFn (fun (i : Fin inputBufs.size) => s!"in{i.val}")
  let outputNames : Array String :=
    Array.ofFn (fun (i : Fin outputBufs.size) => s!"out{i.val}")
  let inputBroadcast : Array Bool := inShapes.map (fun s => s == #[1])
  let shader : ShaderM Unit :=
    lowerScatterMulti inputNames inputBroadcast outputNames dstNumels numel bodies
  let mut namedBufs : List (String × GPUBackend.Buf β) := []
  for i in [0:outputBufs.size] do
    match outputBufs[i]? with
    | some buf => namedBufs := (s!"out{i}", buf) :: namedBufs
    | none     => pure ()
  for i in [0:inputBufs.size] do
    match inputBufs[i]? with
    | some buf => namedBufs := (s!"in{i}", buf) :: namedBufs
    | none     => pure ()
  let config : ExecConfig := ExecConfig.dispatch1D numel
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

/-- Lower a Circuit: execute each Op in order.  Buffers for produced
    tensors are supplied by the caller via `outputBufs` — the caller
    knows which TensorRef is which (they hold the TensorRef handles
    from the builder). -/
def compile [GPUBackend β]
    (ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (outputBufs : List (TensorRef × GPUBackend.Buf β))
    : IO Unit := do
  -- Build the initial buffer map from externals + outputBufs.
  let mut bmap : BufferMap (GPUBackend.Buf β) := []
  for (tr, buf) in state.externals.toList do
    bmap := (tr.id, buf) :: bmap
  for (tr, buf) in outputBufs do
    bmap := (tr.id, buf) :: bmap
  let lookup (id : Nat) : Option (GPUBackend.Buf β) :=
    (bmap.find? (fun e => e.1 == id)).map (·.2)
  -- Execute each op.
  for op in state.ops do
    match op.prim with
    | Prim.matmulQ4K layer =>
      let inTr := op.inputs[0]!
      let outTr := op.outputs[0]!
      match lookup inTr.id, lookup outTr.id with
      | some inputBuf, some outputBuf =>
        Hesper.Layers.Linear.LinearLayer.forward ctx layer inputBuf outputBuf
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for matmul op (in={inTr.id}, out={outTr.id})")
    | Prim.matmulQ4KWithEpilogue layer epiBufferSizes epiReadOffsets epiBody =>
      let inTr := op.inputs[0]!
      let outTr := op.outputs[0]!
      let mut epiBufs : Array (GPUBackend.Buf β) := #[]
      for k in [1 : op.inputs.size] do
        match op.inputs[k]? with
        | some tr =>
          match lookup tr.id with
          | some b => epiBufs := epiBufs.push b
          | none => throw (IO.userError s!"Circuit.compile: missing epi input id={tr.id}")
        | none => pure ()
      match lookup inTr.id, lookup outTr.id with
      | some inputBuf, some outputBuf =>
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runMatmulQ4KWithEpilogueOp ctx layer inputBuf epiBufs
          epiBufferSizes epiReadOffsets epiBody outputBuf 0 cacheRef
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for matmul-epi op (in={inTr.id}, out={outTr.id})")
    | Prim.scatter outShape dstShape inShapes valueExpr addrExpr =>
      let outTr := op.outputs[0]!
      let mut inBufs : Array (GPUBackend.Buf β) := #[]
      for inTr in op.inputs do
        match lookup inTr.id with
        | some b => inBufs := inBufs.push b
        | none   => throw (IO.userError s!"Circuit.compile: missing scatter input {inTr.id}")
      match lookup outTr.id with
      | some outBuf =>
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runScatterOp ctx outShape.numel dstShape.numel inShapes valueExpr addrExpr inBufs outBuf 0 cacheRef
      | none => throw (IO.userError s!"Circuit.compile: missing scatter output {outTr.id}")
    | Prim.scatterMulti outShape inShapes outputs =>
      -- op.inputs layout: [data inputs..., dst_0, dst_1, ...]
      let nOuts := op.outputs.size
      let nIn := op.inputs.size - nOuts
      let mut inBufs : Array (GPUBackend.Buf β) := #[]
      for k in [0:nIn] do
        let tr := op.inputs[k]!
        match lookup tr.id with
        | some b => inBufs := inBufs.push b
        | none => throw (IO.userError s!"Circuit.compile: missing scatterMulti input {tr.id}")
      let mut outBufs : Array (GPUBackend.Buf β) := #[]
      for k in [0:nOuts] do
        let tr := op.outputs[k]!
        match lookup tr.id with
        | some b => outBufs := outBufs.push b
        | none => throw (IO.userError s!"Circuit.compile: missing scatterMulti output {tr.id}")
      let bodies : Array (ScalarExp × ScalarExp) :=
        outputs.map (fun (_, v, a) => (v, a))
      let dstNumels : Array Nat := outputs.map (fun (s, _, _) => s.numel)
      let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
      runScatterMultiOp ctx outShape.numel inShapes bodies dstNumels inBufs outBufs 0 cacheRef
    | Prim.reduceLastAxis rop inShape =>
      let inTr  := op.inputs[0]!
      let outTr := op.outputs[0]!
      match lookup inTr.id, lookup outTr.id with
      | some inBuf, some outBuf =>
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runReduceOp ctx rop inShape.numel inBuf outBuf 0 cacheRef
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for reduce op (in={inTr.id} out={outTr.id})")
    | Prim.reduceLastAxisWithEpilogue rop reduceInShape _epiShapes body =>
      let inTr  := op.inputs[0]!
      let outTr := op.outputs[0]!
      match lookup inTr.id, lookup outTr.id with
      | some reduceInBuf, some outBuf =>
        let mut epiBufs : Array (GPUBackend.Buf β) := #[]
        for k in [1 : op.inputs.size] do
          match op.inputs[k]? with
          | some tr =>
            match lookup tr.id with
            | some b => epiBufs := epiBufs.push b
            | none => throw (IO.userError s!"Circuit.compile: missing epilogue input id={tr.id}")
          | none => pure ()
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runReduceWithEpilogueOp ctx rop reduceInShape.numel body
          reduceInBuf epiBufs outBuf 0 cacheRef
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for reduce-with-epilogue op (in={inTr.id} out={outTr.id})")
    | Prim.reduceScatterEpilogue rop reduceInShape epiShapes dstShape valueExpr addrExpr =>
      -- inputs layout: [reduceIn, epi_1, ..., epi_k, dst]
      let nInputs := op.inputs.size
      let nEpi := epiShapes.size
      let _ := nEpi  -- shape info already in epiShapes
      let reduceTr := op.inputs[0]!
      let dstTr := op.inputs[nInputs - 1]!
      match lookup reduceTr.id, lookup dstTr.id with
      | some reduceInBuf, some dstBuf =>
        let mut epiBufs : Array (GPUBackend.Buf β) := #[]
        for k in [1 : nInputs - 1] do
          match op.inputs[k]? with
          | some tr =>
            match lookup tr.id with
            | some b => epiBufs := epiBufs.push b
            | none => throw (IO.userError s!"Circuit.compile: missing reduceScatter epi input id={tr.id}")
          | none => pure ()
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runReduceScatterEpilogueOp ctx rop reduceInShape.numel dstShape.numel
          valueExpr addrExpr reduceInBuf epiBufs dstBuf 0 cacheRef
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for reduceScatter")

/-! ## Build-once, replay-many — zero-overhead dispatch path

`CompiledCircuit` holds a pre-resolved list of op closures that take
only the caller's per-call buffers (hidden state in, Q out, etc.) and
execute the backing kernels directly.  No CircuitM evaluation, no
buffer-map construction, no pattern-matching on Prim on the hot path.

Usage (cached):
  state.compiledQ : IO.Ref (Option (CompiledCircuit β))
  ...
  let cc ← state.compiledQ.get >>= fun
    | some cc => pure cc
    | none =>
      let cc ← Circuit.compileOnce ctx (buildCircuit ...)
      state.compiledQ.set (some cc)
      pure cc
  cc.replay [(tensorQNormed, state.normedBuf), (tensorQOut, state.qBuf)]
-/

/-- An op compiled down to a single closure.  The closure takes the
    resolver function (input/output id → buffer) and runs the dispatch. -/
structure OpClosure (β : Type) [GPUBackend β] where
  run : (lookup : Nat → Option (GPUBackend.Buf β)) → IO Unit

/-- A circuit compiled into a flat, cache-friendly representation.
    No Lean-side allocation on replay — just a walk over the closures. -/
structure CompiledCircuit (β : Type) [GPUBackend β] where
  ops : Array (OpClosure β)
  /-- TensorRefs the caller needs to supply a buffer for (externals +
      every produced tensor).  Kept so callers know what to pass on
      replay. -/
  externalIds : Array Nat
  producedIds : Array Nat
  /-- Intermediate buffers allocated once at compile time for
      TensorRefs produced by surviving ops that the caller did NOT
      register in `buffers` on replay — e.g. the scalar output of a
      `reduceLastAxis` that feeds a fused pointwise tail.  These are
      prepended to the caller's buffers on every replay so the ops'
      lookup fn sees them. -/
  baseBuffers : List (Nat × GPUBackend.Buf β) := []

/-- Compile a CircuitState into a CompiledCircuit that can be replayed
    with minimal overhead.  Runs once per unique circuit. -/
def compileOnce [GPUBackend β]
    (_ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    : IO (CompiledCircuit β) := do
  -- Convert each Op to an OpClosure that captures its metadata
  -- (layer, tensor ids) but defers buffer resolution to replay time.
  -- Pointwise ops allocate their own cacheRef in IO so the replay
  -- closure can close over it and skip PTX regeneration.
  let mut closures : Array (OpClosure β) := #[]
  for op in state.ops do
    match op.prim with
    | Prim.matmulQ4K layer =>
      let inId := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some inputBuf, some outputBuf =>
              Hesper.Layers.Linear.LinearLayer.forward (β := β) _ctx layer inputBuf outputBuf
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing buffer (in={inId} out={outId})")
        }
    | Prim.matmulQ4KWithEpilogue layer epiBufferSizes epiReadOffsets epiBody =>
      let inId := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      let epiIds : Array Nat :=
        (op.inputs.extract 1 op.inputs.size).map (·.id)
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-matmulQ4K-epi",
              layer.config.inDim, layer.config.outDim,
              reprStr epiBody, reprStr epiBufferSizes.toList,
              reprStr epiReadOffsets.toList)
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some inputBuf, some outputBuf =>
              let mut epiBufs : Array (GPUBackend.Buf β) := #[]
              for id in epiIds do
                match lookup id with
                | some b => epiBufs := epiBufs.push b
                | none => throw (IO.userError s!"CompiledCircuit: missing matmul-epi input id={id}")
              runMatmulQ4KWithEpilogueOp _ctx layer inputBuf epiBufs
                epiBufferSizes epiReadOffsets epiBody outputBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing matmul-epi buffer (in={inId} out={outId})")
        }
    | Prim.scatter outShape dstShape inShapes valueExpr addrExpr =>
      let numel := outShape.numel
      let dstNumel := dstShape.numel
      let inIds := op.inputs.map (·.id)
      let outId := op.outputs[0]!.id
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-scatter", numel, dstNumel,
              reprStr valueExpr, reprStr addrExpr, reprStr inShapes.toList)
      closures := closures.push
        { run := fun lookup => do
            let mut inBufs : Array (GPUBackend.Buf β) := #[]
            for id in inIds do
              match lookup id with
              | some b => inBufs := inBufs.push b
              | none   => throw (IO.userError s!"CompiledCircuit: missing scatter input id={id}")
            match lookup outId with
            | some outBuf =>
              runScatterOp _ctx numel dstNumel inShapes valueExpr addrExpr inBufs outBuf cacheKey cacheRef
            | none => throw (IO.userError s!"CompiledCircuit: missing scatter output id={outId}")
        }
    | Prim.scatterMulti outShape inShapes outputs =>
      let numel := outShape.numel
      let nOuts := op.outputs.size
      let nIn := op.inputs.size - nOuts
      let inIds : Array Nat := (op.inputs.extract 0 nIn).map (·.id)
      let outIds : Array Nat := op.outputs.map (·.id)
      let bodies : Array (ScalarExp × ScalarExp) :=
        outputs.map (fun (_, v, a) => (v, a))
      let dstNumels : Array Nat := outputs.map (fun (s, _, _) => s.numel)
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-scatter-multi", numel, reprStr inShapes.toList,
              reprStr (outputs.map (fun (s, v, a) => (s.numel, reprStr v, reprStr a))).toList)
      closures := closures.push
        { run := fun lookup => do
            let mut inBufs : Array (GPUBackend.Buf β) := #[]
            for id in inIds do
              match lookup id with
              | some b => inBufs := inBufs.push b
              | none   => throw (IO.userError s!"CompiledCircuit: missing scatterMulti input id={id}")
            let mut outBufs : Array (GPUBackend.Buf β) := #[]
            for id in outIds do
              match lookup id with
              | some b => outBufs := outBufs.push b
              | none   => throw (IO.userError s!"CompiledCircuit: missing scatterMulti output id={id}")
            runScatterMultiOp _ctx numel inShapes bodies dstNumels inBufs outBufs cacheKey cacheRef
        }
    | Prim.reduceLastAxis rop inShape =>
      let D := inShape.numel
      let inId  := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 := hash ("circuit-reduce", reprStr rop, D)
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some inBuf, some outBuf =>
              runReduceOp _ctx rop D inBuf outBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing reduce buffer (in={inId} out={outId})")
        }
    | Prim.reduceLastAxisWithEpilogue rop reduceInShape epiShapes body =>
      let D := reduceInShape.numel
      let inId  := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      let epiIds : Array Nat :=
        (op.inputs.extract 1 op.inputs.size).map (·.id)
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-reduce-epi", reprStr rop, D, reprStr body,
              reprStr epiShapes.toList)
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some reduceBuf, some outBuf =>
              let mut epiBufs : Array (GPUBackend.Buf β) := #[]
              for id in epiIds do
                match lookup id with
                | some b => epiBufs := epiBufs.push b
                | none => throw (IO.userError s!"CompiledCircuit: missing epilogue input id={id}")
              runReduceWithEpilogueOp _ctx rop D body reduceBuf epiBufs outBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing reduce-epi buffer (in={inId} out={outId})")
        }
    | Prim.reduceScatterEpilogue rop reduceInShape epiShapes dstShape valueExpr addrExpr =>
      let D := reduceInShape.numel
      let dstSize := dstShape.numel
      let nInputs := op.inputs.size
      let reduceId := op.inputs[0]!.id
      let dstId := op.inputs[nInputs - 1]!.id
      let epiIds : Array Nat :=
        (op.inputs.extract 1 (nInputs - 1)).map (·.id)
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-reduce-scatter", reprStr rop, D, dstSize,
              reprStr valueExpr, reprStr addrExpr, reprStr epiShapes.toList)
      closures := closures.push
        { run := fun lookup => do
            match lookup reduceId, lookup dstId with
            | some reduceBuf, some dstBuf =>
              let mut epiBufs : Array (GPUBackend.Buf β) := #[]
              for id in epiIds do
                match lookup id with
                | some b => epiBufs := epiBufs.push b
                | none => throw (IO.userError s!"CompiledCircuit: missing reduceScatter epi input id={id}")
              runReduceScatterEpilogueOp _ctx rop D dstSize valueExpr addrExpr
                reduceBuf epiBufs dstBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing reduceScatter buffer")
        }
  let externalIds := state.externals.map (fun (tr, _) => tr.id)
  -- producedIds := tensor ids that are produced by some op's outputs
  let producedIds := state.ops.foldl (init := (#[] : Array Nat)) fun acc op =>
    op.outputs.foldl (init := acc) fun acc' tr => acc'.push tr.id
  return { ops := closures, externalIds, producedIds }

/-- Replay a compiled circuit.  `buffers` lists the (tensorId, buffer)
    pairs the caller wants to wire in for this invocation — externals
    AND produced-tensor outputs they want preserved. -/
def CompiledCircuit.replay [GPUBackend β]
    (cc : CompiledCircuit β)
    (buffers : List (Nat × GPUBackend.Buf β))
    : IO Unit := do
  -- Caller buffers win on id collisions (cons'd first); intermediate
  -- `baseBuffers` fill in the tensor ids the caller didn't supply.
  let combined := buffers ++ cc.baseBuffers
  let lookup (id : Nat) : Option (GPUBackend.Buf β) :=
    (combined.find? (fun e => e.1 == id)).map (·.2)
  for op in cc.ops do
    op.run lookup

/-- Build-or-replay helper.  On first call, runs `build` to produce the
    Circuit, compiles it, and stores it in `cacheRef`.  On subsequent
    calls, replays the cached CompiledCircuit directly.  This is the
    zero-overhead production path: the CircuitM builder runs at most once
    per unique cacheRef. -/
def runCached [GPUBackend β]
    (ctx : β)
    (cacheRef : IO.Ref (Option (CompiledCircuit β)))
    (build : CircuitM (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) Unit)
    (buffers : List (Nat × GPUBackend.Buf β))
    : IO Unit := do
  match ← cacheRef.get with
  | some cc => cc.replay buffers
  | none =>
    let (_, st) := CircuitM.run build
    let cc ← compileOnce ctx st
    cacheRef.set (some cc)
    cc.replay buffers

/-- Per-cacheKey store for CompiledCircuits — sibling of
    `Hesper.Models.Gemma4.KernelCacheRefs` but typed for our IR.  Lazily
    creates IO.Refs on demand. -/
structure CircuitCacheRefs (β : Type) [GPUBackend β] where
  store : IO.Ref (Array (UInt64 × IO.Ref (Option (CompiledCircuit β))))

def CircuitCacheRefs.create [GPUBackend β] : IO (CircuitCacheRefs β) := do
  let r ← IO.mkRef #[]
  return ⟨r⟩

def CircuitCacheRefs.getRef [GPUBackend β]
    (ccr : CircuitCacheRefs β) (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β))) := do
  let arr ← ccr.store.get
  match arr.find? (fun e => e.1 == key) with
  | some (_, r) => return r
  | none =>
    let r ← IO.mkRef none
    ccr.store.modify (·.push (key, r))
    return r

/-! ## Module-level fallback cache

Untyped global cache keyed by hash.  Used when callers can't (yet)
thread a `CircuitCacheRefs` through their signatures.  The stored
pointer is `unsafeCast`-erased; the caller must supply matching `β`
on each lookup (typically by including the backend tag in the key). -/
initialize circuitCacheGlobal : IO.Ref (Array (UInt64 × NonScalar)) ← IO.mkRef #[]

/-- Get-or-create a cache slot.  Returns an `IO.Ref (Option (CompiledCircuit β))`. -/
unsafe def getGlobalCircuitRefUnsafe [GPUBackend β] (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β))) := do
  let arr ← circuitCacheGlobal.get
  match arr.find? (fun e => e.1 == key) with
  | some (_, ptr) => return unsafeCast ptr
  | none =>
    let r : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
    circuitCacheGlobal.modify (·.push (key, unsafeCast r))
    return r

@[implemented_by getGlobalCircuitRefUnsafe]
opaque getGlobalCircuitRef [GPUBackend β] (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β)))

end Hesper.Circuit
