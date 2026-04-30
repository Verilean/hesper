import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.Transpile.CUDA.MMQEnv
import Hesper.Transpile.CUDA.AutoEnv
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 12: lower-coverage map across the parsed translation unit

We've reached **128/128 functions parse** in Phase 11. The next
question is: of those 128, how many actually `lowerStmt` cleanly with
a default `Env`? The body of each parsed function is fed to the
lowering pipeline with **`Env.empty`** (no struct fields, no inlines,
no buffer bindings). Then we categorise by error class:

  * `lower-ok`   — `lowerStmt` succeeds. The function body becomes a
                   ShaderM action. We also count the rendered WGSL
                   length so the user gets a feel for body size.
  * `body-empty` — Phase 11 body-recovery fallback fired (header-only
                   parse). Skipped from lowering because there's no
                   body to lower.
  * `lower-fail` — `lowerStmt` returned `Except.error e`. We capture
                   the error class (we group on the first 80 chars of
                   `e`) so the user sees which lowering gaps recur.

This is a research tool, not a pass/fail gate. The default-Env path
**will fail on most real kernels** because they need at minimum a
buffer binding for the `__restrict__` pointers. The point is to map
where the failures cluster — that tells us which env-resolver gaps
are the most worthwhile to close.

Usage:
```
lake exe transpile-cuda-lower-map [path1.cu ...]
```
With no args, runs on the four canonical llama.cpp source files.
-/
namespace Hesper.Transpile.CUDA.LowerMap

open Hesper.Transpile.CUDA Hesper.WGSL Hesper.WGSL.Monad

structure FnRow where
  name      : String
  bodyEmpty : Bool
  lowerErr  : Option String  -- none = OK
  shaderLen : Nat            -- only meaningful when lowerErr = none

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

def isEmptyBlock : CStmt → Bool
  | .block stmts => stmts.size == 0
  | _ => false

def lowerOne (mkEnv : CFunction → Env) (f : CFunction) : FnRow :=
  if isEmptyBlock f.body then
    { name := f.name, bodyEmpty := true, lowerErr := none, shaderLen := 0 }
  else
    let env := mkEnv f
    match lowerStmt env f.body with
    | .ok act =>
      let s := renderShader act
      { name := f.name, bodyEmpty := false, lowerErr := none,
        shaderLen := s.length }
    | .error e =>
      let trimmed := if e.length > 80 then e.take 80 ++ "…" else e
      { name := f.name, bodyEmpty := false, lowerErr := some trimmed,
        shaderLen := 0 }

structure Counts where
  ok        : Nat := 0
  bodyEmpty : Nat := 0
  fail      : Nat := 0

/-- Env-builder strategy. `.fixed` uses the same env for every fn;
    `.auto` synthesises a per-function env from the parsed TU. -/
inductive EnvMode where
  | fixed (env : Env)
  | auto

def runFile (mode : EnvMode) (path : System.FilePath) : IO Counts := do
  let present ← path.pathExists
  if !present then
    IO.println s!"  (skipped — file not found: {path})"
    return {}
  let src ← IO.FS.readFile path
  IO.println ""
  IO.println s!"═══ {path} ═══"
  let items := parseTranslationUnitStr src
  let mkEnv : CFunction → Env := match mode with
    | .fixed e => fun _ => e
    | .auto    => fun f => autoEnvFor items f
  let mut rows : Array FnRow := #[]
  for it in items do
    match it with
    | .function f => rows := rows.push (lowerOne mkEnv f)
    | _ => pure ()
  let mut nOk : Nat := 0
  let mut nEmpty : Nat := 0
  let mut nFail : Nat := 0
  -- Simple tally as (msg, count) pairs; updated linearly. Plenty fast
  -- for ≤200 functions and avoids Std.HashMap which needs a different
  -- import setup in this build.
  let mut errClasses : Array (String × Nat) := #[]
  let mut firstFails : Array FnRow := #[]
  for r in rows do
    if r.bodyEmpty then nEmpty := nEmpty + 1
    else match r.lowerErr with
    | none =>
      nOk := nOk + 1
    | some e =>
      nFail := nFail + 1
      let mut found := false
      let mut newClasses : Array (String × Nat) := #[]
      for (k, v) in errClasses do
        if k == e then
          newClasses := newClasses.push (k, v + 1)
          found := true
        else
          newClasses := newClasses.push (k, v)
      if !found then
        newClasses := newClasses.push (e, 1)
      errClasses := newClasses
      if firstFails.size < 6 then
        firstFails := firstFails.push r
  IO.println s!"  total parsed functions: {rows.size}"
  IO.println s!"  lower-ok    : {nOk}"
  IO.println s!"  body-empty  : {nEmpty}  (header-only fallback)"
  IO.println s!"  lower-fail  : {nFail}"
  if nOk > 0 then
    IO.println ""
    IO.println "  successfully lowered (name, WGSL char count):"
    let mut shown : Nat := 0
    for r in rows do
      if !r.bodyEmpty ∧ r.lowerErr.isNone ∧ shown < 30 then
        IO.println s!"    ✓ {r.name} ({r.shaderLen} chars)"
        shown := shown + 1
    if nOk > 30 then IO.println s!"    … ({nOk - 30} more)"
  if nFail > 0 then
    IO.println ""
    IO.println "  most common lower-fail messages:"
    let sorted := errClasses.qsort (fun a b => a.snd > b.snd)
    let upto := min 8 sorted.size
    for k in [0:upto] do
      let (msg, count) := sorted[k]!
      IO.println s!"    [{count}×] {msg}"
    IO.println ""
    IO.println "  first failing functions:"
    for r in firstFails do
      let e := r.lowerErr.getD "?"
      IO.println s!"    ✖ {r.name}: {e}"
  pure { ok := nOk, bodyEmpty := nEmpty, fail := nFail }

def main (args : List String) : IO Unit := do
  -- Env strategy flags (mutually exclusive; --auto wins if both set):
  --   --mmq   : MMQ default env (helper inlines + threadIdx/blockIdx
  --             builtins, fixed for every function)
  --   --auto  : 2-pass auto env. Per function, build buffers from its
  --             pointer params and inline-rewrite every const-return
  --             helper in the TU on top of mmqHelperInlines.
  let useAuto := args.contains "--auto"
  let useMMQ  := args.contains "--mmq"
  let pathArgs := args.filter (fun a => !a.startsWith "--")
  let paths : List System.FilePath := match pathArgs with
    | [] => [
        "llama.cpp/ggml/src/ggml-cuda/quantize.cu",
        "llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh",
        "llama.cpp/ggml/src/ggml-cuda/mmvq.cu",
        "llama.cpp/ggml/src/ggml-cuda/mmq.cuh"
      ]
    | xs => xs.map System.FilePath.mk
  let mode : EnvMode :=
    if useAuto then .auto
    else if useMMQ then .fixed mmqDefaultEnv
    else .fixed Env.empty
  let label :=
    if useAuto then "auto Env (2-pass)"
    else if useMMQ then "MMQ default"
    else "default Env"
  IO.println s!"═══ Phase 12: lower-coverage map ({label}) ═══"
  let mut totalOk := 0
  let mut totalEmpty := 0
  let mut totalFail := 0
  for p in paths do
    let c ← runFile mode p
    totalOk := totalOk + c.ok
    totalEmpty := totalEmpty + c.bodyEmpty
    totalFail := totalFail + c.fail
  IO.println ""
  IO.println "═══ totals across files ═══"
  IO.println s!"  lower-ok    : {totalOk}"
  IO.println s!"  body-empty  : {totalEmpty}"
  IO.println s!"  lower-fail  : {totalFail}"
  IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.LowerMap

def main (args : List String) : IO Unit :=
  Hesper.Transpile.CUDA.LowerMap.main args
