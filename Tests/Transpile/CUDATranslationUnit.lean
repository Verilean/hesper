import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse

/-! # Phase 11: full `.cu`/`.cuh` translation-unit transpile

Feeds an entire CUDA source file into the new `parseTranslationUnit`,
which walks the file top-to-bottom and reports per-item results:

  * `.function f`  — successfully parsed function (printed: name, attrs, params)
  * `.skipped k`   — top-level item we deliberately skip (struct, typedef,
                     `extern "C"`, namespace, `using`, top-level decl)
  * `.failed h e`  — looked like a function but parsing failed; we report
                     the header snippet and the error so we know what to fix.

The lexer was just extended to drop preprocessor directives
(`#include`, `#define`, `#pragma`, `#if/#endif`, ...) so the input
is the file verbatim — no manual stripping, no helper distillations.

Usage:
```
lake exe transpile-cuda-translation-unit                  # default file set
lake exe transpile-cuda-translation-unit <path-to-.cu>    # one specific file
```
-/
namespace Hesper.Transpile.CUDA.TranslationUnit

open Hesper.Transpile.CUDA

structure Counts where
  parsed : Nat := 0
  skipped : Nat := 0
  failed : Nat := 0

def summarise (items : Array TUItem) : IO Counts := do
  let mut nFn : Nat := 0
  let mut nSkip : Nat := 0
  let mut nFail : Nat := 0
  let mut firstFails : Array (String × String) := #[]
  let mut fnNames : Array String := #[]
  let mut skipExternC : Nat := 0
  let mut skipNamespace : Nat := 0
  let mut skipStruct : Nat := 0
  let mut skipTypedef : Nat := 0
  let mut skipUsing : Nat := 0
  let mut skipDecl : Nat := 0
  let mut skipUnknown : Nat := 0
  let mut skipOther : Nat := 0
  for it in items do
    match it with
    | .function f =>
      nFn := nFn + 1
      fnNames := fnNames.push f.name
    | .skipped k _ =>
      nSkip := nSkip + 1
      if k == "extern-c" ∨ k == "extern-c-open" then
        skipExternC := skipExternC + 1
      else if k == "namespace" ∨ k == "namespace-open" then
        skipNamespace := skipNamespace + 1
      else if k == "struct" then skipStruct := skipStruct + 1
      else if k == "typedef" then skipTypedef := skipTypedef + 1
      else if k == "using" then skipUsing := skipUsing + 1
      else if k == "decl" then skipDecl := skipDecl + 1
      else if k == "unknown" then skipUnknown := skipUnknown + 1
      else skipOther := skipOther + 1
    | .failed header err =>
      nFail := nFail + 1
      if firstFails.size < 8 then
        firstFails := firstFails.push (header, err)
  IO.println s!"  parsed functions : {nFn}"
  IO.println s!"  skipped items    : {nSkip}"
  IO.println s!"  failed functions : {nFail}"
  if nFn > 0 then
    IO.println ""
    IO.println s!"  function names ({fnNames.size}):"
    let upto := min 80 fnNames.size
    for k in [0:upto] do
      IO.println s!"    • {fnNames[k]!}"
    if fnNames.size > upto then
      IO.println s!"    … ({fnNames.size - upto} more)"
  if nSkip > 0 then
    IO.println ""
    IO.println "  skip-reason breakdown:"
    if skipExternC  > 0 then IO.println s!"    • extern-c  : {skipExternC}"
    if skipNamespace > 0 then IO.println s!"    • namespace : {skipNamespace}"
    if skipStruct   > 0 then IO.println s!"    • struct    : {skipStruct}"
    if skipTypedef  > 0 then IO.println s!"    • typedef   : {skipTypedef}"
    if skipUsing    > 0 then IO.println s!"    • using     : {skipUsing}"
    if skipDecl     > 0 then IO.println s!"    • decl      : {skipDecl}"
    if skipUnknown  > 0 then IO.println s!"    • unknown   : {skipUnknown}"
    if skipOther    > 0 then IO.println s!"    • other     : {skipOther}"
  if nFail > 0 then
    IO.println ""
    IO.println "  first parse-failures:"
    for hp in firstFails do
      let (h, e) := hp
      IO.println s!"    ✖ header: {h}"
      IO.println s!"      error : {e}"
  pure { parsed := nFn, skipped := nSkip, failed := nFail }

def runFile (path : System.FilePath) : IO Counts := do
  let present ← path.pathExists
  if !present then
    IO.println s!"  (skipped — file not found: {path})"
    return {}
  let src ← IO.FS.readFile path
  IO.println ""
  IO.println s!"═══ {path} ({src.length} chars) ═══"
  let items := parseTranslationUnitStr src
  IO.println s!"  total top-level items: {items.size}"
  summarise items

def main (args : List String) : IO Unit := do
  let paths : List System.FilePath := match args with
    | [] => [
        "llama.cpp/ggml/src/ggml-cuda/quantize.cu",
        "llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh",
        "llama.cpp/ggml/src/ggml-cuda/mmvq.cu",
        "llama.cpp/ggml/src/ggml-cuda/mmq.cuh"
      ]
    | xs => xs.map System.FilePath.mk
  IO.println "═══ Phase 11: full CUDA translation-unit transpile ═══"
  let mut totalParsed := 0
  let mut totalSkipped := 0
  let mut totalFailed := 0
  for p in paths do
    let c ← runFile p
    totalParsed := totalParsed + c.parsed
    totalSkipped := totalSkipped + c.skipped
    totalFailed := totalFailed + c.failed
  IO.println ""
  IO.println "═══ totals across files ═══"
  IO.println s!"  parsed functions : {totalParsed}"
  IO.println s!"  skipped items    : {totalSkipped}"
  IO.println s!"  failed functions : {totalFailed}"
  IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.TranslationUnit

def main (args : List String) : IO Unit :=
  Hesper.Transpile.CUDA.TranslationUnit.main args
