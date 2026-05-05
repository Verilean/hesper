#!/usr/bin/env bash
# Hesper full regression suite — runs after any DSL/codegen/PTX change.
#
# Coverage:
#   1. Transpiler unit tests (Lex/Parse/Lower) — CPU only, fast
#   2. PTX codegen text emission (incl. WMMA) — CPU only
#   3. Low-level CUDA kernel tests on real GPU (dp4a/fma/bitlinear/Q6_K)
#   4. Gemma 4 layer-0 parity tests against real GGUF weights
#
# Total runtime: ~3-5 min on RTX 4070 Ti when builds are cached.
# Stops at first FAIL. Pass `--continue` to keep going past failures.
#
# Usage:
#   bash scripts/regression.sh             # stop at first FAIL
#   bash scripts/regression.sh --continue  # run all, summarize at end

set -u
cd "$(dirname "$0")/.."

CONTINUE=0
if [[ "${1:-}" == "--continue" ]]; then
  CONTINUE=1
fi

PASS=()
FAIL=()
SKIP=()

run_test() {
  local name="$1"; shift
  local cmd="$*"
  echo ""
  echo "═══════ $name ═══════"
  # Run cmd and capture exit code via PIPESTATUS so a non-zero exit
  # (e.g. "unknown executable" or any other lake/exec failure) is
  # detected even if no FAIL/error string is in the tail.
  eval "$cmd" 2>&1 | tail -20 | tee /tmp/_hesper_regression_last.log
  local rc="${PIPESTATUS[0]}"
  if [[ $rc -ne 0 ]] || grep -qE "FAIL|error: Lean exited|unknown executable" /tmp/_hesper_regression_last.log; then
    echo "✖ FAIL: $name (rc=$rc)"
    FAIL+=("$name")
    if [[ $CONTINUE -eq 0 ]]; then
      echo ""
      echo "Stopped at first failure. Run with --continue to keep going."
      summary
      exit 1
    fi
  else
    echo "✓ PASS: $name"
    PASS+=("$name")
  fi
}

summary() {
  echo ""
  echo "═══════════════════════════════════════════"
  echo "Regression summary"
  echo "═══════════════════════════════════════════"
  echo "  PASS: ${#PASS[@]}"
  echo "  FAIL: ${#FAIL[@]}"
  echo "  SKIP: ${#SKIP[@]}"
  if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    for t in "${FAIL[@]}"; do echo "  - $t"; done
  fi
}

echo "═══════════════════════════════════════════"
echo "Hesper full regression suite"
echo "═══════════════════════════════════════════"

# ─── 1. (Reserved — transpile suite removed 2026-05-05; see
#        feedback_transpile_abandoned.md.  CPU-only Lean unit tests
#        for any new pipeline replace this section.)

# ─── 2. PTX codegen text (CPU) ───────────────
run_test "wmma-ptx-text-test"  "lake exe wmma-ptx-text-test"
run_test "wmma-shaderm-test"   "lake exe wmma-shaderm-test"
run_test "ptx-codegen-test"    "lake exe ptx-codegen-test"
run_test "cuda-ptx-inst-test"  "lake exe cuda-ptx-inst-test"
run_test "test-subgroup-codegen" "lake exe test-subgroup-codegen"

# ─── 3. Low-level GPU kernel tests ───────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "── Skipping GPU tests (no nvidia-smi) ──"
  SKIP+=("(all GPU tests)")
else
  run_test "cuda-backend-test"      "lake exe cuda-backend-test"
  run_test "cuda-dp4a-test"         "lake exe cuda-dp4a-test"
  run_test "cuda-q6k-dp4a-test"     "lake exe cuda-q6k-dp4a-test"
  run_test "cuda-fma-f16x2-test"    "lake exe cuda-fma-f16x2-test"
  run_test "cuda-bitlinear-test"    "lake exe cuda-bitlinear-test"
  run_test "cuda-q6k-4warp-parity"  "lake exe cuda-q6k-4warp-parity"
  run_test "wmma-gpu-parity-test"   "lake exe wmma-gpu-parity-test"

  # ─── 3b. Dual-backend (WebGPU vs CUDA) parity ───
  # These exercise the WGSL codegen path in addition to the PTX path.
  run_test "cuda-bitnet-test"        "lake exe cuda-bitnet-test"
  run_test "cuda-bitnet-golden-test" "lake exe cuda-bitnet-golden-test"
  run_test "cuda-fa-golden-test"     "lake exe cuda-fa-golden-test"
  run_test "cuda-matmul-test"        "lake exe cuda-matmul-test"

  # WebGPU-only DSL/codegen tests
  run_test "test-wgsl-dsl"           "lake exe test-wgsl-dsl"

  # ─── 4. Gemma 4 layer-0 parity (real GGUF) ───
  if [[ -f "data/gemma-4-e4b-it-Q4_K_M.gguf" ]]; then
    run_test "gemma4-qproj-parity"     "lake exe gemma4-qproj-parity"
    run_test "gemma4-qkv-parity"       "lake exe gemma4-qkv-parity"
    run_test "gemma4-ffn-parity"       "lake exe gemma4-ffn-parity"
    run_test "gemma4-q4k-mmq-parity"   "lake exe gemma4-q4k-mmq-parity"
    run_test "gemma4-postffn-parity"   "lake exe gemma4-postffn-parity"
    run_test "gemma4-kv-parity"        "lake exe gemma4-kv-parity"
    run_test "gemma4-k-parity"         "lake exe gemma4-k-parity"
    run_test "gemma4-kv-multi-parity"  "lake exe gemma4-kv-multi-parity"
    run_test "gemma4-ropeq-parity"     "lake exe gemma4-ropeq-parity"
  else
    echo ""
    echo "── Skipping Gemma 4 parity (data/gemma-4-e4b-it-Q4_K_M.gguf not found) ──"
    SKIP+=("(gemma4 parity tests)")
  fi
fi

summary
[[ ${#FAIL[@]} -eq 0 ]]
