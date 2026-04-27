#!/usr/bin/env bash
# kernel_bench.sh — fixed before/after benchmark for hesper kernel changes.
#
# Use case: "I'm changing kernel X.  Did it move TPS / per-kernel time?"
#
# Single canonical command — DO NOT write a new ad-hoc script per session.
# The whole point of this script is to STOP discussing what to run.
#
# Output dir is /dev/shm/bench (tmpfs, RAM-backed).  nsys writes ~120 MB
# sqlite per run; on ext4 the fsync alone pins the host for ~30 s.  tmpfs
# makes the whole iteration finish in seconds.  This file is gone on reboot,
# which is fine — only the docs/llama-fusion-analysis/bench/<tag>-*.txt
# summaries are durable.
#
# Usage:
#   scripts/kernel_bench.sh before <tag>       # capture baseline
#   # ... edit kernel ...
#   lake build gemma4-cuda
#   scripts/kernel_bench.sh after  <tag>       # capture post-change
#   scripts/kernel_bench.sh diff   <tag>       # print before/after diff
#
# Env (rarely needed):
#   MODEL=data/gemma-4-e4b-it-Q4_K_M.gguf      (default)
#   PROMPT="Hello world how are you"           (default)
#   N_TOKENS=60                                (default)
#   GRAPHS=on|off                              (default: on — matches HEAD)
#   LLAMA_SQLITE=/dev/shm/bench/llama_cli.sqlite
#                                              (set once per llama.cpp build)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  sed -n '/^# Usage/,/^# Env/p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

PHASE="$1"
TAG="${2:-default}"

MODEL="${MODEL:-data/gemma-4-e4b-it-Q4_K_M.gguf}"
PROMPT="${PROMPT:-Hello world how are you}"
N_TOKENS="${N_TOKENS:-60}"
GRAPHS="${GRAPHS:-on}"
BENCH_DIR="${BENCH_DIR:-/dev/shm/bench}"
DOCS_DIR="${DOCS_DIR:-docs/llama-fusion-analysis/bench}"
LLAMA_SQLITE="${LLAMA_SQLITE:-${BENCH_DIR}/llama_cli.sqlite}"

mkdir -p "$BENCH_DIR" "$DOCS_DIR"

NSYS_REP="${BENCH_DIR}/${PHASE}_${TAG}.nsys-rep"
NSYS_DB="${BENCH_DIR}/${PHASE}_${TAG}.sqlite"
KERNEL_TXT="${DOCS_DIR}/${TAG}-${PHASE}.txt"
TPS_TXT="${DOCS_DIR}/${TAG}-tps-${PHASE}.txt"

if [[ "$GRAPHS" == "off" ]]; then
  ENV_VARS=(HESPER_USE_MMAP=1 HESPER_DP4A=1 HESPER_CUDA_GRAPHS=0)
else
  ENV_VARS=(HESPER_USE_MMAP=1 HESPER_DP4A=1)
fi

run_capture() {
  if [[ ! -f "$LLAMA_SQLITE" ]]; then
    echo "[kernel_bench] WARNING: \$LLAMA_SQLITE not found at $LLAMA_SQLITE"
    echo "    Capture llama-cli once with:"
    echo "      nsys profile -o ${BENCH_DIR}/llama_cli.nsys-rep --force-overwrite=true \\"
    echo "        --trace=cuda --cuda-graph-trace=node \\"
    echo "        ./llama.cpp/build/bin/llama-cli -m $MODEL -p \"$PROMPT\" -n $N_TOKENS --temp 0"
    echo "      nsys export -t sqlite ${BENCH_DIR}/llama_cli.nsys-rep -o $LLAMA_SQLITE"
    echo "    The kernel_compare table will skip the llama side this run."
  fi

  echo "[kernel_bench] $PHASE / $TAG / graphs=$GRAPHS"
  rm -f "$NSYS_REP" "$NSYS_DB"

  echo "[kernel_bench] nsys profile → $NSYS_REP"
  env "${ENV_VARS[@]}" nsys profile \
    -o "$NSYS_REP" --force-overwrite=true \
    --trace=cuda --cuda-graph-trace=node \
    .lake/build/bin/gemma4-cuda "$MODEL" "$PROMPT" "$N_TOKENS" \
    > "${BENCH_DIR}/${PHASE}_${TAG}.stdout" 2>&1

  echo "[kernel_bench] nsys export → $NSYS_DB"
  nsys export -t sqlite "$NSYS_REP" -o "$NSYS_DB" --force-overwrite=true \
    > /dev/null 2>&1

  if [[ -f "$LLAMA_SQLITE" ]]; then
    echo "[kernel_bench] kernel_compare_graphs.py → $KERNEL_TXT"
    python3 scripts/kernel_compare_graphs.py \
      "$NSYS_DB" "$N_TOKENS" \
      "$LLAMA_SQLITE" "$N_TOKENS" \
      > "$KERNEL_TXT"
  else
    echo "[kernel_bench] (no llama sqlite — skipping kernel_compare)" > "$KERNEL_TXT"
  fi

  echo "[kernel_bench] TPS 3-run avg → $TPS_TXT"
  : > "$TPS_TXT"
  for i in 1 2 3; do
    env "${ENV_VARS[@]}" .lake/build/bin/gemma4-cuda "$MODEL" "$PROMPT" "$N_TOKENS" \
      | grep -E "tokens/sec|TPS" >> "$TPS_TXT" || true
  done

  echo "[kernel_bench] done.  Summary:"
  echo "  per-kernel : $KERNEL_TXT"
  echo "  TPS        : $TPS_TXT"
  tail -3 "$TPS_TXT" || true
}

run_diff() {
  local before="${DOCS_DIR}/${TAG}-before.txt"
  local after="${DOCS_DIR}/${TAG}-after.txt"
  local tps_before="${DOCS_DIR}/${TAG}-tps-before.txt"
  local tps_after="${DOCS_DIR}/${TAG}-tps-after.txt"

  for f in "$before" "$after" "$tps_before" "$tps_after"; do
    if [[ ! -f "$f" ]]; then
      echo "[kernel_bench] missing $f — run 'before' and 'after' first"
      exit 1
    fi
  done

  echo "=== Per-kernel diff (before → after) ==="
  diff -u "$before" "$after" || true
  echo
  echo "=== TPS before ==="
  cat "$tps_before"
  echo "=== TPS after ==="
  cat "$tps_after"
}

case "$PHASE" in
  before|after) run_capture ;;
  diff)         run_diff ;;
  *)
    echo "[kernel_bench] unknown phase: $PHASE (expected before|after|diff)"
    exit 1
    ;;
esac
