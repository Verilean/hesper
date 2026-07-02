#!/bin/bash
# dg_eval.sh — DiffusionGemma decode quality gate.
#
# Runs the decode over a fixed prompt suite with expected-keyword checks and prints
# per-prompt PASS/FAIL + the trimmed TEXT head + a total score + avg per-step ms.
#
# WHY: single-prompt text-equality is too fragile a gate for kernel changes — some prompts
# (e.g. Jupiter) sit at a logical near-tie where ANY numeric change (kernel swap, FMA
# contraction) flips the decoded text. The gate here is the TOTAL score: a config change is
# acceptable if its total ≥ baseline total (individual prompt flips are near-tie tolerance).
#
# Usage:
#   bash scripts/dg_eval.sh                 # current defaults
#   ENV="DG_DENSEDOWNRB=1" bash scripts/dg_eval.sh   # extra env for the config under test
#   MODEL=path/to.gguf bash scripts/dg_eval.sh
set -u
cd "$(dirname "$0")/.."
MODEL="${MODEL:-./diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
BIN=./.lake/build/bin/diffusiongemma-decode
ENV="${ENV:-}"
export HESPER_LOG_LEVEL=quiet

PROMPTS=(
  "The capital of France is"
  "The largest planet in our solar system is"
  "Water is made of"
  "The author of Romeo and Juliet is"
  "2+2="
  "The chemical symbol for gold is"
  "The opposite of hot is"
  "The first man on the moon was"
)
EXPECTS=(
  "[Pp]aris"
  "[Jj]upiter|gas giant"
  "[Hh]ydrogen|H2O|[Oo]xygen"
  "[Ss]hakespeare"
  "4|[Ff]our"
  "Au"
  "[Cc]old"
  "[Aa]rmstrong"
)

score=0
stepms_total=0
stepms_n=0
echo "=== dg_eval  ENV='${ENV}' ==="
for i in "${!PROMPTS[@]}"; do
  p="${PROMPTS[$i]}"; ex="${EXPECTS[$i]}"
  out=$(env $ENV timeout 400 "$BIN" "$MODEL" "$p" 2>&1)
  text=$(echo "$out" | grep -a "TEXT" | tail -1 | sed 's/^.*TEXT[^:]*: //')
  # avg emb+fwd over steps 2.. (skip warmup step 0/1)
  ms=$(echo "$out" | grep -aE "step [0-9]+:" | tail -n +3 | grep -oaE "emb\+fwd [0-9]+" | grep -oaE "[0-9]+" | awk '{s+=$1;n++} END{if(n>0) printf "%d", s/n; else print 0}')
  nsteps=$(echo "$out" | grep -acE "step [0-9]+:")
  if echo "$text" | grep -qaE "$ex"; then verdict="PASS"; score=$((score+1)); else verdict="FAIL"; fi
  [ -n "$ms" ] && [ "$ms" != "0" ] && { stepms_total=$((stepms_total+ms)); stepms_n=$((stepms_n+1)); }
  printf "%s [%2d steps, %4sms/step] %-45s → %.80s\n" "$verdict" "$nsteps" "$ms" "\"$p\"" "$text"
done
avg=0; [ $stepms_n -gt 0 ] && avg=$((stepms_total/stepms_n))
echo "=== TOTAL: $score/${#PROMPTS[@]}  avg emb+fwd ${avg}ms/step ==="
