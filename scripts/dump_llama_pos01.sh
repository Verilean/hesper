#!/usr/bin/env bash
# Dump llama.cpp intermediate tensors for both pos=0 and pos=1.
# Batch 0 (prompt eval) → /tmp/llama_dump/
# Batch 1 (first decode) → /tmp/llama_dump_pos1/
#
# Relies on the hesper_batch_idx counter in llama.cpp/common/debug.cpp.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
MODEL="$REPO_ROOT/data/gemma-4-e4b-it-Q4_K_M.gguf"

echo "[1/4] Cleaning dump directories..."
rm -rf /tmp/llama_dump /tmp/llama_dump_pos1
mkdir -p /tmp/llama_dump

echo "[2/4] Rebuilding llama-eval-callback..."
cd "$LLAMA_DIR/build"
cmake --build . --target llama-eval-callback -j 8 2>&1 | tail -5

echo "[3/4] Running llama-eval-callback (prompt='Hello', -n 2)..."
./bin/llama-eval-callback \
  -m "$MODEL" \
  -p "Hello" \
  -n 2 \
  --seed 42 \
  --temp 0 2>&1 | grep -E '\[DUMP\]|number of input tokens|generated token' | tail -20

echo "[4/4] Dumped file counts:"
echo "  pos=0: $(ls /tmp/llama_dump/ 2>/dev/null | wc -l) files"
echo "  pos=1: $(ls /tmp/llama_dump_pos1/ 2>/dev/null | wc -l) files"
