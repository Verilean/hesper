#!/usr/bin/env bash
# Build llama-eval-callback (with our dump patch) and run it on layer 0,
# saving intermediate tensors to /tmp/llama_dump/

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
DUMP_DIR="/tmp/llama_dump"
MODEL="$REPO_ROOT/data/gemma-4-e4b-it-Q4_K_M.gguf"

echo "[1/4] Cleaning dump directory..."
rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"

echo "[2/4] Rebuilding llama-eval-callback..."
cd "$LLAMA_DIR/build"
cmake --build . --target llama-eval-callback -j 8 2>&1 | tail -20

echo "[3/4] Running llama-eval-callback on prompt 'Hello'..."
./bin/llama-eval-callback \
  -m "$MODEL" \
  -p "Hello" \
  -n 1 \
  --seed 42 \
  --temp 0 \
  --no-warmup 2>&1 | grep -E '\[DUMP\]|number of input tokens' | head -100

echo "[4/4] Listing dumped files..."
ls -la "$DUMP_DIR/" | head -50
echo ""
echo "Total dumped files: $(ls "$DUMP_DIR/" | wc -l)"
