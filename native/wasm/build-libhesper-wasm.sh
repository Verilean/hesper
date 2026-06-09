#!/usr/bin/env bash
# build-libhesper-wasm.sh — emcc build of native/CMakeLists.txt's
# EMSCRIPTEN branch.  Produces `libhesper_wasm.a` for downstream link
# into xeus-lean's wasm xlean kernel.
#
# Requires emcc on PATH and a Lean 4 toolchain (so `lean --print-prefix`
# resolves `<lean/lean.h>`).  See docs/research/wasm-webgpu-plan.md for
# the overall design; this script is the M1 deliverable.

set -euo pipefail

cd "$(dirname "$0")/.."         # cd into native/
NATIVE_DIR="$(pwd)"
BUILD_DIR="${BUILD_DIR:-${NATIVE_DIR}/build-wasm}"

echo "[hesper-wasm] native dir : ${NATIVE_DIR}"
echo "[hesper-wasm] build dir  : ${BUILD_DIR}"

if ! command -v emcc >/dev/null 2>&1; then
    echo "[hesper-wasm] ERROR: emcc not on PATH" >&2
    echo "[hesper-wasm] Install Emscripten (e.g. via pixi / nix-shell)" >&2
    exit 1
fi

LEAN_PREFIX="${LEAN_PREFIX:-$(lean --print-prefix 2>/dev/null || true)}"
if [ -z "${LEAN_PREFIX}" ] || [ ! -d "${LEAN_PREFIX}/include" ]; then
    echo "[hesper-wasm] ERROR: cannot locate Lean headers" >&2
    echo "[hesper-wasm] Set LEAN_PREFIX, or put 'lean' on PATH" >&2
    exit 1
fi
echo "[hesper-wasm] lean prefix: ${LEAN_PREFIX}"

mkdir -p "${BUILD_DIR}"

# We pass `-DEMSCRIPTEN=1` defensively even though emcmake already sets it,
# because some cmake versions don't propagate the platform define through
# until configure is finished.
emcmake cmake \
    -S "${NATIVE_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLEAN_INCLUDE_DIR="${LEAN_PREFIX}/include"

emmake make -C "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)"

if [ ! -f "${BUILD_DIR}/libhesper_wasm.a" ]; then
    echo "[hesper-wasm] FAIL: ${BUILD_DIR}/libhesper_wasm.a not produced" >&2
    exit 2
fi

echo "[hesper-wasm] OK: $(ls -lh "${BUILD_DIR}/libhesper_wasm.a" | awk '{print $5, $9}')"
echo "[hesper-wasm] symbols (sample):"
"${EMSDK}/upstream/emscripten/llvm-nm" "${BUILD_DIR}/libhesper_wasm.a" 2>/dev/null \
    | grep -E "lean_hesper_(init|get_device|create_buffer)" | head -5 \
    || echo "    (llvm-nm not in EMSDK env; skipping symbol listing)"
