#!/usr/bin/env bash
# build-wasm.sh — produce everything xeus-lean's wasm xlean kernel
# needs to link Hesper into a browser build.
#
# Two artifacts come out of this script, in a single staging directory:
#
#   <out-dir>/
#       Hesper.olean ... Hesper/**/*.{olean,olean.server,olean.private,ir,ilean}
#       lib/libhesper_wasm.a
#
# The `Hesper/` tree is the pure-Lean side (WGSL DSL + anything else
# we ask `lake build` for).  `lib/libhesper_wasm.a` is a static
# archive built from `native/bridge_wasm_stub.cpp +
# native/cuda_bridge_stub.cpp` (see `native/CMakeLists.txt`'s
# EMSCRIPTEN branch) and provides every `lean_hesper_*` extern as an
# IO.Error stub — enough to satisfy wasm-ld at xlean link time so the
# Lean interpreter can `import Hesper.WGSL.DSL` at run time.  The
# real WebGPU FFI port lands in M3 (see
# `docs/research/wasm-webgpu-plan.md`).
#
# Usage:
#   build-wasm.sh <out-staging-dir> [target ...]
#
# Defaults:
#   targets = Hesper.WGSL.DSL  Hesper.WGSL.Helpers  Hesper.WGSL.Templates
#             Hesper.WGSL.Shader  Hesper.WGSL.Kernel
#
# Environment:
#   LEAN_TOOLCHAIN_OVERRIDE   path to a `lean-toolchain` file that we
#                             copy on top of the repo's own toolchain
#                             before `lake build`.  Use this from
#                             xeus-lean's CI to force-pin Hesper to
#                             the kernel's Lean version.  Falls back
#                             to whatever `lean-toolchain` is already
#                             checked in if unset.
#   SKIP_LIB                  if set, skip the libhesper_wasm.a step
#                             (used by CI variants that only want the
#                             olean tree, e.g. for parity-only
#                             smoketests).
#   SKIP_OLEAN                if set, skip `lake build` (just produce
#                             the static archive).
#   LEAN_PREFIX               override `lean --print-prefix` for the
#                             emcc CMake invocation (needed if the
#                             toolchain is not on PATH).
#   BUILD_TYPE                CMake build type for libhesper_wasm.a
#                             (default Release).

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

OUT_DIR="$1"
shift
TARGETS=("${@:-Hesper.WGSL.DSL Hesper.WGSL.Helpers Hesper.WGSL.Templates Hesper.WGSL.Shader Hesper.WGSL.Kernel}")
# Word-split the single-element default into the array. If the user
# passed individual targets they're already split.
if [ "${#TARGETS[@]}" -eq 1 ]; then
    # shellcheck disable=SC2206 # intentional split on whitespace
    TARGETS=(${TARGETS[0]})
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

WASM_DIR="$(cd "$(dirname "$0")" && pwd)"        # native/wasm/
NATIVE_DIR="$(cd "$WASM_DIR/.." && pwd)"          # native/
HESPER_DIR="$(cd "$NATIVE_DIR/.." && pwd)"        # repo root

LAKEFILE_OVERRIDE="$WASM_DIR/lakefile-wasm.lean"
if [ ! -f "$LAKEFILE_OVERRIDE" ]; then
    echo "[hesper-wasm] FATAL: missing $LAKEFILE_OVERRIDE" >&2
    exit 1
fi

echo "[hesper-wasm] repo root  : $HESPER_DIR"
echo "[hesper-wasm] staging    : $OUT_DIR"
echo "[hesper-wasm] targets    : ${TARGETS[*]}"

# ----------------------------------------------------------------------------
# Lean build (olean tree)
# ----------------------------------------------------------------------------
if [ -z "${SKIP_OLEAN:-}" ]; then
    cd "$HESPER_DIR"

    # Save originals so we can restore even if `lake build` fails.
    cp -f lakefile.lean lakefile.lean.wasm-bak 2>/dev/null || true
    cp -f lakefile.toml lakefile.toml.wasm-bak 2>/dev/null || true
    cp -f lean-toolchain lean-toolchain.wasm-bak 2>/dev/null || true
    cp -f lake-manifest.json lake-manifest.json.wasm-bak 2>/dev/null || true

    restore() {
        cd "$HESPER_DIR"
        [ -f lakefile.lean.wasm-bak ] && mv -f lakefile.lean.wasm-bak lakefile.lean || rm -f lakefile.lean
        [ -f lakefile.toml.wasm-bak ] && mv -f lakefile.toml.wasm-bak lakefile.toml || rm -f lakefile.toml
        [ -f lean-toolchain.wasm-bak ] && mv -f lean-toolchain.wasm-bak lean-toolchain || true
        [ -f lake-manifest.json.wasm-bak ] && mv -f lake-manifest.json.wasm-bak lake-manifest.json || rm -f lake-manifest.json
    }
    trap restore EXIT

    cp -f "$LAKEFILE_OVERRIDE" "$HESPER_DIR/lakefile.lean"
    rm -f "$HESPER_DIR/lakefile.toml"
    rm -f "$HESPER_DIR/lake-manifest.json"
    rm -rf "$HESPER_DIR/.lake/packages"

    if [ -n "${LEAN_TOOLCHAIN_OVERRIDE:-}" ] && [ -f "${LEAN_TOOLCHAIN_OVERRIDE}" ]; then
        echo "[hesper-wasm] pinning Lean toolchain via $LEAN_TOOLCHAIN_OVERRIDE"
        cp -f "$LEAN_TOOLCHAIN_OVERRIDE" "$HESPER_DIR/lean-toolchain"
    fi

    echo "[hesper-wasm] lake build ${TARGETS[*]}"
    lake build "${TARGETS[@]}"

    LIB="$HESPER_DIR/.lake/build/lib/lean"
    if [ ! -d "$LIB" ]; then
        echo "[hesper-wasm] FATAL: $LIB missing — lake build produced no oleans" >&2
        exit 2
    fi

    copy_match() {
        local src="$1" rel
        [ -e "$src" ] || return 0
        rel="${src#$LIB/}"
        mkdir -p "$OUT_DIR/$(dirname "$rel")"
        cp -f "$src" "$OUT_DIR/$rel"
    }
    for ext in olean olean.server olean.private ir ilean; do
        [ -e "$LIB/Hesper.$ext" ] && copy_match "$LIB/Hesper.$ext"
        while IFS= read -r -d '' f; do
            copy_match "$f"
        done < <(find "$LIB/Hesper" -type f -name "*.$ext" -print0 2>/dev/null)
    done

    NUM=$(find "$OUT_DIR" -type f \( -name '*.olean*' -o -name '*.ir' -o -name '*.ilean' \) | wc -l)
    echo "[hesper-wasm] staged $NUM olean files"

    # Drop the restore trap before falling through to the C++ phase —
    # the rest of the script does not touch $HESPER_DIR's lakefile.
    trap - EXIT
    restore
fi

# ----------------------------------------------------------------------------
# libhesper_wasm.a (C++ FFI stubs)
# ----------------------------------------------------------------------------
if [ -z "${SKIP_LIB:-}" ]; then
    if ! command -v emcc >/dev/null 2>&1; then
        echo "[hesper-wasm] WARN: emcc not on PATH; skipping libhesper_wasm.a" >&2
        echo "[hesper-wasm]       set SKIP_LIB=1 to silence this warning" >&2
    else
        LEAN_PREFIX="${LEAN_PREFIX:-$(lean --print-prefix 2>/dev/null || true)}"
        if [ -z "${LEAN_PREFIX}" ] || [ ! -d "${LEAN_PREFIX}/include" ]; then
            echo "[hesper-wasm] FATAL: cannot locate Lean headers (set LEAN_PREFIX)" >&2
            exit 3
        fi
        echo "[hesper-wasm] lean prefix: ${LEAN_PREFIX}"

        BUILD_DIR="${BUILD_DIR:-${NATIVE_DIR}/build-wasm}"
        mkdir -p "$BUILD_DIR"
        emcmake cmake \
            -S "$NATIVE_DIR" \
            -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
            -DLEAN_INCLUDE_DIR="${LEAN_PREFIX}/include"
        emmake make -C "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

        if [ ! -f "$BUILD_DIR/libhesper_wasm.a" ]; then
            echo "[hesper-wasm] FATAL: $BUILD_DIR/libhesper_wasm.a not produced" >&2
            exit 4
        fi
        mkdir -p "$OUT_DIR/lib"
        cp -f "$BUILD_DIR/libhesper_wasm.a" "$OUT_DIR/lib/"
        echo "[hesper-wasm] staged $(ls -lh "$OUT_DIR/lib/libhesper_wasm.a" | awk '{print $5}') libhesper_wasm.a"

        # ----------------------------------------------------------------
        # Symbol export list for the wasm linker.
        # ----------------------------------------------------------------
        # wasm-ld + LTO drop every symbol nothing in main references, and
        # all of libhesper_wasm.a's entry points are interpreter-resolved
        # at run time — nothing in xlean's C++ code references them.  We
        # write the symbol list to $OUT_DIR/lib/hesper_exports.txt so the
        # downstream consumer (xeus-lean's CMakeLists.txt) can splat it
        # into emcc's `-sEXPORTED_FUNCTIONS=@...` argument verbatim.
        #
        # The set is the union of:
        #   * lean_hesper_* from bridge_wasm_stub.cpp:
        #       - HESPER_WASM_STUB(name)  → all stub entry points
        #       - LEAN_EXPORT ... name(   → the three hand-written helpers
        #   * lean_hesper_cuda_* from cuda_bridge_stub.cpp:
        #       - HESPER_CUDA_STUB(name)
        EXPORTS_TXT="$OUT_DIR/lib/hesper_exports.txt"
        {
            grep -hoE 'HESPER_WASM_STUB\([a-zA-Z_][a-zA-Z_0-9]*\)' \
                "$NATIVE_DIR/bridge_wasm_stub.cpp" \
                | sed 's/HESPER_WASM_STUB(\(.*\))/_\1/'
            grep -hoE 'LEAN_EXPORT [^(]*[ *][a-zA-Z_][a-zA-Z_0-9]*\(' \
                "$NATIVE_DIR/bridge_wasm_stub.cpp" \
                | sed -E 's/.*[ *]([a-zA-Z_][a-zA-Z_0-9]*)\(/_\1/'
            grep -hoE 'HESPER_CUDA_STUB\([a-zA-Z_][a-zA-Z_0-9]*\)' \
                "$NATIVE_DIR/cuda_bridge_stub.cpp" \
                | sed 's/HESPER_CUDA_STUB(\(.*\))/_\1/'
        } | sort -u > "$EXPORTS_TXT"
        N_SYMS=$(wc -l < "$EXPORTS_TXT")
        echo "[hesper-wasm] wrote $N_SYMS exported symbols → $(basename "$EXPORTS_TXT")"
    fi
fi

echo "[hesper-wasm] DONE — staging at $OUT_DIR"
