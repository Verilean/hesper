# Hesper Tutorial — Docker image
#
# Bundles everything the tutorial needs:
#   - Lean 4 (matches `lean-toolchain`) + Lake
#   - Python + Jupyter Lab + jupytext + nbconvert
#   - xeus-lean kernel re-linked with Hesper + its native bridge so
#     `#eval`s that call into Dawn / Highway / CUDA resolve at runtime
#   - xlean-convert CLI for the .md→.ipynb pipeline
#   - Hesper itself, pre-built
#
# Architecture (mirrors Verilean/xeus-lean's Dockerfile.native-sparkle):
#   1. Build xeus-lean's `xlean` kernel against vanilla Lean.
#   2. Build Hesper.  Each Hesper module produces a per-module
#      `.c.o.export` next to its `.olean`; these are the C compiled
#      forms of every Lean function the interpreter might eval.
#   3. Bundle every `.c.o.export` into one static archive
#      `libhesper_olean.a`.
#   4. Re-link `xlean` with `--whole-archive libhesper_olean.a
#      stdLinkArgs` so every `@[extern "lean_hesper_*"]` symbol is
#      embedded in the kernel binary.  Now `#eval` from a notebook
#      cell can resolve them without `--load-dynlib` or per-module
#      shared libraries.
#   5. Patch the kernelspec's `LEAN_PATH` so `import Hesper` works.
#
# Build (from the repo root):
#   docker build -t hesper-tutorial:local -f docker/tutorial/Dockerfile .
#
# Run:
#   docker run --rm -p 8888:8888 hesper-tutorial:local
#   → open http://localhost:8888

# ─── Stage 1 ── base + system packages ─────────────────────────────
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PATH="/root/.elan/bin:/usr/local/bin:${PATH}"

# X11/Wayland/Vulkan dev libs are needed by Dawn's GLFW build path.
# Kept in sync with .github/workflows/ci.yml `linux-vulkan`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        cmake \
        ninja-build \
        pkg-config \
        python3 python3-pip python3-venv \
        libvulkan-dev vulkan-tools mesa-vulkan-drivers \
        libx11-dev libx11-xcb-dev libxrandr-dev libxinerama-dev \
        libxcursor-dev libxi-dev libxext-dev libxcb1-dev \
        libxkbcommon-dev libwayland-dev libgl1-mesa-dev \
        clang libc++-dev libc++abi-dev \
        nlohmann-json3-dev uuid-dev libssl-dev zstd \
    && rm -rf /var/lib/apt/lists/*

# jupyter_client drives smoke tests; jupyterlab is what the user opens.
RUN pip install --break-system-packages --no-cache-dir \
        jupyter_client jupyterlab jupytext nbconvert notebook

# ─── Stage 2 ── Lean toolchain via elan ────────────────────────────
FROM base AS lean

ARG LEAN_VERSION=v4.28.0
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
      | sh -s -- -y --default-toolchain ${LEAN_VERSION}

# ─── Stage 3 ── xeus-lean kernel + xlean-convert CLI ───────────────
FROM lean AS xeus-lean

WORKDIR /opt

ARG XEUS_LEAN_REV=main
# Force the layer cache to bust whenever the upstream xeus-lean HEAD
# advances; without this the COPY+clone would silently re-use stale
# binaries when the user rebuilds.
ADD https://api.github.com/repos/Verilean/xeus-lean/commits/${XEUS_LEAN_REV} \
    /tmp/xeus-lean-rev.json
RUN git clone https://github.com/Verilean/xeus-lean.git \
    && cd xeus-lean \
    && git checkout ${XEUS_LEAN_REV}

WORKDIR /opt/xeus-lean

# Build the C++ FFI bits (xeus, xeus-zmq, libzmq fetched via cmake).
RUN cmake -S . -B build-cmake \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++" && \
    sed -i 's/#define XEUS_KERNEL_PROTOCOL_VERSION_MINOR 3/#define XEUS_KERNEL_PROTOCOL_VERSION_MINOR 4/' \
        build-cmake/_deps/xeus-src/include/xeus/xeus.hpp && \
    grep -q 'XEUS_KERNEL_PROTOCOL_VERSION_MINOR 4' \
        build-cmake/_deps/xeus-src/include/xeus/xeus.hpp && \
    cmake --build build-cmake -j

# Linux glibc C23 shim (clang -> __isoc23_strtoull -> Lean's older glibc).
RUN leanc -c src/glibc_isoc23_compat.c -o build-cmake/glibc_isoc23_compat.o || true

# Vanilla xlean + xlean-convert. xlean is re-linked again in Stage 5
# with Hesper symbols statically baked in, but build it now so the
# xeus-lean stage's cache is reusable for non-Hesper notebooks too.
RUN lake build xlean xlean-convert

RUN install -m 0755 .lake/build/bin/xlean         /usr/local/bin/xlean && \
    install -m 0755 .lake/build/bin/xlean-convert /usr/local/bin/xlean-convert

RUN bash scripts/install-kernelspec.sh --user

# Duplicate the kernelspec under the name notebooks reference.
RUN cp -r /root/.local/share/jupyter/kernels/xlean \
          /root/.local/share/jupyter/kernels/xeus-lean && \
    python3 -c "import json,pathlib; \
p=pathlib.Path('/root/.local/share/jupyter/kernels/xeus-lean/kernel.json'); \
spec=json.loads(p.read_text()); \
spec['display_name']='Lean 4'; \
spec['language']='lean4'; \
p.write_text(json.dumps(spec,indent=2))"

# Auto-shutdown kernels when the notebook tab closes (avoids zombie
# kernels piling up after each chapter).
RUN mkdir -p /usr/local/share/jupyter/lab/settings && \
    python3 -c "import json,pathlib; \
p=pathlib.Path('/usr/local/share/jupyter/lab/settings/overrides.json'); \
p.write_text(json.dumps({'@jupyterlab/notebook-extension:tracker': {'kernelShutdown': True}}, indent=2))"

# ─── Stage 4 ── Hesper source + lake build ─────────────────────────
FROM xeus-lean AS hesper

WORKDIR /workspace/hesper

# Copy only what's needed to resolve dependencies and warm the cache
# before the full source COPY below, so a change to a Lean file
# doesn't bust the dep-fetch layer.
COPY lakefile.lean lake-manifest.json lean-toolchain ./

RUN set -eux; \
    cp lean-toolchain /tmp/lean-toolchain.pinned; \
    lake update; \
    cp /tmp/lean-toolchain.pinned lean-toolchain

# Native build inputs (Dawn / Highway / bridge sources):
COPY native ./native
COPY c_src  ./c_src

# Lean sources:
COPY Hesper      ./Hesper
COPY Hesper.lean ./Hesper.lean
COPY tools       ./tools
COPY Examples    ./Examples
COPY Tests       ./Tests

# Tutorial sources (Markdown master + build script):
COPY docs/tutorial ./docs/tutorial

# Build the library so each module's `.c.o.export` exists for the
# bundling step.  Native deps (Dawn / Highway / CUDA bridge) build
# the first time, then are cached.
RUN lake build Hesper

# ─── Stage 5 ── Re-link xlean against Hesper ──────────────────────
# Bundle every Hesper module's compiled `.c.o.export` into a single
# static archive `libhesper_olean.a`.  Exclude `Main.c.o.export`
# files: each carries a `lean_main` symbol that would clash with
# xlean's own `lean_main`.
RUN cd /workspace/hesper/.lake/build/ir && \
    find . -name "*.c.o.export" ! -name "Main.c.o.export" -print0 \
      | xargs -0 ar rcs /workspace/hesper/libhesper_olean.a && \
    echo "Bundled $(ar t /workspace/hesper/libhesper_olean.a | wc -l) objects into libhesper_olean.a"

# Re-link xlean with `--whole-archive libhesper_olean.a + the same
# native link line every Hesper exe uses (Dawn, Highway, CUDA bridge
# or stub, X11/Vulkan).  `XEUS_LEAN_EXTRA_LIBS` is xeus-lean's generic
# extension hook — anything in this whitespace-separated env var is
# appended to xlean's link line.
RUN cd /opt/xeus-lean && rm -f .lake/build/bin/xlean && \
    XEUS_LEAN_EXTRA_LIBS=" \
      -Wl,--whole-archive \
      /workspace/hesper/libhesper_olean.a \
      -Wl,--no-whole-archive \
      -Wl,--whole-archive \
      /workspace/hesper/.lake/build/native/libhesper_native.a \
      /workspace/hesper/.lake/build/dawn-build/src/dawn/libdawn_proc.a \
      /workspace/hesper/.lake/build/dawn-build/src/dawn/glfw/libdawn_glfw.a \
      -Wl,--no-whole-archive \
      -L/workspace/hesper/.lake/build/dawn-build/src/dawn -ldawn_proc \
      -L/workspace/hesper/.lake/build/dawn-install/lib64 -lwebgpu_dawn \
      -L/workspace/hesper/.lake/build/dawn-build/third_party/glfw3/src/src \
      -lglfw3 \
      /workspace/hesper/.lake/build/simd/libhesper_simd.a \
      /workspace/hesper/.lake/build/simd/_deps/highway-build/libhwy.a \
      /workspace/hesper/.lake/build/native/libhesper_cuda.a \
      -lstdc++ -lvulkan -lX11 -lX11-xcb -lxcb -lwayland-client -ldl -lpthread \
    " \
      lake build xlean

RUN install -m 0755 /opt/xeus-lean/.lake/build/bin/xlean /usr/local/bin/xlean

# Convert .md → .ipynb for every chapter.
RUN bash docs/tutorial/build-from-md.sh

# Wire Hesper's LEAN_PATH into the xeus-lean kernelspec so `import
# Hesper` works inside notebook cells.
RUN python3 - <<'PY'
import json, subprocess, pathlib

out = subprocess.check_output(['lake', 'env', 'env'], cwd='/workspace/hesper').decode()
hesper_lean_path = next(
    line.split('=', 1)[1] for line in out.splitlines()
    if line.startswith('LEAN_PATH=')
)
full_lean_path = '/opt/xeus-lean/.lake/build/lib/lean:' + hesper_lean_path
for spec_path in [
    '/root/.local/share/jupyter/kernels/xlean/kernel.json',
    '/root/.local/share/jupyter/kernels/xeus-lean/kernel.json',
]:
    p = pathlib.Path(spec_path)
    spec = json.loads(p.read_text())
    env = spec.setdefault('env', {})
    env['LEAN_PATH'] = full_lean_path
    p.write_text(json.dumps(spec, indent=2))
    print('Patched', spec_path)
print('LEAN_PATH:', full_lean_path)
PY

# ─── Default command ───────────────────────────────────────────────
EXPOSE 8888
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--MultiKernelManager.default_kernel_name=xeus-lean", \
     "--MappingKernelManager.default_kernel_name=xeus-lean", \
     "--notebook-dir=/workspace/hesper/docs/tutorial/Notebooks/Gen/notebooks"]
