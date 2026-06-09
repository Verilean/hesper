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
# Install everything in a single RUN so the Mesa pin survives.
# `noble-updates` ships Mesa 25.2.8, which segfaults inside
# `dawn::native::Adapter::CreateDevice` on the very first `getDevice()`
# call (reproduced both via xeus-lean and via direct
# `lean --load-dynlib=…`).  24.0.5 (from `noble/main`) matches the
# Mesa 24.x series that the Ch04 GPU regression was verified against
# on the NixOS host.  Use an apt pin file *before* the install so the
# resolver picks the 24.0 line for every Mesa-runtime dep that gets
# pulled in transitively by `libgl1-mesa-dev` etc.
RUN printf 'Package: mesa-vulkan-drivers libgl1-mesa-dri libglapi-mesa libglx-mesa0 libegl-mesa0 libgbm1 mesa-libgallium libxatracker2\nPin: release a=noble-updates\nPin-Priority: -1\n' \
        > /etc/apt/preferences.d/pin-mesa-to-noble-main \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
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
    && apt-mark hold mesa-vulkan-drivers libgl1-mesa-dri \
    && dpkg -l mesa-vulkan-drivers | tail -1 \
    && vulkaninfo --summary 2>&1 | grep -E "deviceName|driverInfo" \
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

# Patch the xlean kernel entrypoint to read `LEAN_DYNLIB_PATH` and
# `Lean.loadDynlib` each entry before initialising the elaborator.
# This is what lets `#eval`-ed notebook cells resolve `@[extern]`
# symbols declared by downstream Lean libraries built with
# `precompileModules := true` — without it the elaborator can
# typecheck the call but the interpreter cannot run it.
#
# Meant to upstream to Verilean/xeus-lean once we've verified end-
# to-end in this image.  Until then the patch lives under
# `docker/tutorial/patches/`.
COPY docker/tutorial/patches/xlean-dynlib-path.patch /tmp/
# Idempotent: upstream may have already merged the patch. We only
# care about the post-condition (the LEAN_DYNLIB_PATH loader is
# present in src/XeusKernel.lean). `patch -N` is a no-op if every
# hunk is already applied; the trailing `grep` makes the layer fail
# loudly if neither path got us there.
RUN (patch -p1 -N --batch < /tmp/xlean-dynlib-path.patch || true) \
    && grep -q 'LEAN_DYNLIB_PATH' src/XeusKernel.lean

# Build xlean + xlean-convert.  xlean now honours `LEAN_DYNLIB_PATH`
# at kernel startup; downstream tutorial chapters can set that env
# var in the kernelspec to load the Hesper precompiled `.so`s.
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

# Native build inputs (Dawn / Highway / bridge sources) — copied
# BEFORE the Lean source tree so the slow CMake build below (Dawn
# 12-15min on first run) lives in its own Docker layer.  Editing a
# Lean file in `Hesper/` invalidates the *Lean* build layer but
# leaves this Dawn layer's cache untouched.
COPY native ./native
COPY c_src  ./c_src

# Pre-build the native dependencies (Dawn + WebGPU bridge + Highway
# SIMD + CUDA stub) in their own cache-friendly layer.  `lake build
# Hesper:nativeDeps` would be the precise target, but Lake doesn't
# expose `nativeDeps` as a top-level build target — instead invoke
# the `Hesper.Logging` module which depends on `nativeDeps` and is
# the smallest module Hesper ships (so this only builds the C side
# + one tiny Lean file).
#
# `HESPER_NO_CUDA=1` tells `stdLinkArgs` to skip `-lcuda` because the
# tutorial image doesn't ship the CUDA user-mode driver — only the
# stub `libhesper_cuda.a` is linked, so CUDA FFI calls resolve to
# error-returning stubs instead of being undefined symbols.
ENV HESPER_NO_CUDA=1

# Lean sources required to invoke a module build:
COPY Hesper/Logging.lean ./Hesper/Logging.lean

RUN lake build +Hesper.Logging

# Now copy the full Lean source tree.  Editing files here only
# invalidates this layer + the `lake build Hesper` layer below,
# never the Dawn / native bridge layer above.
COPY Hesper      ./Hesper
COPY Hesper.lean ./Hesper.lean
COPY tools       ./tools
COPY Examples    ./Examples
COPY Tests       ./Tests

# Tutorial sources (Markdown master + build script):
COPY docs/tutorial ./docs/tutorial

# Build the Hesper library.  `lean_lib «Hesper»` is configured with
# `precompileModules := true` + `moreLinkArgs := stdLinkArgs`, so this
# step produces:
#
#   - `.lake/build/lib/lean/Hesper/**/*.olean`  (typechecked modules)
#   - `.lake/build/lib/lean/Hesper_Hesper_*.so` (per-module dynlibs,
#     each statically linked against the C bridge + Dawn + Vulkan +
#     Highway artefacts).
#
# The `.so`s are what makes `#eval`-ing an `@[extern]` call inside a
# notebook cell resolve at runtime — the patched xlean kernel reads
# `LEAN_DYNLIB_PATH` below and `Lean.loadDynlib`s each entry before
# starting the elaborator.
#
# We also build `Hesper:shared` explicitly: it produces the root
# `.lake/build/lib/libHesper_Hesper.so`, which is what carries the
# `@[extern]` symbols declared directly in `Hesper.lean` (e.g.
# `Hesper.init`, `Hesper.getDevice`).  Without this, `lake build Hesper`
# alone only builds per-submodule `.so`s and leaves root extern
# resolution broken inside notebook cells.
RUN lake build Hesper Hesper:shared

# Convert .md → .ipynb for every chapter.
RUN bash docs/tutorial/build-from-md.sh

# Wire Hesper's LEAN_PATH + LEAN_DYNLIB_PATH into the xeus-lean
# kernelspec so `import Hesper.*` and `#eval`-ed FFI calls both work
# inside notebook cells.
RUN python3 - <<'PY'
import json, subprocess, pathlib, glob

out = subprocess.check_output(['lake', 'env', 'env'], cwd='/workspace/hesper').decode()
hesper_lean_path = next(
    line.split('=', 1)[1] for line in out.splitlines()
    if line.startswith('LEAN_PATH=')
)
full_lean_path = '/opt/xeus-lean/.lake/build/lib/lean:' + hesper_lean_path

# Collect per-module dynlibs in **dependency order**.  Lake emits one
# `*.setup.json` per built module, each carrying a `plugins` field that
# lists every transitive `.so` dependency in topological-sort order.
# Concatenating those lists (dedup-keep-first) yields a global load
# order that satisfies every consumer.  Sorting alphabetically does
# NOT work: e.g. `Hesper_Hesper_Backend_WebGPU.so` references
# `lp_Hesper_Hesper_WebGPU_writeBuffer___boxed` exported from
# `Hesper_Hesper_WebGPU_BufferOps.so`, but the alphabetic name
# `Backend_WebGPU` < `WebGPU_BufferOps` would load the consumer first
# and `dlopen` would fail at relocation time.
import os
seen = set()
dynlib_order = []
setup_dir = '/workspace/hesper/.lake/build/ir'
for setup_path in sorted(pathlib.Path(setup_dir).rglob('*.setup.json')):
    try:
        plugins = json.loads(setup_path.read_text()).get('plugins', [])
    except json.JSONDecodeError:
        continue
    for plug in plugins:
        if plug not in seen and os.path.exists(plug):
            seen.add(plug)
            dynlib_order.append(plug)
# Also add any dangling .so that no setup.json references — append at
# the end so they load after their dependencies (likely transitive).
for extra in sorted(glob.glob('/workspace/hesper/.lake/build/lib/lean/*.so')):
    if extra not in seen:
        seen.add(extra)
        dynlib_order.append(extra)
# The root `Hesper.lean` module also declares `@[extern]` symbols
# (e.g. `Hesper.init`).  Lake links that into
# `.lake/build/lib/libHesper_Hesper.so` (one dir up from the
# per-module dynlibs, with the `lib` prefix that the `:shared`
# facet uses).  Append it last because it depends on every
# submodule loaded above.
root_so = '/workspace/hesper/.lake/build/lib/libHesper_Hesper.so'
if os.path.exists(root_so) and root_so not in seen:
    dynlib_order.append(root_so)
lean_dynlib_path = ':'.join(dynlib_order)

for spec_path in [
    '/root/.local/share/jupyter/kernels/xlean/kernel.json',
    '/root/.local/share/jupyter/kernels/xeus-lean/kernel.json',
]:
    p = pathlib.Path(spec_path)
    spec = json.loads(p.read_text())
    env = spec.setdefault('env', {})
    env['LEAN_PATH'] = full_lean_path
    env['LEAN_DYNLIB_PATH'] = lean_dynlib_path
    p.write_text(json.dumps(spec, indent=2))
    print('Patched', spec_path)
print(f'LEAN_PATH (chars): {len(full_lean_path)}')
print(f'LEAN_DYNLIB_PATH: {len(dynlib_order)} dynlibs')
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
