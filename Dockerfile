# Dockerfile for Hesper CI
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# - clang, cmake, ninja-build: For native compilation (Dawn)
# - curl, git: General utilities
# - pkg-config, libgl1-mesa-dev, libx11-dev: Graphics/Vulkan deps (if needed for build)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    libvulkan-dev \
    libglfw3-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libx11-xcb-dev \
    libxcb1-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Elan (Lean Version Manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable

# Update PATH
ENV PATH="/root/.elan/bin:${PATH}"

# Setup Workspace
WORKDIR /app

# Copy dependency files first (optimizing cache)
COPY lean-toolchain ./
COPY lakefile.lean ./
COPY lakefile.toml ./

# Install Lake dependencies (this downloads standard library)
# Note: We can't fully pre-build without access to local dependencies if any.
# Hesper seems to use Git dependencies mostly.

# Copy source code
COPY . .

# Build Hesper
# 1. Build Native Bridge (takes time, runs CMake)
RUN lake script run buildNative

# 2. Build Lean Library
RUN lake build Hesper

# 3. Build Tests (Optional but good for CI)
RUN lake build Hesper
