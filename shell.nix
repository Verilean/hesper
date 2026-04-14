{ pkgs ? import <nixpkgs> {config.cudaSupport=true;config.allowUnfree=true;} }:

pkgs.mkShell {
  name = "hesper-webgpu-dev-shell";

  # Packages to install in the environment
  buildInputs = with pkgs; [
    # Build tools
    pkg-config
    cmake
    ninja
    git
    curl

    # Lean 4 toolchain
    elan

    # C++ compiler and libraries
    clang
    llvmPackages.libcxx

    # WebGPU/Dawn dependencies
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    vulkan-tools
    libglvnd
    xorg.libX11
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi
    xorg.libxcb
    xorg.libXext
    wayland
    wayland-protocols

    # GLFW for windowing
    glfw3

    # Additional libraries
    nlohmann_json
    libuuid
    openssl

    # Python for utilities and notebooks
    (python3.withPackages (ps: with ps; [
      numpy
      matplotlib
      pyyaml
      pandas
      pip
      jupyter
      jupyterlab
    ]))

    # Node.js for WebAssembly tooling
    nodejs
    emscripten
    cudatoolkit
    cudaPackages.cuda_nsight
    cudaPackages.cuda_nvprof
    # Modern profilers (required for sm_80+; nvprof deprecated since CC 8.0)
    cudaPackages.nsight_systems     # `nsys` — system-wide timeline profiler
    cudaPackages.nsight_compute     # `ncu`  — kernel-level profiler
  ];

  # Environment variables
  shellHook = ''
    # Compiler configuration
    export CMAKE_C_COMPILER=clang
    export CMAKE_CXX_COMPILER=clang++
    export CMAKE_CXX_FLAGS="-stdlib=libc++"
    export CC=clang
    export CXX=clang++

    # Vulkan configuration for Nvidia
    export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
    export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json"

    # WebGPU/Dawn backend selection (Vulkan for Nvidia on Linux)
    export DAWN_BACKEND=vulkan

    # GPU feature selection (if needed)
    # export HESPER_GPU_FEATURES=subgroups,fp16

    # LD_LIBRARY_PATH for Vulkan and OpenGL (runtime)
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.libglvnd}/lib:${pkgs.wayland}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

    # LIBRARY_PATH for linker (leanc uses cc which needs this on NixOS)
    export LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libxcb}/lib:${pkgs.xorg.libXext}/lib:${pkgs.wayland}/lib:${pkgs.libglvnd}/lib:/run/opengl-driver/lib:$LIBRARY_PATH"

    # XDG runtime directory (for Wayland)
    export XDG_RUNTIME_DIR="''${XDG_RUNTIME_DIR:-/tmp}"

    echo "═════════════════════════════════════════════════════"
    echo "  Hesper WebGPU Development Environment (NixOS)"
    echo "═════════════════════════════════════════════════════"
    echo ""
    echo "GPU Information:"
    if command -v nvidia-smi &> /dev/null; then
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "  Could not query GPU"
    else
      echo "  nvidia-smi not available (install nvidia drivers)"
    fi
    echo ""
    echo "Vulkan Support:"
    if command -v vulkaninfo &> /dev/null; then
      echo "  ✓ Vulkan tools available"
      vulkaninfo --summary 2>/dev/null | grep -E "(Vulkan Instance Version|deviceName)" | head -3 || echo "  Vulkan query failed"
    else
      echo "  ✗ vulkaninfo not available"
    fi
    echo ""
    echo "Lean Toolchain:"
    echo "  $(lean --version 2>/dev/null || echo 'Not installed - run: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh')"
    echo ""
    echo "Build Commands:"
    echo "  lake run buildNative  # Build Dawn WebGPU native library"
    echo "  lake build            # Build Hesper"
    echo "  lake test-all         # Run test suite"
    echo ""
    echo "Environment Variables:"
    echo "  DAWN_BACKEND=$DAWN_BACKEND"
    echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
    echo "═════════════════════════════════════════════════════"
  '';
}
