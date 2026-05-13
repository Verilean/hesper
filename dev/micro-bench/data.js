window.BENCHMARK_DATA = {
  "lastUpdate": 1778655251508,
  "repoUrl": "https://github.com/Verilean/hesper",
  "entries": {
    "Hesper Kernel Micro-Benchmark": [
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "ab0e5b030766d2a1b1ad2f5c93fc8236f7a8bf00",
          "message": "feat: subgroup fallback kernels with shared-memory reduction + formal proof\n\nAdd runtime detection of WebGPU subgroup support and shared-memory\nfallback kernels for devices without subgroups (e.g. GitHub Actions CI).\n\n- bridge.cpp: graceful device creation (try subgroups, fallback without)\n- Device.lean: add deviceHasSubgroups FFI binding\n- Execute.lean: cached subgroup support flag, default extensions now empty\n- BitLinear.lean: 4 shared-memory fallback kernels with tree reduction,\n  auto-selection based on device capabilities\n- SubgroupFallbackTests.lean: LSpec tests verifying both kernel paths\n- ReductionEquiv.lean: formal Lean 4 proof that tree reduction equals\n  linear summation (treeReduce k f = rangeSum (2^k) f)",
          "timestamp": "2026-02-19T12:18:19+09:00",
          "tree_id": "44f2327c383c7777277374437ad0ca1d9a0d4c9c",
          "url": "https://github.com/Verilean/hesper/commit/ab0e5b030766d2a1b1ad2f5c93fc8236f7a8bf00"
        },
        "date": 1771477940112,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 4.729843,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.315108,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 0.4309,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 0.647208,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 0.842667,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.340413,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.275079,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 29.112896,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 1.100229,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "a688ce9848d6416b2e958b29a0a3b95518df7505",
          "message": "feat: HESPER_GPU_FEATURES env var for device feature selection\n\nAdd runtime feature level control via environment variable:\n  auto (default) - autodetect, try all features with graceful fallback\n  subgroup_matrix - force subgroups + subgroup matrix\n  subgroup        - force subgroups only, skip subgroup matrix\n  basic           - ShaderF16 only, no subgroups\n\nAlso fixes 3-tier device creation fallback: previously missing\nsubgroup matrix caused subgroups to be lost entirely.",
          "timestamp": "2026-02-19T15:24:29+09:00",
          "tree_id": "d828356132dad8ee9070ac4f1381cb7b83e34082",
          "url": "https://github.com/Verilean/hesper/commit/a688ce9848d6416b2e958b29a0a3b95518df7505"
        },
        "date": 1771482395962,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 3.233085,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.309533,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 0.469237,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.331921,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 0.815646,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.360979,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 1.921237,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 32.169083,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 5.466621,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "cc88e218ad1b951a0f4ec8da8d91b33cfdccd722",
          "message": "Fix lakefile.lean to build google/dawn on 'lake build'",
          "timestamp": "2026-03-29T14:17:46+09:00",
          "tree_id": "118c80f180fd93ccdbb0575c454bc6d69930c257",
          "url": "https://github.com/Verilean/hesper/commit/cc88e218ad1b951a0f4ec8da8d91b33cfdccd722"
        },
        "date": 1774762118022,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 4.191338,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.413642,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 0.503287,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 0.663458,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 0.780783,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.413754,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.356683,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 29.799033,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 1.009429,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "3a1f3ace94e54ef89aa68007f2968a657dd06f70",
          "message": "ci: fix macOS Metal and Windows D3D12 builds\n\n- native/CMakeLists.txt: gate the real CUDA bridge to Linux only; build a\n  stub library on macOS/Windows so Lean modules that re-export\n  Hesper.CUDA.FFI symbols still link without a CUDA SDK present.\n- native/cuda_bridge_stub.c: stubs for all 59 CUDA FFI entry points,\n  returning IO.userError. Never invoked on macOS/Windows; exist purely to\n  satisfy the linker.\n- lakefile.lean:\n    * always build hesper_cuda (real on Linux, stub elsewhere) and link\n      libhesper_cuda.a from stdLinkArgs on macOS.\n    * pass --config Release to cmake --build/--install so multi-config\n      generators (Visual Studio on Windows) produce Release artifacts.\n    * enable D3D12 backend on Windows in compileDawn.\n    * include both upstream Dawn glfw layouts as -L paths so either\n      version (glfw/src or glfw3/src/src) resolves -lglfw3.\n    * sync the nativeDeps Dawn pin to match the CI cache key (3f79f3a).\n    * use platform-appropriate static-lib filename (.lib on Windows) for\n      cache hit detection and tag the Dawn build hash with windows/osx/linux.",
          "timestamp": "2026-05-13T01:38:29+09:00",
          "tree_id": "8e48cb838bf4d67f342e75a901253ee085e9eb1d",
          "url": "https://github.com/Verilean/hesper/commit/3a1f3ace94e54ef89aa68007f2968a657dd06f70"
        },
        "date": 1778604873650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 2.164928,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.846392,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 1.013592,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.361529,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.564508,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.778062,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.804671,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 39.260258,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 2.884929,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "dc995d7bda77d6e168dbcca22e31ddf09eaca01e",
          "message": "ci: drop redundant Lean Build (Ubuntu) job\n\nThe job runs `lake build Hesper / Tests.WGSLDSLTests / Tests.NumericalTests`\nwhich is a strict subset of what Linux + Vulkan already does, but installs\nno system dependencies. Now that `lake build` triggers `nativeDeps` (Dawn +\nGLFW + SIMD), the bare Ubuntu runner fails when Dawn's GLFW tries to find\nX11 dev libs. Remove the job rather than duplicate the Vulkan install step.",
          "timestamp": "2026-05-13T02:03:14+09:00",
          "tree_id": "4eb824b1754a118f2c911a8eee9a2efd9a79d5b8",
          "url": "https://github.com/Verilean/hesper/commit/dc995d7bda77d6e168dbcca22e31ddf09eaca01e"
        },
        "date": 1778605610173,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 1.286632,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.816887,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 3.328029,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.43585,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.487858,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 1.455271,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.866392,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 35.738496,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 1.689242,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "a4a6f46ea2b72e84ef62ba9e0316760a4a3bed41",
          "message": "ci: unblock Linux X11 build and MSVC C++ compile of lean.h\n\n- ci.yml (linux-vulkan): add libx11-dev, libx11-xcb-dev, libxext-dev,\n  libxcb1-dev, libxkbcommon-dev, libwayland-dev. Dawn's GLFW pulls\n  X11/Xlib-xcb.h transitively, which only ships with libx11-xcb-dev.\n- native/bridge.cpp, native/glfw_bridge.cpp: shim `_Noreturn` to\n  `__declspec(noreturn)` before including <lean/lean.h> on MSVC. Lean's\n  header expands LEAN_NORETURN to `_Noreturn` unconditionally on _MSC_VER,\n  but MSVC only recognises `_Noreturn` as a C-language keyword — in C++\n  mode it is undefined, so the Lean header fails to parse.",
          "timestamp": "2026-05-13T09:02:49+09:00",
          "tree_id": "3d02e12542b4db96cb272217e1f3dd12ec7037c3",
          "url": "https://github.com/Verilean/hesper/commit/a4a6f46ea2b72e84ef62ba9e0316760a4a3bed41"
        },
        "date": 1778630770886,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 1.136342,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 3.202233,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 2.127767,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.639625,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.531904,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.614537,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.659917,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 39.412533,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 2.483462,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "4a0592f13fc989330e9541ec2c0703cf5cedd33b",
          "message": "native: auto-detect CUDA, bump to C++20, fix MSVC headers\n\n- native/CMakeLists.txt:\n    * Auto-detect CUDA Toolkit instead of hard-requiring it. On Linux\n      hosts without CUDA (e.g. the CI Vulkan runner) fall back to the\n      stub bridge so cmake configure succeeds. macOS and Windows still\n      always use the stub. Write a marker file (hesper_cuda.found) so\n      downstream tooling can later condition link flags on the decision.\n    * Bump CMAKE_CXX_STANDARD to 20. bridge.cpp uses designated\n      initializers, which are a non-standard extension in C++17 that\n      MSVC rejects (`/std:c++20` required).\n- native/bridge.cpp: explicitly include <array>. MSVC's standard\n  headers do not transitively pull it in, unlike libstdc++/libc++.",
          "timestamp": "2026-05-13T11:00:38+09:00",
          "tree_id": "bf79218f34b7b7c643b30f4a4b85095ee7dedde8",
          "url": "https://github.com/Verilean/hesper/commit/4a0592f13fc989330e9541ec2c0703cf5cedd33b"
        },
        "date": 1778641509874,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 2.716019,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.656579,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 0.796196,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 0.867204,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.227171,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.673412,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.845804,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 39.226708,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 6.103254,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "b757b3cd01dae316f34a99194beb70a7a6b1f1a7",
          "message": "native: enable C11 for the CUDA stub on MSVC\n\nMSVC defaults to C89 for .c sources and refuses to compile\n<vcruntime_c11_stdatomic.h> (pulled in transitively from <lean/lean.h>):\n  error C1189: \"C atomics require C11 or later\"\n\nSet C_STANDARD=11 on the hesper_cuda stub target to add /std:c11.",
          "timestamp": "2026-05-13T12:45:46+09:00",
          "tree_id": "9f42360f4e5e0681bb36b61331a90ea9490c41be",
          "url": "https://github.com/Verilean/hesper/commit/b757b3cd01dae316f34a99194beb70a7a6b1f1a7"
        },
        "date": 1778645354192,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 1.950331,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 1.165712,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 1.22005,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.09825,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 2.056767,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.669829,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.769179,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 35.5789,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 6.388325,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "90bbcbc135d2c6026e151c18f6e008f460c83660",
          "message": "native: compile the CUDA stub as C++ instead of C\n\nMSVC defaults to refusing <stdatomic.h> in C mode even with /std:c11 —\nit requires an additional experimental flag (/experimental:c11atomics):\n  error C1189: \"C atomic support is not enabled\"\n\n<lean/lean.h> pulls in <stdatomic.h> transitively, so the C stub cannot\nbe compiled by MSVC without depending on an experimental flag. Compile\nthe stub as C++ instead; C++ has standard atomics and no extra flags.\n\nThe zero-arg `name()` signatures still satisfy callers from Lean-\ngenerated C with arbitrary arguments: under `extern \"C\"` the linker\nmatches by symbol name only, and the platform ABI delivers the unused\narguments into registers/stack the stub never reads.",
          "timestamp": "2026-05-13T14:44:50+09:00",
          "tree_id": "e994c7e8341c582bfdd08be24a23d12690562280",
          "url": "https://github.com/Verilean/hesper/commit/90bbcbc135d2c6026e151c18f6e008f460c83660"
        },
        "date": 1778651297406,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 2.287941,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.987704,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 0.910963,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.190042,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.2461,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.755392,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.619929,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 36.527338,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 2.006204,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "junji.hashimoto@gree.net",
            "name": "Junji Hashimoto",
            "username": "junjihashimoto"
          },
          "distinct": true,
          "id": "eee64585ce805525fc69652ee4f1a1a90f54d79d",
          "message": "simd: shim _Noreturn for MSVC C++ before <lean/lean.h>\n\nSame fix as the native bridge: Lean's header expands LEAN_NORETURN to\n`_Noreturn`, which MSVC only recognises in C mode. Apply the shim in\nboth simd_ops_highway.cpp and simd_ops_highway.h so any consumer of the\nSIMD code that pulls in lean.h transitively also gets the substitute.",
          "timestamp": "2026-05-13T15:50:26+09:00",
          "tree_id": "34a42ae8407ed57297b5d0a777810855d044361b",
          "url": "https://github.com/Verilean/hesper/commit/eee64585ce805525fc69652ee4f1a1a90f54d79d"
        },
        "date": 1778655249523,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Projected TPS",
            "value": 1.864908,
            "unit": "tokens/sec"
          },
          {
            "name": "RMSNorm (2560)",
            "value": 0.994817,
            "unit": "ms"
          },
          {
            "name": "BitLinear Q (2560->2560)",
            "value": 1.436775,
            "unit": "ms"
          },
          {
            "name": "BitLinear Gate (2560->6912)",
            "value": 1.497867,
            "unit": "ms"
          },
          {
            "name": "BitLinear Down (6912->2560)",
            "value": 1.223129,
            "unit": "ms"
          },
          {
            "name": "Elementwise Add (2560)",
            "value": 0.974313,
            "unit": "ms"
          },
          {
            "name": "ReLU-Sqr-Mul (6912)",
            "value": 0.738192,
            "unit": "ms"
          },
          {
            "name": "MatMul LM Head",
            "value": 37.258108,
            "unit": "ms"
          },
          {
            "name": "GPU Argmax (128256)",
            "value": 2.411708,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}