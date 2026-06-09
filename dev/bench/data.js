window.BENCHMARK_DATA = {
  "lastUpdate": 1780968239403,
  "repoUrl": "https://github.com/Verilean/hesper",
  "entries": {
    "BitNet Inference Benchmark": [
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
        "date": 1771477974553,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 13.755356,
            "unit": "tokens/sec",
            "extra": "ms/token: 2206.051396"
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
        "date": 1771482429358,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 18.14947,
            "unit": "tokens/sec",
            "extra": "ms/token: 1484.162562"
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
        "date": 1774762151097,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 14.552593,
            "unit": "tokens/sec",
            "extra": "ms/token: 1568.817313"
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
        "date": 1778604933216,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 4.092437,
            "unit": "tokens/sec",
            "extra": "ms/token: 3268.819229"
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
        "date": 1778605721245,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 0.121684,
            "unit": "tokens/sec",
            "extra": "ms/token: 8508.951355"
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
        "date": 1778630832294,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 4.783097,
            "unit": "tokens/sec",
            "extra": "ms/token: 5002.488208"
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
        "date": 1778641571731,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 9.565431,
            "unit": "tokens/sec",
            "extra": "ms/token: 1845.669167"
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
        "date": 1778645422711,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 6.36941,
            "unit": "tokens/sec",
            "extra": "ms/token: 2091.452563"
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
        "date": 1778651365493,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 8.168063,
            "unit": "tokens/sec",
            "extra": "ms/token: 1869.284021"
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
        "date": 1778655311706,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 5.8969,
            "unit": "tokens/sec",
            "extra": "ms/token: 2383.883937"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junjihashimoto@users.noreply.github.com",
            "name": "junji hashimoto",
            "username": "junjihashimoto"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "09733e160748a6c48bb1656bd2f9c7d0b26d8ad3",
          "message": "Merge pull request #7 from Verilean/docs/sparkle-style-tutorial\n\nDocs/sparkle style tutorial",
          "timestamp": "2026-06-09T10:07:35+09:00",
          "tree_id": "25fc958ff7a26e8869f30ba94af95d022877aed4",
          "url": "https://github.com/Verilean/hesper/commit/09733e160748a6c48bb1656bd2f9c7d0b26d8ad3"
        },
        "date": 1780968237015,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "BitNet b1.58 2B Inference (macOS Metal)",
            "value": 12.588242,
            "unit": "tokens/sec",
            "extra": "ms/token: 1229.365166"
          }
        ]
      }
    ]
  }
}