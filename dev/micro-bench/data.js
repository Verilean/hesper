window.BENCHMARK_DATA = {
  "lastUpdate": 1778605611564,
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
      }
    ]
  }
}