window.BENCHMARK_DATA = {
  "lastUpdate": 1771482397554,
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
      }
    ]
  }
}