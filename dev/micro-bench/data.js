window.BENCHMARK_DATA = {
  "lastUpdate": 1771477941737,
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
      }
    ]
  }
}