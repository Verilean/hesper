window.BENCHMARK_DATA = {
  "lastUpdate": 1774762152145,
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
      }
    ]
  }
}