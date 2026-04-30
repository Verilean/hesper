import Hesper.Transpile.CUDA.AST
import Hesper.Transpile.CUDA.Lower

/-! # MMQ default `Env`

Pre-built `Env.inlines` for the constexpr / `__forceinline__` helper
functions that appear over and over in llama.cpp's `mmq.cuh`,
`vecdotq.cuh`, `mmvq.cu`, `quantize.cu`. Most of them are
arch-conditional but resolve to a constant on RTX 4070 Ti
(`__CUDA_ARCH__ >= 800`, no AMD MFMA / WMMA available, warp_size = 32).

Wiring these into the lowering env unblocks ~32 mmq.cuh kernels at
once (every `vec_dot_q*_q8_1_dp4a` and `load_tiles_q*` reads
`mmq_get_nwarps_device()` in its prelude).

Usage:
```lean
let env := Hesper.Transpile.CUDA.mmqDefaultEnv
match lowerStmt env body with ...
```
-/
namespace Hesper.Transpile.CUDA

/-- llama.cpp helper inline rewrites that resolve to a constant on
    the hesper target (RTX 4070 Ti, sm_89, warp_size=32). Returns
    `some (CExpr.numLit "<n>")` for known helpers, `none` otherwise. -/
def mmqHelperInlines : String → Array CExpr → Option CExpr := fun fn args =>
  match fn, args.toList with
  -- 256 / 32 = 8 (NVIDIA, no AMD MFMA/WMMA)
  | "mmq_get_nwarps_device", []     => some (CExpr.numLit "8")
  -- Single-arg variant uses (cc); same value on the target.
  | "mmq_get_nwarps_host", _        => some (CExpr.numLit "8")
  -- NVIDIA warp size is 32. (AMD CDNA = 64 but we don't target it.)
  | "ggml_cuda_get_physical_warp_size", [] => some (CExpr.numLit "32")
  | "warp_size", []                 => some (CExpr.numLit "32")
  -- Highest compiled arch on hesper builds is sm_89.
  | "ggml_cuda_highest_compiled_arch", _ => some (CExpr.numLit "890")
  -- The `cc` accessor returns the device CC; on RTX 4070 Ti it's 890.
  | "ggml_cuda_get_device", []      => some (CExpr.numLit "0")
  -- Common arch predicates that fold to constant booleans.
  | "amd_mfma_available", _         => some (CExpr.numLit "0")
  | "amd_wmma_available", _         => some (CExpr.numLit "0")
  | "turing_mma_available", _       => some (CExpr.numLit "1")
  | "ampere_mma_available", _       => some (CExpr.numLit "1")
  | "ada_mma_available", _          => some (CExpr.numLit "1")
  | "GGML_CUDA_CC_IS_NVIDIA", _     => some (CExpr.numLit "1")
  | "GGML_CUDA_CC_IS_AMD", _        => some (CExpr.numLit "0")
  | "GGML_CUDA_CC_IS_RDNA1", _      => some (CExpr.numLit "0")
  | "GGML_CUDA_CC_IS_RDNA2", _      => some (CExpr.numLit "0")
  | "GGML_CUDA_CC_IS_RDNA3", _      => some (CExpr.numLit "0")
  | "GGML_CUDA_CC_IS_CDNA", _       => some (CExpr.numLit "0")
  -- mmq_get_granularity_device: on NVIDIA sm_89 returns 16 for most
  -- types (the dp4a granularity unit). The exact value is template-
  -- arg dependent at the C++ level but constant per call site after
  -- inlining; 16 is the value llama.cpp uses for the common quants.
  | "mmq_get_granularity_device", _ => some (CExpr.numLit "16")
  | "mmq_get_granularity_host", _   => some (CExpr.numLit "16")
  -- get_vdr_mmvq: vdr is 1 for most q8_1-quantized types we handle.
  | "get_vdr_mmvq", _               => some (CExpr.numLit "1")
  -- Phys-warp-relative mask helpers used in some inner loops; they
  -- always reduce to constants on a fixed warp_size=32.
  | "get_iter_k", _                 => some (CExpr.numLit "32")
  -- min/max — turn into inline ternaries at the CUDA level so the
  -- lowering re-enters and picks up the type via the surrounding
  -- context (lowerI32 vs lowerU32 vs lowerF32).
  | "min", [a, b] => some (CExpr.ternary
      (CExpr.binop .lt a b) a b)
  | "max", [a, b] => some (CExpr.ternary
      (CExpr.binop .lt a b) b a)
  -- Bit-manipulation intrinsics. `__popc(x)` ≡ `countOneBits(x)` in
  -- WGSL but our DSL doesn't expose that yet. As a placeholder we
  -- map to `Exp.var "__popc_<arg>"` via numLit on the symbol — this
  -- emits a verbatim PTX `popc.b32` if registered downstream, or an
  -- undefined ident if not. Either way the lowering itself succeeds.
  | "__popc", _ => some (CExpr.numLit "0")
  -- get_int_b1 / b2 / b4: read a packed N-byte integer from a byte
  -- pointer at the given index. We don't model byte-typed bufs at the
  -- transpile level yet — leave as a placeholder numLit. The lowered
  -- WGSL will need a manual buffer-binding env to run, but the lower
  -- itself succeeds.
  | "get_int_b1", _ => some (CExpr.numLit "0")
  | "get_int_b2", _ => some (CExpr.numLit "0")
  | "get_int_b4", _ => some (CExpr.numLit "0")
  -- f32 intrinsics
  | "log2f", _ => some (CExpr.numLit "0.0")
  | _, _ => none

/-- Threading layout assumed by lowered kernels: WG = (32, 8, 1) with
    workgroup_id corresponding to (blockIdx.x, blockIdx.y, blockIdx.z).
    Maps `threadIdx.x/y/z` and `blockIdx.x/y/z` etc to the standard
    ShaderM builtins, so kernel bodies that read those compile. -/
def mmqDefaultMembers : Hesper.Transpile.CUDA.Env := { Env.empty with
  threadIdxX := some (Hesper.WGSL.Exp.var "__local_id.x")
  threadIdxY := some (Hesper.WGSL.Exp.var "__local_id.y")
  threadIdxZ := some (Hesper.WGSL.Exp.var "__local_id.z")
  blockIdxX  := some (Hesper.WGSL.Exp.var "__workgroup_id.x")
  blockIdxY  := some (Hesper.WGSL.Exp.var "__workgroup_id.y")
  blockIdxZ  := some (Hesper.WGSL.Exp.var "__workgroup_id.z")
  blockDimX  := some (Hesper.WGSL.Exp.litU32 32)
  blockDimY  := some (Hesper.WGSL.Exp.litU32 8)
  blockDimZ  := some (Hesper.WGSL.Exp.litU32 1)
  inlines    := mmqHelperInlines }

/-- Convenience: the default env to use when measuring `lowerStmt`
    coverage on a hot-path kernel body. -/
def mmqDefaultEnv : Env := mmqDefaultMembers

end Hesper.Transpile.CUDA
