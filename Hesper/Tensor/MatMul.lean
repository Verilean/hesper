import Hesper.Tensor.Types

/-!
# Matrix Multiplication Kernels

GPU kernels for matrix multiplication.

**Modern Approach**: Use the WGSL DSL and `ComputeShader` type for type-safe shader construction.

**Example**: See `Examples/MainMatmul.lean` for production usage with the WGSL DSL.
-/

namespace Hesper.Tensor.MatMul

open Hesper.Tensor

-- Deprecated string-based shader generation functions have been removed.
-- Use the WGSL DSL instead (see Examples/MainMatmul.lean for reference).

end Hesper.Tensor.MatMul
