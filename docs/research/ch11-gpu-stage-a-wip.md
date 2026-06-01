# Ch11 GPU Stages A + B — LANDED

## Final state

- `Hesper/Training/MSE.lean` — verified `MSEOp` op:
  - `instance : Differentiable MSEOp (Array Float × Array Float) Float`
    (CPU spec; forward = mean of (p-y)², backward returns
    `(dPred, dTarget)` pair).
  - `forwardKernel` + `executeMSEForward` — scalar mean loss on GPU
    (one-workgroup tree reduce, mirrors CrossEntropy in `Loss.lean`).
  - `backwardKernel` + `executeMSEBackward` — per-row
    `dPred[i] = (2/N)(pred − y)` on GPU (one thread per row, guarded
    write).
- `Examples/Tutorial/CaliforniaHousingGPU.lean` — now uses
  `MSE.executeMSEForward` / `MSE.executeMSEBackward` instead of
  inline residual + reduce kernels.  Only kept inline: per-column
  `gradMatmulKernel` (Xᵀ @ dPred) and `sgdUpdateKernel`.
- `lakefile.lean` — `lean_exe «california-housing-gpu»` entry.

## Trajectory parity (1000 GD steps, lr=0.1, full batch)

| iter | CPU AD (fp64) | GPU Stage A (fp32) | GPU Stage B (fp32) |
|------|---------------|---------------------|---------------------|
| 0    | 5.55e10       | 5.55e10 ✓           | 5.55e10 ✓           |
| 100  | 8.50e9        | 8.51e9 ✓            | 8.51e9 ✓            |
| 200  | 6.78e9        | 6.79e9 ✓            | 6.79e9 ✓            |
| 1000 | 5.39e9        | 5.40e9 ✓            | 5.39e9 ✓            |

Bias `w[10]` lands at ~168395 (Stage A had a tail-write bug masking
this until the per-column workaround; Stage B inherits the fix).

## Bug recap (Stage A, still in `docs/research`)

`Hesper.WGSL.MatMul.naiveMatMulKernel` masks the value with
`select inBounds sum 0.0` but the `writeBuffer "c" idx` itself is
unguarded — small-N matmul (`M=1, N=11`) sees OOB writes from threads
11..255 clobbering `c[10]`. Workaround: per-column custom kernel.
Logged in `feedback_naive_matmul_oob_write.md`.

## Bug recap (Stage B)

`target` is a WGSL reserved keyword. Renamed the bind name to `tgt`
inside the MSE kernels. Forward shader compiled cleanly after that.

## Next (optional)

- Wire `executeMSEForward` / `executeMSEBackward` into the Ch11
  notebook (`docs/tutorial/md/Ch11_DataAnalysis.md`) alongside the
  CPU AD version as a "now run it on the GPU" coda.
- Container rebuild + browser smoke-test once user wants to ship.
