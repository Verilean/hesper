# 45 — Q6_K ncu root-cause: memory-bound, tail effect 50%

*Written 2026-04-23. ncu profile of hesper Q6_K ffn_down 4-row kernel.*

## Setup

```bash
sudo rmmod nvidia_uvm nvidia   # reload driver to pick up modprobe options
# NVreg_RestrictProfilingToAdminUsers=0 is already in /etc/modprobe.d/nixos.conf
# (setting was in place but module needed reload)

# NixOS cuda_nsight_compute wrapper hides the real binary; call it directly
NCU=/nix/store/.../cuda12.8-nsight_compute-2025.1.1.2/target/linux-desktop-glibc_2_11_3-x64/ncu

HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0 $NCU \
  --set detailed \
  --kernel-name regex:"k_7031743127946451" \
  --launch-skip 100 --launch-count 5 \
  -o /tmp/q6k_hesper_ncu -f \
  lake exe gemma4-llama-prefill-skeleton \
  data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 3
```

Grid = (640, 1, 1), block = (128, 1, 1), CC 8.9.

## Key findings

| metric                       | value           | implication                          |
|------------------------------|-----------------|--------------------------------------|
| **SoL bottleneck**           | Memory > Compute| memory-bound                         |
| **FP32 peak used**           | 1%              | compute idle; problem is not ALU     |
| **Theoretical occupancy**    | 66.7%           | shared memory is the limiter         |
| **Achieved occupancy**       | 52.5%           | warp scheduling overhead on top      |
| **Warps/scheduler**          | 8 / 12 max      | HW can do 50% more in flight         |
| **Tail effect**              | **50% impact**  | 1 full wave + 160-block partial wave |
| **Uncoalesced global**       | 11%             | ~243k/2.19M sectors wasted           |
| **Uncoalesced shared**       | 30%             | ~205k/672k wavefronts wasted         |
| **ALU utilisation**          | 33.5%           | well-utilised but not bottleneck     |

Hardware (RTX 4070 Ti) has 60 SMs at CC 8.9. Full wave of hesper's
4-row kernel ≈ 480 blocks.  Grid 640 = 480 full + 160 tail.  Those 160
tail blocks execute alone, using ~1/4 the SMs — ncu estimates
**50% of kernel runtime is spent in this tail wave**.

## What this implies for the remaining 1.10× gap

The PTX-level optimisations (doc 41-44) plateaued because ptxas had
already done most of the address-arithmetic collapsing the source-level
bind_var work.  The remaining gap is **not** in the inner-loop compute
or instruction count.  It's in **launch configuration** and
**global-memory access pattern**:

### 1. Tail effect (50% potential win)

Change grid shape so it divides evenly into SM waves.  Options:
- **1-row kernel** → grid = 2560, which is 2560/480 ≈ 5.3 waves.
  Matches llama.cpp's launch config.  Many more full waves, tail
  impact ≈ 5%.
- **8-row kernel** → grid = 320, one partial wave of 320 blocks.
  Better than current 160 but still partial.
- Keep 4-row but tweak block size to get 720-block-per-wave alignment
  (probably requires different SMEM / register mix).

The 1-row variant has worse L1 reuse per-block (no cross-row smem
sharing) but the 50% tail saving dominates.

### 2. Uncoalesced global (11%)

byte-granularity loads (u8/u16) introduced in doc 42 don't coalesce —
32 threads in a warp each request a different byte, HW serves these
as separate sectors.  llama.cpp reads 4 scale bytes via a single u32
load per warp, paying 1 sector for what hesper pays 4.

Fix: batch-read scales as u32, then bfe the byte out.  Trade 2 u8
loads for 1 u32 + 1 bfe, cutting sector count.

### 3. Uncoalesced shared (30%)

The Q8_1 input staging reads from smem.  30% uncoalesced wavefronts
suggests bank conflicts.  Likely cause: `s_input_q8` indexed by
`q8BlockIdx` × 9 elements — the `×9` stride hits bank conflicts on 32
banks.

Fix: pad smem layout to `×10` or `×8` (power-of-two) to break the 9-way
modular conflict.

## Priority (expected wall-clock win)

1. **Tail effect (1-row kernel)**: potential −50% → 0.66 ms/dec
2. **Uncoalesced global (batch scales as u32)**: potential −10% → 1.19 ms/dec
3. **Uncoalesced shared (padding)**: potential −5% → 1.26 ms/dec

Combined expected: 1.32 → ~0.5 ms/dec, passing llama.cpp's 1.20.

But: 1-row loses smem input sharing (each block re-reads 2.56KB),
which adds global traffic.  Need to actually measure instead of
extrapolating.

## ncu setup notes (for future runs)

- `/etc/modprobe.d/nixos.conf` already has `NVreg_RestrictProfilingToAdminUsers=0`
- After adding the option, either reboot or `sudo rmmod nvidia_uvm nvidia`
  then trigger a CUDA program to reload with the new option
- NixOS ncu wrapper script calls the real binary via `$APPDIR/target/...`;
  that path is read-only in the nix store, causing "Failed to add rules
  search path" warnings — these are non-fatal but hide details
- Workaround: invoke the real binary directly
  (`$STORE/.../target/linux-desktop-glibc_2_11_3-x64/ncu`)
- `--csv` output to stderr works even when stdout printing fails
- `--kernel-name regex:"k_HASH"` matches hesper's renamed kernels
