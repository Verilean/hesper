#!/usr/bin/env bash
# perf_compare.sh — nsys CUDA comparison: hesper vs llama-cli
#                  Decompose into load / prefill / decode + host overhead + GPU idle
#
# Per engine, prints three tables:
#   1. CUDA API call counts       — load / prefill-per-tok / decode-per-tok
#   2. Host time in CUDA API      — how much host wall is spent inside the driver
#   3. Decode host-time breakdown — what the host does while the GPU is "idle"
#                                   (cuMemcpyDtoH sync, cuStreamSync, etc.)
#
# Linear regression from 3 nsys runs to subtract one-time `load`:
#   (P=1,     G=1)  →  load + 1*prefill + 1*decode
#   (P=Plong, G=1)  →  load + Plong*prefill + 1*decode
#   (P=Plong, G=N)  →  load + Plong*prefill + N*decode
#
# Does NOT advocate Graphs ON.  Default is GRAPHS=off; flip to `on` only when
# explicitly requested.
#
# llama-bench is **excluded by default**.  Why:
#   tools/llama-bench/llama-bench.cpp:test_gen() picks the next token with
#       token = std::rand() % n_vocab;
#   i.e. it skips real sampling.  The greedy argmax kernel + DtoH readback
#   overhead is therefore NOT measured, which underestimates the host cost
#   relative to a real inference loop.  llama-bench numbers should not be
#   compared apples-to-apples with hesper's greedy path.  Pass `bench`
#   explicitly when you want it as a forward-only baseline.
#
# Usage:
#   scripts/perf_compare.sh [hesper|cli|both|bench|all]   (default: both)
#     hesper : hesper only
#     cli    : llama-cli (real --temp 0 sampling) only
#     both   : hesper + cli (recommended)
#     bench  : llama-bench (no sampling — reference only)
#     all    : everything
#
# Env:
#   MODEL=path/to/model.gguf       (default: data/gemma-4-e4b-it-Q4_K_M.gguf)
#   PROMPT_SHORT="Hi"
#   PROMPT_LONG="Hello world how are you"
#   N_TOKENS=30                    (decode count)
#   P_SHORT_TOK=1, P_LONG_TOK=6    (approx token counts of the strings, used
#                                    for the linear regression and for
#                                    llama-bench's integer prompt-length flag)
#   GRAPHS=off|on                  (default: off)
#   OUT_DIR=/tmp/perf_compare
#   HESPER_BIN=.lake/build/bin/gemma4-cuda
#   LLAMA_DIR=./llama.cpp/build/bin

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WHICH="${1:-both}"

MODEL="${MODEL:-data/gemma-4-e4b-it-Q4_K_M.gguf}"
PROMPT_SHORT="${PROMPT_SHORT:-Hi}"
PROMPT_LONG="${PROMPT_LONG:-Hello world how are you}"
N_TOKENS="${N_TOKENS:-30}"
P_SHORT_TOK="${P_SHORT_TOK:-1}"
P_LONG_TOK="${P_LONG_TOK:-6}"

GRAPHS="${GRAPHS:-off}"
# Default to tmpfs.  nsys writes a ~120 MB sqlite per llama-cli run during
# its post-process pass; on a regular ext4 disk the fsync stage is the
# dominant cost and can hold the host pinned for tens of seconds after the
# GPU stopped.  /dev/shm is a tmpfs (RAM-backed) on Linux, so writes never
# touch the disk.  Override with OUT_DIR=/tmp/foo if you want persistence.
OUT_DIR="${OUT_DIR:-/dev/shm/perf_compare}"
HESPER_BIN="${HESPER_BIN:-.lake/build/bin/gemma4-cuda}"
LLAMA_DIR="${LLAMA_DIR:-./llama.cpp/build/bin}"

mkdir -p "$OUT_DIR"

# nsys's `--start-agent` daemons survive a SIGKILL of the launcher process
# (different process group), and a previous run that was timeout-killed can
# leave 5+ agents running at >90 % CPU for tens of minutes.  Those orphans
# steal cycles from the very workload we are profiling, so kill them up-front
# and again on exit.
cleanup_nsys_agents() {
  pkill -9 -f 'nsys --start-agent' 2>/dev/null || true
  pkill -9 -f 'nsys-tee'           2>/dev/null || true
  pkill -9 -f 'nsys-launcher'      2>/dev/null || true
}
cleanup_nsys_agents
trap cleanup_nsys_agents EXIT INT TERM

if [[ "$GRAPHS" == "off" ]]; then
  GRAPHS_ENV=(GGML_CUDA_DISABLE_GRAPHS=1)
else
  GRAPHS_ENV=()
fi

run_nsys() {
  # $1=tag, then command...
  local tag="$1"; shift
  local rep="$OUT_DIR/$tag.nsys-rep"
  local cap="${NSYS_TIMEOUT:-120}"
  echo "=== running $tag (nsys, ${cap}s cap) ==="
  # Hard cap so a stuck workload (e.g. an older llama-cli without -no-cnv
  # falling into chat mode) cannot freeze the whole experiment.  By the time
  # nsys reports ~95 % its `.nsys-rep` is already on disk, so even a forced
  # kill from `timeout` typically yields a usable report.
  # Three speedups stacked here:
  #  (a) `--stats=false` skips the auto post-process pass that imports the
  #      trace into sqlite + computes summary tables on every run.  The
  #      sqlite import is the dominant cost for high-event runs (llama-cli's
  #      decode hits the driver thousands of times per token); the GPU has
  #      long stopped but nsys keeps the host pinned for tens of seconds.
  #      We call `nsys stats` later in extract_*() ourselves on demand.
  #  (b) `--export=none` likewise skips the `.sqlite` materialisation.
  #  (c) When NSYS_DELAY / NSYS_DURATION are set, the trace only covers the
  #      steady-state decode window (load + prefill are skipped).  This
  #      shrinks the .nsys-rep by ~10× and is what we actually care about
  #      when comparing host overhead per token.
  # Only apply delay/duration to *long*-decode cases (tag ends in `gN`).
  # The `p1g1` and `pLg1` cases generate 1 token each and finish before any
  # NSYS_DELAY would expire, leaving us with an empty trace.
  local extra=()
  if [[ "$tag" == *gN ]]; then
    if [[ -n "${NSYS_DELAY:-}" ]];    then extra+=(--delay="$NSYS_DELAY"); fi
    if [[ -n "${NSYS_DURATION:-}" ]]; then extra+=(--duration="$NSYS_DURATION"); fi
  fi
  timeout --signal=KILL "$cap" \
    env "${GRAPHS_ENV[@]}" nsys profile --trace=cuda --sample=none \
      --stats=false "${extra[@]}" \
      -o "$OUT_DIR/$tag" --force-overwrite=true \
      "$@" </dev/null > "$OUT_DIR/$tag.stdout" 2>&1
  if [[ ! -f "$rep" ]]; then
    echo "  ! nsys failed, see $OUT_DIR/$tag.stdout"
    return 1
  fi
  return 0
}

# perf record on the same workload so we can see *where in the host process*
# CPU cycles go.  paranoid=2 systems can still sample user-space, which is
# all we need (kernel time is already attributed by nsys' driver-API rows).
# Output: $OUT_DIR/$tag.perf.txt with the top-30 user-space symbols by self %.
run_perf() {
  local tag="$1"; shift
  local pdata="$OUT_DIR/$tag.perf.data"
  local prpt="$OUT_DIR/$tag.perf.txt"
  local cap="${PERF_TIMEOUT:-120}"
  echo "=== running $tag (perf, ${cap}s cap) ==="
  timeout --signal=KILL "$cap" \
    env "${GRAPHS_ENV[@]}" \
      perf record -e cycles:u -F 999 --call-graph=fp -q \
        -o "$pdata" -- "$@" </dev/null > "$OUT_DIR/$tag.perf.stdout" 2>&1
  if [[ ! -f "$pdata" ]]; then
    echo "  ! perf record failed, see $OUT_DIR/$tag.perf.stdout"
    return 1
  fi
  # Self-time top symbols (children sort would mix in callees we already see
  # in nsys CUDA-API rows).  Strip dso to keep the column under 80 chars.
  perf report -i "$pdata" --stdio --no-children --sort=symbol \
      --percent-limit=0.5 2>/dev/null \
    | awk '/^#/ || NF<3 {next} {print}' \
    | head -50 > "$prpt"
  return 0
}

extract_api() {
  # Print one line per API row: name|calls|time_ns
  local tag="$1"
  local rep="$OUT_DIR/$tag.nsys-rep"
  nsys stats --force-export=true --report cuda_api_sum "$rep" 2>/dev/null \
    | awk '
        /^[ ]+[0-9]+\.[0-9]+/ {
          time_ns=$2; calls=$3; name=$NF
          gsub(/,/, "", calls); gsub(/,/, "", time_ns)
          printf "%s|%s|%s\n", name, calls, time_ns
        }
      '
}

extract_kernels_total() {
  # Sum of all CUDA kernel time and call count → "ns|calls"
  local tag="$1"
  local rep="$OUT_DIR/$tag.nsys-rep"
  nsys stats --force-export=true --format csv --report cuda_gpu_kern_sum "$rep" 2>/dev/null \
    | awk -F',' 'NR>1 && $2 ~ /^[0-9]+$/ {sum+=$2; calls+=$3} END {printf "%d|%d", sum+0, calls+0}'
}

extract_memcpy_total() {
  # Sum across all memcpy directions → "ns|ops"
  local tag="$1"
  local rep="$OUT_DIR/$tag.nsys-rep"
  nsys stats --force-export=true --format csv --report cuda_gpu_mem_time_sum "$rep" 2>/dev/null \
    | awk -F',' 'NR>1 && $2 ~ /^[0-9]+$/ {sum+=$2; ops+=$3} END {printf "%d|%d", sum+0, ops+0}'
}

#-----------------------------------------------------------------
run_hesper() {
  # HESPER_IGNORE_EOS=1 keeps decode running for the full -n N tokens — the
  # equivalent of llama-cli's `--ignore-eos`.  Without it Gemma 4 hits EOS
  # after ~26 tokens in the long-prompt case.  HESPER_CHAT is dropped here
  # for the same reason as llama-cli's --jinja: chat formatting is
  # irrelevant for a host-overhead microbench.
  local HSP_ENV=(env HESPER_DP4A=1 HESPER_IGNORE_EOS=1)
  if [[ -n "${HESPER_USE_MMAP:-}" ]]; then
    HSP_ENV+=(HESPER_USE_MMAP=1)
  fi
  # Only the long-prompt long-decode case is required.  The 3-case linear
  # regression was unreliable (short-decode runs are post-process-bound,
  # not GPU-bound, so the regression slope was contaminated).  Set
  # HESPER_3CASE=1 to bring the short cases back if you want to re-derive
  # `load` and `prefill/tok` separately.
  if [[ -n "${HESPER_3CASE:-}" ]]; then
    run_nsys "h_p1g1" "${HSP_ENV[@]}" "$HESPER_BIN" "$MODEL" "$PROMPT_SHORT" 1
    run_nsys "h_pLg1" "${HSP_ENV[@]}" "$HESPER_BIN" "$MODEL" "$PROMPT_LONG"  1
  fi
  run_nsys "h_pLgN" "${HSP_ENV[@]}" "$HESPER_BIN" "$MODEL" "$PROMPT_LONG" "$N_TOKENS"
  run_perf "h_pLgN" "${HSP_ENV[@]}" "$HESPER_BIN" "$MODEL" "$PROMPT_LONG" "$N_TOKENS"
}

run_lcli() {
  # CRITICAL: use `--single-turn`.  Despite documentation, `-no-cnv` *does
  # not* prevent llama-cli from re-prompting after EOS when stdin is
  # /dev/null — it just reads EOF immediately and loops forever, producing
  # 1.9 GB of `> ` lines (this was the original "llama.cpp writes a lot to
  # disk" symptom).  `--single-turn` exits cleanly after one generation.
  # `--jinja` is dropped: chat formatting is irrelevant for a host-overhead
  # microbench and prolongs the run with template tokens.
  local CLI_OPTS=(--no-warmup --single-turn -ngl 99 --seed 1 --temp 0 -c 4096)
  if [[ -n "${HESPER_3CASE:-}" ]]; then
    run_nsys "lcli_p1g1" "$LLAMA_DIR/llama-cli" -m "$MODEL" "${CLI_OPTS[@]}" -p "$PROMPT_SHORT" -n 1
    run_nsys "lcli_pLg1" "$LLAMA_DIR/llama-cli" -m "$MODEL" "${CLI_OPTS[@]}" -p "$PROMPT_LONG"  -n 1
  fi
  run_nsys "lcli_pLgN" "$LLAMA_DIR/llama-cli" -m "$MODEL" "${CLI_OPTS[@]}" -p "$PROMPT_LONG"  -n "$N_TOKENS"
  run_perf "lcli_pLgN" "$LLAMA_DIR/llama-cli" -m "$MODEL" "${CLI_OPTS[@]}" -p "$PROMPT_LONG"  -n "$N_TOKENS"
}

run_lbench() {
  # llama-bench takes integer prompt/decode counts.
  run_nsys "lbench_p1g1" "$LLAMA_DIR/llama-bench" -m "$MODEL" -p "$P_SHORT_TOK" -n 1  -r 1
  run_nsys "lbench_pLg1" "$LLAMA_DIR/llama-bench" -m "$MODEL" -p "$P_LONG_TOK"  -n 1  -r 1
  run_nsys "lbench_pLgN" "$LLAMA_DIR/llama-bench" -m "$MODEL" -p "$P_LONG_TOK"  -n "$N_TOKENS" -r 1
}

#-----------------------------------------------------------------
analyze() {
  # $1=label, $2=tag_p1g1, $3=tag_pLg1, $4=tag_pLgN
  #
  # The 3-case linear-regression decomposition was fragile: the short cases
  # ran for less than nsys' post-process timeout, and a 1-token decode trace
  # is dominated by load + prefill anyway, so any noise there poisoned the
  # regression.  When the short cases are missing we fall back to
  # **pLgN-only single-case mode**:
  #   total = N * decode/tok + P_LONG * prefill/tok + load
  # We can't separate prefill/load from decode this way, BUT for N >> P_LONG
  # the per-token contribution from prefill (P_LONG ms) is tiny compared
  # with N*decode_per_tok, so we report only `decode/tok` (computed as
  # `total / N` minus a small prefill correction estimated from the GPU
  # kernel time of the first P_LONG tokens).
  local label="$1" t11="$2" tL1="$3" tLN="$4"

  local d11="" dL1="" k11="" kL1="" m11="" mL1=""
  if [[ -f "$OUT_DIR/$t11.nsys-rep" ]]; then
    d11=$(extract_api "$t11"); k11=$(extract_kernels_total "$t11"); m11=$(extract_memcpy_total "$t11")
  fi
  if [[ -f "$OUT_DIR/$tL1.nsys-rep" ]]; then
    dL1=$(extract_api "$tL1"); kL1=$(extract_kernels_total "$tL1"); mL1=$(extract_memcpy_total "$tL1")
  fi
  local dLN=$(extract_api "$tLN")
  local kLN=$(extract_kernels_total "$tLN")
  local mLN=$(extract_memcpy_total "$tLN")

  python3 - <<PY
def parse(text):
    out = {}
    for line in text.strip().splitlines():
        if not line: continue
        name, c, t = line.split("|")
        out[name] = (int(c), int(t))
    return out

d11 = parse("""$d11""")
dL1 = parse("""$dL1""")
dLN = parse("""$dLN""")

P_LONG = $P_LONG_TOK
P_SHORT = $P_SHORT_TOK
N = $N_TOKENS

def pkern(s):
    if not s: return (0,0)
    a,b = s.split("|"); return (int(a), int(b))

k11 = pkern("$k11"); kL1 = pkern("$kL1"); kLN = pkern("$kLN")
m11 = pkern("$m11"); mL1 = pkern("$mL1"); mLN = pkern("$mLN")

SINGLE_CASE = (not d11) or (not dL1)

def regress_count(v11, vL1, vLN):
    """Returns (load, prefill_per_tok, decode_per_tok).

    In single-case mode (only pLgN is available) we collapse to:
        decode_per_tok = vLN / N      (overestimates by P_LONG/N <≈ 4 % at N=300)
        prefill_per_tok = NaN
        load = NaN
    The bias is bounded by P_LONG / N; for N=300, P_LONG=11 that is 3.7 %,
    smaller than the run-to-run variance we observe."""
    if SINGLE_CASE:
        return float('nan'), float('nan'), vLN / max(1, N)
    decode_per_tok = (vLN - vL1) / max(1, N - 1)
    prefill_per_tok = (vL1 - v11) / max(1, P_LONG - P_SHORT)
    load = v11 - P_SHORT*prefill_per_tok - 1*decode_per_tok
    return load, prefill_per_tok, decode_per_tok

def api_phase(name, idx):
    """idx=0 → call count, idx=1 → time_ns."""
    v11 = d11.get(name,(0,0))[idx]
    vL1 = dL1.get(name,(0,0))[idx]
    vLN = dLN.get(name,(0,0))[idx]
    return regress_count(v11, vL1, vLN)

names = set()
for d in (d11, dL1, dLN):
    names.update(d.keys())
selected = sorted(n for n in names if any(s in n for s in
    ("cuMemcpy","cuLaunchKernel","cuMemAlloc","cudaMemcpy","cudaLaunchKernel",
     "cudaMalloc","cuModule","cudaStreamSync","cuStreamSync")))

print()
print("=" * 78)
print(" $label")
print(f"   (P_short={P_SHORT}, P_long={P_LONG}, N={N})")
print("=" * 78)

# ------ 1. API counts (load / prefill/tok / decode/tok) ------
print()
print("[1] CUDA API call counts (linear-regression decomposition)")
print(f"  {'API':<34} | {'load':>9} | {'prefill/tok':>12} | {'decode/tok':>11}")
print("  " + "-"*72)
for k in selected:
    L, P, D = api_phase(k, 0)
    print(f"  {k:<34} | {L:>9.1f} | {P:>12.2f} | {D:>11.2f}")

# kernel/memcpy total counts
L, P, D = regress_count(k11[1], kL1[1], kLN[1])
print(f"  {'(GPU kernel count, all)':<34} | {L:>9.1f} | {P:>12.2f} | {D:>11.2f}")
L, P, D = regress_count(m11[1], mL1[1], mLN[1])
print(f"  {'(GPU memcpy ops, all)':<34} | {L:>9.1f} | {P:>12.2f} | {D:>11.2f}")

# ------ 2. Host wall vs GPU work ------
print()
print("[2] Host time spent in CUDA API (driver time, includes implicit syncs)")
print(f"  {'API':<34} | {'load(ms)':>9} | {'prefill/tok':>12} | {'decode/tok':>11}")
print("  " + "-"*72)
host_total_decode = 0.0
host_total_prefill = 0.0
for k in selected:
    L, P, D = api_phase(k, 1)  # time_ns
    if abs(D) < 1000 and abs(P) < 1000 and abs(L) < 1000:
        continue  # skip rows that contribute essentially zero
    print(f"  {k:<34} | {L/1e6:>9.2f} | {P/1e6:>12.4f} | {D/1e6:>11.4f}")
    host_total_decode += D
    host_total_prefill += P

# GPU kernel + memcpy time = "GPU busy"
Lk, Pk, Dk = regress_count(k11[0], kL1[0], kLN[0])
Lm, Pm, Dm = regress_count(m11[0], mL1[0], mLN[0])
print()
print(f"  {'GPU kernel time':<34} | {Lk/1e6:>9.2f} | {Pk/1e6:>12.4f} | {Dk/1e6:>11.4f} ms")
print(f"  {'GPU memcpy time':<34} | {Lm/1e6:>9.2f} | {Pm/1e6:>12.4f} | {Dm/1e6:>11.4f} ms")
print(f"  {'GPU busy (kernel+memcpy)':<34} | {(Lk+Lm)/1e6:>9.2f} | {(Pk+Pm)/1e6:>12.4f} | {(Dk+Dm)/1e6:>11.4f} ms")

# ------ 3. Decode host-time breakdown ------
# Where the host spends time per token: launching, copying in, syncing, etc.
print()
print("[3] Decode-time host-side breakdown (per token)")
sync_keys   = [k for k in selected if "Synchronize" in k]
dtoh_keys   = [k for k in selected if "DtoH" in k or "MemcpyDtoH" in k]
host_keys   = [k for k in selected if "cuMemcpyHtoD" in k or "cudaMemcpyAsync" in k]
launch_keys = [k for k in selected if "Launch" in k]

def sum_phase(keys, idx, phase):
    total = 0.0
    for k in keys:
        L, P, D = api_phase(k, idx)
        v = {0: L, 1: P, 2: D}[phase]
        total += v
    return total

decode_launch_ns = sum_phase(launch_keys, 1, 2)
decode_htod_ns   = sum_phase(host_keys,   1, 2)
decode_dtoh_ns   = sum_phase(dtoh_keys,   1, 2)
decode_sync_ns   = sum_phase(sync_keys,   1, 2)
decode_kern_ns   = (kLN[0] - kL1[0]) / max(1, N-1)
decode_memcpy_ns = (mLN[0] - mL1[0]) / max(1, N-1)

print(f"  GPU busy / token        = {(decode_kern_ns+decode_memcpy_ns)/1e6:>8.3f} ms (kernel {decode_kern_ns/1e6:.2f} + memcpy {decode_memcpy_ns/1e6:.2f})")
print(f"  Host: cudaLaunchKernel  = {decode_launch_ns/1e6:>8.3f} ms (launch stub)")
print(f"  Host: HtoD copies       = {decode_htod_ns/1e6:>8.3f} ms (write logits / scalars)")
print(f"  Host: DtoH (sync read)  = {decode_dtoh_ns/1e6:>8.3f} ms (sample readback)")
print(f"  Host: cuStreamSync      = {decode_sync_ns/1e6:>8.3f} ms (explicit drain)")
print(f"  Host total in CUDA API  = {(decode_launch_ns+decode_htod_ns+decode_dtoh_ns+decode_sync_ns)/1e6:>8.3f} ms")
PY

  # ------ 4. perf top symbols (where the host actually spends cycles) ------
  local prpt="$OUT_DIR/${tLN}.perf.txt"
  if [[ -f "$prpt" ]]; then
    echo
    echo "[4] perf record top user-space symbols (cycles:u --no-children, >=0.5 %)"
    echo "    captured during the long-prompt long-decode workload"
    echo "  -------------------------------------------------------------------------"
    awk 'NF >= 2 {print "  " $0}' "$prpt"
  fi
}

run_all() {
  case "$WHICH" in
    hesper) run_hesper ;;
    cli)    run_lcli ;;
    both)   run_hesper; run_lcli ;;
    bench)  run_lbench ;;
    all)    run_hesper; run_lcli; run_lbench ;;
    *)      echo "unknown: $WHICH (hesper|cli|both|bench|all)"; exit 1 ;;
  esac

  case "$WHICH" in
    hesper|both|all)
      [[ -f "$OUT_DIR/h_pLgN.nsys-rep" ]] && analyze "hesper (real argmax sampling)" h_p1g1 h_pLg1 h_pLgN ;;
  esac
  case "$WHICH" in
    cli|both|all)
      [[ -f "$OUT_DIR/lcli_pLgN.nsys-rep" ]] && analyze "llama-cli (real sampling, --temp 0)" lcli_p1g1 lcli_pLg1 lcli_pLgN ;;
  esac
  case "$WHICH" in
    bench|all)
      [[ -f "$OUT_DIR/lbench_pLgN.nsys-rep" ]] && analyze "llama-bench (NO real sampling — std::rand)" lbench_p1g1 lbench_pLg1 lbench_pLgN ;;
  esac

  echo
  echo "Reports under: $OUT_DIR"
}

run_all
