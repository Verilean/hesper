#!/bin/bash
# msl_check.sh — M-Metal Stage 3 gate: static bounds-check the hand-written
# MSL kernels (metal_replace.mm) with the wgsl-check MSL front end, plus the
# intentional-FAIL fixture suite (specs/mslcheck/).
#
# The kernels are extracted from native/metal_replace.mm AT CHECK TIME (source
# of truth stays the .mm) and their @DIM@ template holes are substituted with
# the production DiffusionGemma dims documented below.
#
# Dim provenance (Examples/DiffusionGemmaDecode.lean, France N_tok=277 config):
#   dim=2048  expFF=1408  nExpert=32  nUsed=8  N_tok=P+C=21+256=277
#   maxPadded = ceil((N_tok*nUsed + 32*nExpert + 31)/32)*32
#             = ceil((2216+1024+31)/32)*32 = 3264          (L1417)
#   q4k gate/up (mslQ4kDispatch, L2014): M=maxPadded N=2*expFF=2816 K=dim
#     grid=((2816+31)/32, (3264+31)/32)=(88,102) tg=(128,1,1)
#     src=sMoeN N_tok*dim=567296 | idx=sSortedPos=3264
#     b=guE nExpert*2816 rows x RSU=(K/256)*36=288 u32 = 25952256
#     c=sGatheredGU maxPadded*2816=9191424 | te=trs=maxPadded/32=102
#   q8 down (mslQ8DownDispatch, L2109): M=maxPadded N=dim K=expFF FUSED=0
#     grid=((2048+31)/32,102)=(64,102)
#     a=sGatheredEh maxPadded*expFF=4595712
#     b=dnE 32*2048 rows x RSB=(K/32)*34=1496 B = 98041856 B = 24510464 u32
#     pos=slot=3264 | dst=sDownAll nUsed*N_tok*dim=4538368
#   q5 down: RSB=(K/32)*22=968 B -> b = 15859712 u32; rest as q8.
set -u
cd "$(dirname "$0")/.."
OUT=.lake/build/mslcheck
mkdir -p "$OUT"

python3 - "$OUT" <<'PYEOF'
import re, sys
out = sys.argv[1]
src = open("native/metal_replace.mm").read()

def extract(var):
    m = re.search(re.escape(var) + r'\s*=\s*R"MSL\((.*?)\)MSL"', src, re.S)
    assert m, var
    return m.group(1)

def subst(t, d):
    for k, v in d.items():
        t = t.replace("@%s@" % k, str(v))
    assert "@" not in t, "unsubstituted hole"
    return t

q4k = subst(extract("kQ4kMslTemplate"),
            dict(M=3264, N=2816, K=2048, NEXP=32, SRCROWS=277))
q8  = subst(extract("kQ8DownMslTemplate"),
            dict(M=3264, N=2048, K=1408, NEXP=32, NUSED=8, NTOK=277, FUSED=0))
q5  = subst(extract("kQ5DownMslTemplate"),
            dict(M=3264, N=2048, K=1408, NEXP=32, NUSED=8, NTOK=277, FUSED=0))
open(out + "/q4k_gateup.metal", "w").write(q4k)
open(out + "/q8_down.metal", "w").write(q8)
open(out + "/q5_down.metal", "w").write(q5)

manifest = """{ "dispatches": [
  { "kernel": "q4k_gateup.metal", "entry": "q4k_grouped_reg_indexed",
    "lang": "msl", "grid": [88, 102, 1], "wg": [128, 1, 1],
    "bindings": [
      { "name": "src", "elems": 567296 }, { "name": "idx", "elems": 3264 },
      { "name": "b", "elems": 25952256 }, { "name": "c", "elems": 9191424 },
      { "name": "te", "elems": 102 }, { "name": "trs", "elems": 102 } ] },
  { "kernel": "q8_down.metal", "entry": "q8_down_indexed_scatter",
    "lang": "msl", "grid": [64, 102, 1], "wg": [128, 1, 1],
    "bindings": [
      { "name": "a", "elems": 4595712 }, { "name": "b", "elems": 24510464 },
      { "name": "te", "elems": 102 }, { "name": "trs", "elems": 102 },
      { "name": "pos", "elems": 3264 }, { "name": "slot", "elems": 3264 },
      { "name": "dst", "elems": 4538368 } ] },
  { "kernel": "q5_down.metal", "entry": "q5_down_indexed_scatter",
    "lang": "msl", "grid": [64, 102, 1], "wg": [128, 1, 1],
    "bindings": [
      { "name": "a", "elems": 4595712 }, { "name": "b", "elems": 15859712 },
      { "name": "te", "elems": 102 }, { "name": "trs", "elems": 102 },
      { "name": "pos", "elems": 3264 }, { "name": "slot", "elems": 3264 },
      { "name": "dst", "elems": 4538368 } ] }
] }"""
open(out + "/manifest.json", "w").write(manifest)
print("extracted 3 kernels -> " + out)
PYEOF

echo "=== fixtures (expected: 3 FAIL, ok x1) ==="
./.lake/build/bin/wgsl-check --manifest specs/mslcheck/manifest.json
FIX=$?
echo "=== production hand-MSL kernels ==="
./.lake/build/bin/wgsl-check --manifest "$OUT/manifest.json"
REAL=$?
if [ "$FIX" -ne 1 ]; then echo "FIXTURE SUITE BROKEN (expected exit 1)"; exit 2; fi
exit $REAL
