/-!
# Formal model of the CACHEMODE-1 sliding-window KV ring protocol

Lean port of e4b-webgpu/specs/KVRing.tla — but stronger: where TLC checked
finite instances (W=1024, C=512, RING ∈ {1534,1535,1536}), the theorems here
are PARAMETRIC in all sizes.

Protocol (kernels/attnf32.wgsl + headprep.wgsl + engine-a4b.js):
* writer: position `p`'s K/V land in slot `p % RING`; positions are written
  strictly in order; a prefill chunk's writes all precede its reads.
* reader: for query `qp` with newest written position `mw`, slot `t`'s
  position is recovered as `ps = mw - ((mw + RING - t) % RING)` and the slot
  is dead iff `ps > qp` (causality) or `ps + W ≤ qp` (window).

Results:
* `recover_exact` — a position within the last `RING` writes is recovered
  exactly (core lemma).
* `sound_mem` / `sound_max` (V-P1) — the recovery formula returns exactly
  the slot's true content (the newest position ≤ mw in that residue class),
  so a clobbered position can only DROP OUT via the dead test; stale K/V is
  never misattributed.
* `complete` (V-P2) — `RING ≥ W + C - 1` serves every window position of
  every query, for ALL W, C, RING (subsumes the TLC production runs).
* `tightness` — `RING = W + C - 2` loses a window position (W=4, C=3).
-/

namespace KVRing

/-- attnf32.wgsl: recovered position of slot `t` when the newest written
position is `mw`. -/
def recPs (RING mw t : Nat) : Nat := mw - ((mw + RING - t) % RING)

/-- Core recovery lemma: if `p` is within the last `RING` written positions,
its slot still holds it and the reader recovers it exactly. -/
theorem recover_exact {RING mw p : Nat} (hR : 0 < RING)
    (hpm : p ≤ mw) (hdist : mw - p < RING) :
    recPs RING mw (p % RING) = p := by
  have hd : RING * (p / RING) + p % RING = p := Nat.div_add_mod p RING
  have hmod : p % RING < RING := Nat.mod_lt _ hR
  have hkey : mw + RING - p % RING = RING * (p / RING + 1) + (mw - p) := by
    rw [Nat.mul_succ]; omega
  have h2 : (mw + RING - p % RING) % RING = (mw - p) % RING := by
    rw [hkey, Nat.mul_add_mod]
  have h3 : (mw - p) % RING = mw - p := Nat.mod_eq_of_lt hdist
  unfold recPs
  omega

/-- V-P1 (membership): the recovered position is a real written position in
slot `t`'s residue class. -/
theorem sound_mem {RING mw t : Nat} (hR : 0 < RING) (ht : t < RING)
    (htm : t ≤ mw) :
    recPs RING mw t ≤ mw ∧ recPs RING mw t % RING = t := by
  have hkey : mw + RING - t = RING * 1 + (mw - t) := by omega
  have h2 : (mw + RING - t) % RING = (mw - t) % RING := by
    rw [hkey, Nat.mul_add_mod]
  have hd : RING * ((mw - t) / RING) + (mw - t) % RING = mw - t :=
    Nat.div_add_mod (mw - t) RING
  have hmod : (mw - t) % RING < RING := Nat.mod_lt _ hR
  constructor
  · unfold recPs; omega
  · unfold recPs
    rw [h2]
    have : mw - (mw - t) % RING = t + RING * ((mw - t) / RING) := by omega
    rw [this, Nat.add_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt ht]

/-- V-P1 (maximality): every written position in the same slot is ≤ the
recovered one — the reader always sees the slot's NEWEST content, so an
overwritten position is never misread as older K/V. -/
theorem sound_max {RING mw t p : Nat} (hR : 0 < RING)
    (hpm : p ≤ mw) (hpt : p % RING = t) :
    p ≤ recPs RING mw t := by
  have hd : RING * (p / RING) + p % RING = p := Nat.div_add_mod p RING
  have hmod : p % RING < RING := Nat.mod_lt _ hR
  have hkey : mw + RING - t = RING * (p / RING + 1) + (mw - p) := by
    rw [Nat.mul_succ]; omega
  have h2 : (mw + RING - t) % RING = (mw - p) % RING := by
    rw [hkey, Nat.mul_add_mod]
  have h3 : (mw - p) % RING ≤ mw - p := Nat.mod_le _ _
  unfold recPs
  omega

/-- V-P2, parametric (subsumes every TLC instance): with `RING ≥ W + C - 1`,
every position `p` in the window of every query `qp` of a chunk whose last
write is `mw` is recovered exactly at read time. `mw - qp ≤ C - 1` says `qp`
is in the chunk being read; `qp < p + W` and `p ≤ qp` say `p` is in `qp`'s
live window (the kernel's non-dead condition). -/
theorem complete {RING W C mw qp p : Nat} (hR : 0 < RING)
    (hbound : W + C - 1 ≤ RING) (hC : 0 < C)
    (hq : qp ≤ mw) (hchunk : mw - qp ≤ C - 1)
    (hwin : qp < p + W) (hple : p ≤ qp) :
    recPs RING mw (p % RING) = p :=
  recover_exact hR (Nat.le_trans hple hq) (by omega)

/-- The bound is tight: with `RING = W + C - 2` (W=4, C=3, RING=5), query
`qp = 3` reading after its chunk's last write `mw = 5` needs window position
`p = 0`, but slot 0 has been overwritten by position 5. -/
theorem tightness : recPs 5 5 (0 % 5) = 5 := by decide

/-- Sanity: the production constants satisfy the parametric bound
(W=1024, C=512: minimum safe ring = 1535; shipped ring = 1536). -/
example : 1024 + 512 - 1 ≤ 1536 := by decide

end KVRing
