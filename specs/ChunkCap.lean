/-!
# Chunk-capacity bound for the grouped-MoE counting sort (expgroup.wgsl)

wgsl-check flagged `chunkExp[c + j]` (and `chunkEnt`) as data-dependent: the
write offset `c` accumulates `ceil(n_e / 8)` over 128 experts, where `n_e`
is the number of (token, slot) entries routed to expert `e` — an invariant
the interval domain cannot see. This file discharges it for ALL inputs, not
just traced ones.

Setting: a prefill chunk routes at most `T = MPRE * K = 512 * 8 = 4096`
entries among `E = 128` experts; each expert's entries are split into
8-entry chunks, so expert `e` contributes `ceil(n_e / 8)` chunks. The
`chunkExp` buffer holds `CAP = 640` labels.

Theorems (general in the chunk size `c`):
* `sum_ceil_div_le` — Σ_e ceil(n_e/c) ≤ (Σ_e n_e)/c + #experts.
* `chunkCap_sufficient` — with Σ n_e ≤ 4096, ≤ 128 experts, c = 8:
  Σ ceil(n_e/8) ≤ 640. The shipped capacity IS the provable bound
  (T/c + E = 512 + 128), leaving slack 16 over the attainable maximum
  (T−E)/c + E = 624 — safe for every routing, but only while
  MPRE·K ≤ 4096 and E ≤ 128: raising either silently overflows the buffer
  (the wgsl-check WARN keeps pointing here as the reminder).
-/

namespace ChunkCap

/-- pointwise: `ceil(n/c) ≤ n/c + 1` (Nat division). -/
theorem ceil_div_le (n c : Nat) (hc : 0 < c) : (n + c - 1) / c ≤ n / c + 1 := by
  calc (n + c - 1) / c ≤ (n + c) / c := Nat.div_le_div_right (by omega)
    _ = n / c + 1 := Nat.add_div_right n hc

/-- floor division is superadditive: `a/c + b/c ≤ (a+b)/c`. -/
theorem div_add_div_le (a b c : Nat) (hc : 0 < c) : a / c + b / c ≤ (a + b) / c := by
  apply (Nat.le_div_iff_mul_le hc).mpr
  rw [Nat.add_mul]
  have ha := Nat.div_mul_le_self a c
  have hb := Nat.div_mul_le_self b c
  omega

/-- Σ floor(n/c) over a list ≤ floor(Σ n / c). -/
theorem sum_div_le (l : List Nat) (c : Nat) (hc : 0 < c) :
    (l.map (· / c)).sum ≤ l.sum / c := by
  induction l with
  | nil => simp
  | cons n rest ih =>
    simp only [List.map_cons, List.sum_cons]
    calc n / c + (rest.map (· / c)).sum
        ≤ n / c + rest.sum / c := by omega
      _ ≤ (n + rest.sum) / c := div_add_div_le n rest.sum c hc

/-- Σ ceil(n_e/c) ≤ (Σ n_e)/c + number of experts. -/
theorem sum_ceil_div_le (l : List Nat) (c : Nat) (hc : 0 < c) :
    (l.map (fun n => (n + c - 1) / c)).sum ≤ l.sum / c + l.length := by
  have hpt : (l.map (fun n => (n + c - 1) / c)).sum
      ≤ (l.map (· / c)).sum + l.length := by
    induction l with
    | nil => simp
    | cons n rest ih =>
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      have := ceil_div_le n c hc
      omega
  have := sum_div_le l c hc
  omega

/-- The shipped capacity is sufficient for EVERY routing: with at most 128
experts and at most 4096 = MPRE·K routed entries, the counting sort emits at
most 640 chunks — exactly the `chunkExp` buffer size. -/
theorem chunkCap_sufficient (l : List Nat)
    (hlen : l.length ≤ 128) (hsum : l.sum ≤ 4096) :
    (l.map (fun n => (n + 8 - 1) / 8)).sum ≤ 640 := by
  have h := sum_ceil_div_le l 8 (by omega)
  have : l.sum / 8 ≤ 512 := by omega
  omega

end ChunkCap
