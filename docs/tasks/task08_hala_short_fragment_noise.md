# Task 08 (Hala): extra noise on size-selected fragments

**Goal:** answer R1-M1. In the simulation and in Eq. 4, the depth-independent noise σ hits every fragment length
the same way, so their close agreement checks consistency but can't test that assumption, and the main depth
result rests on it. Here short fragments get **their own extra noise**, and we see how much of the depth result
and of B's advantage survives.
**Decisions:** #7, #8, #16, #17 (new #18 added by Maya). **`git pull` first.**
**Main results must not change:** the new noise is off by default. `check_rerun` on the main grid before and
after, as in Task 07.

---

## 8A. The noise model

Per patient, as now: ε ~ N(0, σ) multiplies the **whole chr21 row**. New: a second, independent term
ε_s ~ N(0, σ_s) multiplies the chr21 cells **inside the 40–156 bp window only**:

```
P21(l) ← P21(l) · (1 + ε) · (1 + ε_s · 1[40 ≤ l ≤ 156])      then renormalise, then draw counts
```

- Draw ε first, then ε_s, from the patient stream. With σ_s = 0, **don't draw ε_s at all**, so the main patients
  stay byte-identical.
- Raise the same error as now if (1 + ε_s) ≤ 0.
- Applies to every group (reference, training, calibration, test).

**Parameter:** k = how much noisier the window's chr21 fraction is than with σ alone:
```
σ_s = σ · sqrt(k² − 1)          so the window fraction has relative SD ≈ σ·k
```
Scenarios: **k = 1.5 and k = 2**, at the **low and realistic** noise levels only (σ = 0 has nothing to scale).

Note for the text, nothing to code: the extra noise also reaches A and W a little, because their chr21 fraction
includes window fragments. For a method with weights u (Task 06), the relative noise becomes
```
σ_u² = σ² + σ_s² · ρ_u²,      ρ_u = E[u · 1_win] / E[u]
```
At FF 4%: for A, ρ = r = 0.203, so σ_A grows by 2.5% (k = 1.5) and 6% (k = 2). For B, ρ = 1. For W, ρ = 0.49
(W leans on short fragments). So the effective ratio σ_B/σ_A is a bit below k: 1.46 and 1.89. Report it.

Cutoffs at or below 156 bp (B, B_120…B_150) take all of the extra noise; only B_165, and B-opt when it picks
160–170 bp, are partly protected. A step at 156 bp is the **worst case for B**: the extra noise lands exactly on B's fragments. A real length-dependent
bias would be smoother. The text will say so; don't build a smooth version.

## 8B. Config and scenarios

```yaml
robustness:
  short_noise:                       # decision #18
    snoise_k15: {window_bp: [40, 156], k: 1.5}
    snoise_k2:  {window_bp: [40, 156], k: 2.0}
```

Reuse the Task 07 `--scenario` mechanism: output to `results/robustness/<name>`, same seeds as the main grid
(partial pairing only: drawing ε_s shifts the stream, so say so in the docstring), G1 fetal profile.
Grid: 8 FF × 4 depths × {low, realistic} = **64 conditions per scenario**. `detection_limits` will skip the
missing σ = 0 groups, so make sure it reports that cleanly rather than erroring.

## 8C. Predictions and QC first (before the grid)

**Model:** extend `unified_model_check` with σ_u (8A) in place of σ, for every method (A included). Self-test:
with σ_s = 0 every number equals Task 06 exactly.

Save `results/tables/robustness_snoise_pred.csv`: ratio_pred vs A for B, W, W_fixedFF, per scenario × FF ×
depth × noise, plus σ_B/σ_A and ρ_W.

My preview at FF 4%, realistic noise, ratio vs A (σ_u for every method, A included). Yours should match within ~0.01:

| k | B: 1M / 2.5M / 5M / 10M | W: 1M / 2.5M / 5M / 10M |
|---|---|---|
| 1 (main) | 1.16 / 1.24 / 1.35 / 1.52 | 1.36 / 1.41 / 1.47 / 1.54 |
| 1.5 | 1.14 / 1.19 / 1.26 / 1.36 | 1.35 / 1.38 / 1.42 / 1.46 |
| 2 | 1.12 / 1.14 / 1.17 / 1.20 | 1.33 / 1.35 / 1.36 / 1.37 |

So B still rises with depth at k = 2, but much less (+7% instead of +31%), and W's lead over B grows at high depth,
since W takes only about half of the extra noise.

**QC (normals, FF 4%, 2,000 patients, 1M and 10M, both k, realistic):** SD of the all-fragment and the window
chr21 fractions vs theory `p0 · sqrt(1/(p0 · N · r_eff) + σ_u²)` (r_eff = 1 for A, r for the window).
Within ~3%, like the Task 04 noise QC.

**Send Maya the prediction table (FF 4% rows) and the QC before running the grid.** If anything differs from the
preview by more than ~0.02, stop there.

## 8D. Run per scenario

`run_condition --scenario`, then `detection_limits`, `fpr_check`, `bootstrap_diff` (B − A),
`breakeven_check`, `unified_model_check` (with σ_u), as in Task 07. About 2 × 64 conditions, plus the bootstrap.

## 8E. Summary

`src/evaluate/robustness_snoise.py` → `results/tables/robustness_snoise_summary.csv`. Answer, per k:

1. **Is B worse than A anywhere?** Number of conditions with sep(B)/sep(A) < 1, and with B − A `sens_at_fpr`
   significantly negative. List them.
2. **Does the depth result survive?** At FF 4%, sep(B)/sep(A) at 1M and 10M for low and realistic noise. The
   prediction is that it still rises, but much less as k grows (the high-depth limit drops from g to g·σ_A/σ_B:
   g/1.46 at k = 1.5, g/1.89 at k = 2, since A takes some of the extra noise too).
3. **Model accuracy:** median |rel_error| for B and W with σ_u.
4. **W vs B vs A:** detection limits per depth × noise. Does W stay below B and A? (W also takes some of the extra
   noise through ρ_W.)
5. **B-opt:** chosen cutoffs. Does extra noise on the window push it to longer cutoffs?
6. **FPR:** pooled per method. It should stay near target (the reference has the same noise).

**Figure** `figS_short_noise.(png|pdf)`: FF 4%, x = depth, y = ratio vs A, one panel per noise level (low,
realistic), lines = prediction, points = simulation, for B and W at k = 1 (main), 1.5 and 2. Dashed line at 1.
Caption as Figure S5.

## 8F. Model only: what k would fit Budiš et al.? (illustration, no simulation)

The manuscript compares with Budiš et al.: trisomic z-score gain from size selection of 1.05× at 5M alignments
and 1.13× at all reads (mean 8.8M). Our main simulation predicts much more (e.g. ~1.35 at 5M at FF 4%), and the
Limitations suggest length-specific noise as a reason (R1-M1 points at the same thing).

With the 8A model at realistic σ, compute B vs A at **5M and 8.8M** for k from 1 to 3 (step 0.1), at FF 4%,
6% and 10% (their cohort FF isn't reported). Save `results/tables/snoise_budis_illustration.csv`, and report which
k range gives ratios near 1.05 and 1.13.

This is an **illustration**, not a fit: their cutoff, library prep, z-score definition and FF differ from ours.
Maya will decide whether it goes in the Discussion. Don't put it in a figure.

**Already found (Hala's quick check):** no single k fits both numbers. The k values depend on FF (1.05 at 5M /
1.13 at 8.8M need k ≈ 2.70 / 2.25 at FF 4%, 2.57 / 2.18 at 6%, 2.34 / 2.00 at 10%), so the text gives no fixed k;
the 5M number always needs the larger k, and above k* ≈ 2.4 the predicted gain *falls* with depth. That follows from the
model: as N grows, the ratio moves monotonically from g·√r (low depth) to g·σ_A/σ_B (high depth), so

```
depth increases the gain of size selection  ⇔  σ_B / σ_A < 1 / √r      (≈ 2.2 for the 40–156 bp window)
```

With A taking its r²σ_s² share, the k that reaches the threshold has a closed form:
`k* = sqrt((1 + r) / r)` → 2.43 / 2.40 / 2.35 at FF 4 / 6 / 10% (1/√r = 2.22 / 2.19 / 2.13).
Add both to the 8F table per FF, and check k* against the numerical crossing (self-test). The text will say that k ≈ 2–3
brings the gains down to Budiš's level, but that no single k reproduces their rise from 5M to 8.8M.

**Four regimes.** With two depth limits, low-depth gain g·√r and high-depth gain g·σ_A/σ_B, each above or below 1:

| g·√r | g·σ_A/σ_B | B vs A |
|---|---|---|
| ≥ 1 | ≥ 1 | never worse (main analysis, Yang window) |
| < 1 | ≥ 1 | worse at low depth, better past N* (the strict cutoffs, Eq. 5) |
| ≥ 1 | < 1 | **better at low depth, worse past N\*** (new, needs σ_B/σ_A > g) |
| < 1 | < 1 | never better |

Add a `regime` column to the extended `breakeven_check` output. The new regime needs σ_B/σ_A > g, i.e.
k² > g²(1 − r²)/(1 − g²r²) (k ≈ 2.7 at FF 4%), so probably **no grid condition hits it** at k ≤ 2. Report the k
where it starts per FF, so the text can say where it would appear. Some fixed cutoffs (e.g. B_165, smaller g) or
higher FF may get closer; list any that do.

## 8G. Reproducibility
- Scenario runs and summary in `scripts/run_all.sh` (same optional robustness block as Task 07, with run times).
- README: the new outputs.
- `check_rerun` on the main grid before and after, and on one `snoise` condition.

## Send Maya
1. After 8C: the FF 4% predictions and the QC (**before the grid**)
2. After 8E: the summary with answers to 1–6 in a few lines each
3. `figS_short_noise.png`
4. The 8F table (which k range matches Budiš)
5. Anything that surprised you
