# Task 06 (Hala): one counting model for all z-score methods

**Goal:** extend the break-even formula (decision #8) from size windows to **any length weighting**, so a single
model predicts A, B, every B_c, W and W_fixedFF. This **supports** the answer to R1-M3 / R2-M2 (W's advantage is
explained by the model, not only observed), but it does **not** answer it: the model, like W, uses the true length
distributions. The actual answer is the mismatch run in **Task 07** (patients from a changed fetal profile, W
with G1 weights). Tasks 07 and 08 use this model for their predictions; for mismatched W, plug u = G1 weights
into the 6A formula while E[·] and w* use the true (changed) profile.
**Decisions:** #8, #14, #15 (new #16 added by Maya). **`git pull` first.**

**No new simulation.** Everything uses `results/tables/results_all.csv` and the post-mappability length
distributions from `origin_matrices`. Expected run time: seconds.

---

## 6A. The general formula

Any z-score method in the study is a **weighted chr21 fraction**: each fragment of length l gets a weight u(l),
the same on both rows (chr21 and other autosomes):

```
F_u = Σ_l u(l) X21(l) / Σ_l u(l) [X21(l) + Xother(l)]
```

| Method | u(l) |
|---|---|
| A | 1 for every length |
| B, B_c | 1 inside [40, c], 0 outside |
| W | w*(l) = FF·p_f(l) / p_mix(l) (true FF) |
| W_fixedFF | same formula with FF = 0.10 in the weight (but the patients still have the condition's FF) |

Notation, for one condition (FF, depth N, noise σ):
- `p_f(l)`, `p_m(l)`: fetal and maternal length distributions **after mappability**, summed over the two rows:
  `matrices["fetal"].sum(axis=0)` and `matrices["maternal"].sum(axis=0)` (the normal fetal matrix, as in
  `posterior_weights`).
- `p_mix(l) = FF·p_f(l) + (1 − FF)·p_m(l)`
- `w*(l) = FF·p_f(l) / p_mix(l)`: the **true** P(fetal | l) for this FF
- `E[x] = Σ_l p_mix(l)·x(l)` (an average over the patient's fragment lengths)
- `p0` = q(chr21) (column `p0` in `results_all.csv`)

**Formula** (first order in FF, p0 ≪ 1, same assumptions as Eq. 4):

```
sep_u = ( E[u·w*] / (2·E[u]) ) / sqrt( E[u²] / (p0·N·E[u]²) + σ² )
ratio_pred(u) = sep_u / sep_A        (sep_A = sep_u with u = 1, which is (FF/2) / sqrt(1/(p0 N) + σ²))
```

Why, in one line each:
- A trisomy multiplies chr21 at length l by about `1 + w*(l)/2` (only the fetal part gains 50%), so the relative
  signal of F_u is `E[u·w*] / (2E[u])`.
- Counting noise: the relative variance of F_u is `E[u²] / (p0·N·E[u]²)`.
- The patient noise multiplies the whole chr21 row by (1 + ε), so it adds σ² to the relative variance whatever u is.

It is **exactly Eq. 4** with an effective g and r:
```
g_eff = E[u·w*] / (FF·E[u])          r_eff = E[u]² / E[u²]
ratio_pred = g_eff · sqrt( (1/(p0 N) + σ²) / (1/(p0 r_eff N) + σ²) )
```
For a window (u = 0 or 1) this gives g_eff = g and r_eff = r, so the B rows must reproduce
`breakeven_check.csv`.

### Self-tests (in the script, run first every time, like `detection_limits.py`)
1. u = 1 → `ratio_pred = 1`, g_eff = 1, r_eff = 1.
2. For B and every B_c: `g_eff`, `r_eff` and `ratio_pred` equal `g`, `r` and `ratio_pred` in
   `breakeven_check.csv` (relative difference < 1e-9).
3. Multiplying u by a constant (e.g. 7) does not change `ratio_pred` (a z-score doesn't care about the scale of the weights).
4. `E[w*] = FF` (to 1e-12).
5. (after 6C) At σ = 0, `u_opt = w*` and `w_efficiency = 1` (to 1e-9) in every condition.

## 6B. Check against the simulation

New script `src/evaluate/unified_model_check.py`. Reuse `CONDITION`, the `low_signal` rule (sep_A < 1) and
`error_summary` / `summarise` from `breakeven_check.py` (import them, don't copy).

For **every one of the 96 conditions** and methods **B, B_120 … B_165, W, W_fixedFF**:
- `sep_obs = (z_mean_t21 − z_mean_normal) / z_sd_normal` (as in `breakeven_check.py`)
- `ratio_obs = sep_obs / sep_obs(A)`
- `ratio_pred`, `g_eff`, `r_eff` from 6A
- `rel_error`, `abs_error`, `low_signal`

Leave out B-opt (its cutoff is chosen from the data) and C1/C2 (not weighted fractions).

**Outputs**
- `results/tables/unified_model_check.csv`: columns
  `ff, depth, noise, sigma, method, p0, g_eff, r_eff, sep_A, sep_obs, ratio_obs, ratio_pred, rel_error, abs_error, low_signal`
- `results/tables/unified_model_summary.csv`: the same summary as `breakeven_summary.csv`, **per method family**
  (`B` = B and all B_c together; `W`; `W_fixedFF`), overall and without `low_signal`.
- **Do not change** `breakeven_check.csv` or `breakeven_summary.csv` (they are in the current manuscript).

**Supplementary figure** `results/figures/figS_unified_model.(png|pdf)`, same style as Fig. 4: FF 4%, one panel per
noise level, x = depth, y = ratio vs A; points = simulation, lines = 6A prediction, for **B (156 bp) and W**.
Dashed line at 1.

**What to expect** (already confirmed by Hala on the committed results; same 10 bp S(l) bins as the code):
- median |rel_error| 0.50% for W, 0.75% for the B family
- W at FF 4%, realistic noise: predicted ratio ≈ 1.36 (1M), 1.41 (2.5M), 1.47 (5M), 1.54 (10M);
  observed in `results_all.csv`: 1.353, 1.410, 1.462, 1.528
- W_fixedFF close to W everywhere (that's why its detection limits moved ≤ 0.08 points)

If W's median error is above ~2%, stop and send me the numbers before going on.

## 6C. How far is W from the best possible weighting? (analytic only)

W's weights w* are the **best possible linear weights when counting noise is the only noise** (matched filter).
With depth-independent noise σ, the best weights are different. The closed form (Sherman–Morrison on the
formula in 6A) is:

```
u_opt(l) = w*(l) − c,     c = FF · k / (1 + k),     k = σ²·p0·N
```

With σ = 0, c = 0 and u_opt = w* (W is optimal). As depth grows, c → FF and the optimal weights start to
**down-weight long fragments**, which is what hard size selection does. This is the likely reason B catches up
with W at 10M.

For every condition, compute `ratio_pred(u_opt)` with the 6A formula and add two columns to
`unified_model_check.csv` on the **W rows only**: `ratio_pred_opt` and `w_efficiency = ratio_pred / ratio_pred_opt`.

Expected at FF 4%, realistic noise: W ≈ 1.36 vs optimum ≈ 1.37 at 1M, but 1.54 vs ≈ 1.70 at 10M
(W ≈ 90% of the optimum). At σ = 0 the efficiency must be exactly 1 (self-test 5).

**Analytic only:** do **not** add u_opt as a new detection method or rerun the grid with it. It's a Discussion
point (and future work), not a new result. Adding a method now would also need a reference, calibration and
detection limits, which is out of scope for the revision.

## 6D. Reproducibility
- Add the script to `scripts/run_all.sh` after the break-even check (note its run time in a comment).
- Add the new outputs to the README table ("Where each figure and table comes from").
- `check_rerun` doesn't need changes (no new patients).

## Send Maya
1. `unified_model_summary.csv` (the overall and without-low_signal rows per family)
2. The 12 W rows at FF 4% (ratio_obs, ratio_pred, rel_error, ratio_pred_opt, w_efficiency)
3. The 3 W rows with the largest |rel_error| outside `low_signal`, with their conditions
4. `figS_unified_model.png`

Maya will write decision #16 and add the derivation to `docs/break_even.md`.
