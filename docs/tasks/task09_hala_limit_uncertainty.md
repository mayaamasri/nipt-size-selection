# Task 09 (Hala): uncertainty in the detection limits

**Goal:** answer R1-M4 and R2-M1 (and R2 minor 1). The detection limits are interpolated point estimates with no
uncertainty, yet the paper ranks methods whose limits differ by 0.01–0.1 points (W 1.99% vs C2 2.00% at 10M no
noise; W 2.89% vs B 2.91% at 10M realistic). After this task, a method is called better **only where the CI of
the difference excludes 0**, and "similar" otherwise.
**Decisions:** #5, #13 (new #19 added by Maya). **`git pull` first.** Doesn't depend on Task 08.
**Main results must not change:** this task only reads and resamples; nothing in `results_all.csv` or
`detection_limits.csv` is rewritten.

---

## 9A. What the bootstrap resamples (R2 asks us to state this exactly)

Two versions, both **paired** (the same resampled patients for every method in a resample):

| Version | Resampled | Held fixed | Methods | Use |
|---|---|---|---|---|
| **test** | test normals and test T21, within each FF | reference statistics, trained trees, tree thresholds, B-opt's cutoff | all: A, B, B-opt, W, W_fixedFF, C1, C2 | every comparison involving C1/C2 |
| **test+ref** | test patients **and** the 2,000 reference normals (→ new mean and SD per statistic) | B-opt's cutoff, z > 3 | z-score methods only: A, B, B-opt, W, W_fixedFF (+ W_G1 in scenarios) | comparisons among z-score methods (W−B, B−A, W_G1−A) |

Why two: each condition's normals share one reference, so the reference's sampling error moves all of them
together (Hala's Task 07 point about the Wilson CIs). The test+ref version includes that. For the trees,
resampling the reference without retraining would be inconsistent, so they stay in the test version. The text will
say both versions condition on the trained models and thresholds, and that training and calibration variability
aren't included.

**Per resample, per (depth, noise) setting:**
1. For each of the 8 FF conditions **independently**: resample the test normals (10,000) and T21 (2,000) with
   replacement; in test+ref also the reference normals (2,000), then recompute each statistic's mean and SD and
   rescore the test patients.
2. Per method and FF: `sens` (at its own threshold: z > 3, or the tree's calibrated t) and `sens_at_fpr`.
3. Detection limits with `detection_limits.crossing` (same rule as the paper), from `sens` and from `sens_at_fpr`.
4. Differences for the pairs below, in the same resample.

**B = 1,000 resamples**, percentile 95% CI. Seeds: a spawn-key family **not used anywhere yet** (list every
family in use first: `(3, …)` is already taken by `bootstrap_diff`, so it can't be reused), so nothing collides.

**Note on the two bases:** for `sens_at_fpr`, test+ref gives the same CIs as test, because a z-score method's
reference only shifts and scales its scores and `sens_at_fpr` depends only on their ranking within the test set.
Reference resampling therefore matters only for the `sens` basis (z > 3). Run test+ref for `sens` only, and the
text will say this. To save time, compute each condition's per-patient raw statistics and
tree probabilities once (regenerate from the seeds, as `bootstrap_diff` does) and cache them under
`data/processed/` (gitignored), not in `results/`.

**Limits outside the grid** ("< 1%", "> 15%"): keep the text. For a difference, report the share of resamples
where both limits are numeric, and give a CI only if that share is ≥ 95%.

## 9B. Pairs

| Grid | Pairs | Versions |
|---|---|---|
| main | W−B, W−C2, W−A, B−A, C2−B, B-opt−B, W_fixedFF−W | W−C2, C2−B: test; others: test+ref (and test too, for comparison) |
| shift_p10, blend50 | W_G1−A, W_G1−B, W_G1−W | test+ref |

Shift_p10 and blend50 are where W_G1's margin over A is only 0.03–0.05 points (Task 07), so "never worse than A"
needs a CI there.

## 9C. Is linear interpolation good enough? (R1 asks for extra FF values)

The CIs cover sampling error, not interpolation error. Check that separately where it matters:
- For every main-grid setting where a pair from 9B differs by **< 0.2 points**, find the grid interval containing
  the two limits and add its **midpoint** FF (e.g. 2.5% for limits between 2 and 3%; 1.5%, 3.5% and 5% as needed).
- Simulate only those extra FF conditions (all groups, same methods). Seeds: an unused family with the same
  structure as the main grid, e.g. `(F, k, i_depth, i_noise, i_group)`, k = position of the extra FF in a fixed
  list in the config, so the main grid's seeds are untouched and every group gets its own stream.
- Recompute those limits on the refined grid and report **interpolated vs refined** per method. If the refined
  limits change a ranking, flag it.

Expect a handful of settings (10M none, 10M realistic, 5M realistic, maybe 5M low): a few minutes of compute.

## 9D. Gate

Run **one setting first: 10M, realistic, main grid**, both versions, and send Maya the limits with CIs and the
W−B, W−C2, B−A differences, plus the run time, before the full run. If a CI is wider than ~1 point, or B-opt
looks odd, stop there.

## 9E. Outputs

- `results/tables/detection_limits_ci.csv`: grid, method, depth, noise, version, basis (`sens` / `sens_at_fpr`),
  limit (the paper's point estimate), ci_lo, ci_hi, share_numeric
- `results/tables/limit_differences_ci.csv`: grid, pair, depth, noise, version, basis, diff, ci_lo, ci_hi,
  share_numeric, `clear` (CI excludes 0)
- `results/tables/limits_refined_grid.csv`: setting, method, interpolated limit, refined limit, difference
- Summary printout: per pair and basis, how many of the 12 settings are `clear` and in which direction

**Expect:** CIs of roughly a tenth of a point. Differences of 0.01–0.03 (W vs C2 at 10M no noise, W vs B at 10M
realistic) will probably **not** be clear; the large gaps at low depth (e.g. W vs A at 1M) will be. If that's what
comes out, "W had the lowest detection limit in 11 of 12 settings" becomes "W's limit was clearly lower than B's in
X of 12 settings and similar in the rest", and that's the honest version.

## 9F. Reproducibility
- Add to `scripts/run_all.sh` after the detection limits (run time in a comment), and the refined-grid run in the
  robustness block.
- README: the new outputs and what each bootstrap version holds fixed.
- `check_rerun` unaffected (no new main-grid patients); confirm it still passes.

## Send Maya
1. After 9D: the single-setting results (**before the full run**)
2. `limit_differences_ci.csv` summary: per pair, clear vs similar across the 12 settings (main), and W_G1−A in
   shift_p10 / blend50
3. The refined-grid comparison, and any ranking it changes
4. Anything that surprised you

Maya will write decision #19 and rewrite the ranking sentences in Section 3.4 and the Abstract from these.
