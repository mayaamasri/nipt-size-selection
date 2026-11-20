# Task 07 (Hala): robustness to the fetal length profile, and W with mismatched weights

**Goal:** answer the two reviewer points that the current paper cannot:
- **R1-M2:** every result uses one fetal length profile (G1, 38 weeks). How do the conclusions change if the
  fetal profile is different?
- **R1-M3 / R2-M2:** W is given the exact length distributions. What happens when its weights come from a
  profile that doesn't match the patient's? (Task 06 explains W's advantage but uses the true distributions, so
  this task gives the actual answer.)

**Decisions:** #1, #9, #14, #15, #16 (new #17 added by Maya). **`git pull` first.**
**Main results must not change:** everything here is an option that is off by default. After this task,
`check_rerun` on the main grid must still reproduce the committed rows exactly.

---

## 7A. Fetal-profile scenarios

Add to `config/config.yaml`:
```yaml
robustness:
  fetal_profiles:              # decision #17; G1 itself (shift 0, blend 0) is the main analysis
    shift_m10: {shift_bp: -10, blend_maternal: 0.0}
    shift_m5:  {shift_bp: -5,  blend_maternal: 0.0}
    shift_p5:  {shift_bp: 5,   blend_maternal: 0.0}
    shift_p10: {shift_bp: 10,  blend_maternal: 0.0}
    blend50:   {shift_bp: 0,   blend_maternal: 0.5}
```

**How a profile is changed** (in `load_length_distributions`, fetal column only, maternal untouched):
1. Steps 1–2 as now (drop length 0, fix the 101 bp bin), on the **full** raw count array.
2. **Shift:** `fetal_new(l) = fetal_raw(l − shift_bp)` (positive = fetal fragments longer, so less
   different from maternal). Lengths with no source value get 0.
3. Step 3 as now: keep 36–500 bp, normalise.
4. **Blend:** `fetal = (1 − λ)·fetal + λ·maternal`, λ = `blend_maternal` (fetal fragments look more like
   maternal ones).

Mappability stays the same: S(l) depends only on length and was the same for both origins (decision #3,
`survival_by_origin.csv`), so **no remapping**. `origin_matrices` then works unchanged.

Why these values: the G1 median fetal length is 157 bp vs 172 bp maternal (a 15 bp gap). ±5 and ±10 bp cover
"a bit more / less different" to "much less different" (+10 leaves a 5 bp gap). We don't claim any value is the
early-pregnancy profile. It's a sensitivity analysis, and the text will say so. Shifting moves both fetal peaks
(143 and 166 bp) together; blending keeps their positions and only weakens the difference. Together they cover
both kinds of change.

**QC table** `results/tables/robustness_profiles_inputs.csv`, one row per scenario (plus G1): fetal median
length, and for the 40–156 bp window at FF 2%, 4% and 10%: g, r, g·√r, g²r (from `window_stats`, exact).

## 7B. Methods in a scenario

Patients (all groups, including the reference) are generated from the **scenario** profile. Methods:

| Method | Weights / features | Role |
|---|---|---|
| A, B, B_c, B-opt | unchanged (B-opt still picks its cutoff on the training set) | B-opt shows whether the best cutoff moves |
| `W` | w*(l) from the **scenario** profile, true FF | oracle: a lab that knows the true profile |
| `W_G1` (**new**) | w(l) from the **G1** profile, true FF | realistic: a lab using a published profile |
| `W_fixedFF` | **G1** profile, FF = 10% | realistic and no FF needed |
| C1 | unchanged | |
| C2 | z_weighted from **`W_G1`** weights | realistic, so C2 isn't given the true profile either |

Implementation suggestion: build `matrices_g1 = origin_matrices(base config)` once and pass it to
`statistics()` for the G1-weighted statistics (`weighted_g1`, `weighted_fixedff`); `weighted` keeps the
scenario matrices. In the main analysis the scenario **is** G1, so `W_G1 = W`. **Don't add `W_G1` to the main
results table**, so the committed files stay byte-identical.

The FF used in the weights is still the condition's true FF (decision #9); `W_fixedFF` covers an unknown FF.

## 7C. Predictions first (before any simulation)

With the Task 06 model, for every scenario × FF × depth × noise: `ratio_pred` vs A for **B, W, W_G1, W_fixedFF**.
For W_G1 and W_fixedFF: u = G1 weights, while E[·] and w* use the **scenario** profile (Task 06 note).
Save `results/tables/robustness_profiles_pred.csv`.

My preview at FF 4%, realistic noise (ratio vs A). Yours should match closely:

| Scenario | g·√r (FF 4%) | 1M: W / W_G1 / B | 10M: W / W_G1 / B |
|---|---|---|---|
| shift_m10 | 1.42 | 1.78 / 1.69 / 1.50 | 2.19 / 1.91 / 1.96 |
| shift_m5 | 1.25 | 1.60 / 1.52 / 1.32 | 1.91 / 1.72 / 1.73 |
| G1 (main) | 1.09 | 1.36 / 1.36 / 1.16 | 1.54 / 1.54 / 1.52 |
| shift_p5 | 0.94 | 1.25 / 1.21 / 0.99 | 1.37 / 1.36 / 1.31 |
| shift_p10 | 0.79 | 1.13 / 1.07 / 0.84 | 1.18 / 1.21 / 1.10 |
| blend50 | 0.77 | 1.11 / 1.08 / 0.82 | 1.15 / 1.21 / 1.07 |

Two things that look like bugs but aren't:
- **shift_p10 at 10M: W_G1 > W.** The oracle weights are only optimal without depth-independent noise (Task 06C).
  The G1 weights are more contrastive for this profile, so they're closer to u_opt. Check it with
  `optimal_weights`.
- **blend50: W_G1 loses very little.** At low FF a blend changes the weights almost by a constant, and a
  z-score ignores constants. That's why shifts are the main test.

**Send Maya the 7A QC table and this prediction table before running the grid.** If anything differs from my
preview by more than ~0.02, stop there.

## 7D. Run the grid per scenario

- Add `--scenario NAME` to `run_condition` (and `run_all.sh`'s grid loop). It loads the config, applies
  `robustness.fetal_profiles[NAME]`, and sets `paths.results` to `results/robustness/<NAME>`, so nothing in
  `results/tables` is touched.
- Run the **full 96-condition grid** for each of the 5 scenarios (~5 min each).
- **Seeds: the same spawn keys as the main grid** (no scenario index). Because the multinomial probabilities change
  with the profile, this gives only **partial** pairing between scenarios; the docstring should say that and not
  claim more.
- `--scenario` also goes into `detection_limits`, `fpr_check`, `bootstrap_diff` (it regenerates patients, so it
  must use the scenario profile), `breakeven_check` (needed by `unified_model_check` self-test 2) and
  `unified_model_check` (extended to W_G1). Run all of them per scenario; `bootstrap_diff` is ~7.5 min each.
- `W_G1` is added to `method_order` only when a scenario is set. Run `check_rerun` on the main grid **before and
  after** the change.

Total ≈ 1 h of compute.

## 7E. Summary across scenarios

`src/evaluate/robustness_profiles.py` → `results/tables/robustness_profiles_summary.csv` and a supplementary
figure. It must answer these questions (the manuscript text will follow them):

1. **Is B ever worse than A?** Per scenario: the number of conditions with sep(B)/sep(A) < 1, and the number with
   B − A `sens_at_fpr` **significantly** negative (bootstrap CI below 0). List the conditions.
2. **Does the depth result survive?** At FF 4%: is sep(B)/sep(A) flat with depth at σ = 0 and rising with depth at
   σ > 0? (ratio at 1M and 10M per noise level)
3. **Does the model still predict the simulation?** Median |rel_error| per scenario for B, W, W_G1 (from
   `unified_model_check`).
4. **What does a wrong profile cost W?** Detection limits of W vs W_G1 vs W_fixedFF per scenario × depth × noise,
   and the separation loss W_G1 / W.
5. **Ranking:** W_G1 vs B detection limits per scenario × depth × noise. Where does B beat W_G1? (Expected:
   high depth with realistic noise, and in the shift_m scenarios.)
6. **False positives:** pooled FPR of W_G1 and C2 per scenario. A wrong profile should **not** raise the FPR,
   because the reference uses the same weights. This is a key point for the response letter, so check it.
7. **B-opt:** which cutoffs it chooses per scenario (does the best cutoff move away from 156 bp?).

**Supplementary figure** `figS_profiles.(png|pdf)`: FF 4%, realistic noise, x = depth, y = ratio vs A, one panel
per scenario (6 including G1), lines = prediction, points = simulation, for B, W and W_G1. Dashed line at 1.

## 7F. Reproducibility
- Add the scenario runs and summary to `scripts/run_all.sh` (an optional block with run times in comments).
- README: the new outputs and the `--scenario` flag.
- `check_rerun` must still pass on the main grid, and should also accept `--scenario` (check one scenario
  condition reproduces exactly).

## Send Maya
1. After 7C: the QC table and the predictions (**before the grid**)
2. After 7E: `robustness_profiles_summary.csv`, with the answers to questions 1–7 in a few lines each
3. `figS_profiles.png`
4. Anything that surprised you

Maya will write decision #17, and the new Results section ("Robustness of the conclusions") will be built from
question 1–7 answers.
