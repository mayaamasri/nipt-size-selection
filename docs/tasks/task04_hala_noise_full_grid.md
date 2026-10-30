# Task 04 (Hala): noise, extra analyses, and the full experiment

**Goal:** add the noise model (proposal step 5), add three cheap extra analyses that make the conclusions
much stronger, and run all 96 conditions.
**Decisions:** `docs/decisions.md` #7, #8, #15. **`git pull` first.**

---

## 4A. Noise (decision #7)

Config: keep `noise_sigma: [none, low, realistic]` in that order (seed keys use the index) and add
```yaml
  noise_sigma_values: [0.0, 0.0015, 0.0035]   # relative SD of each patient's chr21 share, decision #7
```

In `generate_patients`, per patient:
1. Build the patient's probability matrix as now (FF mixture, trisomy if T21).
2. Draw `eps ~ Normal(0, sigma)` (patient's own value, from the patient stream).
3. Multiply the **whole chr21 row** (all lengths) by `(1 + eps)`, renormalise the matrix to sum to 1.
4. `counts = rng.multinomial(depth, P)`.

Noise applies to **every** group (reference, training, calibration, test; normals and T21).
With `sigma = 0` the output must stay **byte-identical** to the current patients (please check).

**QC (normals, FF 4%):** SD of the chr21 fraction across 2,000 normals vs. theory
`p0 · sqrt(1/(p0·N) + sigma²)` (p0 = q(chr21)), for each noise level at 1M and 10M.

## 4B. Extra methods (decision #15)

All computed from the same patients, no retraining except where stated:

| Name | What | Why |
|---|---|---|
| `B_120`, `B_130`, `B_135`, `B_140`, `B_145`, `B_150`, `B_165` | Method B with a **fixed** window `[40, c]` (main B stays `[40, 156]`) | Lets us map where size selection stops paying off as the cutoff gets stricter; the break-even formula predicts a real break-even depth only for cutoffs below ~145 bp |
| `W` | `z_weighted > 3` (a plain threshold, no tree) | Tells us whether C2's gain comes from the weighting or from the tree |
| `W_fixedFF` | same as `W`, but the weights use **FF = 10% for every condition** instead of the true FF | Tests the "true FF is idealised" limitation. At low FF the FF mostly cancels out of the weight, so we expect W ≈ W_fixedFF |

Evaluate each exactly like A and B (sens, CIs, FPR, AUC, `sens_at_fpr`). For every `B_c` also store its exact `g` and `r`.

## 4C. Extra columns for the break-even check

For every z-score method (A, B, B_c, B-opt, W, W_fixedFF) add to `results_all.csv`:
- `z_mean_t21`: mean z of the **test T21** patients
- `z_sd_normal`: SD of z among the **test normals**

These give the observed signal-to-noise ratio, which we compare with the formula in Task 05.
Also add `p0` (q(chr21)) and `sigma` (numeric) columns.

## 4D. Run everything

- Sanity rerun of the milestone (FF 4%, 5M, none) with `target_fpr: 0.00135`: numbers should barely change.
- Then **all 96 conditions** (8 FF × 4 depths × 3 noise). Log each in `run_log.csv` as now.
- Then `detection_limits.py` on the full results (for all methods, including the new ones).
- Send Maya: total run time, any condition with `nobody_positive`, any detection-limit flags,
  and the QC table from 4A.

---

### Commit
Code, `results/tables/*.csv`, QC figures. Not the patients (too big).
