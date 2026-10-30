# Task 05 (Hala): figures, tables, and a reproducible repository

**Goal:** turn the full results into the figures and tables of the manuscript, and make the repository
something a reviewer can rerun.
Uses the outputs of Task 04. **Decisions:** #8, #13, #15.

All figures: one script each in `src/evaluate/figures/`, saved as PNG (300 dpi) **and** PDF in `results/figures/`,
same style as your QC plots, readable in greyscale (use line styles/markers, not only colour).

---

## Figure 2: inputs of the simulation
Three panels:
(a) maternal and fetal length distributions **after mappability** (with the 40–156 bp window shaded);
(b) survival S(l) by length;
(c) exact `g` (FF enrichment) and `r` (fraction kept) as a function of the upper cutoff (120–170 bp, lower = 40),
at FF 4%, with Yang's 156 bp marked.

## Figure 3: when does size selection help? (main result, research question 1)
Heatmaps of **`sens_at_fpr(B) − sens_at_fpr(A)`** over FF (rows) × depth (columns), **one panel per noise level**,
diverging colour scale centred on 0, values written in the cells.

## Figure 4: does the break-even formula predict the simulation? (research question 1)
For FF 4%, one panel per noise level: x = depth (log scale), y = `zB/zA` = `z_mean_t21(B_c) / z_mean_t21(A)`,
one line per cutoff (130, 140, 150, 156).
- **Points** = simulation (from `results_all.csv`).
- **Lines** = prediction (decision #8):
  `ratio(N) = g · sqrt( (1/(p0·N) + σ²) / (1/(p0·r·N) + σ²) )` using that cutoff's exact `g`, `r`, and `p0`.
- Dashed horizontal line at 1 (B = A).
Also save the predicted vs observed ratios for all FF × depth × noise × cutoff as
`results/tables/breakeven_check.csv` with the relative error.

## Figure 5: which method is best? (research question 2)
`sens_at_fpr` against FF, one panel per depth, **realistic noise**: A, B, W, C1, C2
(same figure for the other noise levels saved as supplementary).

## Tables
- **Table 1:** detection limits (main, conservative, check flag) for A, B, B-opt, W, C1, C2,
  per depth × noise, compared with the 4% commercial minimum → `results/tables/table1_detection_limits.csv`.
- **Table 2:** realism check: our g and r at FF 4% and 10% next to Yang et al. (FF ~2×, 75% of reads discarded).
- **Supplementary:** B-opt chosen cutoff per condition; C2 feature importances per condition; W vs W_fixedFF.

## Reproducibility
1. `scripts/run_all.sh`: from a fresh clone (after `download_reference.sh`) runs every step in order:
   pool → mappability → full grid → detection limits → figures. Note the expected run times in comments.
2. `environment.lock.yml` from `conda env export --no-builds` (keep `environment.yml` as the readable version).
3. Update `README.md`: what the project does, how to reproduce, where each figure/table comes from.
4. Check that rerunning one condition reproduces its `results_all.csv` rows exactly.

## Figure 1 (pipeline schematic)
Not code: Maya will draw it. Just send her the list of steps with the key numbers (pool size, survival,
patients per group) so the schematic matches the code.
