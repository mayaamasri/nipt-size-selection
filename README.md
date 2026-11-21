# When does fragment size selection help NIPT detection of trisomy 21?

CMPS 355 (Bioinformatics Algorithms), AUB, Fall 2026 - Hala Fatfat, Maya El Masri
A simulation study of when hard fragment-size selection improves non-invasive prenatal detection of trisomy 21, across fetal fraction (FF), sequencing depth and noise, and whether keeping all fragments and weighting them by length works better.

## What the study does

1. **Fragment lengths.** Maternal and fetal cell-free DNA lengths come from the exact per-bp counts behind
   Rabinowitz et al. Fig. 1 (family G1), 36–500 bp (`docs/decisions.md` #1, #2).
2. **Mappability.** A pool of 10M maternal + 10M fetal fragments is simulated on GRCh38, mapped with
   Bowtie2 (2×36 bp, proper pairs, MAPQ ≥ 40) and used only to estimate survival by length S(l) and the
   chromosome share of survivors q(c) (#3, #11, #12). 86.2% of fragments survive.
3. **Virtual patients.** Each patient is an independent multinomial draw of surviving fragments
   (chr21 / other autosomes × length), mixed by FF; T21 adds 50% fetal chr21; optional per-patient noise
   σ on the chr21 share (#3, #7). Per FF × depth × noise condition: 2,000 reference normals, 10,000 + 2,000
   training, 10,000 calibration normals (only to set the trees' threshold), 10,000 + 2,000 test (#4).
4. **Methods.** A: chr21 z-score on all fragments. B: the same on 40–156 bp fragments (Yang et al.);
   B_c: fixed cutoffs 120–165 bp; B-opt: cutoff chosen on the training set. W: length-weighted z-score.
   C1 / C2: decision trees on zA / on zA + z_short + z_weighted (#5, #6, #9, #10, #15).
5. **Grid.** FF 1, 2, 3, 4, 6, 8, 10, 15% × depth 1, 2.5, 5, 10M × noise σ = 0, 0.15, 0.35% (96 conditions).
6. **Evaluation.** Sensitivity, FPR and AUC on the test set, sensitivity at FPR 0.135% from the ROC,
   detection limits (#13), and a check of the break-even formula for size selection (#8, #14).

All parameters are in `config/config.yaml`; every design choice and its reason is in `docs/decisions.md`.

## Repository layout

```
config/config.yaml     all study parameters (single source of truth)
docs/decisions.md      design decision log + work split
docs/tasks/            task briefs
docs/captions.md       generated figure and table captions (do not edit)
scripts/               setup, mapping and run_all.sh
src/simulate/          fragment generation, mapping, mappability pool, virtual patients
src/methods/           z-score methods (A, B, B_c, B-opt, W, W_fixedFF) and decision trees (C1, C2)
src/evaluate/          one condition end to end, detection limits, bootstrap, break-even check, tables
src/evaluate/figures/  one script per manuscript figure
data/                  NOT in git (large); recreate with scripts/download_reference.sh and run_all.sh
results/figures,tables final outputs used in the manuscript
results/robustness/    per-scenario tables of the robustness analyses (Task 07 fetal profiles, Task 08 short-fragment noise)
results/refined_grid/  extra FF conditions: the 14-point detection-limit grid (Task 09, decision #19)
```

## Setup

### 1. Prerequisites
- **macOS**: Terminal is enough.
- **Windows**: install WSL2 (Ubuntu) and do everything inside it:
  https://learn.microsoft.com/windows/wsl/install
- Git: https://git-scm.com/downloads
- Miniforge (conda): https://github.com/conda-forge/miniforge

### 2. Clone and create the environment
```bash
git clone https://github.com/mayaamasri/nipt-size-selection.git
cd nipt-size-selection
conda env create -f environment.yml
conda activate nipt
```

`environment.yml` is the readable list of what the project needs. `environment.lock.yml` pins the exact
versions the results were produced with (`conda env export --no-builds`; Python 3.11.17, Bowtie2 2.5.5,
samtools 1.24, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, Matplotlib 3.11.2). It was exported on
**macOS 26.6 (Apple silicon, arm64)**; on that platform use it for an exact rebuild:
```bash
conda env create -f environment.lock.yml
```
On Linux/WSL some pinned versions may not exist for the platform; use `environment.yml` there.

### 3. Download the reference genome and Bowtie2 index (~5 GB download)
```bash
bash scripts/download_reference.sh
```

### 4. Check everything works
```bash
python scripts/check_env.py
```

The Methods' 10k-fragment mapping check (every simulated read re-fetched from the reference with an independent
reverse complement, including the overlapping-reads case at short lengths) is
`python -m src.simulate.check_fragments --out-prefix data/processed/task01/sim10k`; it is a one-off validation of
the fragment simulator and is not part of `run_all.sh`.

## Reproduce all results

```bash
bash scripts/run_all.sh
```

Runs every step in order: length QC → mappability pool → mappability table → patient and noise QC → full
96-condition grid → detection limits, FPR check, bootstrap, break-even and unified-model checks → the 14-point
detection-limit grid → robustness scenarios (Tasks 07, 08) → detection-limit CIs (Task 09) → figures → tables,
LaTeX tables, numbers sheet and captions → rerun check. About 1 h from scratch on an Apple-silicon MacBook
(32 min of it is the pool); `bash scripts/run_all.sh --robustness` also reruns the 7 robustness scenarios
(~2.5–3 h in all). Expected run times are noted per step in the script. Everything is seeded from
`config/config.yaml`, so a rerun reproduces the committed files in `results/` byte for byte; the only files that
change are the `run_log.csv` files (commit, time and dirty flag of each run) and the timing line of
`tables/patients_qc.csv`.

**Two FF grids.** The simulated grid has 8 FFs (1, 2, 3, 4, 6, 8, 10, 15%): per-condition results, Figs. 3–5 and
all robustness-scenario limits use it. Every main-grid detection limit in the paper (Table 2, Tables S4, S6–S8)
uses a 14-point grid that adds 1.5, 2.5, 3.5, 5, 7 and 9% (decision #19; `results/refined_grid`). The pool step was checked on one chunk: chunk 0 (1M fragments), regenerated and
remapped from scratch in a separate folder, matched the committed run's chunk exactly (137 s).

To check that one condition reproduces its saved rows exactly (a few seconds):
```bash
python -m src.evaluate.check_rerun --ff 0.04 --depth 5000000 --noise realistic
```

### Robustness to the fetal length profile (`--scenario`)

`config/config.yaml` → `robustness.fetal_profiles` defines five fetal-profile scenarios (decision #17): the G1
fetal length distribution shifted by −10, −5, +5 or +10 bp, or blended 50:50 with the maternal one. They are
off by default; the main analysis (G1) does not change. `run_condition`, `check_rerun`, `detection_limits`,
`fpr_check`, `bootstrap_diff`, `breakeven_check` and `unified_model_check` accept `--scenario NAME`, which loads
the config through `load_config(scenario=NAME)` (`src/config.py`): all patients come from the scenario profile,
the method W_G1 (weights from the G1 profile) is added, W_fixedFF and C2 use the G1 weights, and every output
goes to `results/robustness/NAME/tables` instead of `results/tables`. Patient seeds are the same as the main
grid's, which pairs scenarios only partially.

```bash
bash scripts/run_all.sh --skip-pool --robustness
```
reruns all five scenarios as well (sequential time ~13 min per scenario, ~65 min for all five; they can also run in
parallel, one job per scenario). Without `--robustness`, `run_all.sh` uses the committed scenario tables. One
scenario condition can be checked like a main one:
```bash
python -m src.evaluate.check_rerun --scenario shift_p10 --ff 0.04 --depth 10000000 --noise realistic
```

### Extra noise on size-selected fragments (`--scenario snoise_k15`, `snoise_k2`)

`robustness.short_noise` defines two more scenarios (decision #18): besides the per-patient noise on the whole
chr21 row, the chr21 fragments of 40–156 bp get a second, independent noise σ_s = σ·√(k² − 1), so the window's
chr21 fraction is about k times noisier (k = 1.5 and 2). They use the same `--scenario` flag and output folders,
the G1 fetal profile, and only the low and realistic noise levels (64 conditions; `none` is skipped through
`src.config.scenario_noise_levels`, never by editing the grid, because patient seeds use the level's index).
`bash scripts/run_all.sh --robustness` also reruns them: sequential time ~9 min per scenario (~18 min for both);
they can also run in parallel, one job per scenario.

### Uncertainty of the detection limits (Task 09)

`python -m src.evaluate.limit_ci --jobs 6` bootstraps the detection limits (1,000 paired resamples: every method
is scored on the same resampled patients; percentile 95% CIs). What each version holds fixed:

| Version | Resampled | Held fixed | Methods |
|---|---|---|---|
| `test` | test normals (10,000) and test T21 (2,000), per FF | reference mean and SD of every statistic, trained trees, tree thresholds, B-opt's cutoff | all |
| `test+ref` | the same test resamples **and** the 2,000 reference normals (new mean and SD) | B-opt's cutoff and the z > 3 threshold (trees not included) | z-score methods only |

Both versions condition on the trained models and thresholds, so training and calibration variability are not
included. `test+ref` is the main version for z-score comparisons (it includes the reference's sampling error,
which moves all of a condition's normals together); `test` is kept for comparison and is the only version for the
trees. `sens_at_fpr` depends only on the ranking within the test set, so it is identical in both versions (checked
exactly). Each limit is reported on two bases: `sens` (the method at its own threshold, the paper's limit) and
`sens_at_fpr` (all methods at the same FPR). For the trees (C1, C2) the two differ in an important way: on `sens`
they are compared at their own achieved FPRs (`pooled_test_fpr`), so a clear difference may come from a different
FPR rather than better detection; `sens_at_fpr` is the fair comparison, but with only a few distinct leaf scores
its CIs are lumpy (the limit jumps when the number of resampled normals in the top leaf crosses the target FPR).
A tree pair is therefore never called clear from the `sens` basis alone. A difference gets a CI only if both
limits are numeric in at least 95% of resamples. Seeds: family 5 (test and reference streams per setting).

**14-point limit grid (decision #19).** The main grid's detection limits in the paper (Table 2, Tables S4 and
S6–S8) use a 14-point FF grid: the 8 grid FFs plus
`evaluation.limit_extra_ff` (1.5, 2.5, 3.5, 5, 7, 9%), each simulated at every depth × noise setting
(`python -m src.evaluate.refined_grid --all-settings`, 72 conditions in `results/refined_grid`, seeds family 7).
`python -m src.evaluate.detection_limits_refined` writes `detection_limits_refined.csv` and `limits_dip_flags.csv`
(limits that move because one new FF dips below 95% and the curve recovers), and
`python -m src.evaluate.limit_ci --refined --jobs 6` the CIs on that grid. The scenarios, the per-condition
results, Figs. 3–5 and `detection_limits.csv` stay on the 8 simulated grid FFs.

`python -m src.evaluate.refined_grid` checks the linear interpolation between grid FFs: it simulates the midpoint
FF of the interval holding two close limits (16 conditions, seeds family 7, listed in `uncertainty.refine_conditions`)
and recomputes those limits on the refined grid.

## Where each figure and table comes from

| Output (`results/`) | Script | Content |
|---|---|---|
| Fig. 1 (drawn by hand) | — | pipeline schematic; numbers from `tables/survival_by_origin.csv` and `config/config.yaml` |
| `figures/fig2_inputs` | `src/evaluate/figures/fig2_inputs.py` | length distributions after mappability, S(l), exact g, r and g·√r against the cutoff |
| `figures/fig3_b_minus_a` | `src/evaluate/figures/fig3_b_minus_a.py` | sens_at_fpr(B) − sens_at_fpr(A) heatmaps (from `tables/fig3_b_minus_a.csv`) |
| `figures/fig4_breakeven` | `src/evaluate/figures/fig4_breakeven.py` | break-even formula vs simulation at FF 4% (from `tables/breakeven_check.csv`) |
| Fig. 5, Figs. S1, S2: `figures/fig5_methods`, `figS1_methods_none`, `figS2_methods_low` | `src/evaluate/figures/fig5_methods.py` | sensitivity at FPR 0.135% by method, realistic / no / low noise |
| Fig. S3: `figures/figS3_unified_model` | `src/evaluate/figures/figS_unified_model.py` | unified counting model vs simulation for B and W at FF 4% (from `tables/unified_model_check.csv`) |
| Fig. S4: `figures/figS4_profiles` | `src/evaluate/figures/figS_profiles.py` | B, W and W_G1 vs A under each fetal-profile scenario, FF 4%, realistic noise: model vs simulation |
| Fig. S5: `figures/figS5_short_noise` | `src/evaluate/figures/figS_short_noise.py` | B and W vs A with extra noise on short fragments (k = 1, 1.5, 2), FF 4%: model vs simulation |
| `tables/detection_limits_ci.csv`, `limit_differences_ci.csv` | `src/evaluate/limit_ci.py` | bootstrap 95% CIs of the detection limits (with each method's pooled test FPR) and of method differences (`clear` = CI excludes 0), main grid and shift_p10 / blend50 |
| `tables/detection_limits_refined.csv`, `limits_dip_flags.csv` | `src/evaluate/detection_limits_refined.py` | main-grid detection limits on the 14-point FF grid (used by Table 2 and Tables S4, S6–S8), and limits moved by a dip at a new FF |
| `tables/detection_limits_ci_refined.csv`, `limit_differences_ci_refined.csv` | `src/evaluate/limit_ci.py --refined` | bootstrap CIs of the 14-point limits and of the method differences |
| `tables/limits_refined_grid.csv`, `limits_refined_pairs.csv` | `src/evaluate/refined_grid.py` | interpolated vs refined-grid limits where pairs differ by < 0.2 points, and the rankings that change |
| `figures/qc_*` | `src/simulate/lengths.py`, `mappability.py`, `noise_qc.py` | QC plots |
| `tables/results_all.csv` and per-condition tables | `src/evaluate/run_condition.py` | all metrics per condition × method |
| `tables/detection_limits.csv` | `src/evaluate/detection_limits.py` | detection limits for all 14 methods |
| `tables/fpr_check.csv` | `src/evaluate/fpr_check.py` | pooled FPR vs target |
| `tables/fig3_b_minus_a.csv` | `src/evaluate/bootstrap_diff.py` | B − A with paired-bootstrap SE and CI |
| `tables/breakeven_check.csv`, `breakeven_summary.csv` | `src/evaluate/breakeven_check.py` | predicted vs observed separation ratios |
| `tables/unified_model_check.csv`, `unified_model_summary.csv` | `src/evaluate/unified_model_check.py` | unified counting model (any length weighting) vs observed ratios for B, B_c, W, W_fixedFF; W vs the optimal weights (analytic) |
| `tables/robustness_profiles_inputs.csv`, `robustness_profiles_pred.csv` | `src/evaluate/robustness_inputs.py` | per fetal-profile scenario: median lengths, g, r of the 40–156 bp window; model predictions for B, W, W_G1, W_fixedFF (before any simulation) |
| `tables/robustness_profiles_summary.csv` | `src/evaluate/robustness_profiles.py` | robustness questions 1–7 across scenarios (B vs A, depth effect, model error, cost of a wrong profile, ranking, FPR, B-opt cutoffs) |
| `tables/robustness_snoise_pred.csv`, `robustness_snoise_qc.csv`, `snoise_budis_illustration.csv` | `src/evaluate/robustness_snoise_inputs.py` | short-fragment noise: model predictions with σ_u (before any simulation), noise QC on the reference normals, and the k that would match Budiš et al. (illustration, model only) |
| `tables/robustness_snoise_summary.csv` | `src/evaluate/robustness_snoise.py` | short-fragment noise questions 1–6 across k (B vs A, depth effect, model error, W vs B vs A, B-opt, FPR) |
| `robustness/<NAME>/tables/*` | the scripts above with `--scenario NAME` | the main-grid tables (results_all, detection limits, FPR, bootstrap, break-even, unified model) for one scenario |
| `tables/breakeven_ratio_noise.csv` | `src/evaluate/bootstrap_ratio.py` | bootstrap SD of the observed ratio (sampling noise of Fig. 4) |
| Table 1: `tables/table1_realism_yang.csv` | `src/evaluate/tables.py` | realism check vs Yang et al. |
| Table 2: `tables/table2_detection_limits.csv` | `src/evaluate/tables.py` | detection limits with 95% CIs, bold (lowest or not clearly different) and tree flags (14-point grid) |
| Tables S1–S4: `tables/tableS1_bopt_cutoff.csv`, `tableS2_c2_importances.csv`, `tableS3_w_vs_wfixed_sens.csv`, `tableS4_w_vs_wfixed_limits.csv` | `src/evaluate/tables.py` | B-opt cutoffs, C2 importances, W vs W_fixedFF (sensitivity; limits on the 14-point grid) |
| Table S5: `tables/tableS5_auc.csv` | `src/evaluate/tables.py` | AUC of every method and condition (the LaTeX table shows the 7 main methods) |
| Tables S6–S8: `tables/tableS6_limits_conservative_fpr.csv`, `tableS7_limits_8_vs_14.csv`, `tableS8_limit_differences.csv` | `src/evaluate/tables.py` | conservative and matched-FPR limits (CIs for the latter), 8- vs 14-point limits, method differences with CIs (14-point grid) |
| Table S9 (parts a–d): `tables/tableS9a_profile_inputs.csv` … `tableS9d_wg1_vs_a.csv` | `src/evaluate/tables.py` | fetal-profile scenarios (8-point grid) |
| Table S10 (parts a–c): `tables/tableS10a_snoise_rise.csv` … `tableS10c_snoise_regimes.csv` | `src/evaluate/tables.py` | short-fragment noise and the four regimes (8-point grid) |
| `tables/snoise_regimes.csv` | `src/evaluate/snoise_regimes.py` | regime (i)–(iv) per cutoff × FF × k from g, r and the two-σ model (model only) |
| `tables/latex/*.tex` | `src/evaluate/latex_tables.py` | every table above as LaTeX (booktabs; `table1_realism_yang.tex` and `table2_detection_limits.tex` are the bare tabular, caption in main.tex; `tableS1_…` to `tableS10_…` with their captions, S9 and S10 one table each; reader labels such as W (FF 10%) and +10 bp; needs booktabs, longtable, amsmath) |
| `supplement/additional_file_1.tex` | `src/evaluate/additional_file.py` | Additional file 1: Supplementary Note S1 (empty, written in the manuscript), Figs. S1–S5 with captions, Tables S1–S10. Compile with `cd supplement && tectonic additional_file_1.tex` (any LaTeX with amsmath, booktabs, longtable, graphicx works); the submitted version, with Supplementary Note S1 filled in, is `supplement/Additional_file_1_submitted.pdf` |
| `docs/revision_numbers.md` | `src/evaluate/revision_numbers.py` | every number quoted in the revised text, with its source file |
| `tables/mappability.csv`, `survival_by_origin.csv` | `src/simulate/mappability.py`, `survival_by_origin.py` | S(l), q(c), survival by origin |
| `tables/patients_qc.csv`, `noise_qc.csv` | `src/simulate/patients.py`, `noise_qc.py` | patient and noise QC |

All figure and table captions are in `docs/captions.md`, generated by `python -m src.evaluate.captions` from
the caption functions of the scripts above; captions describe what is shown, and the numbers quoted in the
text are in `docs/revision_numbers.md`. Do not edit either by hand.

## Reproducibility rules
- Every parameter lives in `config/config.yaml`; every random step uses the configured seed.
- Final figures and tables are produced by scripts, never edited by hand.
- Large data is never committed; it must be regenerable from scripts.
- Record design decisions in `docs/decisions.md` as you make them.
