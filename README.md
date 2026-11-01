# When does fragment size selection help NIPT detection of trisomy 21?

CMPS 355 (Bioinformatics Algorithms), AUB, Fall 2026 - Hala Fatfat, Maya El Masri

A simulation study of when hard fragment-size selection improves non-invasive
prenatal detection of trisomy 21, across fetal fraction (FF), sequencing depth
and noise, and whether a classifier combining counts and fragment lengths
outperforms size selection.

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
scripts/               setup, mapping and run_all.sh
src/simulate/          fragment generation, mapping, mappability pool, virtual patients
src/methods/           z-score methods (A, B, B_c, B-opt, W, W_fixedFF) and decision trees (C1, C2)
src/evaluate/          one condition end to end, detection limits, bootstrap, break-even check, tables
src/evaluate/figures/  one script per manuscript figure
notebooks/             exploration only - final results must come from scripts
data/                  NOT in git (large); recreate with scripts/download_reference.sh and run_all.sh
results/figures,tables final outputs used in the manuscript
manuscript/            manuscript source
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

## Reproduce all results

```bash
bash scripts/run_all.sh
```

Runs every step in order: length QC → mappability pool → mappability table → patient and noise QC → full
96-condition grid → detection limits and FPR check → bootstrap → break-even check → figures → tables →
rerun check. About 45 min from scratch on an Apple-silicon MacBook (32 min of it is the pool), about
13 min if the pool already exists. Expected run times are noted per step in the script. Everything is
seeded from `config/config.yaml`, so a rerun reproduces the committed files in `results/` byte for byte;
the only files that change are `tables/run_log.csv` (commit and time of each run) and the timing line of
`tables/patients_qc.csv`. The pool step was checked on one chunk: chunk 0 (1M fragments), regenerated and
remapped from scratch in a separate folder, matched the committed run's chunk exactly (137 s).

To check that one condition reproduces its saved rows exactly (a few seconds):
```bash
python -m src.evaluate.check_rerun --ff 0.04 --depth 5000000 --noise realistic
```

## Where each figure and table comes from

| Output (`results/`) | Script | Content |
|---|---|---|
| Fig. 1 (drawn by hand) | — | pipeline schematic; numbers from `tables/survival_by_origin.csv` and `config/config.yaml` |
| `figures/fig2_inputs` | `src/evaluate/figures/fig2_inputs.py` | length distributions after mappability, S(l), exact g, r and g·√r against the cutoff |
| `figures/fig3_b_minus_a` | `src/evaluate/figures/fig3_b_minus_a.py` | sens_at_fpr(B) − sens_at_fpr(A) heatmaps (from `tables/fig3_b_minus_a.csv`) |
| `figures/fig4_breakeven` | `src/evaluate/figures/fig4_breakeven.py` | break-even formula vs simulation at FF 4% (from `tables/breakeven_check.csv`) |
| `figures/fig5_methods`, `figS_methods_none`, `figS_methods_low` | `src/evaluate/figures/fig5_methods.py` | sensitivity at FPR 0.135% by method, realistic / no / low noise |
| `figures/qc_*` | `src/simulate/lengths.py`, `mappability.py`, `noise_qc.py` | QC plots |
| `tables/results_all.csv` and per-condition tables | `src/evaluate/run_condition.py` | all metrics per condition × method |
| `tables/detection_limits.csv` | `src/evaluate/detection_limits.py` | detection limits for all 14 methods |
| `tables/fpr_check.csv` | `src/evaluate/fpr_check.py` | pooled FPR vs target |
| `tables/fig3_b_minus_a.csv` | `src/evaluate/bootstrap_diff.py` | B − A with paired-bootstrap SE and CI |
| `tables/breakeven_check.csv`, `breakeven_summary.csv` | `src/evaluate/breakeven_check.py` | predicted vs observed separation ratios |
| `tables/breakeven_ratio_noise.csv` | `src/evaluate/bootstrap_ratio.py` | bootstrap SD of the observed ratio (Fig. 4 caption) |
| `tables/table1_detection_limits.csv` | `src/evaluate/tables.py` | Table 1 |
| `tables/table2_realism_yang.csv` | `src/evaluate/tables.py` | Table 2 |
| `tables/tableS_*.csv` | `src/evaluate/tables.py` | supplementary: B-opt cutoffs, C2 importances, W vs W_fixedFF |
| `tables/mappability.csv`, `survival_by_origin.csv` | `src/simulate/mappability.py`, `survival_by_origin.py` | S(l), q(c), survival by origin |
| `tables/patients_qc.csv`, `noise_qc.csv` | `src/simulate/patients.py`, `noise_qc.py` | patient and noise QC |

All figure and table captions are in `docs/captions.md`, generated by `python -m src.evaluate.captions` from
the caption functions of the scripts above (numbers filled in from `results/`); do not edit it by hand.

## Reproducibility rules
- Every parameter lives in `config/config.yaml`; every random step uses the configured seed.
- Final figures and tables are produced by scripts, never edited by hand.
- Large data is never committed; it must be regenerable from scripts.
- Record design decisions in `docs/decisions.md` as you make them.
