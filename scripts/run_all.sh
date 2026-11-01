#!/usr/bin/env bash
# Task 05: run the whole study, in order, from a fresh clone.
#
# Before: conda env create -f environment.yml (or environment.lock.yml); conda activate nipt;
#         bash scripts/download_reference.sh   (GRCh38 analysis set + Bowtie2 index, ~5 GB)
# Run from anywhere:  bash scripts/run_all.sh
#
# Expected run times (measured on an Apple-silicon MacBook, macOS 26, nipt env) are noted per step;
# total ~45 min from scratch, ~13 min when the pool already exists (measured: 793 s). Every step is deterministic (seeds
# in config/config.yaml), so a rerun reproduces the committed results/ files. Steps that already have
# their output are not skipped, except pool chunks (pool.py skips chunks whose Parquet exists).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, so config paths resolve

if ! python -c "import pysam, sklearn, matplotlib" 2>/dev/null; then
  echo "error: activate the nipt conda env first (conda activate nipt)" >&2
  exit 1
fi
if [ ! -f data/reference/hg38.analysisSet.fa ]; then
  echo "error: reference missing; run bash scripts/download_reference.sh first" >&2
  exit 1
fi

step() {   # step "<label>" <command...>: run, then print the elapsed time
  local label="$1"; shift
  echo "==> $label"
  local start=$SECONDS
  "$@"
  echo "    done in $((SECONDS - start)) s"
}

# --- 1. Simulation inputs -------------------------------------------------------------------------
# Length distributions QC (decision #1): seconds
step "length distributions QC" python -m src.simulate.lengths
# Mappability pool: 10M maternal + 10M fetal fragments, 20 chunks, simulated, mapped with Bowtie2 and
# filtered at MAPQ >= 40 (decision #3): ~32 min (~135 s per chunk); finished chunks are skipped
step "mappability pool" python -u -m src.simulate.pool
# S(l) and q(c) from the pool, QC plots: ~6 s
step "mappability table" python -m src.simulate.mappability
# Survival by origin per length bin (Fig. 1, Fig. 2b): ~5 s
step "survival by origin" python -m src.simulate.survival_by_origin
# Virtual-patient QC (Task 02) and noise QC (decision #7): a few seconds. patients_qc.csv also records a
# timing, so that one value changes on every run
step "patients QC" python -m src.simulate.patients
step "noise QC" python -m src.simulate.noise_qc

# --- 2. Full grid: 8 FF x 4 depths x 3 noise levels (96 conditions) -------------------------------
# Each condition generates its patients, runs A, B, B_c, B-opt, W, W_fixedFF, C1, C2 and replaces its
# rows in results/tables/: ~3 s per condition, ~5.5 min in all (measured 319 s). run_log.csv records the
# git commit, time and dirty flag of each run, so it changes on every run
grid_conditions() {
  python -c '
import yaml
g = yaml.safe_load(open("config/config.yaml"))["grid"]
for noise in g["noise_sigma"]:
    for depth in g["depth"]:
        for ff in g["fetal_fraction"]:
            print(ff, depth, noise)
'
}
run_grid() {
  local n=0
  while read -r ff depth noise; do
    n=$((n + 1))
    echo "    condition $n: FF $ff, depth $depth, noise $noise"
    python -u -m src.evaluate.run_condition --ff "$ff" --depth "$depth" --noise "$noise" > /dev/null
  done < <(grid_conditions)
}
step "full grid (96 conditions)" run_grid

# --- 3. Evaluation --------------------------------------------------------------------------------
# Detection limits (decision #13) and pooled FPR check: seconds
step "detection limits" python -m src.evaluate.detection_limits
step "FPR check" python -m src.evaluate.fpr_check
# Paired bootstrap of sens_at_fpr(B) - sens_at_fpr(A), 2,000 resamples x 96 conditions (Fig. 3): ~7.5 min (measured 446 s)
step "bootstrap B - A" python -u -m src.evaluate.bootstrap_diff
# Break-even formula vs simulation (decision #8, Fig. 4): seconds
step "break-even check" python -m src.evaluate.breakeven_check

# --- 4. Figures and tables (results/figures, results/tables): seconds each -----------------------
step "Figure 2" python -m src.evaluate.figures.fig2_inputs
step "Figure 3" python -m src.evaluate.figures.fig3_b_minus_a
step "Figure 4" python -m src.evaluate.figures.fig4_breakeven
step "Figure 5 + supplementary" python -m src.evaluate.figures.fig5_methods
step "Tables 1, 2 + supplementary" python -m src.evaluate.tables

# --- 5. Reproducibility check: rerun one condition and compare with its saved rows: ~3 s ---------
step "rerun check" python -m src.evaluate.check_rerun

echo "All steps finished in $SECONDS s."
