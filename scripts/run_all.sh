#!/usr/bin/env bash
# Task 05: run the whole study, in order, from a fresh clone.
#
# Before: conda env create -f environment.yml (or environment.lock.yml); conda activate nipt;
#         bash scripts/download_reference.sh   (GRCh38 analysis set + Bowtie2 index, ~5 GB)
# Run from anywhere:  bash scripts/run_all.sh
#                     bash scripts/run_all.sh --skip-pool   (use an existing data/processed/pool as is)
#                     bash scripts/run_all.sh --robustness  (also rerun the Task 07 and 08 robustness scenarios)
#
# Expected run times (measured on an Apple-silicon MacBook, macOS 26, nipt env) are noted per step;
# total ~45 min from scratch, ~13 min when the pool already exists (measured: 793 s). Every step is deterministic (seeds
# in config/config.yaml), so a rerun reproduces the committed results/ files. Steps that already have
# their output are not skipped, except pool chunks (pool.py skips chunks whose Parquet exists).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, so config paths resolve

SKIP_POOL=0
ROBUSTNESS=0
for arg in "$@"; do
  case "$arg" in
    --skip-pool) SKIP_POOL=1 ;;
    --robustness) ROBUSTNESS=1 ;;
    *) echo "usage: bash scripts/run_all.sh [--skip-pool] [--robustness]" >&2; exit 1 ;;
  esac
done

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
if [ "$SKIP_POOL" = 1 ]; then
  echo "==> mappability pool: skipped (--skip-pool), using data/processed/pool as is"
else
  step "mappability pool" python -u -m src.simulate.pool
fi
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
grid_conditions() {   # grid_conditions [NAME]: "ff depth noise" per condition of the main grid or scenario NAME
  # Noise levels come from src.config.scenario_noise_levels (short-noise scenarios skip sigma = 0)
  python -c '
import sys
from src.config import load_config, scenario_noise_levels
c = load_config(scenario=sys.argv[1] if len(sys.argv) > 1 else None)
for noise in scenario_noise_levels(c):
    for depth in c["grid"]["depth"]:
        for ff in c["grid"]["fetal_fraction"]:
            print(ff, depth, noise)
' "$@"
}
run_grid() {   # run_grid [NAME]: every grid condition, main analysis or scenario NAME
  local n=0
  while read -r ff depth noise; do
    n=$((n + 1))
    echo "    condition $n: FF $ff, depth $depth, noise $noise"
    python -u -m src.evaluate.run_condition --ff "$ff" --depth "$depth" --noise "$noise" ${1:+--scenario "$1"} > /dev/null
  done < <(grid_conditions "$@")
}
run_scenario() {   # run_scenario NAME: grid and evaluation of one robustness scenario (results/robustness/NAME)
  local sc="$1"
  step "[$sc] full grid" run_grid "$sc"
  step "[$sc] detection limits" python -m src.evaluate.detection_limits --scenario "$sc"
  step "[$sc] FPR check" python -m src.evaluate.fpr_check --scenario "$sc"
  step "[$sc] bootstrap B - A" python -u -m src.evaluate.bootstrap_diff --scenario "$sc"
  step "[$sc] break-even check" python -m src.evaluate.breakeven_check --scenario "$sc"
  step "[$sc] unified model check" python -m src.evaluate.unified_model_check --scenario "$sc"
  step "[$sc] rerun check" python -m src.evaluate.check_rerun --scenario "$sc"
}
scenarios() {   # scenarios KIND: names in robustness.KIND (fetal_profiles or short_noise)
  python -c 'import sys; from src.config import load_config, scenario_names; print(*scenario_names(load_config(), sys.argv[1]))' "$1"
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
# One counting model for B, B_c, W and W_fixedFF, plus W vs the optimal weights (Task 06, Fig. S3): ~2 s
step "unified model check" python -m src.evaluate.unified_model_check
# Paired bootstrap of the observed ratio sep(B_c)/sep(A) on 4 conditions (Fig. 4 caption): ~5 s
step "bootstrap ratio" python -m src.evaluate.bootstrap_ratio

# --- 4. Figures and tables (results/figures, results/tables): seconds each -----------------------
step "Figure 2" python -m src.evaluate.figures.fig2_inputs
step "Figure 3" python -m src.evaluate.figures.fig3_b_minus_a
step "Figure 4" python -m src.evaluate.figures.fig4_breakeven
step "Figure 5 + supplementary" python -m src.evaluate.figures.fig5_methods
step "Figure S3 (unified model)" python -m src.evaluate.figures.figS_unified_model
step "Tables 1, 2 + supplementary" python -m src.evaluate.tables

# --- 4b. Robustness to the fetal length profile (Task 07, decision #17) ---------------------------
# Scenario inputs and the model's predictions (7A, 7C), no simulation: seconds
step "robustness inputs and predictions" python -m src.evaluate.robustness_inputs
# Optional (--robustness): the full grid and its evaluation for each scenario in robustness.fetal_profiles,
# written to results/robustness/<NAME>/tables. Sequential time per scenario ~13 min (grid ~5.5 min, bootstrap
# ~7.5 min, the rest seconds), ~65 min for all five here. The scenarios write to separate folders and do not
# depend on each other, so they can also run in parallel (one job per scenario: ~14 min wall time for all five
# on a 10-core Mac). Without --robustness the committed scenario tables are used as they are.
if [ "$ROBUSTNESS" = 1 ]; then
  for sc in $(scenarios fetal_profiles); do run_scenario "$sc"; done
else
  echo "==> fetal-profile scenarios: skipped (no --robustness), using results/robustness as is"
fi
# Summary across scenarios (questions 1-7) and Fig. S4: seconds
step "robustness summary" python -m src.evaluate.robustness_profiles
step "Figure S4 (fetal profiles)" python -m src.evaluate.figures.figS_profiles

# --- 4c. Extra noise on size-selected fragments (Task 08, decision #18) ---------------------------
# Predictions with sigma_u, noise QC on the reference normals, and the Budis illustration (8C, 8F): ~2 s
step "short-noise inputs, QC and Budis illustration" python -m src.evaluate.robustness_snoise_inputs
# Optional (--robustness): grid and evaluation per short-noise scenario (k = 1.5, 2), low and realistic noise
# only (64 conditions). Sequential time per scenario ~9 min (grid ~3.5 min, bootstrap ~5 min, the rest seconds),
# ~18 min for both. They can also run in parallel, one job per scenario (measured: 531 s wall time for both).
if [ "$ROBUSTNESS" = 1 ]; then
  for sc in $(scenarios short_noise); do run_scenario "$sc"; done
else
  echo "==> short-noise scenarios: skipped (no --robustness), using results/robustness as is"
fi
# Summary across k (questions 1-6) and Fig. S5: seconds
step "short-noise summary" python -m src.evaluate.robustness_snoise
step "Figure S5 (short-fragment noise)" python -m src.evaluate.figures.figS_short_noise
# All figure and table captions, from the scripts' caption functions, to docs/captions.md: seconds
step "captions" python -m src.evaluate.captions

# --- 5. Reproducibility check: rerun one condition and compare with its saved rows: ~3 s ---------
step "rerun check" python -m src.evaluate.check_rerun

echo "All steps finished in $SECONDS s."
