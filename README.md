# When does fragment size selection help NIPT detection of trisomy 21?

CMPS 355 (Bioinformatics Algorithms), AUB, Fall 2026 - Hala Fatfat, Maya El Masri

A simulation study of when hard fragment-size selection improves non-invasive
prenatal detection of trisomy 21, across fetal fraction (FF), sequencing depth
and noise, and whether a classifier combining counts and fragment lengths
outperforms size selection.

## Repository layout

```
config/config.yaml     all study parameters (single source of truth)
docs/decisions.md      design decision log + work split
scripts/               setup and run scripts
src/simulate/          fragment generation, read simulation, mapping, pools, virtual patients
src/methods/           Method A (count z), B (size-selected z), C1/C2 (decision trees)
src/evaluate/          sensitivity, FPR, AUC, detection limit, break-even, plots
notebooks/             exploration only - final results must come from scripts
data/                  NOT in git (large); recreate with scripts/download_reference.sh
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

### 3. Download the reference genome and Bowtie2 index (~5 GB download)
```bash
bash scripts/download_reference.sh
```

### 4. Check everything works
```bash
python scripts/check_env.py
```

## Reproducibility rules
- Every parameter lives in `config/config.yaml`; every random step uses the configured seed.
- Final figures and tables are produced by scripts, never edited by hand.
- Large data is never committed; it must be regenerable from scripts.
- Record design decisions in `docs/decisions.md` as you make them.
