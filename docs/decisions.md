# Design Decision Log

Record every design decision here with the date, the choice, and the reason.
This file becomes the backbone of the Methods section and of your rebuttal
if a reviewer asks "why did you do X?".

## Work split

| Area | Owner |
|------|-------|
| Decisions, literature, theory (break-even), daily task briefs | Maya |
| All code: simulation, mapping, methods A/B/C1/C2, evaluation, figures | Hala |

## Decisions

| # | Decision | Options considered | Choice | Reason | Date |
|---|----------|--------------------|--------|--------|------|
| 1 | Fragment-length distributions source | Rabinowitz supplementary data / digitised figure / parametric fit | **Exact per-bp counts behind Rabinowitz Fig. 1 (family G1), taken from the authors' Hoobari code** (github.com/nshomron/hoobari, `src/freebayes_patch.py`, commit 74ed24a). Saved as `data/raw/rabinowitz_G1_length_counts.csv`. Processing: drop length 0; replace the 101 bp bin by the mean of its neighbours (read-length artifact); keep 36–500 bp; normalise. | Matches the proposal (Rabinowitz [3]); exact counts instead of a digitised figure. Validated: the counts reproduce G1's published FF (30.1% vs 30.2% in Rabinowitz Table 1). Realism check: a 40–156 bp window keeps ~20% of reads and raises FF ~2.2–2.5×, in line with Yang [7] (75% discarded, FF ~doubled). **Limitation:** one pregnancy, sampled at 38 weeks (NIPT is done at 10–20 weeks). | 2026-10-26 |
| 2 | Read length and fragment length range | 2×36, 2×50, 2×75, 2×100 | **2×36 bp paired-end; fragment lengths 36–500 bp** | 2×36 matches typical NIPT runs (Yu [8] 2×36; Yang [7] 2×37). Reads can't exceed the fragment, so 36 bp is the minimum length (<0.06% of fragments are lost). 500 bp matches Rabinowitz's own max length and Bowtie2's default maximum insert (-X 500); excludes ~0.8% maternal / 0.1% fetal fragments. | 2026-10-26 |
| 3 | Pool size and sampling scheme | fresh fragments per patient / disjoint sub-pools, sampling with replacement / count-level simulation | *pending: needs timing from Task 01* | | |
| 4 | Patients per condition (ref / train / test) | | *pending* | | |
| 5 | FPR handling | report observed FPR + CI / fix specificity | *pending* | | |
| 6 | Size-selection cutoff for Method B | literature value / chosen on training set | *pending* | | |
| 7 | Noise model: where sigma is applied and its values | | *pending: depends on #8* | | |
| 8 | Break-even formula (derivation in docs/break_even.md) | | *pending (Maya)* | | |
| 9 | Decision-tree features (C1, C2) | | *pending* | | |
| 10 | One tree per condition vs one tree overall | | *pending* | | |
| 11 | Chromosomes included | all / autosomes only | **Autosomes chr1–chr22** | Fetal sex is not modelled; sex chromosomes and chrM would add sex-dependent noise to the reference. Standard practice for chr21 z-scores. | 2026-10-26 |
