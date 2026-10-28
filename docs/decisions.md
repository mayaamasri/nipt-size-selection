# Design Decision Log

Record every design decision here with the date, the choice, and the reason.
This file becomes the backbone of the Methods section and of your rebuttal
if a reviewer asks "why did you do X?".

## Work split

| Area | Owner |
|------|-------|
| Decisions, literature, theory (break-even), daily task briefs | Maya |
| All code: simulation, mapping, methods A/B/C1/C2, evaluation, figures | Hala |
| Writing (all manuscript sections) | Maya |

## Decisions

| # | Decision | Options considered | Choice | Reason | Date |
|---|----------|--------------------|--------|--------|------|
| 1 | Fragment-length distributions source | Rabinowitz supplementary data / digitised figure / parametric fit | **Exact per-bp counts behind Rabinowitz Fig. 1 (family G1), taken from the authors' Hoobari code** (github.com/nshomron/hoobari, `src/freebayes_patch.py`, commit 74ed24a). Saved as `data/raw/rabinowitz_G1_length_counts.csv`. Processing: drop length 0; replace the 101 bp bin by the mean of its neighbours (read-length artifact); keep 36–500 bp; normalise. | Matches the proposal (Rabinowitz [3]); exact counts instead of a digitised figure. Validated: the counts reproduce G1's published FF (30.1% vs 30.2% in Rabinowitz Table 1). Realism check: a 40–156 bp window keeps ~20% of reads and raises FF ~2.2–2.5×, in line with Yang [7] (75% discarded, FF ~doubled). **Limitation:** one pregnancy, sampled at 38 weeks (NIPT is done at 10–20 weeks). | 2026-10-26 |
| 2 | Read length and fragment length range | 2×36, 2×50, 2×75, 2×100 | **2×36 bp paired-end; fragment lengths 36–500 bp** | 2×36 matches typical NIPT runs (Yu [8] 2×36; Yang [7] 2×37). Reads can't exceed the fragment, so 36 bp is the minimum length (<0.06% of fragments are lost). 500 bp matches Rabinowitz's own max length and Bowtie2's default maximum insert (-X 500); excludes ~0.8% maternal / 0.1% fetal fragments. | 2026-10-26 |
| 3 | Pool size and sampling scheme | fresh fragments per patient / disjoint sub-pools, sampling with replacement / count-level simulation | **Map one pool of 10M maternal + 10M fetal fragments and use it only to estimate mappability: survival probability by length, S(l), and chromosome share of survivors, q(c). Each virtual patient is an independent multinomial draw of N surviving fragments from P_o(chrom, length) ∝ P_o(length)·S(length)·q(chrom), mixed by FF.** Patients are stored as a 2 × 465 count matrix (chr21 / other autosomes × length 36–500 bp). | Disjoint finite sub-pools (proposal wording) give each set a slightly different chr21 share by chance, which shifts all test z-scores against the reference by ~√(N·(1/P_ref + 1/P_test)) SD: ≈0.8 SD at 10M depth with 30M-fragment pools. That is an artifact (real patients share one genome) and it grows with depth, so it would bias the break-even result. Independent draws from one model remove it, keep steps 1–3 (Rabinowitz lengths, GRCh38, Bowtie2, MAPQ ≥ 40) intact, honour the intent of "no fragments shared between sets", and are fast enough for thousands of patients per condition. Task 01: 100k pairs map in ~5 s, 86% survive. **To state in Methods as a deviation from the proposal.** | 2026-10-28 |
| 4 | Patients per condition (ref / train / test) | | **Reference 2,000 normals; training 10,000 normal + 2,000 T21; test 10,000 normal + 2,000 T21**, per FF × depth × noise condition | Cheap under #3. 10,000 training normals are needed to set the tree threshold at an FPR of 0.13% (#5). 10,000 test normals estimate an FPR near 0.13% with ~13 expected false positives (95% CI roughly 0.07–0.22%); 2,000 T21 give sensitivity to about ±1%. | 2026-10-28 |
| 5 | FPR handling | report observed FPR + CI / fix specificity | **A and B use z > 3 (target FPR 0.13%). Tree thresholds are set on the training normals as the smallest threshold with FPR ≤ 0.13%. All observed FPRs reported with Wilson 95% CIs.** | Follows the proposal (tree cutoff matched to method A's FPR); 0.13% is the one-sided tail of z > 3. | 2026-10-28 |
| 6 | Size-selection cutoff for Method B | literature value / chosen on training set | **Main: fixed 40–156 bp window (Yang [7]). Extra analysis (B-opt): upper cutoff chosen on the training set from 120–170 bp, lower bound 40.** | The fixed window answers the question as labs practise it and matches the realism check; B-opt checks whether a better cutoff would change the conclusions, and gives B the same tuning chance as the trees. | 2026-10-28 |
| 7 | Noise model: where sigma is applied and its values | | *pending: depends on #8* | | |
| 8 | Break-even formula (derivation in docs/break_even.md) | | *pending (Maya)* | | |
| 9 | Decision-tree features (C1, C2) | | **C1: zA. C2: zA + z_short (Yu [8]: excess of ≤150 bp fragments on chr21 vs other autosomes) + z_weighted (Rabinowitz [3]: chr21 fraction with each fragment weighted by P(fetal \| length)).** B's z-score deliberately excluded from C2. | C1 vs C2 isolates the value of length information. Each length feature comes from a cited study. Excluding zB keeps C2 a 'keep all fragments' method, as in the research question. The raw chr21 fraction is dropped because within a condition it is a linear rescaling of zA. | 2026-10-28 |
| 10 | One tree per condition vs one tree overall | | **One tree per FF × depth × noise condition; max depth 3–5 chosen by 5-fold CV** | Same per-condition setting as the z-score reference, so the comparison with A and B is like for like. | 2026-10-28 |
| 11 | Chromosomes included | all / autosomes only | **Autosomes chr1–chr22** | Fetal sex is not modelled; sex chromosomes and chrM would add sex-dependent noise to the reference. Standard practice for chr21 z-scores. | 2026-10-26 |
| 12 | Definition of depth and FF | before / after the mapping filter | **Both defined on fragments that pass the MAPQ ≥ 40 filter** | Matches what a lab actually counts (unique mapped reads) and measures (FF of the analysed reads). | 2026-10-28 |

## Literature support (for citing in Methods)

| Choice | Source | What the source says/did |
|---|---|---|
| Fragment-length distributions | Rabinowitz [3] | Measured fetal and maternal lengths (Fig. 1, family G1); counts from their Hoobari code reproduce their FF of 30.2% |
| Read length 2×36 | Yu [8]; Yang [7] | 36- and 50-cycle paired-end (Yu); 37+37 paired-end (Yang) |
| Bowtie2, MAPQ ≥ 40 | Budis [1] | "Reads with mapping quality of 40 or higher were retained"; mapped with Bowtie 2 |
| Method A: chr21 reads / autosomal reads, z vs reference, z > 3 | Yang [7] | GR = chromosome reads / total autosomal reads; mean and SD from reference samples; "Z score > 3" |
| Method B window 40–156 bp, reference also filtered | Yang [7] | "40–156 bp fragments were selected"; Filter Tag Z score > 3 |
| B-opt cutoff sweep | Minárik [2] | Tested several upper cutoffs (140, 145, 150 bp, ...) |
| z_short (≤150 bp excess on chr21) | Yu [8] | ΔF = P(≤150)chr21 − P(≤150)reference chromosomes, z-scored against euploid references |
| z_weighted (P(fetal \| length) weights) | Rabinowitz [3] | Per-length fetal fraction used as a weight instead of a hard cutoff |
| Decision trees, shallow | Teder [4]; Tu [5] | Decision tree with max_depth 3 (Teder); ML classifiers for trisomy at low FF (Tu) |
| Reference set of euploid samples | Xie [6] | 412 euploid pregnancies as reference set |
| Our own reasoning (justify in text) | — | #3 mappability-table design, #4 patient numbers, #10 per-condition trees, #12 definitions, 101 bp artifact smoothing, Wilson CIs (Brown, Cai & DasGupta 2001) |
