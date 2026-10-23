# Design Decision Log

Record every design decision here with the date, the choice, and the reason.
This file becomes the backbone of the Methods section and of your rebuttal
if a reviewer asks "why did you do X?".

| # | Decision | Options considered | Choice | Reason | Date |
|---|----------|--------------------|--------|--------|------|
| 1 | Fragment-length distributions source | Rabinowitz supplementary data / digitized figure / parametric fit | | | |
| 2 | Read length for simulated pairs | must be <= shortest fragment | | | |
| 3 | Pool size and sampling scheme | fresh fragments per patient / disjoint sub-pools, sampling with replacement / count-level simulation | | | |
| 4 | Patients per condition (ref / train / test) | | | | |
| 5 | FPR handling | report observed FPR + CI / fix specificity | | | |
| 6 | Size-selection cutoff for Method B | literature value / chosen on training set | | | |
| 7 | Noise model: where sigma is applied and its values | | | | |
| 8 | Break-even formula (derivation in docs/break_even.md) | | | | |
| 9 | Decision-tree features (C1, C2) | | | | |
| 10 | One tree per condition vs one tree overall | | | | |
| 11 | Chromosomes included (autosomes only? exclude chrX/Y?) | | | | |

## Work split

| Area | Owner |
|------|-------|
| Simulation pipeline (fragments, mapping, pools) | |
| Detection methods (A, B, C1, C2) and evaluation | |
| Break-even derivation | |
| Introduction + Discussion | |
| Methods + Results | |
| Figures | |
