# Task 03 (Hala): detection methods A, B, C1, C2 and the evaluation

**Goal:** turn virtual patients into results: the four detection methods from the proposal and every
evaluation measure, first for **one condition end-to-end**, then ready to run on the full grid.
**Target:** Saturday Oct 31. You can start before Task 02 is finished: write the code against a stub
`generate_patients` that returns random counts of the right shape `(n, 2, 465)`, then swap in the real one.

Decisions this task relies on: `docs/decisions.md` #4, #5, #6, #9, #10.

---

## Data per patient
A count matrix `X` of shape `(2, 465)`: row 0 = chr21, row 1 = other autosomes; column `j` = length `36 + j` bp.

## 3A. Per-condition reference statistics (`src/methods/reference.py`)
For each condition (FF × depth × noise), use the **2,000 reference normals** only:
- mean and SD of the chr21 fraction using **all** fragments → used by method A
- mean and SD of the chr21 fraction using fragments **40–156 bp** only → method B
- mean and SD of the C2 length features below (so they can be z-scored)

## 3B. Methods A and B (`src/methods/zscore.py`)
- **A:** `zA = (chr21 fraction − ref mean) / ref SD`, all fragments. Positive call if `zA > 3`.
- **B:** same, but every count (patient and reference) restricted to **40–156 bp** (Yang [7]). Positive if `zB > 3`.
- **B-opt (extra analysis):** same as B, but the upper cutoff is chosen on the **training set** from
  {120, 125, …, 170} bp (lower bound stays 40) as the one maximising the mean `zB` of training T21 patients
  minus that of training normals. Applied unchanged to the test set. Record the chosen cutoff per condition.

## 3C. Features (`src/methods/features.py`)
| Feature | Used by | Definition |
|---|---|---|
| `zA` | C1, C2 | method A's z-score |
| `z_short` | C2 | Yu et al. [8]: `S = (short fraction on chr21) − (short fraction on other autosomes)`, short = ≤ 150 bp; z-scored against the reference normals |
| `z_weighted` | C2 | Rabinowitz et al. [3]: weighted chr21 fraction `Σ w(l)·X[0,l] / Σ w(l)·(X[0,l]+X[1,l])` with `w(l) = FF·p_f(l) / (FF·p_f(l) + (1−FF)·p_m(l))`, i.e. P(fetal \| length) as in Rabinowitz, using the condition's FF and the length distributions **after mappability** (decision #9); z-scored against the reference normals |

C1 uses `zA` only (within one condition the raw chr21 fraction is a linear rescaling of `zA`, so adding it gives no new information).

## 3D. Decision trees C1, C2 (`src/methods/trees.py`)
- One tree per condition, `sklearn.tree.DecisionTreeClassifier`, trained on that condition's training set
  (**10,000 normal + 2,000 T21**), `class_weight="balanced"`, seed from config.
- `max_depth` chosen from {3, 4, 5} by 5-fold cross-validation on the training set (AUC).
- **Threshold (decision #5):** fit the tree on the whole training set, then compute its predicted probabilities
  for a separate **calibration set of 10,000 normals** (generated like the reference set, never used for training).
  Candidate thresholds = the tree's own distinct leaf probabilities; call positive if `prob ≥ threshold`, and pick the
  smallest candidate whose calibration FPR is **≤ 0.13%**. Record the threshold and the achieved calibration FPR.
  A shallow tree has few distinct outputs, so the achieved FPR may be below 0.13%: keep it and report it
  (the `sens_at_fpr` column below keeps the comparison fair).
- Save C2 feature importances per condition.

## 3E. Evaluation on the test set (`src/evaluate/metrics.py`)
Test set per condition: **10,000 normal + 2,000 T21**. For each method (A, B, B-opt, C1, C2):
- sensitivity and FPR, each with a **Wilson 95% CI**
- AUC (scores: `zA`, `zB`, `zB-opt`, tree probabilities)

Also per condition, for method B (inputs to Maya's break-even formula):
- `g` = FF within 40–156 bp ÷ overall FF, and `r` = fraction of fragments in 40–156 bp, both computed exactly from
  the fetal/maternal probability matrices **after mappability** (normal fetal matrix, decision #12), no sampling needed

Write everything to one tidy table `results/tables/results_all.csv`, one row per condition × method:
`ff, depth, noise, method, sens, sens_lo, sens_hi, fpr, fpr_lo, fpr_hi, auc, sens_at_fpr, g, r, cutoff, threshold`.

`sens_at_fpr` = sensitivity at **FPR = 0.13% read from the test ROC curve**, linearly interpolated between the two
nearest achievable points (decision #5). This compares all methods at exactly the same FPR, so a tree is not
penalised just because its operating point is stricter than `z > 3`.

**Detection limit (decision #13):** separate small script, for each method × depth × noise, using the FF grid
1, 2, 3, 4, 6, 8, 10, 15%:
- **main:** based on `sens` (each method's real cutoff: z > 3 for A/B, the calibration threshold for the trees):
  the FF where sensitivity crosses 95% **for the last time** (so it stays ≥ 95% at every higher grid FF),
  linearly interpolated between the two grid FFs around the crossing; point estimate
- **conservative:** the same using `sens_lo` (Wilson lower bound)
- **check:** the same using `sens_at_fpr` (point estimate only, no CI or bootstrap needed). Flag any method/condition
  where this differs from the main value by more than one grid step.
- report "< 1%" if already ≥ 95% at FF 1%, and "> 15%" if never reached
→ `results/tables/detection_limits.csv`

## 3F. Milestone: one condition end-to-end
Run **FF 4%, depth 5M, sigma 0** and send Maya the results rows for all five methods.
Sanity checks (should hold):
- method A's test FPR is close to 0.13% (CI should include it)
- `zA` of test normals has mean ≈ 0 and SD ≈ 1
- C1 performs about the same as A (it has the same information)

**Do not run the full grid yet**: the noise levels (decision #7) are still being decided.

---

### Config keys to add
```yaml
methods:
  z_threshold: 3.0
  size_window_bp: [40, 156]          # decision #6
  size_opt_upper_grid: [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170]
  short_fragment_bp: 150             # Yu et al., decision #9
  target_fpr: 0.0013                 # decision #5
  tree_max_depth: [3, 4, 5]
  cv_folds: 5
```
These replace the current `methods` block (remove `size_selection_cutoff_bp: TODO`).

### Conventions (same as Task 01)
Modules run as `python -m src.<package>.<module>` from the repo root, parameters via `src.config.load_config()`,
paths via `repo_path()`. Put the per-condition driver in `src/evaluate/run_condition.py`.
