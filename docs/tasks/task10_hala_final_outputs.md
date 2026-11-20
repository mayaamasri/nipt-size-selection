# Task 10 (Hala): final figures, tables, supplement and repository

**Goal:** produce every figure and table of the revised manuscript and Additional file 1, fix the presentation
points the reviewers raised, and leave the repository reproducible. No new analysis.
**Decisions:** all; Maya adds #19 (limit CIs and the 14-point grid). **`git pull` first.**

**The two grids** (state them in every caption that has limits):
- **14-point FF grid** (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15%): every **main-grid detection limit**
  (Table 2, Tables S4, S6–S8).
- **8-point grid**: Figs. 3–5, per-condition results, and **all robustness-scenario limits** (Tasks 07–08).

---

## 10A. Numbering: follow the manuscript

The manuscript has **Table 1 = comparison with Yang et al.** and **Table 2 = detection limits** (the repo has them
the other way round). Rename outputs, captions and every cross-reference (`captions.md`, the README table,
figure captions such as Fig. 5's "detection limits in Table 1") to the manuscript's numbering. Final list:

| Main | Content |
|---|---|
| Fig. 1 | pipeline (Maya, unchanged) |
| Figs. 2–5 | as now (8-point grid) |
| Table 1 | realism check vs Yang |
| Table 2 | detection limits with 95% CIs (14-point grid) |

| Additional file 1 | Content | Grid |
|---|---|---|
| Fig. S1–S2 | Fig. 5 at no / low noise | 8 |
| Fig. S3 | unified model (Task 06) | – |
| Fig. S4 | fetal-profile scenarios (Task 07) | – |
| Fig. S5 | short-fragment noise (Task 08) | – |
| Table S1 | B-opt chosen cutoffs | – |
| Table S2 | C2 feature importances | – |
| Table S3 | W vs W_fixedFF, sensitivity | 8 |
| Table S4 | W vs W_fixedFF, detection limits | 14 |
| Table S5 | AUC, all methods and conditions | – |
| Table S6 | detection limits: conservative (Wilson lower bound) and matched-FPR, with CIs | 14 |
| Table S7 | 8-point vs 14-point limits per method × setting (interpolation check) | both |
| Table S8 | method differences with CIs and clear / similar (W−B, W−A, B−A, W−C2, C2−B, B-opt−B, W_fixedFF−W) | 14 |
| Table S9 | fetal-profile scenarios: g, r, g·√r per FF; B<A counts; limits of A, B, B-opt, W, W_G1, W_fixedFF; W_G1 vs A CIs | 8 |
| Table S10 | short-fragment noise: B/A rise by depth (sim / model); limits of A, B, W; k* and the four regimes | 8 |

If a table above already exists under another number, renumber it. Ask Maya before adding or dropping any.

## 10B. LaTeX table files

The manuscript `\input`s `tables/table1_….tex` and `tables/table2_….tex`. Generate **every** main and
supplementary table as a `.tex` file in `results/tables/latex/` from the CSVs (a script, not by hand), with
`booktabs`, **en-dashes for every range** (36–500, 40–156, 4.09–4.48), and a thin space before `bp`.

**Table 1 (Yang):** same content; widen the first column so labels don't wrap mid-word (R1 presentation point).

**Table 2 (detection limits), the main one:**
- Rows: noise level × method (A, B, B-opt, W, C1, C2); columns: 1M, 2.5M, 5M, 10M.
- Cell: limit at the method's own threshold, with its 95% CI underneath or in brackets, e.g. `6.90 [6.71–7.12]`.
  CIs: test+ref for z-score methods, test for trees (Task 09).
- **Bold** the lowest limit per setting; also bold any method whose difference from the lowest is **not clear**
  (CI includes 0), so a bold pair means "similar".
- `†` = limit ≤ 4% (keep). New symbol for tree limits with a nobody-positive tree at the lower end of the crossing
  interval (your new column), explained in the caption.
- Caption: the 14-point grid, the CI versions (what's resampled, what's held fixed), "matched-FPR and conservative
  limits in Table S6".
- If cells with CIs make it too wide for the page, send Maya a compiled draft and propose a layout (e.g. CIs in a
  second line). Don't silently drop the CIs.

## 10C. Figure fixes

- **Fig. 3 (R2 minor 3):** the B − A difference at FF 1%, 1M, low noise is −0.05 points but shows as "0.0", so the
  figure has 4 visible negatives and the text says 5. Show values with |x| < 0.1 with two decimals (`−0.05`), or
  keep the sign (`−0.0`). Your call; the text will say "five negative differences" either way.
- **Figs. 3 and 5:** add "10,000 euploid and 2,000 trisomic test patients per condition" to the generated captions
  (already in the manuscript's own captions).
- **Fig. 5 caption:** the W_fixedFF note ("within 0.1") is on the 8-point grid; say so, or point to Table S4.
- Regenerate Figs. S3–S5 only if a caption number changed.

## 10D. Numbers sheet for Maya (most useful part for the writing)

Generate `docs/revision_numbers.md` from the results (script, rerun by `run_all.sh`): every number the revised
text and the response letter will quote, each with its source file. At least:
- **Abstract / 3.4:** W, B, A limits at 1M realistic (14-point) with CIs; W−B / W−A / B−A clear counts per basis;
  10M realistic W vs B and C2 (both bases); A at 10M realistic vs 4%; W vs W_fixedFF max difference (0.115).
- **8 → 14 point changes:** per method, the largest drop at 1M and 2.5M (e.g. W 1M realistic 7.33 → 6.90).
- **Unified model:** median errors (W 0.50%, B family 0.92%), W bias (+0.27%, t ≈ 2.6), W efficiency (0.997 / 0.902).
- **Profiles:** B<A and significantly-worse counts per scenario; B-opt<A counts and worst sensitivity loss;
  W_G1 cost (median, IQR, max); W_G1 vs A and vs B clear counts; FPRs; B-opt median cutoffs.
- **Short noise:** B/A rise per k and noise (sim / model); W's lead over B at 10M; k* per FF; the Budiš k values
  per FF; B-opt cutoffs.
- **Trees:** nobody-positive counts (8-point: 40 C1 / 31 C2 conditions; 14-point: 60 of 144 extra trees), the
  flagged limits, C2's FPR at 10M realistic, FF 2% (0.23%).

## 10E. Reproducibility

1. `scripts/run_all.sh`: main pipeline → 14-point grid → Task 09 bootstrap → robustness block (Tasks 07–08) →
   figures → LaTeX tables → numbers sheet → rerun checks. Run times in comments.
2. **One full run from a fresh clone** (pool included; long, so start it and do the rest meanwhile). Every
   committed file must reproduce byte for byte, apart from the run logs. Report the total time.
3. README: the two grids, `--scenario`, the outputs table in the new numbering, total run time.
4. Remove anything unused (old scratch outputs, dead code paths).
5. Tag the commit used for the revised manuscript, e.g. `v2-revision`, so the paper can cite it.

## Gate
Before 10E's full run, send Maya: the compiled **Table 2** draft (PDF snippet) and the numbering list from 10A.

## Send Maya
1. Table 2 draft + numbering (gate)
2. `results/tables/latex/` (all `.tex` tables) and the final figures
3. `docs/revision_numbers.md`
4. The fresh-clone result (time, byte-identical or not) and the tag
