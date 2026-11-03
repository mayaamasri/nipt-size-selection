# The break-even model: when does size selection beat counting all fragments?

This note derives the formula behind decision #8, lists its assumptions, and records how well it
predicts the simulation. It is the source for the "Break-even model" subsection of the Methods.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| N | depth: number of fragments that pass the MAPQ ≥ 40 filter |
| f | fetal fraction (FF) among those fragments |
| p0 | share of fragments on chr21 in a euploid pregnancy (here q(chr21) = 0.01257) |
| σ | per-patient extra noise: relative SD of the chr21 share that does **not** shrink with depth |
| window | fragments of length 40–c bp kept by size selection (Yang: c = 156) |
| r | fraction of fragments inside the window |
| g | FF enrichment: FF inside the window ÷ f |

g and r depend on the cutoff c **and** on f; both are computed exactly from the post-mappability
length distributions (decision #14). At FF 4% and c = 156: g = 2.43, r = 0.203.

---

## 2. Signal

In a trisomy 21 pregnancy the fetus contributes 1.5× its normal share of chr21 fragments.
Since fetal fragments are a fraction f of all fragments, the patient's chr21 share rises from p0 to

    p_T21 = p0 · (1 + f/2) / (1 + p0·f/2)  ≈  p0 · (1 + f/2)

so the **relative signal** is δ = f/2 (the denominator is within 0.1% of 1 because p0 ≈ 1.3% and f ≤ 15%).

Inside the size-selection window the fetal fraction is g·f, so the relative signal of method B is

    δ_B = g·f/2.

The chr21 share inside the window is still p0: mappability does not depend on the chromosome
(Task 02 QC: chr21 share is flat across length bins), so filtering by length removes chr21 and
other fragments in the same proportion.

## 3. Noise

Two independent sources of variation in a euploid patient's measured chr21 share:

1. **Counting noise.** With N fragments, the chr21 count is binomial, so the share has SD
   `sqrt(p0(1 − p0)/N)`, i.e. relative SD `sqrt((1 − p0)/(p0·N)) ≈ 1/sqrt(p0·N)`.
   It shrinks as depth grows.
2. **Extra noise.** Each patient's chr21 share is multiplied by (1 + ε), ε ~ N(0, σ²)
   (decision #7: GC bias, library and run effects). Relative SD = σ, whatever the depth.

Total relative noise of method A, and of method B (which keeps only r·N fragments, while the same
ε affects all lengths):

    noise_A = sqrt( 1/(p0·N)   + σ² )
    noise_B = sqrt( 1/(p0·r·N) + σ² )

## 4. Separation and the ratio

The z-score of a patient is (share − reference mean) / reference SD, and the reference SD estimates
the noise above. The **standardised separation** between trisomic and euploid patients is therefore
signal ÷ noise:

    sep_A = (f/2)   / sqrt( 1/(p0·N)   + σ² )
    sep_B = (g·f/2) / sqrt( 1/(p0·r·N) + σ² )

Dividing, f/2 cancels:

    R(N) = sep_B / sep_A = g · sqrt( (1/(p0·N) + σ²) / (1/(p0·r·N) + σ²) )        (1)

Size selection helps when R > 1.

## 5. Consequences

**No extra noise (σ = 0).** Both noise terms are counting noise, the N's cancel, and

    R = g · √r                                                                  (2)

**Depth does not matter at all.** For Yang's window at FF 4%, R = 2.43 · √0.203 = 1.09:
a fixed ~9% gain. Enrichment (g) and the loss of reads (√r) almost cancel.

**Very high depth (N → ∞).** Counting noise vanishes, both methods are limited by σ, and

    R → g                                                                       (3)

so the gain approaches the full FF enrichment (2.43 for Yang's window at FF 4%).

**In between,** R moves monotonically from g·√r towards g as N grows (for σ > 0 and r < 1):
the extra noise is what makes depth matter.

**Break-even condition.** Squaring (1), B beats A exactly when

    σ² (g² − 1)  >  (1/(p0·N)) · (1/r − g²)                                     (4)

- If **g²r ≥ 1**, the right side is ≤ 0 while the left side is ≥ 0 (g > 1), so B is **never worse** than A,
  at any depth or noise level.
- If **g²r < 1**, B loses at low depth and wins above the **break-even depth**

      N* = (1/r − g²) / ( p0 · σ² · (g² − 1) )                                  (5)

  Without extra noise (σ = 0), N* is infinite: a strict cutoff never pays off.

## 6. What this predicts for our design

| Cutoff (40–c bp) | FF 4%: g, r, g²r | N* at σ = 0.15% | N* at σ = 0.35% |
|---|---|---|---|
| 120 | 3.38, 0.028, 0.32 | 83.7M | 15.4M |
| 130 | 3.67, 0.042, 0.57 | 29.2M | **5.4M** |
| 135 | 3.66, 0.057, 0.77 | 11.6M | **2.1M** |
| 140 | 3.49, 0.075, 0.91 | **3.8M** | 0.7M |
| 145 | 3.16, 0.106, 1.06 | never worse | never worse |
| 150 | 2.82, 0.143, 1.14 | never worse | never worse |
| **156 (Yang)** | 2.43, 0.203, 1.20 | never worse | never worse |
| 165 | 1.80, 0.351, 1.14 | never worse | never worse |

- At FF 4% the boundary g²r = 1 (equivalently g√r = 1) lies at **143 bp** (Fig. 2e).
- g falls with FF while r barely changes, so g²r falls and the boundary moves up: at FF 10%,
  the 145 bp window already has g²r = 0.94 (N* = 0.5M at σ = 0.35%), and at FF 15%
  the 150 bp window has g²r ≈ 0.95. Yang's 156 bp window stays ≥ 1 across FF 1–15% (1.03 at 15%).

## 7. Approximations

1. p0 is small, so (1 − p0) ≈ 1 and the T21 renormalisation term ≈ 1.
2. Counts are approximated as normal (the separation uses means and SDs only).
3. The T21 patients' own SD is slightly larger than the euploid SD; the separation uses the euploid SD,
   as the z-score does.
4. The same relative σ applies to all fragment lengths, so size selection neither removes nor adds
   extra noise. **This is an assumption of the model**, not something the simulation tests (it
   generates noise the same way); a size-dependent bias (e.g. GC) would make B worse than predicted.

## 8. How well it predicts the simulation

Compared on the observed standardised separations, sep = (z_mean_t21 − z_mean_normal) / z_sd_normal,
for B at 8 cutoffs × 8 FF × 4 depths × 3 noise levels (`results/tables/breakeven_summary.csv`):

| Rows | n | median relative error | 95th percentile | max |
|---|---|---|---|---|
| all | 768 | 0.92% | 3.5% | 10.8% |
| A's separation ≥ 1 | 712 | **0.87%** | 2.9% | 6.4% |

No systematic bias (mean signed error +0.2%, t = 1.6). The largest errors are at FF 1%, where A's
separation is below 1 and the ratio is unstable. Observed break-even depths match equation (5):
≈ 5.4M for 130 bp and ≈ 2.1M for 135 bp at realistic noise (Fig. 4).

## 9. Link to the literature

Equation (1) explains the depth dependence reported in real data: with σ > 0, the gain of size
selection is small at low depth (R ≈ 1.16–1.24 at 1–2.5M for Yang's window, hard to detect in a cohort)
and large at high depth (R ≈ 1.50 at 10M). This matches no significant gain at 3.1M reads but a significant
gain at 5.8M (Minárik et al.), and no gain at 5M but a significant gain at 8.8M (Budiš et al.).
