"""Task 3E: evaluation measures on the test set (decisions #4, #5).

Per method:
  sens, sens_lo, sens_hi   sensitivity on the test T21 patients, Wilson 95% CI
  fpr, fpr_lo, fpr_hi      false positive rate on the test normals, Wilson 95% CI
  auc                      ROC AUC of the method's score
  sens_at_fpr              sensitivity at FPR exactly target_fpr, read from the test ROC by linear
                           interpolation between the two nearest ROC points
Per condition, for every size-selection method (B, B_c, B-opt): g = FF inside its window / overall FF and
r = fraction of fragments inside the window, exact from the post-mappability matrices (normal fetal matrix),
no sampling.

Run from the repo root to check sens_at_fpr on hand-computed examples and print g and r at FF 4% and 10%:
    python -m src.evaluate.metrics
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import load_config
from src.simulate.mappability import wilson_interval
from src.simulate.patients import origin_matrices, window_stats


def rate(calls):
    """Share of True in calls with its Wilson 95% CI: (p, lo, hi)."""
    k, n = int(np.sum(calls)), len(calls)
    lo, hi = (float(v) for v in wilson_interval(k, n))
    # The formula gives 0 and 1 at the edges only up to float rounding (hi = 0.9999999999999998 at k = n)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return k / n, lo, hi


def sens_at_fpr(y, scores, target_fpr):
    """Sensitivity at FPR = target_fpr on the ROC of (y, scores).

    ROC points come from roc_curve: one point per distinct score (tied patients move together), starting
    at (0, 0). Left point: the last with FPR <= target (highest TPR at that FPR). Right point: the first
    with FPR > target. Linear interpolation between them; exact if the left point sits at the target.
    """
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    left = np.flatnonzero(fpr <= target_fpr)[-1]
    if fpr[left] == target_fpr or left == len(fpr) - 1:
        return float(tpr[left])
    right = left + 1
    return float(tpr[left] + (target_fpr - fpr[left]) * (tpr[right] - tpr[left]) / (fpr[right] - fpr[left]))


def evaluate(y, scores, calls, target_fpr):
    """All test-set measures of one method. y: 1 = T21, 0 = normal."""
    y = np.asarray(y).astype(bool)
    calls = np.asarray(calls).astype(bool)
    sens, sens_lo, sens_hi = rate(calls[y])
    fpr, fpr_lo, fpr_hi = rate(calls[~y])
    return {"sens": sens, "sens_lo": sens_lo, "sens_hi": sens_hi,
            "fpr": fpr, "fpr_lo": fpr_lo, "fpr_hi": fpr_hi,
            "auc": float(roc_auc_score(y, scores)), "sens_at_fpr": sens_at_fpr(y, scores, target_fpr)}


def g_and_r(matrices, ff, config, upper=None):
    """g and r for the window lo..upper, lo from size_selection_window_bp (B's window if upper is None)."""
    lo, hi = config["methods"]["size_selection_window_bp"]
    w = window_stats(matrices, ff, lo, hi if upper is None else upper)
    return float(w["gain"]), float(w["r"])


def main():
    config = load_config()

    # Hand-computed check: 10 normals, 4 T21, three distinct scores (like a small tree)
    y = np.r_[np.zeros(10, int), np.ones(4, int)]
    s = np.r_[np.zeros(8), [0.5, 0.5], [0.0, 0.5, 1.0, 1.0]]
    # ROC points (0,0), (0,0.5), (0.2,0.75), (1,1)
    cases = [(0.1, 0.625), (0.2, 0.75), (0.0, 0.5), (0.6, 0.875)]
    for target, expected in cases:
        got = sens_at_fpr(y, s, target)
        print(f"toy ROC, FPR {target}: sens_at_fpr {got:.6f} (hand {expected})")
        if not np.isclose(got, expected, rtol=0, atol=1e-12):
            raise ValueError(f"sens_at_fpr({target}) = {got}, expected {expected}")

    # Edge cases of the Wilson CI
    for k, n in ((0, 10000), (2000, 2000)):
        print(f"rate k={k}, n={n}: {rate(np.r_[np.ones(k, bool), np.zeros(n - k, bool)])}")

    matrices = origin_matrices(config)
    for ff in (0.04, 0.10):
        g, r = g_and_r(matrices, ff, config)
        print(f"FF {ff:.0%}: g {g:.4f}, r {r:.4f}")


if __name__ == "__main__":
    main()
