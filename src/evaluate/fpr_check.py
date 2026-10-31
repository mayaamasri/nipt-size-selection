"""Task 4D: false positive rate of the z-score methods across all conditions (z > z_threshold, target_fpr).

For each z-score method (A, B, B_c, B-opt, W, W_fixedFF), false positives are counted on the test normals
(fpr * test_normal per condition) and pooled:
  level "method"              all 96 conditions (conditions are independent, so this tests a consistent excess)
  level "method_depth"        per depth, over FF and noise
  level "method_noise"        per noise level, over FF and depth
  level "method_depth_noise"  per depth and noise, over the 8 FFs
Each row: pooled FPR with Wilson 95% CI, and over its conditions how many have FPR above the target, a Wilson
CI entirely above it, and a CI entirely below it. Within one condition the windowed methods share patients and
reference, so their FPRs are correlated; across conditions they are independent.

Expected FPR from skew (A, B, B_c, B-opt; NaN for W, W_fixedFF): a normal's windowed chr21 count is binomial,
n = N * r chr21-or-other fragments in the window with chr21 share p0, so its chr21 fraction is right-skewed
with skewness (1 - 2 p0) / sqrt(n p0 (1 - p0)). The noise term (relative SD sigma * (1 - p0), normal) adds
variance but no skew, so skew = binomial skew * (var_binomial / var_total)^(3/2). First-order
Cornish-Fisher: P(z > t) ~ Phi_bar(t) + phi(t) * skew * (t^2 - 1) / 6. r = 1 for A; the reference's own
estimation error is ignored (it adds ~0.002 percentage points on average).

Reads results/tables/results_all.csv, writes results/tables/fpr_check.csv. Run from the repo root:
    python -m src.evaluate.fpr_check
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import load_config, repo_path
from src.evaluate.run_condition import method_order
from src.simulate.mappability import wilson_interval

Z_METHODS_EXCLUDED = ("C1", "C2")  # trees: threshold set on calibration normals, not z > z_threshold
LEVELS = {"method": [], "method_depth": ["depth"], "method_noise": ["noise"], "method_depth_noise": ["depth", "noise"]}
COLUMNS = ["level", "method", "depth", "noise", "n_conditions", "n_normals", "false_positives", "fpr", "fpr_lo",
           "fpr_hi", "expected_fpr_skew", "n_above_target", "n_ci_above_target", "n_ci_below_target"]


def skew_expected_fpr(results, threshold):
    """Per row: first-order Cornish-Fisher P(z > threshold) for a normal; NaN for the weighted methods."""
    r = np.where(results["method"] == "A", 1.0, results["r"])  # NaN for W, W_fixedFF
    p0, n = results["p0"], results["depth"] * r
    var_bin = p0 * (1 - p0) / n
    var_noise = (p0 * (1 - p0) * results["sigma"]) ** 2
    skew = (1 - 2 * p0) / np.sqrt(n * p0 * (1 - p0)) * (var_bin / (var_bin + var_noise)) ** 1.5
    return norm.sf(threshold) + norm.pdf(threshold) * skew * (threshold ** 2 - 1) / 6


def fpr_check(results, config):
    target, t = config["methods"]["target_fpr"], config["methods"]["z_threshold"]
    n_normal = config["patients_per_condition"]["test_normal"]
    res = results[~results["method"].isin(Z_METHODS_EXCLUDED)].copy()
    fp = res["fpr"] * n_normal
    if not np.allclose(fp, fp.round(), rtol=0, atol=1e-6):
        raise ValueError("fpr * test_normal is not a whole number of patients for some rows")
    res["false_positives"] = fp.round().astype(int)
    res["expected_fpr_skew"] = skew_expected_fpr(res, t)

    rows = []
    for level, by in LEVELS.items():
        for key, g in res.groupby(["method"] + by, sort=False):
            key = dict(zip(["method"] + by, key if isinstance(key, tuple) else (key,)))
            k, n = int(g["false_positives"].sum()), len(g) * n_normal
            lo, hi = (float(v) for v in wilson_interval(k, n))
            rows.append({"level": level, "depth": "all", "noise": "all", **{c: str(v) for c, v in key.items()},
                         "n_conditions": len(g), "n_normals": n, "false_positives": k, "fpr": k / n,
                         "fpr_lo": lo, "fpr_hi": hi, "expected_fpr_skew": g["expected_fpr_skew"].mean(),
                         "n_above_target": int((g["fpr"] > target).sum()),
                         "n_ci_above_target": int((g["fpr_lo"] > target).sum()),
                         "n_ci_below_target": int((g["fpr_hi"] < target).sum())})
    table = pd.DataFrame(rows, columns=COLUMNS)
    order = {m: i for i, m in enumerate(method_order(config))}
    depth_order = {str(d): i for i, d in enumerate(config["grid"]["depth"])} | {"all": -1}
    noise_order = {str(v): i for i, v in enumerate(config["grid"]["noise_sigma"])} | {"all": -1}
    level_order = {lv: i for i, lv in enumerate(LEVELS)}
    return table.sort_values(["level", "method", "depth", "noise"], kind="stable", key=lambda c: c.map(
        {"level": level_order, "method": order, "depth": depth_order, "noise": noise_order}[c.name])
    ).reset_index(drop=True)


def main():
    config = load_config()
    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv", dtype={"noise": str})
    table = fpr_check(results, config)
    target = config["methods"]["target_fpr"]

    def pct(v):
        return f"{100 * v:.3f}"

    print(f"FPR of the z-score methods, target {100 * target:.3f}%, z > {config['methods']['z_threshold']}")
    print("\nAll conditions pooled (% ; CI = Wilson 95%):")
    print(f"{'method':>9} {'conds':>5} {'FP':>5} {'fpr':>6} {'CI':>15} {'exp_skew':>8} {'>tgt':>4} "
          f"{'CI>tgt':>6} {'CI<tgt':>6}")
    for r in table[table["level"] == "method"].itertuples():
        print(f"{r.method:>9} {r.n_conditions:5d} {r.false_positives:5d} {pct(r.fpr):>6} "
              f"[{pct(r.fpr_lo)}, {pct(r.fpr_hi)}] "
              f"{pct(r.expected_fpr_skew) if np.isfinite(r.expected_fpr_skew) else '':>8} "
              f"{r.n_above_target:4d} {r.n_ci_above_target:6d} {r.n_ci_below_target:6d}")
    for level, col in (("method_depth", "depth"), ("method_noise", "noise")):
        sub = table[table["level"] == level]
        print(f"\nPooled FPR (%) by {col} (expected from skew in brackets):")
        obs = sub.pivot(index="method", columns=col, values="fpr").reindex(index=sub["method"].unique(),
                                                                          columns=sub[col].unique())
        exp = sub.pivot(index="method", columns=col, values="expected_fpr_skew").reindex_like(obs)
        print(f"{'method':>9} " + " ".join(f"{c:>17}" for c in obs.columns))
        for m in obs.index:
            print(f"{m:>9} " + " ".join(
                f"{pct(obs.loc[m, c]):>7} [{pct(exp.loc[m, c]) if np.isfinite(exp.loc[m, c]) else '  -  '}]".rjust(17)
                for c in obs.columns))
    out = tables / "fpr_check.csv"
    table.to_csv(out, index=False, float_format="%.8g")
    print(f"\nWrote {len(table)} rows to {out}")


if __name__ == "__main__":
    main()
