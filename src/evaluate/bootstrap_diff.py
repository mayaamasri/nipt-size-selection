"""Task 5B: sampling noise of sens_at_fpr(B) - sens_at_fpr(A), for Figure 3.

Per condition (FF x depth x noise): regenerate the reference and test groups with the grid seeds, score A
(frac_all) and B (Yang window) as in run_condition, and check both sens_at_fpr values against
results_all.csv. Then a paired bootstrap over the test patients: resample the test normals and the test T21
with replacement (same patients for A and B) and recompute the difference.

The reference set adds no noise here: a z-score is an affine map of the chr21 fraction, and sens_at_fpr
depends only on the ranking within the test set. Most of the noise comes from the threshold, which at
FPR 0.135% rests on ~13 of the 10,000 test normals, so the percentile CIs are approximate.

Columns: diff, se (bootstrap SD), ci_lo/ci_hi (2.5/97.5 percentiles), within_noise (CI includes 0, not
saturated), saturated (both methods >= 99.95%). Bootstrap seeds: spawn key (3, i_ff, i_depth, i_noise),
apart from the patient (1, ...) and QC (2, ...) keys.

Writes results/tables/fig3_b_minus_a.csv. ~7 min for the 96 conditions. Run from the repo root:
    python -m src.evaluate.bootstrap_diff
"""

import time

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.metrics import sens_at_fpr
from src.methods.reference import build_reference, statistics, zscores
from src.methods.zscore import window_name
from src.simulate.patients import _grid_index, generate_group, origin_matrices

N_BOOT = 2000
SATURATED = 0.9995


def bootstrap_seed(config, ff, depth, noise):
    grid = config["grid"]
    key = (3,
           _grid_index(grid["fetal_fraction"], ff, "fetal_fraction"),
           _grid_index(grid["depth"], depth, "depth"),
           _grid_index(grid["noise_sigma"], noise, "noise_sigma"))
    return np.random.SeedSequence(config["seed"], spawn_key=key)


def condition_scores(config, ff, depth, noise, matrices):
    """Test labels and the A and B scores of one condition, exactly as in run_condition."""
    reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config, matrices)
    z = {g: zscores(statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices), reference)
         for g in ("test_normal", "test_t21")}
    y = np.r_[np.zeros(len(z["test_normal"]["frac_all"]), int), np.ones(len(z["test_t21"]["frac_all"]), int)]
    scores = {m: np.concatenate([z["test_normal"][k], z["test_t21"][k]])
              for m, k in (("A", "frac_all"), ("B", window_name(config)))}
    return y, scores


def bootstrap_diff(y, scores, target, rng, n_boot=N_BOOT):
    normal, t21 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.r_[rng.choice(normal, len(normal)), rng.choice(t21, len(t21))]
        diffs[i] = sens_at_fpr(y[idx], scores["B"][idx], target) - sens_at_fpr(y[idx], scores["A"][idx], target)
    return diffs


def main():
    config = load_config()
    matrices = origin_matrices(config)
    target = config["methods"]["target_fpr"]
    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv")
    grid = config["grid"]

    rows, start = [], time.time()
    for noise in grid["noise_sigma"]:
        for depth in grid["depth"]:
            for ff in grid["fetal_fraction"]:
                y, scores = condition_scores(config, ff, depth, noise, matrices)
                sens = {m: sens_at_fpr(y, s, target) for m, s in scores.items()}
                saved = results[np.isclose(results["ff"], ff) & (results["depth"] == depth)
                                & (results["noise"] == str(noise))].set_index("method")["sens_at_fpr"]
                for m in ("A", "B"):
                    if not np.isclose(sens[m], saved[m], rtol=0, atol=1e-12):
                        raise ValueError(f"FF {ff} depth {depth} {noise}: sens_at_fpr({m}) = {sens[m]} "
                                         f"but results_all.csv has {saved[m]}")
                diffs = bootstrap_diff(y, scores, target, np.random.default_rng(bootstrap_seed(config, ff, depth,
                                                                                               noise)))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                saturated = min(sens.values()) >= SATURATED
                rows.append({"ff": ff, "depth": depth, "noise": str(noise),
                             "sens_at_fpr_A": sens["A"], "sens_at_fpr_B": sens["B"],
                             "diff": sens["B"] - sens["A"], "se": diffs.std(ddof=1), "ci_lo": lo, "ci_hi": hi,
                             "within_noise": (not saturated) and lo <= 0 <= hi, "saturated": saturated})
                r = rows[-1]
                print(f"FF {ff:5.0%} {depth / 1e6:>4g}M {str(noise):>9}: diff {100 * r['diff']:+6.1f} "
                      f"SE {100 * r['se']:5.2f} CI [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  ({time.time() - start:.0f} s)")

    out = tables / "fig3_b_minus_a.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print(f"Saved {out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
