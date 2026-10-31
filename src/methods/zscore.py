"""Task 3B / 4B: z-score methods A, B, B-opt, B_c, W and W_fixedFF (decisions #5, #6, #15).

  A      zA = z-score of the chr21 fraction, all fragments. Positive if zA > z_threshold.
  B      zB = z-score of the chr21 fraction, fragments size_selection_window_bp only (40-156 bp).
         Positive if zB > z_threshold.
  B-opt  same as B with window lo..u, lo = size_selection_window_bp[0]. u is chosen from
         size_opt_upper_grid on the TRAINING set as the one maximising
         mean zB_u(training T21) - mean zB_u(training normals), then used unchanged on the test set.
  B_c    same as B with the fixed window lo..c, one method per c in size_fixed_upper_extra (decision #15a).
  W      z_weighted > z_threshold, no tree (decision #15b).
  W_fixedFF  same as W with the weights at FF = weight_fixed_ff in every condition (decision #15c).

All z-scores are against the condition's reference normals (src.methods.reference).

Run from the repo root to print the B-opt separation on the training set of the milestone condition
(FF 4%, depth 5M, no noise):
    python -m src.methods.zscore
"""

import numpy as np

from src.config import load_config
from src.methods.reference import build_reference, statistics, zscores
from src.simulate.patients import generate_group, origin_matrices


def window_name(config, upper=None):
    """Statistic name of the windowed chr21 fraction (B's window if upper is None)."""
    lo, hi = config["methods"]["size_selection_window_bp"]
    return f"frac_{lo}_{hi if upper is None else upper}"


def separation(train_normal_z, train_t21_z, name):
    """Mean z of training T21 minus mean z of training normals, for one statistic."""
    return float(train_t21_z[name].mean() - train_normal_z[name].mean())


def opt_separation(train_normal_z, train_t21_z, config):
    """Per B-opt upper cutoff: mean z of training T21 minus mean z of training normals."""
    return {u: separation(train_normal_z, train_t21_z, window_name(config, u))
            for u in config["methods"]["size_opt_upper_grid"]}


def choose_opt_cutoff(train_normal_z, train_t21_z, config):
    """Upper cutoff with the largest separation (first in the grid if tied), and the full separation table."""
    sep = opt_separation(train_normal_z, train_t21_z, config)
    grid = list(sep)
    return grid[int(np.argmax([sep[u] for u in grid]))], sep


def fixed_cutoff_methods(config):
    """B_c method names, in size_fixed_upper_extra order: dict name -> upper cutoff (bp)."""
    return {f"B_{c}": c for c in config["methods"]["size_fixed_upper_extra"]}


def zscore_methods(test_z, train_normal_z, train_t21_z, config):
    """Scores, calls and cutoff for every z-score method on the test patients: A, B, B_c, B-opt, W, W_fixedFF.

    Returns (methods, sep): methods is dict method -> {"scores": array, "calls": bool array,
    "cutoff": upper bp of the size window, or None if the method keeps all lengths}; sep is the B-opt
    separation table {upper_bp: separation}.
    """
    t = config["methods"]["z_threshold"]
    hi_b = config["methods"]["size_selection_window_bp"][1]
    u, sep = choose_opt_cutoff(train_normal_z, train_t21_z, config)
    scores = {"A": (test_z["frac_all"], None),
              "B": (test_z[window_name(config)], hi_b),
              **{m: (test_z[window_name(config, c)], c) for m, c in fixed_cutoff_methods(config).items()},
              "B-opt": (test_z[window_name(config, u)], u),
              "W": (test_z["weighted"], None),
              "W_fixedFF": (test_z["weighted_fixedff"], None)}
    return {m: {"scores": s, "calls": s > t, "cutoff": c} for m, (s, c) in scores.items()}, sep


def main():
    config = load_config()
    ff, depth, noise = 0.04, 5_000_000, "none"  # milestone condition (brief 3F); not a study parameter
    matrices = origin_matrices(config)
    reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config, matrices)
    z = {g: zscores(statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices), reference)
         for g in ("train_normal", "train_t21")}

    u, sep = choose_opt_cutoff(z["train_normal"], z["train_t21"], config)
    print(f"B-opt on training set: FF {ff:.0%}, depth {depth:,}, noise {noise}")
    for cutoff, s in sep.items():
        print(f"  40-{cutoff} bp: mean zB(T21) - mean zB(normal) = {s:.3f}{'  <- chosen' if cutoff == u else ''}")
    print(f"  A (all fragments): {separation(z['train_normal'], z['train_t21'], 'frac_all'):.3f}")
    print(f"  B ({window_name(config)}): {separation(z['train_normal'], z['train_t21'], window_name(config)):.3f}")


if __name__ == "__main__":
    main()
