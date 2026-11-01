"""Task 5C: sampling noise of the observed separation ratio sep(B_c) / sep(A), for the Fig. 4 caption.

For a few conditions (FF 4% at 1M, 2.5M and 10M, realistic noise; and FF 1%, 10M, no noise, the largest
error outside low_signal in breakeven_check.csv): regenerate the reference and test groups with the grid
seeds, recompute the observed ratio for cutoffs 130, 140, 150 and 156 bp (checked against breakeven_check.csv)
and run a paired bootstrap over the test patients (same resampled patients for A and B_c). The reference
set adds no noise: sep is unchanged by the affine map from chr21 fraction to z.

Columns: condition, cutoff, sep_A, ratio_obs, ratio_pred, abs_error, rel_error, boot_sd (bootstrap SD of the
ratio), boot_sd_rel (boot_sd / ratio_obs) and err_over_sd (abs_error / boot_sd). Bootstrap seeds: spawn key
(4, i_ff, i_depth, i_noise), apart from the patient (1, ...), QC (2, ...) and bootstrap_diff (3, ...) keys.

Writes results/tables/breakeven_ratio_noise.csv. ~10 s. Run from the repo root (after breakeven_check):
    python -m src.evaluate.bootstrap_ratio
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.methods.reference import build_reference, statistics, zscores
from src.methods.zscore import window_name
from src.simulate.patients import _grid_index, generate_group, origin_matrices

CONDITIONS = [(0.04, 1_000_000, "realistic"), (0.04, 2_500_000, "realistic"), (0.04, 10_000_000, "realistic"),
              (0.01, 10_000_000, "none")]
CUTOFFS = (130, 140, 150, 156)
N_BOOT = 1000


def bootstrap_seed(config, ff, depth, noise):
    grid = config["grid"]
    key = (4,
           _grid_index(grid["fetal_fraction"], ff, "fetal_fraction"),
           _grid_index(grid["depth"], depth, "depth"),
           _grid_index(grid["noise_sigma"], noise, "noise_sigma"))
    return np.random.SeedSequence(config["seed"], spawn_key=key)


def separation(normal, t21):
    return (t21.mean() - normal.mean()) / normal.std(ddof=1)


def main():
    config = load_config()
    matrices = origin_matrices(config)
    tables = repo_path(config["paths"]["results"]) / "tables"
    check = pd.read_csv(tables / "breakeven_check.csv")
    fixed = config["methods"]["size_selection_window_bp"][1]

    rows = []
    for ff, depth, noise in CONDITIONS:
        reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config,
                                    matrices)
        z = {g: zscores(statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices),
                        reference) for g in ("test_normal", "test_t21")}
        n_normal, n_t21 = len(z["test_normal"]["frac_all"]), len(z["test_t21"]["frac_all"])
        rng = np.random.default_rng(bootstrap_seed(config, ff, depth, noise))
        a = (z["test_normal"]["frac_all"], z["test_t21"]["frac_all"])
        for cutoff in CUTOFFS:
            key = window_name(config, None if cutoff == fixed else cutoff)
            b = (z["test_normal"][key], z["test_t21"][key])
            ratio = separation(*b) / separation(*a)
            saved = check[np.isclose(check["ff"], ff) & (check["depth"] == depth) & (check["noise"] == noise)
                          & (check["cutoff"] == cutoff)].iloc[0]
            if not np.isclose(ratio, saved["ratio_obs"], rtol=1e-9, atol=0):
                raise ValueError(f"FF {ff} depth {depth} {noise} cutoff {cutoff}: ratio {ratio} but "
                                 f"breakeven_check.csv has {saved['ratio_obs']}")
            boot = np.empty(N_BOOT)
            for i in range(N_BOOT):
                i_n, i_t = rng.integers(0, n_normal, n_normal), rng.integers(0, n_t21, n_t21)
                boot[i] = separation(b[0][i_n], b[1][i_t]) / separation(a[0][i_n], a[1][i_t])
            sd = boot.std(ddof=1)
            rows.append({"ff": ff, "depth": depth, "noise": noise, "cutoff": cutoff, "sep_A": saved["sep_A"],
                         "ratio_obs": ratio, "ratio_pred": saved["ratio_pred"], "abs_error": saved["abs_error"],
                         "rel_error": saved["rel_error"], "boot_sd": sd, "boot_sd_rel": sd / ratio,
                         "err_over_sd": saved["abs_error"] / sd})
            r = rows[-1]
            print(f"FF {ff:.0%} {depth / 1e6:g}M {noise:>9} cutoff {cutoff}: ratio {ratio:.4f} "
                  f"(pred {r['ratio_pred']:.4f}, error {100 * r['rel_error']:+.2f}%) | bootstrap SD "
                  f"{100 * r['boot_sd_rel']:.2f}% | error / SD {r['err_over_sd']:+.2f}")

    out = tables / "breakeven_ratio_noise.csv"
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False, float_format="%.10g")
    print(f"Bootstrap SD of the ratio: {100 * table['boot_sd_rel'].min():.2f}–{100 * table['boot_sd_rel'].max():.2f}%")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
