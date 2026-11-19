"""Task 09 follow-up (decision #19): detection limits of the main grid on the 14-point FF grid.

The 14-point grid is grid.fetal_fraction (1, 2, 3, 4, 6, 8, 10, 15%) plus evaluation.limit_extra_ff (1.5, 2.5, 3.5,
5, 7, 9%), every point simulated at every depth x noise setting (the extra FFs in results/refined_grid/tables, from
python -m src.evaluate.refined_grid --all-settings; patient seeds family 7). The limits use the paper's rule
unchanged (detection_limits.detection_limits and crossing: the FF where sensitivity crosses 95% for the last time,
linear between neighbouring grid points), on sens (main), sens_lo (conservative) and sens_at_fpr (check).

The at_fpr check flag keeps its meaning from the 8-FF grid: "more than one grid step" is measured on the
original 8-FF positions (detection_limits.grid_position on grid.fetal_fraction), since a step on the 14-point grid
would be 0.5 points below 4% but 5 points at the top. at_fpr_pos_diff_14 (position difference on the 14-point
grid) is kept for reference.

Dip flags (limits_dip_flags.csv): on any basis, a limit whose last point below 95% on the 14-point grid is one of
the new FFs while the original grid FF just below it is already at or above 95%, i.e. one new point dips and the
curve recovers at the next FF (sampling noise rather than interpolation). Listed with the 8-FF and 14-point limits.

The scenarios (Tasks 07, 08), the per-condition results and Figs. 3-5 stay on the 8-FF grid (detection_limits.csv).

Writes results/tables/detection_limits_refined.csv (detection_limits.csv's columns plus at_fpr_pos_diff_14) and
limits_dip_flags.csv. The self-tests run first: detection_limits' own self-test, and on the original 8 FFs this
code reproduces detection_limits.csv exactly. Run from the repo root (seconds):
    python -m src.evaluate.detection_limits_refined
"""

import copy
import io

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.detection_limits import FLAG_STEPS, GROUP, detection_limits, grid_position, self_test

EXTRA_DIR = "results/refined_grid/tables"
BASES = {"sens": "ff_limit", "sens_lo": "ff_limit_cons", "sens_at_fpr": "ff_limit_at_fpr"}


def as_written(table):
    buf = io.StringIO()
    table.to_csv(buf, index=False, float_format="%.10g")
    return pd.read_csv(io.StringIO(buf.getvalue()), dtype=str, keep_default_na=False)


def grid14(config):
    return sorted(float(v) for v in config["grid"]["fetal_fraction"] + config["evaluation"]["limit_extra_ff"])


def config14(config):
    out = copy.deepcopy(config)
    out["grid"]["fetal_fraction"] = grid14(config)
    return out


def merged_results(config):
    """Main-grid results_all.csv plus every extra-FF condition; raises if any of the 72 is missing."""
    tables = repo_path(config["paths"]["results"]) / "tables"
    main = pd.read_csv(tables / "results_all.csv", dtype={"noise": str})
    extra = pd.read_csv(repo_path(EXTRA_DIR) / "results_all.csv", dtype={"noise": str})
    g = config["grid"]
    for noise in g["noise_sigma"]:
        for depth in g["depth"]:
            for ff in config["evaluation"]["limit_extra_ff"]:
                have = (np.isclose(extra["ff"], ff, rtol=1e-9, atol=0) & (extra["depth"] == depth)
                        & (extra["noise"] == str(noise)))
                if have.sum() != main["method"].nunique():
                    raise ValueError(f"extra condition FF {ff} {depth} {noise} missing or incomplete in {EXTRA_DIR}; "
                                     f"run python -m src.evaluate.refined_grid --all-settings")
    return pd.concat([main, extra], ignore_index=True)


def refined_limits(config, results):
    table, skipped = detection_limits(results, config14(config))
    if skipped:
        raise ValueError("incomplete groups on the 14-point grid: " + "; ".join(skipped))
    grid8 = np.array(sorted(config["grid"]["fetal_fraction"]), float)
    table["at_fpr_pos_diff_14"] = table["at_fpr_pos_diff"]
    pos = [abs(grid_position(r.ff_limit, r.ff_limit_text, grid8) - grid_position(r.ff_limit_at_fpr,
                                                                                r.ff_limit_at_fpr_text, grid8))
           for r in table.itertuples()]
    table["at_fpr_pos_diff"] = pos
    table["at_fpr_flag"] = [bool(d > FLAG_STEPS + 1e-9) for d in pos]
    return table


def dip_flags(config, results, refined, original):
    target = config["evaluation"]["sensitivity_target"]
    grid8 = sorted(float(v) for v in config["grid"]["fetal_fraction"])
    extra = {float(v) for v in config["evaluation"]["limit_extra_ff"]}
    cols = {"sens": "sens", "sens_lo": "sens_lo", "sens_at_fpr": "sens_at_fpr"}
    ref, orig = refined.set_index(GROUP), original.set_index(GROUP)
    rows = []
    for key, g in results.groupby(GROUP, sort=False):
        g = g.sort_values("ff")
        ffs = g["ff"].to_numpy(float)
        for basis, col in cols.items():
            s = g[col].to_numpy(float)
            below = np.flatnonzero(s < target)
            if not len(below) or below[-1] == len(s) - 1:
                continue
            i = below[-1]
            if not any(np.isclose(ffs[i], e, rtol=1e-9, atol=0) for e in extra):
                continue
            lower = [f for f in grid8 if f < ffs[i] - 1e-12]
            if not lower:
                continue
            j = int(np.flatnonzero(np.isclose(ffs, lower[-1], rtol=1e-9, atol=0))[0])
            if s[j] >= target:
                limit_col = BASES[basis]
                rows.append({**dict(zip(GROUP, key)), "basis": basis, "dip_ff": ffs[i], "sens_at_dip": s[i],
                             "sens_at_grid_ff_below": s[j], "grid_ff_below": ffs[j],
                             "limit_8ff": orig.loc[key, limit_col], "limit_8ff_text": orig.loc[key, f"{limit_col}_text"],
                             "limit_14": ref.loc[key, limit_col], "limit_14_text": ref.loc[key, f"{limit_col}_text"]})
    return pd.DataFrame(rows, columns=GROUP + ["basis", "dip_ff", "sens_at_dip", "sens_at_grid_ff_below",
                                              "grid_ff_below", "limit_8ff", "limit_8ff_text", "limit_14",
                                              "limit_14_text"])


def main():
    config = load_config()
    self_test(config)
    tables = repo_path(config["paths"]["results"]) / "tables"
    main_results = pd.read_csv(tables / "results_all.csv", dtype={"noise": str})
    original = pd.read_csv(tables / "detection_limits.csv", dtype={"noise": str})
    # self-test: on the original 8 FFs, this path reproduces detection_limits.csv exactly
    eight, _ = detection_limits(main_results, config)
    if not as_written(eight).equals(as_written(original)):
        raise ValueError("self-test: the 8-FF limits differ from detection_limits.csv")
    print(f"self-test: 8-FF limits == detection_limits.csv ({len(eight)} rows) ok")

    results = merged_results(config)
    refined = refined_limits(config, results)
    dips = dip_flags(config, results, refined, original)
    refined.to_csv(tables / "detection_limits_refined.csv", index=False, float_format="%.10g")
    dips.to_csv(tables / "limits_dip_flags.csv", index=False, float_format="%.6g")
    print(f"14-point grid {[f'{100 * f:g}' for f in grid14(config)]}: {len(refined)} limits, "
          f"{int(refined['at_fpr_flag'].sum())} at_fpr flags (8-FF steps), {len(dips)} dip flags")
    if len(dips):
        with pd.option_context("display.width", 250):
            print(dips.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved {tables / 'detection_limits_refined.csv'} and {tables / 'limits_dip_flags.csv'}")


if __name__ == "__main__":
    main()
