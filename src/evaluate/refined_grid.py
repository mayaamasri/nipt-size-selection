"""Task 09, 9C: is linear interpolation between grid FFs good enough for the close rankings?

The bootstrap CIs (src.evaluate.limit_ci) cover sampling error, not interpolation error. For every main-grid setting
(depth x noise) where a 9B pair's detection limits (the paper's, sens basis) differ by less than
uncertainty.refine_max_diff, the grid interval holding each of the two limits gets its midpoint FF as an extra
condition (both midpoints if the limits lie in different intervals). The resulting list must equal
uncertainty.refine_conditions in the config (self-test), which fixes the patient seeds: family 7,
(7, k, i_depth, i_noise, i_group), k = position of the FF in uncertainty.refine_ff (src.simulate.patients), so the
main grid's seeds are untouched.

Each extra condition is run end to end with run_condition (all groups, all methods, new trees) and written to
results/refined_grid/tables (results_all.csv etc. and run_log.csv). The limits of the settings are then recomputed
with detection_limits.crossing on the refined FF grid (the 8 grid FFs plus that setting's extra FFs).

Writes:
  results/tables/limits_refined_grid.csv   per setting and method: extra FFs, the pairs that triggered them,
      interpolated (paper) and refined limit and their difference (points), and moved_interval: the refined limit
      is outside the grid interval of the interpolated one (a dip in sensitivity at a new FF moved the last
      crossing), flagged separately from ranking changes
  results/tables/limits_refined_pairs.csv  per setting and 9B pair: interpolated and refined difference and
      ranking_changed and change_type: flip (the order reverses), became_tie (the refined difference is within
      LIMIT_TIE = 0.05 points) or tie_resolved (a tie on the interpolated grid is ordered on the refined one)
Run from the repo root (~1 min; conditions already in results/refined_grid are not rerun unless --rerun):
    python -m src.evaluate.refined_grid
"""

import argparse
import copy
import math

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.detection_limits import crossing
from src.evaluate.limit_ci import MAIN, PAIRS
from src.evaluate.run_condition import git_state, replace_condition_rows, run_condition

OUT = "results/refined_grid"
LIMIT_TIE = 0.0005         # refined differences within 0.05 points count as a tie for the ranking flag


def needed_conditions(config):
    """[(ff, depth, noise, [pairs])] from detection_limits.csv: midpoints of the intervals of close pairs."""
    lim = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "detection_limits.csv",
                      dtype={"noise": str}).set_index(["method", "depth", "noise"])
    grid = sorted(config["grid"]["fetal_fraction"])
    max_diff = config["uncertainty"]["refine_max_diff"]
    need = {}
    for noise in [str(n) for n in config["grid"]["noise_sigma"]]:
        for depth in config["grid"]["depth"]:
            for a, b in PAIRS[MAIN]:
                la, lb = lim.loc[(a, depth, noise), "ff_limit"], lim.loc[(b, depth, noise), "ff_limit"]
                if not (np.isfinite(la) and np.isfinite(lb)) or abs(la - lb) >= max_diff:
                    continue
                for v in (la, lb):
                    i = max(int(np.searchsorted(grid, v, side="left")) - 1, 0)   # interval (grid[i], grid[i+1]]
                    mid = round((grid[i] + grid[i + 1]) / 2, 6)
                    need.setdefault((mid, depth, noise), set()).add(f"{a}-{b}")
    return [(ff, d, n, sorted(p)) for (ff, d, n), p in sorted(need.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))]


def self_test(config, needed):
    listed = sorted((float(ff), int(d), str(n)) for ff, d, n in config["uncertainty"]["refine_conditions"])
    derived = sorted((ff, d, n) for ff, d, n, _ in needed)
    if listed != derived:
        raise ValueError(f"uncertainty.refine_conditions {listed} != conditions derived from detection_limits.csv "
                         f"{derived}")
    if not set(ff for ff, _, _ in derived) <= set(float(v) for v in config["uncertainty"]["refine_ff"]):
        raise ValueError("an extra FF is missing from uncertainty.refine_ff")
    print(f"self-test: {len(derived)} extra conditions derived from detection_limits.csv == config list ok")


def run_extra(config, needed, rerun):
    out_cfg = copy.deepcopy(config)
    out_cfg["paths"]["results"] = OUT
    tables_dir = repo_path(OUT) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(tables_dir / "results_all.csv", dtype={"noise": str}) if (tables_dir / "results_all.csv").exists() \
        else pd.DataFrame(columns=["ff", "depth", "noise"])
    commit, dirty = git_state()
    for ff, depth, noise, _ in needed:
        have = (np.isclose(done["ff"].astype(float), ff) & (done["depth"].astype(int) == depth)
                & (done["noise"] == noise)).any()
        if have and not rerun:
            continue
        tables, _ = run_condition(config, ff, depth, noise)
        tables["run_log.csv"] = pd.DataFrame([{"ff": ff, "depth": depth, "noise": noise, "commit": commit,
                                               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                               "dirty": dirty}])
        for name, table in tables.items():
            replace_condition_rows(tables_dir / name, table, out_cfg)
        print(f"ran FF {ff:g}, depth {depth:,}, noise {noise}")
    return pd.read_csv(tables_dir / "results_all.csv", dtype={"noise": str})


def refined_tables(config, needed, extra):
    main = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "results_all.csv", dtype={"noise": str})
    lim = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "detection_limits.csv",
                      dtype={"noise": str}).set_index(["method", "depth", "noise"])
    target = config["evaluation"]["sensitivity_target"]
    grid = sorted(config["grid"]["fetal_fraction"])
    settings = {}
    for ff, d, n, pairs in needed:
        s = settings.setdefault((d, n), {"ffs": [], "pairs": set()})
        s["ffs"].append(ff)
        s["pairs"].update(pairs)
    rows, prow = [], []
    for (depth, noise), s in settings.items():
        both = pd.concat([main[(main["depth"] == depth) & (main["noise"] == noise)],
                          extra[(extra["depth"] == depth) & (extra["noise"] == noise)]])
        refined = {}
        for m in sorted({x for p in PAIRS[MAIN] for x in p}):
            g = both[both["method"] == m].sort_values("ff")
            v, text = crossing(g["ff"].to_numpy(float), g["sens"].to_numpy(float), target)
            p = lim.loc[(m, depth, noise)]
            moved = False
            if np.isfinite(v) and np.isfinite(p["ff_limit"]):
                i = max(int(np.searchsorted(grid, p["ff_limit"], side="left")) - 1, 0)
                moved = not (grid[i] <= v <= grid[i + 1])
            refined[m] = v
            rows.append({"depth": depth, "noise": noise, "method": m, "extra_ff": " ".join(f"{f:g}" for f in s["ffs"]),
                         "triggered_by": " ".join(sorted(s["pairs"])),
                         "limit_interpolated": p["ff_limit"], "limit_interpolated_text": p["ff_limit_text"],
                         "limit_refined": v, "limit_refined_text": text,
                         "difference_pts": 100 * (v - p["ff_limit"]), "moved_interval": moved})
        for a, b in PAIRS[MAIN]:
            di = lim.loc[(a, depth, noise), "ff_limit"] - lim.loc[(b, depth, noise), "ff_limit"]
            dr = refined[a] - refined[b]
            sign = lambda x: 0 if abs(x) <= LIMIT_TIE else int(np.sign(x))  # noqa: E731
            change = "none"
            if np.isfinite(di) and np.isfinite(dr) and sign(di) != sign(dr):
                change = ("flip" if sign(di) * sign(dr) == -1 else "became_tie" if sign(dr) == 0
                          else "tie_resolved")
            prow.append({"depth": depth, "noise": noise, "pair": f"{a}-{b}",
                         "triggered": f"{a}-{b}" in s["pairs"], "diff_interpolated_pts": 100 * di,
                         "diff_refined_pts": 100 * dr, "ranking_changed": change != "none", "change_type": change})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun", action="store_true", help="rerun extra conditions that already have rows")
    args = parser.parse_args()
    config = load_config()
    needed = needed_conditions(config)
    self_test(config, needed)
    extra = run_extra(config, needed, args.rerun)
    limits, pairs = refined_tables(config, needed, extra)
    tables = repo_path(config["paths"]["results"]) / "tables"
    limits.to_csv(tables / "limits_refined_grid.csv", index=False, float_format="%.6g")
    pairs.to_csv(tables / "limits_refined_pairs.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print("\nExtra FFs per setting and the pairs that triggered them:")
        for ff, d, n, p in needed:
            print(f"  {d / 1e6:>4g}M {n:>9}: FF {100 * ff:g}% <- {', '.join(p)}")
        print("\nInterpolated vs refined limits (FF %):")
        show = limits.assign(**{c: 100 * limits[c] for c in ("limit_interpolated", "limit_refined")})
        print(show.drop(columns=["limit_interpolated_text", "limit_refined_text", "triggered_by"])
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        moved = limits[limits["moved_interval"]]
        print(f"\nLimits moved to another grid interval (dip at a new FF): {len(moved)}")
        print(moved.to_string(index=False) if len(moved) else "  none")
        flips = pairs[pairs["ranking_changed"]].sort_values("change_type", kind="stable")
        counts = flips["change_type"].value_counts().to_dict()
        print(f"\nRanking changes on the refined grid: {len(flips)} {counts}")
        print(flips.to_string(index=False, float_format=lambda v: f"{v:+.3f}") if len(flips) else "  none")
    print(f"\nSaved {tables / 'limits_refined_grid.csv'} and {tables / 'limits_refined_pairs.csv'}")


if __name__ == "__main__":
    main()
