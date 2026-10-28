"""Task 3E: detection limit per method x depth x noise (decision #13).

For one group, with sensitivity s(FF) on the config FF grid (grid.fetal_fraction):
  - the limit is where s crosses evaluation.sensitivity_target (0.95) for the LAST time: s >= target at every
    higher grid FF. Let i be the last grid FF with s < target; the limit is linearly interpolated between
    FF_i and FF_(i+1). If s(FF_(i+1)) equals the target exactly, the limit is FF_(i+1).
  - s >= target at every grid FF  -> "< 1%" (below the grid), no number
  - s < target at the highest FF  -> "> 15%" (above the grid), no number
Two versions: point estimate from sens, conservative from sens_lo (Wilson 95% lower bound).
A group missing any grid FF is skipped and reported, never computed from an incomplete grid.

Reads results/tables/results_all.csv and writes results/tables/detection_limits.csv. The hand-checked
self-test runs first, every time:
    python -m src.evaluate.detection_limits
"""

import math

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.run_condition import METHODS

GROUP = ["method", "depth", "noise"]
LIMIT_COLUMNS = GROUP + ["ff_limit", "ff_limit_text", "ff_limit_cons", "ff_limit_cons_text"]


def crossing(ffs, sens, target):
    """Last crossing of target, as (ff or NaN, text). ffs ascending, sens aligned with ffs."""
    ffs, sens = np.asarray(ffs, float), np.asarray(sens, float)
    below = sens < target
    if not below.any():
        return math.nan, f"< {ffs[0]:.0%}"
    if below[-1]:
        return math.nan, f"> {ffs[-1]:.0%}"
    i = int(np.flatnonzero(below)[-1])
    if sens[i + 1] == target:
        ff = float(ffs[i + 1])
    else:
        ff = float(ffs[i] + (target - sens[i]) * (ffs[i + 1] - ffs[i]) / (sens[i + 1] - sens[i]))
    return ff, f"{100 * ff:.2f}%"


def detection_limits(results, config):
    """One row per complete method x depth x noise group. Returns (table, list of skipped-group messages)."""
    grid = np.array(sorted(config["grid"]["fetal_fraction"]), float)
    target = config["evaluation"]["sensitivity_target"]
    rows, skipped = [], []
    for key, g in results.groupby(GROUP, sort=False):
        g = g.sort_values("ff")
        ffs = g["ff"].to_numpy(float)
        matched = [np.flatnonzero(np.isclose(grid, f, rtol=1e-9, atol=0)) for f in ffs]
        if any(len(m) != 1 for m in matched) or len(set(int(m[0]) for m in matched)) != len(ffs):
            raise ValueError(f"{dict(zip(GROUP, key))}: FF values {ffs.tolist()} are off the grid or duplicated")
        missing = sorted(set(range(len(grid))) - {int(m[0]) for m in matched})
        if missing:
            skipped.append(f"skipped {dict(zip(GROUP, key))}: missing FF {[float(grid[j]) for j in missing]}")
            continue
        ff, text = crossing(ffs, g["sens"], target)
        ff_c, text_c = crossing(ffs, g["sens_lo"], target)
        rows.append(dict(zip(GROUP, key), ff_limit=ff, ff_limit_text=text, ff_limit_cons=ff_c,
                         ff_limit_cons_text=text_c))
    table = pd.DataFrame(rows, columns=LIMIT_COLUMNS)
    noise_order = [str(v) for v in config["grid"]["noise_sigma"]]
    table = table.sort_values(GROUP, key=lambda c: c.map({m: i for i, m in enumerate(METHODS)}) if c.name == "method"
                              else c.map({n: i for i, n in enumerate(noise_order)}) if c.name == "noise" else c,
                              kind="stable").reset_index(drop=True)
    return table, skipped


def self_test(config):
    """Made-up table on the config FF grid; every expected limit is computed by hand in the comments."""
    grid = sorted(config["grid"]["fetal_fraction"])
    if grid != [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15] or config["evaluation"]["sensitivity_target"] != 0.95:
        raise ValueError("self-test expectations assume the FF grid 1-15% and target 0.95")
    cases = {
        # clean: sens crosses between 4% (0.93) and 6% (0.97): 0.04 + 0.02 * (0.95-0.93)/(0.97-0.93) = 0.05
        #        sens_lo hits 0.95 exactly at 6% (last below: 4%, 0.91) -> 0.06
        "clean": ([0.50, 0.70, 0.85, 0.93, 0.97, 0.99, 1.00, 1.00],
                  [0.48, 0.68, 0.83, 0.91, 0.95, 0.97, 0.98, 0.98], (0.05, "5.00%"), (0.06, "6.00%")),
        # dip: above at 3% (0.96), back below at 4% (0.94): last crossing 4%->6%: 0.04 + 0.02 * 0.01/0.04 = 0.045
        #      sens_lo: last below 4% (0.92), 6% 0.96: 0.04 + 0.02 * 0.03/0.04 = 0.055
        "dip": ([0.60, 0.80, 0.96, 0.94, 0.98, 0.99, 1.00, 1.00],
                [0.58, 0.78, 0.94, 0.92, 0.96, 0.97, 0.98, 0.98], (0.045, "4.50%"), (0.055, "5.50%")),
        # already: sens >= 0.95 everywhere -> "< 1%"; sens_lo 0.94 at 1%, exactly 0.95 at 2% -> 0.02
        "already": ([0.96, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00],
                    [0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.98, 0.98], (math.nan, "< 1%"), (0.02, "2.00%")),
        # never: sens 0.94 at 15% -> "> 15%" for both
        "never": ([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.94],
                  [0.08, 0.18, 0.28, 0.38, 0.53, 0.68, 0.83, 0.92], (math.nan, "> 15%"), (math.nan, "> 15%")),
        # exact: sens exactly 0.95 at 4% (3% is 0.90) -> 0.04; sens_lo exactly 0.95 at 6% (4% is 0.93) -> 0.06
        "exact": ([0.50, 0.70, 0.90, 0.95, 0.97, 0.99, 1.00, 1.00],
                  [0.48, 0.68, 0.88, 0.93, 0.95, 0.97, 0.98, 0.98], (0.04, "4.00%"), (0.06, "6.00%")),
    }
    rows = [{"ff": f, "depth": 1_000_000, "noise": "none", "method": name, "sens": s, "sens_lo": lo}
            for name, (sens, sens_lo, _, _) in cases.items() for f, s, lo in zip(grid, sens, sens_lo)]
    # incomplete: FF 8% missing -> must be skipped
    rows += [{"ff": f, "depth": 1_000_000, "noise": "none", "method": "incomplete", "sens": 0.99, "sens_lo": 0.98}
             for f in grid if f != 0.08]
    table, skipped = detection_limits(pd.DataFrame(rows), config)  # the made-up methods sort as NaN; order unused

    by = table.set_index("method")
    for name, (_, _, point, cons) in cases.items():
        for col, (ff, text) in (("ff_limit", point), ("ff_limit_cons", cons)):
            got_ff, got_text = by.loc[name, col], by.loc[name, f"{col}_text"]
            ok = got_text == text and (math.isnan(ff) and math.isnan(got_ff) or math.isclose(got_ff, ff, abs_tol=1e-12))
            print(f"self-test {name:>8} {col:>13}: {got_text:>6} (hand {text}) {'ok' if ok else 'FAIL'}")
            if not ok:
                raise ValueError(f"self-test {name} {col}: got {got_ff!r} {got_text!r}, expected {ff!r} {text!r}")
    if "incomplete" in by.index or len(skipped) != 1 or "incomplete" not in skipped[0]:
        raise ValueError(f"self-test: the incomplete group was not skipped as expected ({skipped})")
    print(f"self-test incomplete: {skipped[0]}")


def main():
    config = load_config()
    self_test(config)

    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv", dtype={"noise": str})
    table, skipped = detection_limits(results, config)
    for message in skipped:
        print(message)
    out = tables / "detection_limits.csv"
    table.to_csv(out, index=False, float_format="%.10g")
    print(f"Wrote {len(table)} rows to {out} ({len(skipped)} groups skipped as incomplete)")


if __name__ == "__main__":
    main()
