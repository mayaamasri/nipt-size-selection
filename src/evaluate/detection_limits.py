"""Task 3E: detection limit per method x depth x noise (decision #13).

For one group, with sensitivity s(FF) on the config FF grid (grid.fetal_fraction):
  - the limit is where s crosses evaluation.sensitivity_target (0.95) for the LAST time: s >= target at every
    higher grid FF. Let i be the last grid FF with s < target; the limit is linearly interpolated between
    FF_i and FF_(i+1). If s(FF_(i+1)) equals the target exactly, the limit is FF_(i+1).
  - s >= target at every grid FF  -> "< 1%" (below the grid), no number
  - s < target at the highest FF  -> "> 15%" (above the grid), no number
Main result: point estimate from sens, conservative from sens_lo (Wilson 95% lower bound).
Check: the same rule on sens_at_fpr (sensitivity at FPR exactly target_fpr), point estimate only. The two
point estimates are compared on fractional grid positions (grid FF i -> i, linear in between, so 4.5% -> 3.5;
"< 1%" -> 0 and "> 15%" -> the last position, i.e. the grid edges). A group is flagged if they differ by more
than one grid step.
A group missing any grid FF is skipped and reported, never computed from an incomplete grid.

Reads results/tables/results_all.csv and writes results/tables/detection_limits.csv. The hand-checked
self-test runs first, every time:
    python -m src.evaluate.detection_limits
"""

import math

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.run_condition import method_order

GROUP = ["method", "depth", "noise"]
LIMIT_COLUMNS = GROUP + ["ff_limit", "ff_limit_text", "ff_limit_cons", "ff_limit_cons_text",
                         "ff_limit_at_fpr", "ff_limit_at_fpr_text", "at_fpr_pos_diff", "at_fpr_flag"]
FLAG_STEPS = 1  # flag if the two limits are more than this many grid steps apart


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


def grid_position(ff, text, grid):
    """Fractional grid position: grid FF i -> i, linear in between (4.5% -> 3.5 on the 1-15% grid).
    Limits outside the grid sit on its edges: "< lowest" -> 0, "> highest" -> len(grid) - 1."""
    if text.startswith("<"):
        return 0.0
    if text.startswith(">"):
        return float(len(grid) - 1)
    i = int(np.searchsorted(grid, ff, side="right")) - 1
    if i == len(grid) - 1:
        return float(i)
    return i + (ff - grid[i]) / (grid[i + 1] - grid[i])


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
        ff_f, text_f = crossing(ffs, g["sens_at_fpr"], target)
        diff = abs(grid_position(ff, text, grid) - grid_position(ff_f, text_f, grid))
        rows.append(dict(zip(GROUP, key), ff_limit=ff, ff_limit_text=text, ff_limit_cons=ff_c,
                         ff_limit_cons_text=text_c, ff_limit_at_fpr=ff_f, ff_limit_at_fpr_text=text_f,
                         at_fpr_pos_diff=diff, at_fpr_flag=bool(diff > FLAG_STEPS + 1e-9)))
    table = pd.DataFrame(rows, columns=LIMIT_COLUMNS)
    noise_order = [str(v) for v in config["grid"]["noise_sigma"]]
    methods = method_order(config)
    table = table.sort_values(GROUP, key=lambda c: c.map({m: i for i, m in enumerate(methods)}) if c.name == "method"
                              else c.map({n: i for i, n in enumerate(noise_order)}) if c.name == "noise" else c,
                              kind="stable").reset_index(drop=True)
    return table, skipped


def self_test(config):
    """Made-up table on the config FF grid; every expected limit is computed by hand in the comments.
    Grid positions: 1%=0, 2%=1, 3%=2, 4%=3, 6%=4, 8%=5, 10%=6, 15%=7."""
    grid = sorted(config["grid"]["fetal_fraction"])
    if grid != [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15] or config["evaluation"]["sensitivity_target"] != 0.95:
        raise ValueError("self-test expectations assume the FF grid 1-15% and target 0.95")
    clean = [0.50, 0.70, 0.85, 0.93, 0.97, 0.99, 1.00, 1.00]
    clean_lo = [0.48, 0.68, 0.83, 0.91, 0.95, 0.97, 0.98, 0.98]
    exact = [0.50, 0.70, 0.90, 0.95, 0.97, 0.99, 1.00, 1.00]
    exact_lo = [0.48, 0.68, 0.88, 0.93, 0.95, 0.97, 0.98, 0.98]
    already = [0.96, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00]
    already_lo = [0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.98, 0.98]
    never = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.94]
    never_lo = [0.08, 0.18, 0.28, 0.38, 0.53, 0.68, 0.83, 0.92]
    # name: (sens, sens_lo, sens_at_fpr or None = same as sens, main (ff, text), conservative (ff, text),
    #        check (ff, text) or None = same as main, position difference, flag)
    cases = {
        # clean: sens crosses between 4% (0.93) and 6% (0.97): 0.04 + 0.02 * (0.95-0.93)/(0.97-0.93) = 0.05
        #        sens_lo hits 0.95 exactly at 6% (last below: 4%, 0.91) -> 0.06
        "clean": (clean, clean_lo, None, (0.05, "5.00%"), (0.06, "6.00%"), None, 0.0, False),
        # dip: above at 3% (0.96), back below at 4% (0.94): last crossing 4%->6%: 0.04 + 0.02 * 0.01/0.04 = 0.045
        #      sens_lo: last below 4% (0.92), 6% 0.96: 0.04 + 0.02 * 0.03/0.04 = 0.055
        "dip": ([0.60, 0.80, 0.96, 0.94, 0.98, 0.99, 1.00, 1.00], [0.58, 0.78, 0.94, 0.92, 0.96, 0.97, 0.98, 0.98],
                None, (0.045, "4.50%"), (0.055, "5.50%"), None, 0.0, False),
        # already: sens >= 0.95 everywhere -> "< 1%"; sens_lo 0.94 at 1%, exactly 0.95 at 2% -> 0.02
        "already": (already, already_lo, None, (math.nan, "< 1%"), (0.02, "2.00%"), None, 0.0, False),
        # never: sens 0.94 at 15% -> "> 15%" for both
        "never": (never, never_lo, None, (math.nan, "> 15%"), (math.nan, "> 15%"), None, 0.0, False),
        # exact: sens exactly 0.95 at 4% (3% is 0.90) -> 0.04; sens_lo exactly 0.95 at 6% (4% is 0.93) -> 0.06
        "exact": (exact, exact_lo, None, (0.04, "4.00%"), (0.06, "6.00%"), None, 0.0, False),
        # flagged: main 5% (position 3.5); check last below 8% (0.93), 10% 0.97: 0.08 + 0.02 * 0.5 = 0.09
        #          (position 5.5); difference 2.0 -> flagged
        "flagged": (clean, clean_lo, [0.30, 0.50, 0.70, 0.80, 0.90, 0.93, 0.97, 1.00],
                    (0.05, "5.00%"), (0.06, "6.00%"), (0.09, "9.00%"), 2.0, True),
        # close: main 5% (3.5); check last below 6% (0.94), 8% 0.98: 0.06 + 0.02 * 0.25 = 0.065 (4.25); 0.75
        "close": (clean, clean_lo, [0.40, 0.60, 0.80, 0.90, 0.94, 0.98, 1.00, 1.00],
                  (0.05, "5.00%"), (0.06, "6.00%"), (0.065, "6.50%"), 0.75, False),
        # boundary: main exactly 4% (3); check exactly 0.95 at 6% (4); difference exactly 1 -> not flagged
        "boundary": (exact, exact_lo, exact_lo, (0.04, "4.00%"), (0.06, "6.00%"), (0.06, "6.00%"), 1.0, False),
        # above grid: main "> 15%" (edge, 7); check last below 10% (0.90), 15% 0.96:
        #             0.10 + 0.05 * 0.05/0.06 = 0.141667 (6 + 5/6); difference 1/6 -> not flagged
        "above": (never, never_lo, [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.90, 0.96],
                  (math.nan, "> 15%"), (math.nan, "> 15%"), (0.10 + 0.05 * 5 / 6, "14.17%"), 1 / 6, False),
        # below grid: main "< 1%" (edge, 0); check last below 1% (0.90), 2% 0.96: 0.01 + 0.01 * 5/6 = 0.018333
        #             (5/6); difference 5/6 -> not flagged
        "below": (already, already_lo, [0.90, 0.96, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00],
                  (math.nan, "< 1%"), (0.02, "2.00%"), (0.01 + 0.01 * 5 / 6, "1.83%"), 5 / 6, False),
    }
    rows = [{"ff": f, "depth": 1_000_000, "noise": "none", "method": name, "sens": s, "sens_lo": lo, "sens_at_fpr": a}
            for name, (sens, sens_lo, at_fpr, *_) in cases.items()
            for f, s, lo, a in zip(grid, sens, sens_lo, sens if at_fpr is None else at_fpr)]
    # incomplete: FF 8% missing -> must be skipped
    rows += [{"ff": f, "depth": 1_000_000, "noise": "none", "method": "incomplete", "sens": 0.99, "sens_lo": 0.98,
              "sens_at_fpr": 0.99} for f in grid if f != 0.08]
    table, skipped = detection_limits(pd.DataFrame(rows), config)  # the made-up methods sort as NaN; order unused

    def same(got, expected):
        return math.isnan(expected) and math.isnan(got) or math.isclose(got, expected, abs_tol=1e-12)

    by = table.set_index("method")
    for name, (_, _, _, point, cons, check, diff, flag) in cases.items():
        expected = (("ff_limit", point), ("ff_limit_cons", cons), ("ff_limit_at_fpr", point if check is None else check))
        for col, (ff, text) in expected:
            got_ff, got_text = by.loc[name, col], by.loc[name, f"{col}_text"]
            ok = got_text == text and same(got_ff, ff)
            print(f"self-test {name:>8} {col:>15}: {got_text:>6} (hand {text}) {'ok' if ok else 'FAIL'}")
            if not ok:
                raise ValueError(f"self-test {name} {col}: got {got_ff!r} {got_text!r}, expected {ff!r} {text!r}")
        got_diff, got_flag = by.loc[name, "at_fpr_pos_diff"], by.loc[name, "at_fpr_flag"]
        ok = same(got_diff, diff) and got_flag == flag
        print(f"self-test {name:>8}  position diff: {got_diff:.6f} flag {got_flag} (hand {diff:.6f}, {flag}) "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            raise ValueError(f"self-test {name}: position diff {got_diff!r} flag {got_flag}, expected {diff!r} {flag}")
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
    for r in table[table["at_fpr_flag"]].itertuples():
        print(f"flagged {r.method}, depth {r.depth:,}, noise {r.noise}: limit {r.ff_limit_text} from sens vs "
              f"{r.ff_limit_at_fpr_text} from sens_at_fpr ({r.at_fpr_pos_diff:.2f} grid steps apart)")
    out = tables / "detection_limits.csv"
    table.to_csv(out, index=False, float_format="%.10g")
    print(f"Wrote {len(table)} rows to {out} ({len(skipped)} groups skipped as incomplete, "
          f"{int(table['at_fpr_flag'].sum())} flagged)")


if __name__ == "__main__":
    main()
