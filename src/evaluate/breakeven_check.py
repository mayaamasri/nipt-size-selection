"""Task 5C: does the break-even formula (decision #8) predict the simulation?

For every FF x depth x noise x fixed cutoff (B_c at 120, 130, 135, 140, 145, 150, 165 bp and B itself at
156 bp; B-opt is left out because its cutoff is chosen from the data), from results/tables/results_all.csv:

  sep          (z_mean_t21 - z_mean_normal) / z_sd_normal, the standardised separation of a method
  ratio_obs    sep(B_c) / sep(A)
  ratio_pred   g * sqrt((1/(p0 N) + sigma^2) / (1/(p0 r N) + sigma^2)), with the exact g and r of that cutoff
               at that FF (decision #14, stored in results_all.csv) and p0 = q(chr21); N = depth
  rel_error    (ratio_obs - ratio_pred) / ratio_pred;  abs_error = ratio_obs - ratio_pred
  low_signal   sep(A) < 1: the ratio is unstable there
  breakeven_depth   N* = (1/r - g^2) / (p0 sigma^2 (g^2 - 1)) where g^2 r < 1 and sigma > 0, else empty

Short-noise scenarios (Task 08, decision #18): the chr21 cells of the window_bp window (40-156 bp) get an extra
relative noise sigma_s = sigma sqrt(k^2 - 1), so A and B no longer share one sigma:
  sigma_A^2 = sigma^2 + sigma_s^2 r_win^2,   sigma_B^2 = sigma^2 + sigma_s^2 rho_c^2,   rho_c = r(window and
  [40, c]) / r_c  (1 for every cutoff inside the window, < 1 for 165 bp; r from window_stats, exact)
  ratio_pred = g sqrt((1/(p0 N) + sigma_A^2) / (1/(p0 r N) + sigma_B^2))
B beats A iff a > b / N, with a = g^2 sigma_A^2 - sigma_B^2 and b = (1/r - g^2) / p0, so N* = b / a and
breakeven_direction is "above" (B wins above N*), "below" (B wins below N*), "always" or "never". These columns
(sigma_A, sigma_B, rho_B, breakeven_direction) are added only in short-noise scenarios; without short noise the
formulas above reduce to #8's and the output is unchanged (#8's code is used as is).

The formula is used exactly as written in #8; the exact binomial and renormalisation factors ((1 - p0) on
both noise terms, see src.simulate.noise_qc) change the ratio by ~0.01% and are left out.

Writes results/tables/breakeven_check.csv and breakeven_summary.csv: median, 95th percentile and maximum of
|rel_error| and |abs_error|, and the mean signed rel_error with an SE clustered by condition (t = mean / SE);
overall and per noise level (all rows and without low_signal), per FF and per depth (without low_signal). Run from the
repo root:
    python -m src.evaluate.breakeven_check
    python -m src.evaluate.breakeven_check --scenario shift_p10   (Task 07: results/robustness/shift_p10)
    python -m src.evaluate.breakeven_check --scenario snoise_k2   (Task 08)
"""

import argparse

import numpy as np
import pandas as pd

from src.config import add_scenario_argument, load_config, repo_path, short_noise
from src.simulate.patients import origin_matrices, short_noise_params, window_stats

CONDITION = ["ff", "depth", "noise"]


def predicted_ratio(g, r, p0, depth, sigma):
    return g * np.sqrt((1 / (p0 * depth) + sigma ** 2) / (1 / (p0 * r * depth) + sigma ** 2))


def breakeven_depth(g, r, p0, sigma):
    """N* where B and A have equal separation; NaN if B is never worse (g^2 r >= 1) or there is no noise."""
    g, r, sigma = np.asarray(g, float), np.asarray(r, float), np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = (1 / r - g ** 2) / (p0 * sigma ** 2 * (g ** 2 - 1))
    return np.where((g ** 2 * r < 1) & (sigma > 0), n, np.nan)


def fixed_cutoff_methods(config):
    return ["B"] + [f"B_{c}" for c in config["methods"]["size_fixed_upper_extra"]]


def predicted_ratio_two_sigma(g, r, p0, depth, sigma_a, sigma_b):
    """Decision #8's ratio with A and B having their own depth-independent noise (short-noise scenarios)."""
    return g * np.sqrt((1 / (p0 * depth) + sigma_a ** 2) / (1 / (p0 * r * depth) + sigma_b ** 2))


def breakeven_two_sigma(g, r, p0, sigma_a, sigma_b):
    """(N*, direction) with separate sigma_A, sigma_B: B beats A iff a > b / N, a = g^2 sigma_A^2 - sigma_B^2,
    b = (1/r - g^2) / p0. direction: "above" / "below" (B wins above / below N*), "always", "never"."""
    g, r, sigma_a, sigma_b = (np.asarray(v, float) for v in (g, r, sigma_a, sigma_b))
    a = g ** 2 * sigma_a ** 2 - sigma_b ** 2
    b = (1 / r - g ** 2) / p0
    direction = np.where(b > 0, np.where(a > 0, "above", "never"), np.where(a >= 0, "always", "below"))
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(np.isin(direction, ["above", "below"]), b / a, np.nan)
    return n, direction


def short_noise_sigmas(table, config):
    """sigma_A, sigma_B and rho_B per row of a short-noise scenario (see the module docstring)."""
    matrices = origin_matrices(config)
    w_lo, w_hi = short_noise(config)["window_bp"]
    lo = config["methods"]["size_selection_window_bp"][0]
    out = {"sigma_A": [], "sigma_B": [], "rho_B": []}
    for row in table.itertuples():
        sigma_s, _ = short_noise_params(config, row.noise, matrices["lengths"])
        r_win = window_stats(matrices, row.ff, w_lo, w_hi)["r"]
        r_c = window_stats(matrices, row.ff, lo, row.cutoff)["r"]
        lo_x, hi_x = max(lo, w_lo), min(row.cutoff, w_hi)
        rho = (window_stats(matrices, row.ff, lo_x, hi_x)["r"] if hi_x >= lo_x else 0.0) / r_c
        out["sigma_A"].append(np.sqrt(row.sigma ** 2 + (sigma_s * r_win) ** 2))
        out["sigma_B"].append(np.sqrt(row.sigma ** 2 + (sigma_s * rho) ** 2))
        out["rho_B"].append(rho)
    return pd.DataFrame(out, index=table.index)


def breakeven_table(results, config):
    sep = results.assign(sep=(results["z_mean_t21"] - results["z_mean_normal"]) / results["z_sd_normal"])
    a = sep[sep["method"] == "A"].set_index(CONDITION)["sep"].rename("sep_A")
    b = sep[sep["method"].isin(fixed_cutoff_methods(config))].join(a, on=CONDITION)
    table = pd.DataFrame({
        **{c: b[c] for c in CONDITION + ["sigma", "cutoff", "method", "p0", "g", "r"]},
        "g2r": b["g"] ** 2 * b["r"],
        "sep_A": b["sep_A"], "sep_B": b["sep"],
        "ratio_obs": b["sep"] / b["sep_A"],
        "ratio_pred": predicted_ratio(b["g"], b["r"], b["p0"], b["depth"], b["sigma"]),
    })
    table["rel_error"] = (table["ratio_obs"] - table["ratio_pred"]) / table["ratio_pred"]
    table["abs_error"] = table["ratio_obs"] - table["ratio_pred"]
    table["low_signal"] = table["sep_A"] < 1
    table["breakeven_depth"] = breakeven_depth(table["g"], table["r"], table["p0"], table["sigma"])
    if short_noise(config) is not None:   # Task 08: A and B have their own sigma; recompute the model columns
        sig = short_noise_sigmas(table, config)
        table = pd.concat([table, sig], axis=1)
        table["ratio_pred"] = predicted_ratio_two_sigma(table["g"], table["r"], table["p0"], table["depth"],
                                                        table["sigma_A"], table["sigma_B"])
        table["rel_error"] = (table["ratio_obs"] - table["ratio_pred"]) / table["ratio_pred"]
        table["abs_error"] = table["ratio_obs"] - table["ratio_pred"]
        table["breakeven_depth"], table["breakeven_direction"] = breakeven_two_sigma(
            table["g"], table["r"], table["p0"], table["sigma_A"], table["sigma_B"])
    return table.sort_values(["noise", "cutoff", "ff", "depth"], key=lambda s: s.map(
        {n: i for i, n in enumerate(config["grid"]["noise_sigma"])}) if s.name == "noise" else s).reset_index(drop=True)


def error_summary(d):
    """|error| quantiles plus the mean signed relative error with an SE clustered by condition: the cutoffs of
    one condition share sep(A), so their errors are not independent."""
    rel, ab = d["rel_error"].abs(), d["abs_error"].abs()
    per_condition = d.groupby(CONDITION)["rel_error"].mean()
    n_cond = len(per_condition)
    se = per_condition.std(ddof=1) / np.sqrt(n_cond) if n_cond > 1 else np.nan
    mean = d["rel_error"].mean()
    return {"n": len(d), "n_conditions": n_cond, "rel_median": rel.median(), "rel_p95": rel.quantile(0.95),
            "rel_max": rel.max(), "abs_median": ab.median(), "abs_p95": ab.quantile(0.95), "abs_max": ab.max(),
            "rel_mean_signed": mean, "rel_mean_se": se, "rel_mean_t": mean / se if se else np.nan}


def summarise(table):
    """Rows: overall and per noise level (all rows and without low_signal); per FF and per depth (without
    low_signal, all cutoffs)."""
    rows = []
    for noise in [None] + list(table["noise"].unique()):
        part = table if noise is None else table[table["noise"] == noise]
        for subset, d in (("all", part), ("without_low_signal", part[~part["low_signal"]])):
            rows.append({"group": "overall" if noise is None else "noise", "value": "all" if noise is None else noise,
                         "rows": subset, **error_summary(d)})
    clean = table[~table["low_signal"]]
    for group in ("ff", "depth"):
        for value, d in clean.groupby(group):
            rows.append({"group": group, "value": value, "rows": "without_low_signal", **error_summary(d)})
    return pd.DataFrame(rows)


def main():
    parser = add_scenario_argument(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    config = load_config(scenario=parser.parse_args().scenario)
    tables = repo_path(config["paths"]["results"]) / "tables"
    table = breakeven_table(pd.read_csv(tables / "results_all.csv"), config)
    summary = summarise(table)
    table.to_csv(tables / "breakeven_check.csv", index=False, float_format="%.10g")
    summary.to_csv(tables / "breakeven_summary.csv", index=False, float_format="%.4g")
    print(f"{len(table)} rows, {int(table['low_signal'].sum())} low_signal")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved {tables / 'breakeven_check.csv'} and {tables / 'breakeven_summary.csv'}")


if __name__ == "__main__":
    main()
