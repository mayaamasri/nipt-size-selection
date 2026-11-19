"""Task 10 (Table S10): the four regimes of size selection under short-fragment noise (decision #18), model only.

For every cutoff c (B at 156 bp and every B_c), FF and k (1, i.e. the main analysis, and each short-noise
scenario), from committed values only (no simulation): g and r of the window [40, c] at that FF from
results/tables/breakeven_check.csv (exact, decision #14), and the two-sigma model of Task 08:
  sigma_A^2 = sigma^2 + sigma_s^2 r_win^2,   sigma_B^2 = sigma^2 + sigma_s^2 rho_c^2,   sigma_s^2 = sigma^2 (k^2 - 1)
r_win = fraction of fragments in the noisy 40-156 bp window (B's r), rho_c = the cutoff's overlap with it
(1 for c <= 156, r_win / r_c above). sigma cancels in every ratio below, so the regime does not depend on the
noise level (sigma > 0) or the depth.

The ratio sep(B_c)/sep(A) moves monotonically from g sqrt(r) (low depth) to g sigma_A / sigma_B (high depth):
  (i)   B always better            g^2 r >= 1 and g sigma_A >= sigma_B
  (ii)  worse at low depth, better at high (classic break-even)   g^2 r < 1 and g sigma_A > sigma_B
  (iii) better at low depth, worse at high                         g^2 r >= 1 and g sigma_A < sigma_B
  (iv)  always worse               g^2 r < 1 and g sigma_A <= sigma_B
k_high_equal: the k at which g sigma_A = sigma_B (where (iii) begins when g^2 r >= 1, or (ii) turns into (iv)):
  k^2 = 1 + (1 - g^2) / (g^2 r_win^2 - rho^2),  real only if g r_win < rho; for B at 156 bp it reduces to
  k^2 = g^2 (1 - r^2) / (1 - g^2 r^2) (self-test).
k_star: the k at which depth stops raising the gain, sigma_B / sigma_A = 1 / sqrt(r):
  k^2 = 1 + (1 - r) / (r rho^2 - r_win^2), real only if r rho^2 > r_win^2; for B it is sqrt((1 + r) / r) and must
  equal snoise_budis_illustration.csv (self-test).
Cutoffs where a k does not exist get NaN (for k_star: depth always raises the gain).

Writes results/tables/snoise_regimes.csv: one row per cutoff x FF x k, with the regime and the number of simulated
grid conditions it covers (4 depths x the noise levels the scenario runs: 3 at k = 1, 2 at k > 1). Run from the
repo root (seconds):
    python -m src.evaluate.snoise_regimes
"""

import math

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names, scenario_noise_levels

REGIMES = {"i": "B always better", "ii": "worse at low depth, better at high",
           "iii": "better at low depth, worse at high", "iv": "always worse"}


def k_values(config):
    """{k: (scenario name or 'main', number of noise levels it runs)}, k = 1 first."""
    out = {1.0: ("main", len(config["grid"]["noise_sigma"]))}
    for name in scenario_names(config, "short_noise"):
        cfg = load_config(scenario=name)
        out[float(cfg["short_noise"]["k"])] = (name, len(scenario_noise_levels(cfg)))
    return out


def sigma_ratio(k, r_win, rho):
    """sigma_A / sigma_B (sigma cancels)."""
    return math.sqrt(1 + (k ** 2 - 1) * r_win ** 2) / math.sqrt(1 + (k ** 2 - 1) * rho ** 2)


def k_high_equal(g, r_win, rho):
    den = g ** 2 * r_win ** 2 - rho ** 2
    if den >= 0:
        return math.nan
    k2 = 1 + (1 - g ** 2) / den
    return math.sqrt(k2) if k2 > 1 else math.nan


def k_star(r, r_win, rho):
    den = r * rho ** 2 - r_win ** 2
    if den <= 0:
        return math.nan
    return math.sqrt(1 + (1 - r) / den)


def regime(g, r, ratio_a_b):
    low_better = g ** 2 * r >= 1
    high = g * ratio_a_b          # g sigma_A / sigma_B
    if low_better:
        return "i" if high >= 1 else "iii"
    return "ii" if high > 1 else "iv"


def regime_table(config):
    tables = repo_path(config["paths"]["results"]) / "tables"
    be = pd.read_csv(tables / "breakeven_check.csv", dtype={"noise": str})
    gr = be.drop_duplicates(["cutoff", "ff"])[["cutoff", "ff", "g", "r"]].sort_values(["cutoff", "ff"])
    b_cut = config["methods"]["size_selection_window_bp"][1]
    r_win = gr[gr["cutoff"] == b_cut].set_index("ff")["r"]
    n_depth = len(config["grid"]["depth"])
    rows = []
    for row in gr.itertuples():
        rw = float(r_win.loc[row.ff])
        rho = 1.0 if row.cutoff <= b_cut else rw / row.r
        k_eq, k_st = k_high_equal(row.g, rw, rho), k_star(row.r, rw, rho)
        for k, (name, n_noise) in k_values(config).items():
            ratio = sigma_ratio(k, rw, rho)
            reg = regime(row.g, row.r, ratio)
            rows.append({"cutoff": int(row.cutoff), "ff": row.ff, "k": k, "scenario": name, "g": row.g, "r": row.r,
                         "r_win": rw, "rho": rho, "g2r": row.g ** 2 * row.r, "g_sqrt_r": row.g * math.sqrt(row.r),
                         "sigma_B_over_A": 1 / ratio, "g_sigmaA_over_sigmaB": row.g * ratio,
                         "regime": reg, "regime_text": REGIMES[reg], "iii_possible": bool(row.g ** 2 * row.r >= 1),
                         "k_high_equal": k_eq, "k_star": k_st, "n_simulated_conditions": n_depth * n_noise})
    return pd.DataFrame(rows)


def self_test(config, table):
    def check(name, worst, tol):
        ok = worst < tol
        print(f"self-test {name}: worst {worst:.2e} (tolerance {tol:g}) {'ok' if ok else 'FAIL'}")
        if not ok:
            raise ValueError(f"self-test {name} failed: {worst!r} >= {tol!r}")
    b_cut = config["methods"]["size_selection_window_bp"][1]
    b = table[(table["cutoff"] == b_cut)].drop_duplicates("ff")
    # 1. general k_high_equal == Maya's closed form for B: k^2 = g^2 (1 - r^2) / (1 - g^2 r^2)
    worst = 0.0
    for r in b.itertuples():
        den = 1 - r.g ** 2 * r.r ** 2
        closed = math.sqrt(r.g ** 2 * (1 - r.r ** 2) / den) if den > 0 and r.g ** 2 * (1 - r.r ** 2) / den > 1 \
            else math.nan
        if math.isnan(closed) != math.isnan(r.k_high_equal):
            raise ValueError(f"self-test 1: FF {r.ff}: general {r.k_high_equal} vs closed {closed}")
        if not math.isnan(closed):
            worst = max(worst, abs(r.k_high_equal / closed - 1))
    check("1 (k where g sigma_A = sigma_B, general == closed form for B at 156 bp)", worst, 1e-12)
    # 2. general k_star for B == sqrt((1 + r) / r) and == snoise_budis_illustration.csv
    budis = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "snoise_budis_illustration.csv")
    budis = budis.drop_duplicates("ff").set_index("ff")
    worst = max(abs(r.k_star / math.sqrt((1 + r.r) / r.r) - 1) for r in b.itertuples())
    check("2a (k* for B == sqrt((1 + r) / r))", worst, 1e-12)
    worst = max(abs(b.set_index("ff").loc[ff, "k_star"] / budis.loc[ff, "k_star"] - 1) for ff in budis.index)
    check("2b (k* for B == snoise_budis_illustration.csv, FF 4/6/10%)", worst, 1e-5)
    # 3. at k = 1 the regime follows g^2 r alone ((i) or (ii)), since sigma_A = sigma_B
    k1 = table[table["k"] == 1]
    bad = k1[k1["regime"] != np.where(k1["g2r"] >= 1, "i", "ii")]
    check("3 (k = 1: regime (i) iff g^2 r >= 1, else (ii))", float(len(bad)), 0.5)


def main():
    config = load_config()
    table = regime_table(config)
    self_test(config, table)
    out = repo_path(config["paths"]["results"]) / "tables" / "snoise_regimes.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    counts = table.groupby(["k", "regime"])["n_simulated_conditions"].sum().unstack(fill_value=0)
    print("\nSimulated grid conditions per regime (cutoff x FF x depth x noise):")
    print(counts.to_string())
    iii = table[(table["regime"] == "iii")]
    print(f"\nRegime (iii) at simulated k (1, 1.5, 2): {len(iii)} cutoff x FF combinations"
          + ("" if iii.empty else ":\n" + iii[["cutoff", "ff", "k", "g2r", "g_sigmaA_over_sigmaB"]].to_string(index=False)))
    print("\nk where g sigma_A = sigma_B, cutoffs where (iii) is possible (g^2 r >= 1), and k*, per FF:")
    show = table[table["k"] == 1].pivot_table(index="cutoff", columns="ff", values=["k_high_equal", "k_star"])
    print(show.round(2).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
