"""Task 06: one counting model for all z-score methods (extends the break-even formula, decision #8).

Every z-score method in the study is a weighted chr21 fraction: a fragment of length l gets weight u(l) on both
rows (chr21 and other autosomes), F_u = sum_l u(l) X21(l) / sum_l u(l) [X21(l) + Xother(l)].
  A          u = 1
  B, B_c     u = 1 inside [40, c], 0 outside
  W          u = w*(l) = FF p_f(l) / p_mix(l), the true P(fetal | l)
  W_fixedFF  u = the same with FF = weight_fixed_ff (0.10) in the weight

With p_f, p_m the post-mappability length distributions (normal fetal and maternal matrices from origin_matrices,
summed over the two rows), p_mix = FF p_f + (1 - FF) p_m and E[x] = sum_l p_mix(l) x(l), to first order in FF
(same assumptions as decision #8):
  sep_u      = (E[u w*] / (2 E[u])) / sqrt(E[u^2] / (p0 N E[u]^2) + sigma^2)
  ratio_pred = sep_u / sep_A        (sep_A: u = 1)
This is decision #8's formula with g_eff = E[u w*] / (FF E[u]) and r_eff = E[u]^2 / E[u^2]; for a window
g_eff = g and r_eff = r.

Analytic optimum (6C, NOT a detection method): with depth-independent noise sigma the weights that maximise
sep_u are u_opt = w* - c, c = FF k / (1 + k), k = sigma^2 p0 N (Sherman-Morrison); u_opt = w* when sigma = 0.
On the W rows: ratio_pred_opt = ratio_pred(u_opt) and w_efficiency = ratio_pred / ratio_pred_opt.

For every condition and methods B, B_120 ... B_165, W, W_fixedFF (B-opt is left out: its cutoff is chosen from
the data; C1 and C2 are not weighted fractions), from results/tables/results_all.csv:
  sep_obs    (z_mean_t21 - z_mean_normal) / z_sd_normal;  ratio_obs = sep_obs / sep_A (A's sep_obs)
  rel_error  (ratio_obs - ratio_pred) / ratio_pred;  abs_error = ratio_obs - ratio_pred
  low_signal sep_A < 1 (same rule as breakeven_check.py)

No simulation: only results_all.csv and origin_matrices. Writes results/tables/unified_model_check.csv and
unified_model_summary.csv (breakeven_check's summary per method family: B = B and every B_c, W, W_fixedFF).
breakeven_check.csv is only read (self-test 2). The self-test runs first, every time. Run from the repo root
(seconds):
    python -m src.evaluate.unified_model_check
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.breakeven_check import CONDITION, fixed_cutoff_methods, predicted_ratio, summarise
from src.methods.reference import posterior_weights
from src.simulate.patients import origin_matrices

COLUMNS = CONDITION + ["sigma", "method", "p0", "g_eff", "r_eff", "sep_A", "sep_obs", "ratio_obs", "ratio_pred",
                       "rel_error", "abs_error", "low_signal", "ratio_pred_opt", "w_efficiency"]
FAMILIES = {"B": None, "W": ["W"], "W_fixedFF": ["W_fixedFF"]}   # None: B and every B_c


def length_distributions(matrices):
    """p_f, p_m: post-mappability length distributions (normal fetal matrix), summed over the two rows."""
    return matrices["fetal"].sum(axis=0), matrices["maternal"].sum(axis=0)


def mixture_terms(ff, p_f, p_m):
    """p_mix and the true w* = P(fetal | l) for fetal fraction ff."""
    p_mix = ff * p_f + (1 - ff) * p_m
    if not (p_mix > 0).all():
        raise ValueError("p_mix has non-positive lengths; w* is undefined there")
    return p_mix, ff * p_f / p_mix


def separation(u, p_f, p_m, ff, p0, depth, sigma):
    """sep_u from the 6A formula. p_f, p_m are the TRUE length distributions of the patients (for a mismatch
    run, pass the changed profile here and the mismatched weights as u)."""
    p_mix, w = mixture_terms(ff, p_f, p_m)
    e_u, e_uw, e_uu = p_mix @ u, p_mix @ (u * w), p_mix @ (u * u)
    return (e_uw / (2 * e_u)) / np.sqrt(e_uu / (p0 * depth * e_u ** 2) + sigma ** 2)


def model_terms(u, p_f, p_m, ff, p0, depth, sigma):
    """g_eff, r_eff and ratio_pred = sep_u / sep_A for weights u (array over lengths)."""
    u = np.asarray(u, float)
    p_mix, w = mixture_terms(ff, p_f, p_m)
    e_u = p_mix @ u
    ratio = (separation(u, p_f, p_m, ff, p0, depth, sigma)
             / separation(np.ones_like(u), p_f, p_m, ff, p0, depth, sigma))
    return {"g_eff": (p_mix @ (u * w)) / (ff * e_u), "r_eff": e_u ** 2 / (p_mix @ (u * u)), "ratio_pred": ratio}


def optimal_weights(w, ff, p0, depth, sigma):
    """u_opt = w* - FF k / (1 + k), k = sigma^2 p0 N: the best linear weights (analytic only, 6C)."""
    k = sigma ** 2 * p0 * depth
    return w - ff * k / (1 + k)


def method_weights(config, matrices, ff):
    """method -> u(l) for B, every B_c, W (true FF) and W_fixedFF, in output order."""
    lengths = matrices["lengths"]
    lo = config["methods"]["size_selection_window_bp"][0]
    out = {}
    for method in fixed_cutoff_methods(config):
        hi = config["methods"]["size_selection_window_bp"][1] if method == "B" else int(method.split("_")[1])
        out[method] = ((lengths >= lo) & (lengths <= hi)).astype(float)
    out["W"] = posterior_weights(ff, matrices)
    out["W_fixedFF"] = posterior_weights(config["methods"]["weight_fixed_ff"], matrices)
    return out


def unified_table(results, config, matrices):
    p_f, p_m = length_distributions(matrices)
    sep = results.assign(sep=(results["z_mean_t21"] - results["z_mean_normal"]) / results["z_sd_normal"])
    sep_a = sep[sep["method"] == "A"].set_index(CONDITION)["sep"]
    methods = list(method_weights(config, matrices, 0.1))
    rows = []
    for (ff, depth, noise), cond in sep[sep["method"].isin(methods)].groupby(CONDITION, sort=False):
        sigma, p0 = float(cond["sigma"].iloc[0]), float(cond["p0"].iloc[0])
        weights = method_weights(config, matrices, ff)
        s_a = sep_a.loc[(ff, depth, noise)]
        for r in cond.itertuples():
            terms = model_terms(weights[r.method], p_f, p_m, ff, p0, depth, sigma)
            row = {"ff": ff, "depth": depth, "noise": noise, "sigma": sigma, "method": r.method, "p0": p0,
                   "g_eff": terms["g_eff"], "r_eff": terms["r_eff"], "sep_A": s_a, "sep_obs": r.sep,
                   "ratio_obs": r.sep / s_a, "ratio_pred": terms["ratio_pred"],
                   "ratio_pred_opt": np.nan, "w_efficiency": np.nan}
            if r.method == "W":
                u_opt = optimal_weights(weights["W"], ff, p0, depth, sigma)
                row["ratio_pred_opt"] = model_terms(u_opt, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
                row["w_efficiency"] = row["ratio_pred"] / row["ratio_pred_opt"]
            rows.append(row)
    table = pd.DataFrame(rows)
    table["rel_error"] = (table["ratio_obs"] - table["ratio_pred"]) / table["ratio_pred"]
    table["abs_error"] = table["ratio_obs"] - table["ratio_pred"]
    table["low_signal"] = table["sep_A"] < 1   # same rule as breakeven_check.breakeven_table (sep(A) < 1)
    noise_order = {n: i for i, n in enumerate(config["grid"]["noise_sigma"])}
    method_order = {m: i for i, m in enumerate(methods)}
    order = {"noise": noise_order, "method": method_order}
    table = table.sort_values(["noise", "method", "ff", "depth"],
                              key=lambda s: s.map(order[s.name]) if s.name in order else s)
    return table[COLUMNS].reset_index(drop=True)


def family_summary(table):
    """breakeven_check.summarise per method family, with a family column in front."""
    parts = []
    for family, methods in FAMILIES.items():
        d = table[~table["method"].isin(["W", "W_fixedFF"])] if methods is None else table[table["method"].isin(methods)]
        parts.append(summarise(d).assign(family=family))
    out = pd.concat(parts, ignore_index=True)
    return out[["family"] + [c for c in out.columns if c != "family"]]


def self_test(config, matrices, results, breakeven):
    """Identities of the model on the real length distributions, in every grid condition."""
    p_f, p_m = length_distributions(matrices)
    conditions = results[results["method"] == "A"][CONDITION + ["sigma", "p0"]].itertuples(index=False)
    conditions = [(float(c.ff), int(c.depth), c.noise, float(c.sigma), float(c.p0)) for c in conditions]

    def check(name, worst, tol):
        ok = worst < tol
        print(f"self-test {name}: worst {worst:.2e} (tolerance {tol:g}) {'ok' if ok else 'FAIL'}")
        if not ok:
            raise ValueError(f"self-test {name} failed: worst {worst!r} >= {tol!r}")

    # 1. u = 1 -> ratio_pred = g_eff = r_eff = 1
    worst = 0.0
    for ff, depth, _, sigma, p0 in conditions:
        t = model_terms(np.ones_like(p_f), p_f, p_m, ff, p0, depth, sigma)
        worst = max(worst, *(abs(t[k] - 1) for k in ("ratio_pred", "g_eff", "r_eff")))
    check("1 (u = 1: ratio_pred, g_eff, r_eff = 1, 96 conditions)", worst, 1e-12)

    # 2. windows reproduce breakeven_check.csv (g, r and Eq. 4's ratio_pred): tests 6A == Eq. 4 with g_eff, r_eff
    worst = 0.0
    keyed = breakeven.set_index(CONDITION + ["method"])
    for ff, depth, noise, sigma, p0 in conditions:
        for method, u in method_weights(config, matrices, ff).items():
            if method in ("W", "W_fixedFF"):
                continue
            ref = keyed.loc[(ff, depth, noise, method)]
            t = model_terms(u, p_f, p_m, ff, p0, depth, sigma)
            worst = max(worst, *(abs(t[a] / ref[b] - 1) for a, b in
                                 (("g_eff", "g"), ("r_eff", "r"), ("ratio_pred", "ratio_pred"))))
    check(f"2 (B and B_c vs breakeven_check.csv: g, r, ratio_pred, {len(breakeven)} rows)", worst, 1e-9)

    # 3. scaling the weights by 7 does not change ratio_pred
    worst = 0.0
    for ff, depth, _, sigma, p0 in conditions:
        for u in method_weights(config, matrices, ff).values():
            a = model_terms(u, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
            b = model_terms(7 * u, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
            worst = max(worst, abs(b / a - 1))
    check("3 (u -> 7u leaves ratio_pred unchanged, all methods)", worst, 1e-12)

    # 4. E[w*] = FF; w* is also exactly the weight method W used (posterior_weights)
    worst = 0.0
    for ff in config["grid"]["fetal_fraction"]:
        p_mix, w = mixture_terms(ff, p_f, p_m)
        worst = max(worst, abs(p_mix @ w - ff), np.abs(w - posterior_weights(ff, matrices)).max())
    check("4 (E[w*] = FF and w* = posterior_weights, every FF)", worst, 1e-12)

    # 5. sigma = 0 -> u_opt = w*, w_efficiency = 1 (every FF x depth)
    worst = 0.0
    for ff, depth, _, _, p0 in conditions:
        _, w = mixture_terms(ff, p_f, p_m)
        u_opt = optimal_weights(w, ff, p0, depth, 0.0)
        eff = (model_terms(w, p_f, p_m, ff, p0, depth, 0.0)["ratio_pred"]
               / model_terms(u_opt, p_f, p_m, ff, p0, depth, 0.0)["ratio_pred"])
        worst = max(worst, np.abs(u_opt - w).max(), abs(eff - 1))
    check("5 (sigma = 0: u_opt = w*, w_efficiency = 1, 96 conditions)", worst, 1e-9)

    # 5b. sigma > 0: the closed form equals the brute-force optimum, and no random change of u_opt beats it.
    # sep_u^2 = (u'Dw)^2 / (4 u'Mu), D = diag(p_mix), M = D / (p0 N) + sigma^2 p_mix p_mix'; maximised by
    # u ∝ M^-1 D w*. Solved here with np.linalg.solve (no Sherman-Morrison); compared after scaling.
    rng = np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=(6, 0)))   # test-only stream
    worst_u, worst_ratio, worst_gain = 0.0, 0.0, -np.inf
    n_tested = 0
    for ff, depth, _, sigma, p0 in conditions:
        if sigma == 0:
            continue
        n_tested += 1
        p_mix, w = mixture_terms(ff, p_f, p_m)
        u_opt = optimal_weights(w, ff, p0, depth, sigma)
        M = np.diag(p_mix) / (p0 * depth) + sigma ** 2 * np.outer(p_mix, p_mix)
        u_num = np.linalg.solve(M, p_mix * w)
        scale = (u_num @ u_opt) / (u_opt @ u_opt)
        worst_u = max(worst_u, np.abs(u_num / scale - u_opt).max() / np.abs(u_opt).max())
        best = model_terms(u_opt, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
        worst_ratio = max(worst_ratio, abs(model_terms(u_num, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"] / best - 1))
        for _ in range(20):
            delta = rng.normal(size=u_opt.size) * 0.01 * np.abs(u_opt).max()
            worst_gain = max(worst_gain, model_terms(u_opt + delta, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
                             / best - 1)
    check(f"5b (sigma > 0: closed-form u_opt vs np.linalg.solve, {n_tested} conditions, max rel diff)", worst_u, 1e-9)
    check("5b (ratio_pred of the solved optimum vs closed form)", worst_ratio, 1e-12)
    print(f"self-test 5b (20 random 1% changes of u_opt per condition): best relative gain {worst_gain:+.2e} "
          f"{'ok' if worst_gain <= 1e-12 else 'FAIL'}")
    if worst_gain > 1e-12:
        raise ValueError(f"self-test 5b failed: a random change of u_opt raised ratio_pred by {worst_gain!r}")


def main():
    config = load_config()
    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv", dtype={"noise": str})
    breakeven = pd.read_csv(tables / "breakeven_check.csv", dtype={"noise": str})
    matrices = origin_matrices(config)
    self_test(config, matrices, results, breakeven)

    table = unified_table(results, config, matrices)
    summary = family_summary(table)
    table.to_csv(tables / "unified_model_check.csv", index=False, float_format="%.10g")
    summary.to_csv(tables / "unified_model_summary.csv", index=False, float_format="%.4g")
    print(f"\n{len(table)} rows, {int(table['low_signal'].sum())} low_signal")

    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    print("\n1. Summary per family, overall:")
    print(summary[summary["group"] == "overall"].to_string(**fmt))
    w = table[table["method"] == "W"]
    cols = CONDITION + ["ratio_obs", "ratio_pred", "rel_error", "ratio_pred_opt", "w_efficiency"]
    print("\n2. W rows at FF 4%:")
    print(w[np.isclose(w["ff"], 0.04)][cols].to_string(**fmt))
    clean = w[~w["low_signal"]]
    print("\n3. W rows with the largest |rel_error| outside low_signal:")
    print(clean.loc[clean["rel_error"].abs().nlargest(3).index, CONDITION + ["sep_A", "ratio_obs", "ratio_pred",
                                                                            "rel_error"]].to_string(**fmt))
    w_median = w["rel_error"].abs().median()
    if w_median > 0.02:
        print(f"\nWARNING: W's median |rel_error| is {w_median:.2%}, above 2%")
    print(f"\nSaved {tables / 'unified_model_check.csv'} and {tables / 'unified_model_summary.csv'}")


if __name__ == "__main__":
    main()
