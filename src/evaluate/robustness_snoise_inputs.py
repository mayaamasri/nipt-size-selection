"""Task 08, 8C and 8F: extra noise on size-selected fragments: predictions, noise QC, Budis illustration.

Short-noise scenarios (robustness.short_noise, decision #18): besides eps (whole chr21 row, sigma), each patient's
chr21 cells inside the 40-156 bp window get (1 + eps_s), eps_s ~ N(0, sigma_s), sigma_s = sigma sqrt(k^2 - 1)
(src.simulate.patients). k = 1 is the main analysis. For weights u (Task 06 model, src.evaluate.
unified_model_check) the depth-independent noise becomes sigma_u^2 = sigma^2 + sigma_s^2 rho_u^2,
rho_u = E[u 1_win] / E[u], for every method, A included.

8C, results/tables/robustness_snoise_pred.csv: one row per scenario (main, k = 1, and each short-noise scenario)
x FF x depth x noise (low, realistic) x method (B, W, W_fixedFF): ratio_pred vs A, g_eff, r_eff, rho_u, sigma_u,
sigma_B / sigma_A and rho_W of that condition, ratio_low_depth = g_eff sqrt(r_eff) (N -> 0) and
ratio_high_depth = g_eff sigma_A / sigma_u (N -> infinity), and on W rows the optimum ratio_pred_opt
(numerical u_opt with short noise, closed form at k = 1).

QC, results/tables/robustness_snoise_qc.csv: the study's own reference normals (2,000 per condition) at FF 4%,
1M and 10M, realistic noise, for k = 1, 1.5, 2: SD of the all-fragment and of the window (40-156 bp) chr21
fraction vs p0 sqrt(1/(p0 N r_eff) + sigma_u^2) (r_eff = 1 and sigma_A for all fragments, r and sigma_B for the
window); ratio_se ~ 1/sqrt(2 (n - 1)). Note on the committed run: one of the 12 rows, the k = 1 control (the main
analysis's unchanged reference normals) at 1M for the window, is 4.0% low (2.5 SE); it is treated as chance. All
eight short-noise rows are within 1.8%.

8F, results/tables/snoise_budis_illustration.csv (model only, an illustration, not a fit): sep(B)/sep(A) at
5M and 8.8M, realistic sigma, k = 1.0 ... 3.0 (step 0.1), FF 4%, 6%, 10%, plus per FF: 1/sqrt(r), the closed-form
k* = sqrt((1 + r) / r) above which the gain falls with depth (sigma_B / sigma_A = 1/sqrt(r)), the same k* found
numerically (where the 5M and 8.8M ratios are equal), and the k that gives 1.05 at 5M and 1.13 at 8.8M
(Budis et al.).

No new patients except the QC, which regenerates existing reference groups. The self-test runs first, every
time. Run from the repo root (~1 min, mostly the QC patients at 10M):
    python -m src.evaluate.robustness_snoise_inputs
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.config import load_config, repo_path, scenario_names
from src.evaluate.breakeven_check import predicted_ratio_two_sigma
from src.evaluate.unified_model_check import (best_weights, length_distributions, method_weights, mixture_terms,
                                              model_terms, window_share)
from src.simulate.patients import CHR21, generate_group, noise_sigma, origin_matrices, window_stats

MAIN = "main"
PRED_METHODS = ("B", "W", "W_fixedFF")
QC_FF, QC_DEPTHS, QC_NOISE, QC_GROUP = 0.04, (1_000_000, 10_000_000), "realistic", "reference"
BUDIS = {"depths": (5_000_000, 8_800_000), "targets": (1.05, 1.13), "ffs": (0.04, 0.06, 0.10),
         "k": np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 1), "noise": "realistic"}
PREVIEW_DEPTHS = (1_000_000, 2_500_000, 5_000_000, 10_000_000)


def scenarios(config):
    """{name: (k, window_bp)}: the main analysis (k = 1) and every short-noise scenario."""
    out = {MAIN: (1.0, config["methods"]["size_selection_window_bp"])}
    for name in scenario_names(config, "short_noise"):
        s = load_config(scenario=name)["short_noise"]
        out[name] = (float(s["k"]), s["window_bp"])
    return out


def window_mask(lengths, window_bp):
    return ((lengths >= window_bp[0]) & (lengths <= window_bp[1])).astype(float)


def short_arg(sigma, k, window):
    """The model's short-noise argument: None at k = 1 (exactly Task 06), else (sigma_s, window)."""
    return None if k == 1 else (sigma * math.sqrt(k ** 2 - 1), window)


def b_vs_a(p_f, p_m, ff, p0, depth, sigma, k, window):
    """ratio_pred of B (u = the window) vs A at one setting."""
    return model_terms(window, p_f, p_m, ff, p0, depth, sigma, short_arg(sigma, k, window))["ratio_pred"]


def prediction_table(config, matrices):
    p_f, p_m = length_distributions(matrices)
    p0 = float(matrices["maternal"][CHR21].sum())
    lengths = matrices["lengths"]
    levels = [n for n, s in zip(config["grid"]["noise_sigma"], config["grid"]["noise_sigma_values"]) if s > 0]
    rows = []
    for name, (k, window_bp) in scenarios(config).items():
        window = window_mask(lengths, window_bp)
        for ff in config["grid"]["fetal_fraction"]:
            weights = method_weights(config, matrices, ff)
            p_mix, _ = mixture_terms(ff, p_f, p_m)
            rho_w = window_share(weights["W"], p_mix, window)
            for noise in levels:
                sigma = noise_sigma(config, noise)
                short = short_arg(sigma, k, window)
                for depth in config["grid"]["depth"]:
                    a = model_terms(np.ones_like(p_f), p_f, p_m, ff, p0, depth, sigma, short)
                    b = model_terms(weights["B"], p_f, p_m, ff, p0, depth, sigma, short)
                    u_opt = best_weights(p_f, p_m, ff, p0, depth, sigma, short)
                    opt = model_terms(u_opt, p_f, p_m, ff, p0, depth, sigma, short)["ratio_pred"]
                    for method in PRED_METHODS:
                        t = model_terms(weights[method], p_f, p_m, ff, p0, depth, sigma, short)
                        rows.append({
                            "scenario": name, "k": k, "ff": ff, "depth": depth, "noise": str(noise), "sigma": sigma,
                            "sigma_s": 0.0 if short is None else short[0], "method": method, "p0": p0,
                            "g_eff": t["g_eff"], "r_eff": t["r_eff"],
                            "rho_u": window_share(weights[method], p_mix, window), "sigma_u": t["sigma_u"],
                            "ratio_pred": t["ratio_pred"],
                            "ratio_low_depth": t["g_eff"] * math.sqrt(t["r_eff"]),
                            "ratio_high_depth": t["g_eff"] * a["sigma_u"] / t["sigma_u"],
                            "sigma_B_over_A": b["sigma_u"] / a["sigma_u"], "rho_W": rho_w,
                            "ratio_pred_opt": opt if method == "W" else np.nan})
    return pd.DataFrame(rows)


def qc_table(config, matrices):
    p0 = float(matrices["maternal"][CHR21].sum())
    lengths = matrices["lengths"]
    p_f, p_m = length_distributions(matrices)
    p_mix, _ = mixture_terms(QC_FF, p_f, p_m)
    rows = []
    for name, (k, window_bp) in scenarios(config).items():
        cfg = config if name == MAIN else load_config(scenario=name)
        window = window_mask(lengths, window_bp)
        win = window.astype(bool)
        r = float(p_mix @ window)
        sigma = noise_sigma(cfg, QC_NOISE)
        sigma_s = 0.0 if k == 1 else sigma * math.sqrt(k ** 2 - 1)
        for depth in QC_DEPTHS:
            X = generate_group(cfg, QC_FF, depth, QC_NOISE, QC_GROUP, matrices).astype(float)
            fractions = {"all": (X[:, CHR21].sum(axis=1) / X.sum(axis=(1, 2)), 1.0, r),
                         "window": (X[:, CHR21][:, win].sum(axis=1) / X[:, :, win].sum(axis=(1, 2)), r, 1.0)}
            for which, (f, r_eff, rho) in fractions.items():
                sigma_u = math.sqrt(sigma ** 2 + (sigma_s * rho) ** 2)
                theory = p0 * math.sqrt(1 / (p0 * depth * r_eff) + sigma_u ** 2)
                sd = float(f.std(ddof=1))
                rows.append({"scenario": name, "k": k, "ff": QC_FF, "depth": depth, "noise": QC_NOISE, "sigma": sigma,
                             "sigma_s": sigma_s, "fraction": which, "n": len(f), "r_eff": r_eff, "rho": rho,
                             "sigma_u": sigma_u, "sd_obs": sd, "sd_theory": theory, "ratio": sd / theory,
                             "ratio_se": 1 / math.sqrt(2 * (len(f) - 1))})
    return pd.DataFrame(rows)


def k_star_numeric(p_f, p_m, ff, p0, sigma, window):
    """k where the B/A ratio is the same at the two Budis depths (the gain stops rising with depth)."""
    lo, hi = BUDIS["depths"]
    f = lambda k: (b_vs_a(p_f, p_m, ff, p0, hi, sigma, k, window)  # noqa: E731
                   - b_vs_a(p_f, p_m, ff, p0, lo, sigma, k, window))
    return brentq(f, 1.0001, 10.0, xtol=1e-12)


def budis_table(config, matrices):
    p_f, p_m = length_distributions(matrices)
    p0 = float(matrices["maternal"][CHR21].sum())
    window = window_mask(matrices["lengths"], config["methods"]["size_selection_window_bp"])
    sigma = noise_sigma(config, BUDIS["noise"])
    (d5, d88), (t5, t88) = BUDIS["depths"], BUDIS["targets"]
    rows = []
    for ff in BUDIS["ffs"]:
        p_mix, _ = mixture_terms(ff, p_f, p_m)
        r = float(p_mix @ window)
        per_ff = {
            "inv_sqrt_r": 1 / math.sqrt(r), "k_star": math.sqrt((1 + r) / r),
            "k_star_numeric": k_star_numeric(p_f, p_m, ff, p0, sigma, window),
            "k_for_105_at_5M": brentq(lambda k: b_vs_a(p_f, p_m, ff, p0, d5, sigma, k, window) - t5, 1.0001, 10),
            "k_for_113_at_8_8M": brentq(lambda k: b_vs_a(p_f, p_m, ff, p0, d88, sigma, k, window) - t88, 1.0001, 10)}
        for k in BUDIS["k"]:
            r5, r88 = (b_vs_a(p_f, p_m, ff, p0, d, sigma, float(k), window) for d in (d5, d88))
            rows.append({"ff": ff, "k": float(k), "sigma": sigma, "r": r, "ratio_5M": r5, "ratio_8_8M": r88,
                         "change_5M_to_8_8M": r88 / r5 - 1, **per_ff})
    return pd.DataFrame(rows)


def self_test(config, matrices, pred, budis):
    def check(name, worst, tol):
        ok = worst < tol
        print(f"self-test {name}: worst {worst:.2e} (tolerance {tol:g}) {'ok' if ok else 'FAIL'}")
        if not ok:
            raise ValueError(f"self-test {name} failed: worst {worst!r} >= {tol!r}")

    # 1. k = 1 reproduces the main unified_model_check.csv (B, W, W_fixedFF, low and realistic noise)
    saved = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "unified_model_check.csv",
                        dtype={"noise": str}).set_index(["ff", "depth", "noise", "method"])["ratio_pred"]
    main = pred[pred["scenario"] == MAIN]
    worst = max(abs(r.ratio_pred / saved.loc[(r.ff, r.depth, r.noise, r.method)] - 1) for r in main.itertuples())
    check(f"1 (k = 1 == unified_model_check.csv, {len(main)} rows)", worst, 1e-9)
    # 2. sigma_B / sigma_A = k / sqrt(1 + (k^2 - 1) r^2) (rho_B = 1, rho_A = r)
    b = pred[pred["method"] == "B"]
    p_f, p_m = length_distributions(matrices)
    window = window_mask(matrices["lengths"], config["methods"]["size_selection_window_bp"])
    r_by_ff = {ff: float(mixture_terms(ff, p_f, p_m)[0] @ window) for ff in b["ff"].unique()}
    expected = b["k"] / np.sqrt(1 + (b["k"] ** 2 - 1) * b["ff"].map(r_by_ff) ** 2)
    check("2 (sigma_B / sigma_A == k / sqrt(1 + (k^2 - 1) r^2))", float((b["sigma_B_over_A"] / expected - 1).abs().max()),
          1e-12)
    # 3. B's model ratio == breakeven_check's two-sigma formula with the exact g and r of the window
    worst = 0.0
    lo, hi = config["methods"]["size_selection_window_bp"]
    for row in b.itertuples():
        w = window_stats(matrices, row.ff, lo, hi)
        a_sigma = row.sigma_u / row.sigma_B_over_A
        ref = predicted_ratio_two_sigma(w["gain"], w["r"], row.p0, row.depth, a_sigma, row.sigma_u)
        worst = max(worst, abs(row.ratio_pred / ref - 1))
    check("3 (B ratio_pred == predicted_ratio_two_sigma with exact g, r)", worst, 1e-12)
    # 4. closed-form k* == the numerical crossing (equal ratios at 5M and 8.8M)
    per_ff = budis.drop_duplicates("ff")
    check("4 (k* closed form vs numerical crossing, every FF)",
          float((per_ff["k_star"] - per_ff["k_star_numeric"]).abs().max()), 1e-6)


def main():
    config = load_config()
    matrices = origin_matrices(config)
    pred = prediction_table(config, matrices)
    budis = budis_table(config, matrices)
    self_test(config, matrices, pred, budis)
    qc = qc_table(config, matrices)

    tables = repo_path(config["paths"]["results"]) / "tables"
    pred.to_csv(tables / "robustness_snoise_pred.csv", index=False, float_format="%.10g")
    qc.to_csv(tables / "robustness_snoise_qc.csv", index=False, float_format="%.8g")
    budis.to_csv(tables / "snoise_budis_illustration.csv", index=False, float_format="%.6g")

    p = pred[np.isclose(pred["ff"], 0.04) & (pred["noise"] == "realistic")]
    print("\n8C ratio_pred vs A, FF 4%, realistic noise:")
    print(f"{'scenario':>11} {'k':>4} {'B: ' + ' / '.join(f'{d / 1e6:g}M' for d in PREVIEW_DEPTHS):>26}"
          f" {'W: ' + ' / '.join(f'{d / 1e6:g}M' for d in PREVIEW_DEPTHS):>26} {'sB/sA':>6} {'rho_W':>6}")
    for name, g in p.groupby("scenario", sort=False):
        by = g.set_index(["method", "depth"])["ratio_pred"]
        cells = [" / ".join(f"{by.loc[(m, d)]:.2f}" for d in PREVIEW_DEPTHS) for m in ("B", "W")]
        print(f"{name:>11} {g['k'].iloc[0]:4g} {cells[0]:>26} {cells[1]:>26} {g['sigma_B_over_A'].iloc[0]:6.3f} "
              f"{g['rho_W'].iloc[0]:6.3f}")
    print("\nQC (reference normals, FF 4%, realistic): SD observed / theory")
    print(qc[["scenario", "k", "depth", "fraction", "sigma_u", "sd_obs", "sd_theory", "ratio", "ratio_se"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("\n8F per FF (realistic sigma):")
    print(budis.drop_duplicates("ff")[["ff", "r", "inv_sqrt_r", "k_star", "k_star_numeric", "k_for_105_at_5M",
                                       "k_for_113_at_8_8M"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved robustness_snoise_pred.csv, robustness_snoise_qc.csv and snoise_budis_illustration.csv in {tables}")


if __name__ == "__main__":
    main()
