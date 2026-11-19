"""Task 07, 7A and 7C: fetal-profile scenarios, their inputs and the model's predictions (before any grid).

Profiles: G1 (the main analysis) and every scenario in robustness.fetal_profiles (decision #17), each a shift
and/or a blend of the fetal length distribution (src.simulate.lengths.load_length_distributions; maternal
unchanged, mappability unchanged, so no remapping).

7A, results/tables/robustness_profiles_inputs.csv, one row per profile:
  fetal and maternal median length, and for the 40-156 bp window (B) at FF 2%, 4% and 10%: g, r, g*sqrt(r)
  and g^2*r, exact from window_stats (decision #14).
7C, results/tables/robustness_profiles_pred.csv, one row per profile x FF x depth x noise x method:
  ratio_pred vs A from the Task 06 model (src.evaluate.unified_model_check) for B, W, W_G1 and W_fixedFF.
  E[.] and w* always use the profile's own distributions (the patients' true profile); u is the method's
  weights: W = the profile's own w*, W_G1 = G1 weights at the true FF, W_fixedFF = G1 weights at FF 10%.
  ratio_pred_opt: the same model with the optimal weights u_opt (Task 06C, analytic only), and
  efficiency = ratio_pred / ratio_pred_opt. In G1, W_G1 = W.

No simulation. The self-test runs first, every time. Run from the repo root (seconds):
    python -m src.evaluate.robustness_inputs
"""

import copy

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names
from src.evaluate.unified_model_check import (length_distributions, method_weights, mixture_terms, model_terms,
                                              optimal_weights)
from src.simulate.lengths import distribution_median, load_length_distributions
from src.simulate.patients import CHR21, noise_sigma, origin_matrices, window_stats

MAIN = "G1"
INPUT_FF = (0.02, 0.04, 0.10)
PRED_METHODS = ("B", "W", "W_G1", "W_fixedFF")
PREVIEW = {"ff": 0.04, "noise": "realistic", "depths": (1_000_000, 10_000_000)}


def profile_configs(config):
    """{profile: config}: G1 (the main config) first, then every scenario in config order."""
    return {MAIN: config, **{name: load_config(scenario=name) for name in scenario_names(config)}}


def inputs_table(configs):
    rows = []
    for name, cfg in configs.items():
        dist = load_length_distributions(cfg)
        matrices = origin_matrices(cfg)
        lo, hi = cfg["methods"]["size_selection_window_bp"]
        profile = cfg.get("fetal_profile", {"shift_bp": 0, "blend_maternal": 0.0})
        row = {"profile": name, "shift_bp": profile["shift_bp"], "blend_maternal": profile["blend_maternal"],
               "fetal_median_bp": distribution_median(dist, "fetal"),
               "maternal_median_bp": distribution_median(dist, "maternal")}
        for ff in INPUT_FF:
            w = window_stats(matrices, ff, lo, hi)
            tag = f"ff{round(100 * ff)}"
            row.update({f"g_{tag}": w["gain"], f"r_{tag}": w["r"], f"g_sqrt_r_{tag}": w["gain"] * np.sqrt(w["r"]),
                        f"g2r_{tag}": w["gain"] ** 2 * w["r"]})
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_table(configs):
    main = configs[MAIN]
    matrices_g1 = origin_matrices(main)
    grid = main["grid"]
    rows = []
    for name, cfg in configs.items():
        matrices = origin_matrices(cfg)
        p_f, p_m = length_distributions(matrices)
        p0 = float(matrices["maternal"][CHR21].sum())
        for ff in grid["fetal_fraction"]:
            weights = method_weights(cfg, matrices, ff, matrices_g1 if name != MAIN else None)
            if name == MAIN:
                weights["W_G1"] = weights["W"]
            _, w_star = mixture_terms(ff, p_f, p_m)
            for noise in grid["noise_sigma"]:
                sigma = noise_sigma(main, noise)
                for depth in grid["depth"]:
                    u_opt = optimal_weights(w_star, ff, p0, depth, sigma)
                    opt = model_terms(u_opt, p_f, p_m, ff, p0, depth, sigma)["ratio_pred"]
                    for method in PRED_METHODS:
                        t = model_terms(weights[method], p_f, p_m, ff, p0, depth, sigma)
                        rows.append({"profile": name, "ff": ff, "depth": depth, "noise": str(noise), "sigma": sigma,
                                     "method": method, "p0": p0, "g_eff": t["g_eff"], "r_eff": t["r_eff"],
                                     "ratio_pred": t["ratio_pred"], "ratio_pred_opt": opt,
                                     "efficiency": t["ratio_pred"] / opt})
    return pd.DataFrame(rows)


def self_test(config, configs, pred):
    def check(name, worst, tol):
        ok = worst < tol
        print(f"self-test {name}: worst {worst:.2e} (tolerance {tol:g}) {'ok' if ok else 'FAIL'}")
        if not ok:
            raise ValueError(f"self-test {name} failed: worst {worst!r} >= {tol!r}")

    g1 = load_length_distributions(config)
    # 1. a scenario with shift 0 and blend 0 is exactly G1
    null = copy.deepcopy(config)
    null["fetal_profile"] = {"name": "null", "shift_bp": 0, "blend_maternal": 0.0}
    check("1 (shift 0, blend 0 == G1, both columns)",
          float(np.abs(load_length_distributions(null)[["maternal", "fetal"]] - g1[["maternal", "fetal"]]).max().max()),
          1e-300)
    # 2. maternal column never changes; both columns sum to 1
    worst_m, worst_sum = 0.0, 0.0
    for cfg in configs.values():
        d = load_length_distributions(cfg)
        worst_m = max(worst_m, float(np.abs(d["maternal"] - g1["maternal"]).max()))
        worst_sum = max(worst_sum, abs(d["fetal"].sum() - 1), abs(d["maternal"].sum() - 1))
    check("2a (maternal column identical in every profile)", worst_m, 1e-300)
    check("2b (fetal and maternal sum to 1 in every profile)", worst_sum, 1e-12)
    # 3. a shift moves the fetal profile by exactly shift_bp where the source stays inside 36-500 bp
    worst = 0.0
    for cfg in configs.values():
        prof = cfg.get("fetal_profile")
        if prof is None or prof["blend_maternal"] != 0:
            continue
        s = int(prof["shift_bp"])
        d = load_length_distributions(cfg).set_index("length_bp")["fetal"]
        src = g1.set_index("length_bp")["fetal"]
        inner = [l for l in d.index if 36 <= l - s <= 500]
        ratio = d.loc[inner].to_numpy() / src.loc[[l - s for l in inner]].to_numpy()
        worst = max(worst, float(ratio.max() / ratio.min() - 1))   # proportional (only the normalisation differs)
    check("3 (shifted fetal(l) proportional to G1 fetal(l - shift))", worst, 1e-12)
    # 4. blend: fetal = (1 - lam) fetal_G1 + lam maternal
    worst = 0.0
    for cfg in configs.values():
        prof = cfg.get("fetal_profile")
        if prof is None or prof["shift_bp"] != 0:
            continue
        lam = prof["blend_maternal"]
        d = load_length_distributions(cfg)
        worst = max(worst, float(np.abs(d["fetal"] - ((1 - lam) * g1["fetal"] + lam * g1["maternal"])).max()))
    check("4 (blend = (1 - lambda) G1 fetal + lambda maternal)", worst, 1e-15)
    # 5. G1 predictions reproduce the main unified_model_check.csv (B, W, W_fixedFF)
    saved = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "unified_model_check.csv",
                        dtype={"noise": str}).set_index(["ff", "depth", "noise", "method"])["ratio_pred"]
    mine = pred[(pred["profile"] == MAIN) & pred["method"].isin(["B", "W", "W_fixedFF"])]
    worst = max(abs(r.ratio_pred / saved.loc[(r.ff, r.depth, r.noise, r.method)] - 1) for r in mine.itertuples())
    check(f"5 (G1 ratio_pred == unified_model_check.csv, {len(mine)} rows)", worst, 1e-9)


def main():
    config = load_config()
    configs = profile_configs(config)
    pred = prediction_table(configs)
    self_test(config, configs, pred)
    inputs = inputs_table(configs)

    tables = repo_path(config["paths"]["results"]) / "tables"
    inputs.to_csv(tables / "robustness_profiles_inputs.csv", index=False, float_format="%.6g")
    pred.to_csv(tables / "robustness_profiles_pred.csv", index=False, float_format="%.10g")

    print("\n7A inputs (40-156 bp window):")
    print(inputs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    p = pred[np.isclose(pred["ff"], PREVIEW["ff"]) & (pred["noise"] == PREVIEW["noise"])]
    p = p.set_index(["profile", "depth", "method"])["ratio_pred"]
    print(f"\n7C ratio_pred vs A, FF {PREVIEW['ff']:.0%}, {PREVIEW['noise']} noise:")
    print(f"{'profile':>10} {'g*sqrt(r)':>9} " + " ".join(f"{f'{d / 1e6:g}M: W / W_G1 / B':>22}"
                                                       for d in PREVIEW["depths"]))
    order = [MAIN] + scenario_names(config)
    for name in sorted(order, key=lambda n: (configs[n].get("fetal_profile", {}).get("blend_maternal", 0),
                                             configs[n].get("fetal_profile", {}).get("shift_bp", 0))):
        gsr = inputs.set_index("profile").loc[name, "g_sqrt_r_ff4"]
        cells = [" / ".join(f"{p.loc[(name, d, m)]:.2f}" for m in ("W", "W_G1", "B")) for d in PREVIEW["depths"]]
        print(f"{name:>10} {gsr:9.2f} " + " ".join(f"{c:>22}" for c in cells))
    print(f"\nSaved {tables / 'robustness_profiles_inputs.csv'} and {tables / 'robustness_profiles_pred.csv'}")


if __name__ == "__main__":
    main()
