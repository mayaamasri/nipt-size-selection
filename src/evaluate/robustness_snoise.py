"""Task 08, 8E: what is left of the depth result and of B's advantage with extra noise on short fragments? (R1-M1)

Reads, for the main analysis (results/tables) and each short-noise scenario in robustness.short_noise
(results/robustness/<NAME>/tables, from run_condition --scenario and the evaluation scripts run with --scenario):
results_all.csv, detection_limits.csv, fig3_b_minus_a.csv and unified_model_check.csv, plus
results/tables/robustness_snoise_pred.csv (python -m src.evaluate.robustness_snoise_inputs). Short-noise
scenarios have no sigma = 0 level, so the main analysis is restricted to the same noise levels (low and
realistic: 64 conditions each) wherever it is compared.

Scenario patients use the same seeds as the main grid, but drawing eps_s shifts each patient's stream, so the
pairing with the main grid is only partial; comparisons across k are descriptive. The only test is each
scenario's own paired bootstrap of B - A (bootstrap_diff).

Writes results/tables/robustness_snoise_summary.csv, long format like robustness_profiles_summary.csv (question,
profile = main / scenario, depth, noise, item, value, text). Detection-limit differences are in FF percentage
points (items ending in _pts); separation ratios start with sep_ratio. Questions:
  1  Is B worse than A anywhere? Conditions with sep(B)/sep(A) < 1, and with B - A sens_at_fpr significantly
     negative (bootstrap 95% CI entirely below 0); both listed (robustness_profiles.q1).
  2  Does the depth result survive? sep(B)/sep(A) at FF 4%, 1M and 10M, low and realistic noise, observed and
     model, with the model's limits g sqrt(r) (N -> 0) and g sigma_A / sigma_B (N -> infinity).
  3  Model accuracy with sigma_u: median |rel_error| for B (B and every B_c), W and W_fixedFF
     (unified_model_check.csv, summarised like unified_model_summary).
  4  W vs B vs A: detection limits per depth x noise, and whether W stays below B and below A (limits compared
     with the grid-edge rule of decision #13's check; ties within LIMIT_TIE).
  5  B-opt: chosen cutoffs per noise level (robustness_profiles.q7); 5b: B-opt vs A (q7b).
  6  FPR: pooled over the 64 conditions per method (A, B, W, W_fixedFF, C2), Wilson 95% CI (robustness_profiles.q6;
     the CI treats the normals as independent, so it is too narrow).

Prints the answers. No simulation. Run from the repo root (seconds):
    python -m src.evaluate.robustness_snoise
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names, scenario_noise_levels
from src.evaluate import robustness_profiles as rp
from src.evaluate.unified_model_check import family_summary

MAIN = "main"
FIG_FF = 0.04
FAMILIES = ("B", "W", "W_fixedFF")
FPR_METHODS = ("A", "B", "W", "W_fixedFF", "C2")
QUESTIONS = ["1", "2", "3", "4", "5", "5b", "6"]
RELABEL = {"7": "5", "7b": "5b"}        # robustness_profiles' B-opt questions are 5 and 5b here


def profiles(config):
    """{name: (config, tables directory)}: main first, then every short-noise scenario."""
    out = {MAIN: (config, repo_path(config["paths"]["results"]) / "tables")}
    for name in scenario_names(config, "short_noise"):
        cfg = load_config(scenario=name)
        out[name] = (cfg, repo_path(cfg["paths"]["results"]) / "tables")
    return out


def load(config):
    levels = [str(n) for n in scenario_noise_levels(load_config(scenario=scenario_names(config, "short_noise")[0]))]
    data = {}
    for name, (cfg, d) in profiles(config).items():
        read = {k: pd.read_csv(d / f"{k}.csv", dtype={"noise": str}) for k in
                ("results_all", "detection_limits", "fig3_b_minus_a", "unified_model_check")}
        read = {k: v[v["noise"].isin(levels)].reset_index(drop=True) for k, v in read.items()}
        res = read["results_all"]
        res["sep"] = (res["z_mean_t21"] - res["z_mean_normal"]) / res["z_sd_normal"]
        n_cond = len(cfg["grid"]["fetal_fraction"]) * len(cfg["grid"]["depth"]) * len(levels)
        if len(res) != n_cond * res["method"].nunique():
            raise ValueError(f"{name}: results_all.csv has {len(res)} rows, not {n_cond} per method")
        read["unified_model_summary"] = family_summary(read["unified_model_check"])
        data[name] = read
    pred = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "robustness_snoise_pred.csv",
                       dtype={"noise": str})
    return data, pred, levels


def q2(name, d, pred, levels, depths):
    res = d["results_all"]
    ratio = rp.per_condition(res, "B")["sep"] / rp.per_condition(res, "A")["sep"]
    p = pred[(pred["scenario"] == name) & (pred["method"] == "B") & np.isclose(pred["ff"], FIG_FF)]
    lo, hi = min(depths), max(depths)
    rows = []
    for noise in levels:
        obs = [ratio.loc[(FIG_FF, dep, noise)] for dep in (lo, hi)]
        q = p[p["noise"] == noise].set_index("depth")
        mod = [q.loc[dep, "ratio_pred"] for dep in (lo, hi)]
        rows.append({"question": "2", "profile": name, "depth": f"{lo / 1e6:g}M->{hi / 1e6:g}M", "noise": noise,
                     "item": "sep_ratio_B_A_ff4_change_obs", "value": obs[1] / obs[0] - 1,
                     "text": f"observed {obs[0]:.3f} -> {obs[1]:.3f}; model {mod[0]:.3f} -> {mod[1]:.3f} "
                             f"({mod[1] / mod[0] - 1:+.1%})"})
        rows.append({"question": "2", "profile": name, "depth": "limits", "noise": noise,
                     "item": "sep_ratio_B_A_ff4_high_depth_limit", "value": q["ratio_high_depth"].iloc[0],
                     "text": f"g sigma_A / sigma_B = {q['ratio_high_depth'].iloc[0]:.3f} (g = {q['g_eff'].iloc[0]:.3f}, "
                             f"sigma_B / sigma_A = {q['sigma_B_over_A'].iloc[0]:.3f}); low-depth limit g sqrt(r) = "
                             f"{q['ratio_low_depth'].iloc[0]:.3f}"})
    return rows


def q4(name, d, levels, depths):
    lim = d["detection_limits"].set_index(["method", "depth", "noise"])
    rows = []
    for noise in levels:
        for depth in depths:
            key = lambda m: (m, depth, noise)  # noqa: E731
            num = {m: rp.numeric_limit(lim.loc[key(m)]) for m in ("A", "B", "W")}
            text = {m: lim.loc[key(m), "ff_limit_text"] for m in ("A", "B", "W", "W_fixedFF")}

            def verdict(other):
                diff = num[other] - num["W"]
                return (f"W lower than {other}" if diff > rp.LIMIT_TIE else f"W higher than {other}"
                        if diff < -rp.LIMIT_TIE else f"W equal to {other}")
            setting = {"question": "4", "profile": name, "depth": depth, "noise": noise}
            rows.append({**setting, "item": "limit_B_minus_W_pts", "value": rp.pts(num["B"] - num["W"]),
                         "text": f"{verdict('B')}: A {text['A']}, B {text['B']}, W {text['W']}, "
                                 f"W_fixedFF {text['W_fixedFF']}"})
            rows.append({**setting, "item": "limit_A_minus_W_pts", "value": rp.pts(num["A"] - num["W"]),
                         "text": verdict("A")})
    return rows


def summary(config):
    data, pred, levels = load(config)
    grid = {**config["grid"], "noise_sigma": levels}
    n_normal, target = config["patients_per_condition"]["test_normal"], config["methods"]["target_fpr"]
    rows = []
    for name, d in data.items():
        rows += rp.q1(name, d) + q2(name, d, pred, levels, grid["depth"]) + rp.q3(name, d, FAMILIES) \
            + q4(name, d, levels, grid["depth"]) + rp.q7(name, d, grid) + rp.q7b(name, d) \
            + rp.q6(name, d, n_normal, target, FPR_METHODS)
    out = pd.DataFrame(rows, columns=rp.COLUMNS)
    out["question"] = out["question"].astype(str).replace(RELABEL)
    order = {"question": {q: i for i, q in enumerate(QUESTIONS)}, "profile": {n: i for i, n in enumerate(data)}}
    return out.sort_values(["question", "profile"], key=lambda s: s.map(order[s.name]),
                           kind="stable").reset_index(drop=True)


def report(table):
    def rows(q, item=None):
        t = table[table["question"] == q]
        return (t if item is None else t[t["item"] == item]).itertuples()

    print("\nQ1. Is B worse than A anywhere? (low and realistic noise, 64 conditions per k)")
    for r in rows("1"):
        print(f"  {r.profile:>10} {r.item:>30}: {int(r.value):2d}  {r.text}")
    print("\nQ2. sep(B)/sep(A) at FF 4%, 1M -> 10M:")
    for r in rows("2"):
        what, tag = (("1M->10M", f"{100 * r.value:+6.1f}%") if r.item.endswith("change_obs")
                     else ("high-depth limit", f"{r.value:7.3f}"))
        print(f"  {r.profile:>10} {r.noise:>9} {what:>16}: {tag}  {r.text}")
    print("\nQ3. Median |rel_error| with sigma_u (all / without low_signal):")
    for name, g in table[table["question"] == "3"].groupby("profile", sort=False):
        v = g.set_index("item")["value"]
        print(f"  {name:>10}: " + "  ".join(f"{f} {100 * v[f'median_abs_rel_error_{f}_all']:.2f}/"
                                            f"{100 * v[f'median_abs_rel_error_{f}_without_low_signal']:.2f}%"
                                            for f in FAMILIES))
    print("\nQ4. Detection limits: W vs B and A (points: B - W, A - W):")
    for r in rows("4", "limit_B_minus_W_pts"):
        a = table[(table["item"] == "limit_A_minus_W_pts") & (table["profile"] == r.profile)
                  & (table["depth"] == r.depth) & (table["noise"] == r.noise)].iloc[0]
        print(f"  {r.profile:>10} {r.depth / 1e6:>4g}M {r.noise:>9}: B-W {r.value:+.2f}, A-W {a.value:+.2f} | {r.text}; "
              f"{a.text}")
    print("\nQ5. B-opt chosen cutoffs:")
    for r in rows("5"):
        print(f"  {r.profile:>10} {r.noise:>9}: median {r.value:g} bp ({r.text})")
    print("\nQ5b. B-opt vs A:")
    for r in rows("5b"):
        print(f"  {r.profile:>10} {r.item:>38}: {r.value:g}  {r.text}")
    print("\nQ6. Pooled FPR (64 conditions):")
    for r in rows("6"):
        print(f"  {r.profile:>10} {r.item[11:]:>9}: {r.text}")


def main():
    config = load_config()
    table = summary(config)
    out = repo_path(config["paths"]["results"]) / "tables" / "robustness_snoise_summary.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 250):
        report(table)
    print(f"\nSaved {out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
