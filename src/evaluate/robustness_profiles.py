"""Task 07, 7E: how do the conclusions change with the fetal length profile? (reviewer points R1-M2, R1-M3/R2-M2)

Reads, for G1 (the main analysis, results/tables) and every scenario in robustness.fetal_profiles
(results/robustness/<NAME>/tables, from run_condition --scenario and the evaluation scripts run with --scenario):
results_all.csv, detection_limits.csv, fig3_b_minus_a.csv, unified_model_summary.csv, plus
results/tables/robustness_profiles_pred.csv (python -m src.evaluate.robustness_inputs). In G1, W_G1 is W (the
G1 weights are the true weights), so G1's W rows are used as its W_G1 rows.

Scenario patients use the same seeds as the main grid but different multinomial probabilities, so scenarios are
only partially paired; comparisons across scenarios are descriptive. The only test is each scenario's own
paired bootstrap of B - A (bootstrap_diff).

Writes results/tables/robustness_profiles_summary.csv, long format, one row per answer:
  question  1-7, 4b, 7b (below)  profile  G1, a scenario, or "scenarios" (pooled over the five scenarios)
  depth, noise  "all" where pooled   item  what the row is   value  number   text  readable form / conditions
Detection limits are compared as numbers with the grid-edge rule of decision #13's check ('< 1%' counts as 1%,
'> 15%' as 15%); differences of limits are in FF percentage points (items ending in _pts). Separation ratios
(items starting with sep_ratio) are ratios of standardised separations, not detection limits.
Questions:
  1  Is B ever worse than A? Conditions with sep(B)/sep(A) < 1, and with B - A sens_at_fpr significantly
     negative (bootstrap 95% CI entirely below 0); both listed.
  2  Does the depth result survive? sep(B)/sep(A) at FF 4%, 1M and 10M, per noise level (observed and model).
  3  Does the model still predict the simulation? Median |rel_error| per family (unified_model_summary).
  4  What does a wrong profile cost W? Per depth x noise: detection limits of W, W_G1, W_fixedFF and the cost
     W_G1 limit - W limit (points); the separation ratio sep(W_G1)/sep(W) (median and range over the 8 FFs).
     Over the 60 scenario settings (5 scenarios x 4 depths x 3 noise levels, profile "scenarios"): median,
     IQR and maximum of the cost in points, and the worst separation ratio (setting median and single
     condition).
  4b W_G1 vs A: margin A limit - W_G1 limit (points; positive = W_G1 detects at lower FF) per depth x noise,
     with the conservative (sens_lo) limits in the text, and per profile the smallest and median margin.
  5  Ranking: detection limit of B minus that of W_G1 per depth x noise; B is better where it is negative by
     more than LIMIT_TIE.
  6  False positives: pooled FPR over the 96 conditions (fpr x test normals) of A, W, W_G1, W_fixedFF, C2,
     Wilson 95% CI. C2 is pooled here because fpr_check leaves out the trees. The CI treats the normals as
     independent, but each condition's normals share one reference, so it is too narrow.
  7  B-opt: chosen upper cutoffs per profile and noise level (counts over the 32 FF x depth conditions).
  7b B-opt vs A: conditions with sep(B-opt)/sep(A) < 1, and the largest loss in sens_at_fpr (B-opt - A,
     points) with its condition.

Prints the answers. No simulation. Run from the repo root (seconds):
    python -m src.evaluate.robustness_profiles
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names
from src.simulate.mappability import wilson_interval

MAIN = "G1"
CONDITION = ["ff", "depth", "noise"]
FIG_FF = 0.04
LIMIT_TIE = 0.001          # detection limits closer than 0.1 percentage point count as equal
FPR_METHODS = ("A", "W", "W_G1", "W_fixedFF", "C2")
FAMILIES = ("B", "W", "W_G1", "W_fixedFF")
COLUMNS = ["question", "profile", "depth", "noise", "item", "value", "text"]
QUESTIONS = ["1", "2", "3", "4", "4b", "5", "6", "7", "7b"]
POOLED = "scenarios"       # profile label of rows pooled over the five scenarios


def profiles(config):
    """{profile: its tables directory}, G1 first, then the scenarios in config order."""
    out = {MAIN: repo_path(config["paths"]["results"]) / "tables"}
    for name in scenario_names(config, "fetal_profiles"):
        out[name] = repo_path(load_config(scenario=name)["paths"]["results"]) / "tables"
    return out


def with_w_g1(table, method_col="method"):
    """G1 only: add copies of the W rows as W_G1 (in G1 the G1 weights are the true weights)."""
    w = table[table[method_col] == "W"].assign(**{method_col: "W_G1"})
    return pd.concat([table, w], ignore_index=True)


def load(config):
    data = {}
    for name, d in profiles(config).items():
        read = {k: pd.read_csv(d / f"{k}.csv", dtype={"noise": str}) for k in
                ("results_all", "detection_limits", "fig3_b_minus_a", "unified_model_summary")}
        res = read["results_all"]
        res["sep"] = (res["z_mean_t21"] - res["z_mean_normal"]) / res["z_sd_normal"]
        if name == MAIN:
            for k in ("results_all", "detection_limits"):
                read[k] = with_w_g1(read[k])
            s = read["unified_model_summary"]
            read["unified_model_summary"] = pd.concat([s, s[s["family"] == "W"].assign(family="W_G1")],
                                                      ignore_index=True)
        if len(res) != 96 * res["method"].nunique():
            raise ValueError(f"{name}: results_all.csv has {len(res)} rows, not 96 per method")
        data[name] = read
    pred = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "robustness_profiles_pred.csv",
                       dtype={"noise": str})
    return data, pred


def label(ff, depth, noise):
    return f"FF {100 * ff:g}% {depth / 1e6:g}M {noise}"


def numeric_limit(row):
    """Detection limit as a number, with the grid-edge rule of decision #13's check."""
    text = row["ff_limit_text"]
    return 0.01 if text == "< 1%" else 0.15 if text == "> 15%" else float(row["ff_limit"])


def per_condition(results, method):
    return results[results["method"] == method].set_index(CONDITION)


def q1(name, d):
    res = d["results_all"]
    ratio = (per_condition(res, "B")["sep"] / per_condition(res, "A")["sep"]).rename("ratio")
    below = ratio[ratio < 1]
    boot = d["fig3_b_minus_a"].set_index(CONDITION)
    sig = boot[boot["ci_hi"] < 0]
    return [
        {"question": "1", "profile": name, "depth": "all", "noise": "all", "item": "n_sep_ratio_B_A_below_1",
         "value": len(below), "text": "; ".join(f"{label(*k)} ({v:.3f})" for k, v in below.sort_index().items())},
        {"question": "1", "profile": name, "depth": "all", "noise": "all", "item": "n_sens_diff_B_A_sig_negative",
         "value": len(sig), "text": "; ".join(f"{label(*k)} ({100 * r['diff']:+.1f} pts, CI {100 * r['ci_lo']:+.1f} to "
                                              f"{100 * r['ci_hi']:+.1f})" for k, r in sig.sort_index().iterrows())},
    ]


def q2(name, d, pred, grid):
    res = d["results_all"]
    ratio = per_condition(res, "B")["sep"] / per_condition(res, "A")["sep"]
    p = pred[(pred["profile"] == name) & (pred["method"] == "B")].set_index(CONDITION)["ratio_pred"]
    lo, hi = min(grid["depth"]), max(grid["depth"])
    rows = []
    for noise in grid["noise_sigma"]:
        obs = [ratio.loc[(FIG_FF, dep, str(noise))] for dep in (lo, hi)]
        mod = [p.loc[(FIG_FF, dep, str(noise))] for dep in (lo, hi)]
        rows.append({"question": "2", "profile": name, "depth": f"{lo / 1e6:g}M->{hi / 1e6:g}M", "noise": str(noise),
                     "item": "sep_ratio_B_A_ff4_change_obs", "value": obs[1] / obs[0] - 1,
                     "text": f"observed {obs[0]:.3f} -> {obs[1]:.3f}; model {mod[0]:.3f} -> {mod[1]:.3f}"})
    return rows


def q3(name, d, families=FAMILIES):
    s = d["unified_model_summary"]
    s = s[s["group"] == "overall"]
    rows = []
    for fam in families:
        for subset in ("all", "without_low_signal"):
            r = s[(s["family"] == fam) & (s["rows"] == subset)].iloc[0]
            rows.append({"question": "3", "profile": name, "depth": "all", "noise": "all",
                         "item": f"median_abs_rel_error_{fam}_{subset}", "value": r["rel_median"],
                         "text": f"n {int(r['n'])}, p95 {100 * r['rel_p95']:.2f}%, mean signed "
                                 f"{100 * r['rel_mean_signed']:+.2f}% (t = {r['rel_mean_t']:.1f})"})
    return rows


def pts(row_or_value):
    return 100 * row_or_value


def q4_q5(name, d, grid):
    lim = d["detection_limits"].set_index(["method", "depth", "noise"])
    res = d["results_all"]
    loss = (per_condition(res, "W_G1")["sep"] / per_condition(res, "W")["sep"]).rename("loss")
    rows = []
    for noise in grid["noise_sigma"]:
        for depth in grid["depth"]:
            key = lambda m: (m, depth, str(noise))  # noqa: E731
            texts = {m: lim.loc[key(m), "ff_limit_text"] for m in ("A", "W", "W_G1", "W_fixedFF", "B")}
            cons = {m: lim.loc[key(m), "ff_limit_cons_text"] for m in ("A", "W_G1")}
            num = {m: numeric_limit(lim.loc[key(m)]) for m in ("A", "W", "W_G1", "B")}
            l_loss = loss.xs((depth, str(noise)), level=["depth", "noise"])
            setting = {"profile": name, "depth": depth, "noise": str(noise)}
            rows.append({"question": "4", **setting, "item": "limit_W_G1_minus_W_pts",
                         "value": pts(num["W_G1"] - num["W"]),
                         "text": f"W {texts['W']}, W_G1 {texts['W_G1']}, W_fixedFF {texts['W_fixedFF']}"})
            rows.append({"question": "4", **setting, "item": "sep_ratio_W_G1_over_W_median",
                         "value": l_loss.median(),
                         "text": f"separation ratio, range {l_loss.min():.3f}-{l_loss.max():.3f} over FF"})
            rows.append({"question": "4b", **setting, "item": "limit_A_minus_W_G1_pts",
                         "value": pts(num["A"] - num["W_G1"]),
                         "text": f"A {texts['A']}, W_G1 {texts['W_G1']} (conservative: A {cons['A']}, "
                                 f"W_G1 {cons['W_G1']})"})
            diff = num["B"] - num["W_G1"]
            verdict = "B better" if diff < -LIMIT_TIE else "W_G1 better" if diff > LIMIT_TIE else "equal"
            rows.append({"question": "5", **setting, "item": "limit_B_minus_W_G1_pts", "value": pts(diff),
                         "text": f"{verdict}: B {texts['B']}, W_G1 {texts['W_G1']}"})
    margins = [r for r in rows if r["item"] == "limit_A_minus_W_G1_pts"]
    worst = min(margins, key=lambda r: r["value"])
    rows.append({"question": "4b", "profile": name, "depth": "all", "noise": "all",
                 "item": "limit_A_minus_W_G1_pts_min", "value": worst["value"],
                 "text": f"at {worst['depth'] / 1e6:g}M {worst['noise']}: {worst['text']}; median over the 12 "
                         f"settings {np.median([r['value'] for r in margins]):+.2f} pts"})
    return rows


def q4_pooled(rows, data):
    """Cost of the wrong profile over the 60 scenario settings (G1 left out: there W_G1 is W)."""
    cost = [r for r in rows if r["item"] == "limit_W_G1_minus_W_pts" and r["profile"] != MAIN]
    values = np.array([r["value"] for r in cost])
    top = max(cost, key=lambda r: r["value"])
    out = [{"question": "4", "profile": POOLED, "depth": "all", "noise": "all", "item": f"limit_W_G1_minus_W_pts_{k}",
            "value": v, "text": f"over {len(values)} settings"}
           for k, v in (("median", np.median(values)), ("q25", np.percentile(values, 25)),
                        ("q75", np.percentile(values, 75)))]
    out.append({"question": "4", "profile": POOLED, "depth": "all", "noise": "all",
                "item": "limit_W_G1_minus_W_pts_max", "value": top["value"],
                "text": f"{top['profile']} {top['depth'] / 1e6:g}M {top['noise']}: {top['text']}"})
    sep = [r for r in rows if r["item"] == "sep_ratio_W_G1_over_W_median" and r["profile"] != MAIN]
    low = min(sep, key=lambda r: r["value"])
    out.append({"question": "4", "profile": POOLED, "depth": "all", "noise": "all",
                "item": "sep_ratio_W_G1_over_W_worst_setting", "value": low["value"],
                "text": f"separation ratio (median over FF) at {low['profile']} {low['depth'] / 1e6:g}M {low['noise']}"})
    cond = []
    for name, d in data.items():
        if name == MAIN:
            continue
        res = d["results_all"]
        ratio = per_condition(res, "W_G1")["sep"] / per_condition(res, "W")["sep"]
        cond += [(v, name, k) for k, v in ratio.items()]
    v, name, k = min(cond)
    out.append({"question": "4", "profile": POOLED, "depth": "all", "noise": "all",
                "item": "sep_ratio_W_G1_over_W_worst_condition", "value": v,
                "text": f"separation ratio at {name} {label(*k)}"})
    return out


def q6(name, d, n_normal, target):
    res = d["results_all"]
    rows = []
    for m in FPR_METHODS:
        r = res[res["method"] == m]
        fp = r["fpr"] * n_normal
        if not np.allclose(fp, fp.round(), rtol=0, atol=1e-6):
            raise ValueError(f"{name} {m}: fpr x test normals is not a whole number")
        k, n = int(fp.round().sum()), n_normal * len(r)
        lo, hi = wilson_interval(k, n)
        where = "above" if lo > target else "below" if hi < target else "contains"
        rows.append({"question": "6", "profile": name, "depth": "all", "noise": "all", "item": f"pooled_fpr_{m}",
                     "value": k / n, "text": f"{k}/{n} = {100 * k / n:.3f}% [{100 * lo:.3f}, {100 * hi:.3f}]; "
                                             f"CI {where} the target {100 * target:.3f}%"})
    return rows


def q7(name, d, grid):
    b = d["results_all"]
    b = b[b["method"] == "B-opt"]
    rows = []
    for noise in ["all"] + [str(n) for n in grid["noise_sigma"]]:
        part = b if noise == "all" else b[b["noise"] == noise]
        counts = part["cutoff"].astype(int).value_counts().sort_index()
        rows.append({"question": "7", "profile": name, "depth": "all", "noise": noise, "item": "bopt_cutoff_median",
                     "value": float(part["cutoff"].median()),
                     "text": ", ".join(f"{c} bp: {n}" for c, n in counts.items())})
    return rows


def q7b(name, d):
    res = d["results_all"]
    bopt, a = per_condition(res, "B-opt"), per_condition(res, "A")
    ratio = bopt["sep"] / a["sep"]
    diff = (bopt["sens_at_fpr"] - a["sens_at_fpr"]).sort_index()
    k = diff.idxmin()
    return [
        {"question": "7b", "profile": name, "depth": "all", "noise": "all", "item": "n_sep_ratio_Bopt_A_below_1",
         "value": int((ratio < 1).sum()), "text": f"of {len(ratio)} conditions"},
        {"question": "7b", "profile": name, "depth": "all", "noise": "all",
         "item": "max_sens_at_fpr_loss_Bopt_minus_A_pts", "value": pts(diff.loc[k]),
         "text": "B-opt is never below A" if diff.loc[k] >= 0 else
                 f"{label(*k)}: B-opt {100 * bopt.loc[k, 'sens_at_fpr']:.1f}% (cutoff {int(bopt.loc[k, 'cutoff'])} bp) "
                 f"vs A {100 * a.loc[k, 'sens_at_fpr']:.1f}%"},
    ]


def summary(config):
    data, pred = load(config)
    grid = config["grid"]
    n_normal, target = config["patients_per_condition"]["test_normal"], config["methods"]["target_fpr"]
    rows = []
    for name, d in data.items():
        rows += q1(name, d) + q2(name, d, pred, grid) + q3(name, d) + q4_q5(name, d, grid) \
            + q6(name, d, n_normal, target) + q7(name, d, grid) + q7b(name, d)
    rows += q4_pooled(rows, data)
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["question"] = out["question"].astype(str)
    order = {"question": {q: i for i, q in enumerate(QUESTIONS)},
             "profile": {n: i for i, n in enumerate([*data, POOLED])}}
    return out.sort_values(["question", "profile"], key=lambda s: s.map(order[s.name]),
                           kind="stable").reset_index(drop=True)


def report(table):
    """Short answers to questions 1-7 (with 4b, 7b), printed."""
    def rows(q, item=None):
        t = table[table["question"] == q]
        return (t if item is None else t[t["item"] == item]).itertuples()

    print("\nQ1. Is B ever worse than A?")
    for r in rows("1"):
        print(f"  {r.profile:>9} {r.item:>30}: {int(r.value):2d}  {r.text}")
    print("\nQ2. B/A separation ratio at FF 4%, 1M -> 10M:")
    for r in rows("2"):
        print(f"  {r.profile:>9} {r.noise:>9}: {100 * r.value:+6.1f}%  ({r.text})")
    print("\nQ3. Median |rel_error| of the model (all / without low_signal):")
    for name, g in table[table["question"] == "3"].groupby("profile", sort=False):
        v = g.set_index("item")["value"]
        print(f"  {name:>9}: " + "  ".join(f"{f} {100 * v[f'median_abs_rel_error_{f}_all']:.2f}/"
                                           f"{100 * v[f'median_abs_rel_error_{f}_without_low_signal']:.2f}%"
                                           for f in FAMILIES))
    print("\nQ4. Detection limits W / W_G1 / W_fixedFF, cost W_G1 - W (points) | separation ratio sep(W_G1)/sep(W):")
    for r in rows("4", "limit_W_G1_minus_W_pts"):
        sep = table[(table["item"] == "sep_ratio_W_G1_over_W_median") & (table["profile"] == r.profile)
                    & (table["depth"] == r.depth) & (table["noise"] == r.noise)].iloc[0]
        print(f"  {r.profile:>9} {r.depth / 1e6:>4g}M {r.noise:>9}: {r.text}; cost {r.value:+.2f} pts | "
              f"separation {sep.value:.3f} ({sep.text[17:]})")
    print("  Pooled over the 60 scenario settings:")
    for r in table[(table["question"] == "4") & (table["profile"] == POOLED)].itertuples():
        print(f"    {r.item:>40}: {r.value:+.3f}  {r.text}")
    print("\nQ4b. W_G1 vs A detection-limit margin (A - W_G1, points):")
    for r in rows("4b", "limit_A_minus_W_G1_pts"):
        print(f"  {r.profile:>9} {r.depth / 1e6:>4g}M {r.noise:>9}: {r.value:+5.2f}  {r.text}")
    for r in rows("4b", "limit_A_minus_W_G1_pts_min"):
        print(f"  {r.profile:>9} smallest margin {r.value:+.2f} pts {r.text}")
    print("\nQ5. B vs W_G1 detection limits:")
    for r in rows("5"):
        print(f"  {r.profile:>9} {r.depth / 1e6:>4g}M {r.noise:>9}: {r.text}")
    print("\nQ6. Pooled FPR over 96 conditions:")
    for r in rows("6"):
        print(f"  {r.profile:>9} {r.item[11:]:>9}: {r.text}")
    print("\nQ7. B-opt chosen cutoffs:")
    for r in rows("7"):
        print(f"  {r.profile:>9} {r.noise:>9}: median {r.value:g} bp ({r.text})")
    print("\nQ7b. B-opt vs A:")
    for r in rows("7b"):
        print(f"  {r.profile:>9} {r.item:>38}: {r.value:g}  {r.text}")


def main():
    config = load_config()
    table = summary(config)
    out = repo_path(config["paths"]["results"]) / "tables" / "robustness_profiles_summary.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 250):
        report(table)
    print(f"\nSaved {out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
