"""Task 5E: manuscript and supplementary tables.

Table 1  table1_detection_limits.csv   detection limits (decision #13) of A, B, B-opt, W, C1, C2 per depth x
                                        noise: main (sens), conservative (sens_lo), the sens_at_fpr check and
                                        its flag, and whether each limit is at or below the 4% commercial
                                        minimum ('< 1%' counts as below, '> 15%' as above)
Table 2  table2_realism_yang.csv        realism check against Yang et al. [7]: exact r and g (decision #14) of
                                        the 40-156 bp window vs the values Yang report (decisions.md literature
                                        table)
S        tableS_bopt_cutoff.csv         B-opt: chosen upper cutoff per condition with its training separation,
                                        and B-opt vs B (156 bp) on the test set
S        tableS_c2_importances.csv      C2: feature importances and chosen max depth per condition
S        tableS_w_vs_wfixed.csv         W vs W_fixedFF per condition (sens, sens_at_fpr, AUC and differences)
S        tableS_w_vs_wfixed_limits.csv  W vs W_fixedFF detection limits per depth x noise

Reads results_all.csv, detection_limits.csv, bopt_separation.csv, c2_feature_importances.csv and the
length/mappability model; writes all tables to results/tables/. Run from the repo root:
    python -m src.evaluate.tables
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.simulate.patients import origin_matrices, window_stats

CONDITION = ["ff", "depth", "noise"]
TABLE1_METHODS = ["A", "B", "B-opt", "W", "C1", "C2"]
COMMERCIAL_MIN_FF = 0.04
TREES = ("C1", "C2")

CAPTION_TABLE1 = (
    "Table 1. Detection limits (FF at which sensitivity reaches 95% for the last time, decision #13) per depth "
    "and noise level: main estimate from the sensitivity at each method's own threshold, conservative estimate "
    "from its Wilson 95% lower bound, and a check from the sensitivity at FPR 0.135% (flagged if more than one grid step from the "
    "main limit; {n_flags} flags). ✓: at or below the 4% commercial minimum. W has the lowest limit in "
    "{w_lowest} of {n_rows} depth × noise rows ({exceptions}). Tree limits use each tree's real threshold, so "
    "in {n_nobody} of {n_tree_groups} tree groups the limit is interpolated from a grid point where the tree "
    "flags nobody (sensitivity 0); this raises the limit by up to {max_shift:.2f} FF points compared with "
    "interpolating from the sensitivity at FPR 0.135%."
)
CAPTION_TABLE2 = (
    "Table 2. Realism check of the 40–156 bp window against Yang et al. [7]: exact fraction of fragments kept "
    "(r) and FF enrichment (g) from the post-mappability length model (decision #14)."
)
CAPTION_BOPT = (
    "Table S1. B-opt: upper cutoff chosen per condition on the training set (grid 120–170 bp in 5 bp steps, "
    "lower bound 40 bp; decision #6), with its training separation, and B-opt vs B (40–156 bp) on the test set "
    "(standardised separation and sensitivity at FPR 0.135%). Chosen cutoffs: {counts}. B-opt − B in "
    "sensitivity at FPR 0.135%: mean "
    "{mean:+.2f} points (range {lo:+.1f} to {hi:+.1f})."
)
CAPTION_C2 = (
    "Table S2. C2 feature importances (scikit-learn impurity importance) and chosen max depth per condition. At "
    "FF ≥ 10% every method is at or near 100% sensitivity (all ≥ {min_sens:.1%}), so the importances there say "
    "little about the value of each feature."
)
CAPTION_W = (
    "Table S3. W (weights from the true FF) vs W with the weights fixed at FF {fixed_ff:.0%} (W_fixedFF columns) "
    "per condition: sensitivity at the z > 3 threshold, sensitivity at FPR 0.135% and AUC, and the differences "
    "fixed − true (sensitivity at FPR 0.135%: "
    "mean {mean:+.2f} points, largest {largest:.1f} points)."
)
CAPTION_W_LIMITS = (
    "Table S4. Detection limits of W with weights from the true FF and with weights fixed at FF {fixed_ff:.0%} (W_fixedFF "
    "columns) per depth and noise level (main and conservative); they "
    "differ by at most {max_diff:.2f} FF points."
)


def grid_order(table, config, by):
    """Sort by config grid order of noise, then the other keys."""
    noise_rank = {n: i for i, n in enumerate(config["grid"]["noise_sigma"])}
    method_rank = {m: i for i, m in enumerate(TABLE1_METHODS)}
    def key(col):
        if col.name == "noise":
            return col.map(noise_rank)
        if col.name == "method":
            return col.map(lambda m: method_rank.get(m, len(method_rank)))
        return col
    return table.sort_values(by, key=key).reset_index(drop=True)


def at_or_below(limit, text, threshold):
    """Is a detection limit at or below threshold? '< 1%' (NaN, below the grid) yes, '> 15%' no."""
    if np.isfinite(limit):
        return bool(limit <= threshold + 1e-12)
    return text.startswith("<")


def table1(limits, config):
    t = limits[limits["method"].isin(TABLE1_METHODS)].copy()
    t["main_le_4pct"] = [at_or_below(v, s, COMMERCIAL_MIN_FF) for v, s in zip(t["ff_limit"], t["ff_limit_text"])]
    t["cons_le_4pct"] = [at_or_below(v, s, COMMERCIAL_MIN_FF)
                         for v, s in zip(t["ff_limit_cons"], t["ff_limit_cons_text"])]
    cols = ["noise", "depth", "method", "ff_limit", "ff_limit_text", "ff_limit_cons", "ff_limit_cons_text",
            "ff_limit_at_fpr", "ff_limit_at_fpr_text", "at_fpr_flag", "main_le_4pct", "cons_le_4pct"]
    return grid_order(t[cols], config, ["noise", "depth", "method"])


def table1_caption_values(t1, results):
    """Numbers quoted in the Table 1 caption, computed from the data."""
    main = t1.pivot_table(index=["noise", "depth"], columns="method", values="ff_limit")
    lowest = main.idxmin(axis=1)
    exceptions = [f"{n} {d / 1e6:g}M: {m} {100 * main.loc[(n, d), m]:.2f}% vs W {100 * main.loc[(n, d), 'W']:.2f}%"
                  for (n, d), m in lowest[lowest != "W"].items()]
    # Tree groups whose last grid FF below the target is a nobody_positive row
    target = 0.95
    shifts, n_groups = [], 0
    for (method, depth, noise), g in results[results["method"].isin(TREES)].groupby(["method", "depth", "noise"]):
        g = g.sort_values("ff")
        below = np.flatnonzero(g["sens"].to_numpy() < target)
        if len(below) == 0 or below[-1] == len(g) - 1:
            continue
        n_groups += 1
        if bool(g.iloc[below[-1]]["nobody_positive"]):
            row = t1[(t1["method"] == method) & (t1["depth"] == depth) & (t1["noise"] == noise)].iloc[0]
            shifts.append(row["ff_limit"] - row["ff_limit_at_fpr"])
    return {"n_flags": int(t1["at_fpr_flag"].sum()), "w_lowest": int((lowest == "W").sum()), "n_rows": len(lowest),
            "exceptions": "exception: " + "; ".join(exceptions) if exceptions else "no exceptions",
            "n_nobody": len(shifts), "n_tree_groups": n_groups, "max_shift": 100 * max(shifts, default=0)}


def table2(config):
    """Exact r and g of Yang's window from the post-mappability model, next to Yang's reported values."""
    matrices = origin_matrices(config)
    lo, hi = config["methods"]["size_selection_window_bp"]
    s = {ff: window_stats(matrices, ff, lo, hi) for ff in (0.0285, 0.03, 0.04, 0.10)}
    yang_r = "24.76% of the original reads kept"
    rows = [
        ("Reads kept in 40–156 bp (r)", "FF 4%", f"{100 * s[0.04]['r']:.1f}%", yang_r),
        ("Reads kept in 40–156 bp (r)", "FF 10%", f"{100 * s[0.10]['r']:.1f}%", yang_r),
        ("FF gain, low FF (g)", "FF 2.85% (Yang's low-FF median)", f"{s[0.0285]['gain']:.2f}×",
         "2.75× (median 2.85% → 7.84%)"),
        ("FF gain, low FF (g)", "FF 3%", f"{s[0.03]['gain']:.2f}×", "3× when FF < 3%"),
        ("FF gain, typical FF (g)", "FF 10%", f"{s[0.10]['gain']:.2f}×", "~2× on average (cohort FF not reported)"),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "at", "ours", "yang"])
    out["ours_value"] = [s[0.04]["r"], s[0.10]["r"], s[0.0285]["gain"], s[0.03]["gain"], s[0.10]["gain"]]
    return out


def bopt_table(sep, results, config):
    """Chosen cutoff and its training separation (grid 120-170 bp in 5 bp steps, so 156 itself is not a
    candidate), next to B-opt and B (156 bp) on the test set: standardised separation and sens_at_fpr."""
    chosen = sep[sep["chosen"]].set_index(CONDITION)[["upper_bp", "separation"]]
    chosen.columns = ["chosen_upper_bp", "train_separation"]
    test = results[results["method"].isin(["B-opt", "B"])].assign(
        test_separation=lambda d: (d["z_mean_t21"] - d["z_mean_normal"]) / d["z_sd_normal"])
    test = test.pivot_table(index=CONDITION, columns="method", values=["test_separation", "sens_at_fpr"])
    test.columns = [f"{v}_{'Bopt' if m == 'B-opt' else 'B'}" for v, m in test.columns]
    out = chosen.join(test).reset_index()
    out["diff_sens_at_fpr"] = out["sens_at_fpr_Bopt"] - out["sens_at_fpr_B"]
    cols = CONDITION + ["chosen_upper_bp", "train_separation", "test_separation_Bopt", "test_separation_B",
                        "sens_at_fpr_Bopt", "sens_at_fpr_B", "diff_sens_at_fpr"]
    return grid_order(out[cols], config, ["noise", "depth", "ff"])


def c2_table(imp, results, config):
    wide = imp.pivot_table(index=CONDITION, columns="feature", values="importance").reset_index()
    depth = results[results["method"] == "C2"][CONDITION + ["max_depth", "nobody_positive"]]
    out = wide.merge(depth, on=CONDITION)
    return grid_order(out[CONDITION + ["zA", "z_short", "z_weighted", "max_depth", "nobody_positive"]],
                      config, ["noise", "depth", "ff"])


def w_tables(results, limits, config):
    cols = ["sens", "sens_at_fpr", "auc"]
    w = results[results["method"] == "W"].set_index(CONDITION)[cols]
    f = results[results["method"] == "W_fixedFF"].set_index(CONDITION)[cols]
    per = w.add_prefix("W_").join(f.add_prefix("Wfixed_"))
    for c in cols:
        per[f"diff_{c}"] = per[f"Wfixed_{c}"] - per[f"W_{c}"]
    lim = limits[limits["method"].isin(["W", "W_fixedFF"])]
    lim = lim.pivot_table(index=["depth", "noise"], columns="method",
                          values=["ff_limit", "ff_limit_text", "ff_limit_cons_text"], aggfunc="first")
    lim.columns = [f"{m}_{v}" for v, m in lim.columns]
    lim["diff_ff_limit"] = lim["W_fixedFF_ff_limit"].astype(float) - lim["W_ff_limit"].astype(float)
    lim = lim[["W_ff_limit_text", "W_fixedFF_ff_limit_text", "W_ff_limit_cons_text", "W_fixedFF_ff_limit_cons_text",
               "W_ff_limit", "W_fixedFF_ff_limit", "diff_ff_limit"]]
    return (grid_order(per.reset_index(), config, ["noise", "depth", "ff"]),
            grid_order(lim.reset_index(), config, ["noise", "depth"]))


def build_tables(config):
    """{file name: table} for every table this module writes."""
    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv")
    limits = pd.read_csv(tables / "detection_limits.csv")
    sep = pd.read_csv(tables / "bopt_separation.csv")
    imp = pd.read_csv(tables / "c2_feature_importances.csv")
    w_per, w_lim = w_tables(results, limits, config)
    return {
        "table1_detection_limits.csv": table1(limits, config),
        "table2_realism_yang.csv": table2(config),
        "tableS_bopt_cutoff.csv": bopt_table(sep, results, config),
        "tableS_c2_importances.csv": c2_table(imp, results, config),
        "tableS_w_vs_wfixed.csv": w_per,
        "tableS_w_vs_wfixed_limits.csv": w_lim,
    }


def captions(config, out=None):
    """{table file name: caption}, with every number computed from the data."""
    out = build_tables(config) if out is None else out
    results = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "results_all.csv")
    high = results[(results["ff"] >= 0.10 - 1e-12) & results["method"].isin(TABLE1_METHODS)]
    bopt, w_per, w_lim = out["tableS_bopt_cutoff.csv"], out["tableS_w_vs_wfixed.csv"], out["tableS_w_vs_wfixed_limits.csv"]
    counts = bopt["chosen_upper_bp"].value_counts().sort_index()
    return {
        "table1_detection_limits.csv": CAPTION_TABLE1.format(
            **table1_caption_values(out["table1_detection_limits.csv"], results)),
        "table2_realism_yang.csv": CAPTION_TABLE2,
        "tableS_bopt_cutoff.csv": CAPTION_BOPT.format(
            counts=", ".join(f"{c} bp ×{n}" for c, n in counts.items()), mean=100 * bopt["diff_sens_at_fpr"].mean(),
            lo=100 * bopt["diff_sens_at_fpr"].min(), hi=100 * bopt["diff_sens_at_fpr"].max()),
        "tableS_c2_importances.csv": CAPTION_C2.format(min_sens=high["sens"].min()),
        "tableS_w_vs_wfixed.csv": CAPTION_W.format(
            fixed_ff=config["methods"]["weight_fixed_ff"], mean=100 * w_per["diff_sens_at_fpr"].mean(),
            largest=100 * w_per["diff_sens_at_fpr"].abs().max()),
        "tableS_w_vs_wfixed_limits.csv": CAPTION_W_LIMITS.format(fixed_ff=config["methods"]["weight_fixed_ff"], max_diff=100 * w_lim["diff_ff_limit"].abs().max()),
    }


def main():
    config = load_config()
    tables = repo_path(config["paths"]["results"]) / "tables"
    out = build_tables(config)
    for name, table in out.items():
        table.to_csv(tables / name, index=False, float_format="%.6g")
        print(f"Saved {tables / name} ({len(table)} rows)")
    for name, text in captions(config, out).items():
        print(f"\nCaption ({name}):\n{text}")


if __name__ == "__main__":
    main()
