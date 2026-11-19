"""Task 5E / Task 10: the manuscript's tables and the tables of Additional file 1, as CSV (results/tables/).

Numbering follows the manuscript (Task 10). Two FF grids (decision #19):
  14-point grid (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15%): every main-grid detection limit (Table 2,
           Tables S4, S6, S7, S8), from detection_limits_refined.csv and the 14-point bootstrap CIs.
  8-point grid (the simulated grid FFs): per-condition results and every robustness-scenario limit (S3, S9, S10).

Table 1   table1_realism_yang.csv            realism check of the 40-156 bp window against Yang et al.
Table 2   table2_detection_limits.csv        detection limits of A, B, B-opt, W, C1, C2 per depth x noise, with the
                                              95% CI (test+ref for z-score methods, test for trees), the bold flag
                                              (lowest limit, or not clearly different from it) and the tree flag
                                              (nobody-positive condition at the lower end of the crossing interval)
Table S1  tableS1_bopt_cutoff.csv            B-opt chosen cutoff per condition
Table S2  tableS2_c2_importances.csv         C2 feature importances per condition
Table S3  tableS3_w_vs_wfixed_sens.csv       W vs W_fixedFF per condition (sens, sens_at_fpr, AUC)
Table S4  tableS4_w_vs_wfixed_limits.csv     W vs W_fixedFF detection limits
Table S5  tableS5_auc.csv                    AUC of every method and condition (the .tex table shows 7 main methods)
Table S6  tableS6_limits_conservative_fpr.csv  conservative (Wilson lower bound, no CI) and matched-FPR limits (CI)
Table S7  tableS7_limits_8_vs_14.csv         8-point vs 14-point limits per method x setting (interpolation check)
Table S8  tableS8_limit_differences.csv      method differences with CIs, clear / similar (main version, both bases)
Table S9  tableS9a_profile_inputs.csv, tableS9b_profile_b_vs_a.csv, tableS9c_profile_limits.csv,
          tableS9d_wg1_vs_a.csv              fetal-profile scenarios (Task 07), parts (a)-(d) of one table
Table S10 tableS10a_snoise_rise.csv, tableS10b_snoise_limits.csv, tableS10c_snoise_regimes.csv
                                              short-fragment noise (Task 08) and the four regimes (snoise_regimes.py),
                                              parts (a)-(c) of one table

Every table is built from committed CSVs (and, for Table 1, the length/mappability model); nothing is typed by
hand. captions() returns {label: (files, caption)}; captions describe what is shown and quote no results. docs/captions.md is
written from it by src.evaluate.captions, and the LaTeX tables by src.evaluate.latex_tables. Run from the repo root:
    python -m src.evaluate.tables
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names
from src.evaluate.run_condition import method_order
from src.simulate.patients import origin_matrices, window_stats

CONDITION = ["ff", "depth", "noise"]
REFINED_DIR = "results/refined_grid/tables"
TABLE2_METHODS = ["A", "B", "B-opt", "W", "C1", "C2"]
MAIN_METHODS = ["A", "B", "B-opt", "W", "W_fixedFF", "C1", "C2"]
S8_PAIRS = ["W-B", "W-A", "B-A", "W-C2", "C2-B", "B-opt-B", "W_fixedFF-W"]
PROFILE_METHODS = ["A", "B", "B-opt", "W", "W_G1", "W_fixedFF"]
SNOISE_METHODS = ["A", "B", "W"]
COMMERCIAL_MIN_FF = 0.04
TREES = ("C1", "C2")
FIG_FF = 0.04

GRID14 = "14-point FF grid (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15%)"
GRID8 = "8-point FF grid of the simulated conditions (1, 2, 3, 4, 6, 8, 10, 15%)"
CI_TEXT = ("95% CIs from a paired bootstrap (1,000 resamples): for the z-score methods the test patients and the "
           "2,000 reference normals are resampled (new reference mean and SD), for the trees only the test patients "
           "(trained trees and their thresholds held fixed); B-opt's chosen cutoff is held fixed throughout")

CAPTION_TABLE1 = (
    "Table 1. Realism check of the 40–156 bp window against Yang et al.: exact fraction of fragments kept "
    "($r$) and FF enrichment ($g$) from the post-mappability length model (see Methods)."
)
CAPTION_TABLE2 = (
    "Table 2. Detection limits (FF, %, at which sensitivity reaches 95% for the last time; see Methods) per depth "
    "and noise level, on the {grid}; every point is simulated and limits are interpolated linearly between "
    "neighbouring points. Limits use each method's own threshold ($z > 3$, or the tree's calibrated threshold). "
    "{ci_text}. Bold: the lowest limit per depth and noise level, and every limit whose difference from it is not "
    "clear (the 95% CI of the difference includes 0; same versions, own-threshold basis), so bold entries are "
    "similar. † limit ≤ 4% (commercial minimum). ‡ tree limit whose crossing interval starts at an FF where the "
    "tree flags nobody (no leaf meets the target FPR on the calibration normals, sensitivity 0), which pulls the "
    "limit towards the upper end of that interval. Matched-FPR and conservative limits in Table S6; the 8-point "
    "limits in Table S7; differences in Table S8."
)
CAPTION_BOPT = (
    "Table S1. B-opt: upper cutoff chosen per condition on the training set (grid 120–170 bp in 5 bp steps, "
    "lower bound 40 bp; see Methods), with its training separation, and B-opt vs B (40–156 bp) on the test set: "
    "standardised separation, sensitivity at FPR 0.135% and the difference B-opt − B (points)."
)
CAPTION_C2 = (
    r"Table S2. C2 feature importances (scikit-learn impurity importance) of $z_A$, $z_\text{{short}}$ and $z_W$, "
    "chosen max depth and whether the tree's threshold flags nobody, per condition."
)
CAPTION_W = (
    "Table S3. W (weights from the true FF) vs W (FF {fixed_ff:.0%}), W with the weights fixed at FF "
    "{fixed_ff:.0%}, per condition: sensitivity at the $z > 3$ threshold, sensitivity at FPR 0.135% (matched FPR) "
    "and AUC, and the difference W (FF {fixed_ff:.0%}) − W in sensitivity at FPR 0.135% (points)."
)
CAPTION_W_LIMITS = (
    "Table S4. Detection limits of W (weights from the true FF) and W (FF {fixed_ff:.0%}) (weights fixed at FF "
    "{fixed_ff:.0%}) per depth and noise level (own threshold and conservative, on the {grid} as in Table 2), and "
    "the difference W (FF {fixed_ff:.0%}) − W of the own-threshold limits (FF points)."
)
CAPTION_AUC = (
    "Table S5. AUC of the test ROC (10,000 euploid and 2,000 trisomic test patients per condition) per condition "
    "of the {grid8}. AUC values for the main methods; all cutoffs in the repository (tableS5_auc.csv)."
)
CAPTION_S6 = (
    "Table S6. Detection limits on the {grid}: conservative estimate from the Wilson 95% lower bound of each "
    "sensitivity (plain values, no CI), and the matched-FPR limit from the sensitivity at FPR 0.135% read from the "
    "test ROC, which compares all methods at the same FPR, with its 95% CI ({ci_short})."
)
CAPTION_S7 = (
    "Table S7. Interpolation check: detection limits on the {grid8} (as originally reported) and on the {grid} "
    "(Table 2), per method and setting, and the change (points)."
)
CAPTION_S8 = (
    "Table S8. Differences between detection limits (first method minus second, FF points) on the {grid}, with "
    "95% CIs: test + reference version for pairs of z-score methods, test version for pairs with a tree; "
    "own-threshold basis (each method at its own threshold, as in Table 2) and matched-FPR basis (all methods at "
    "FPR 0.135%, Table S6). Clear: the CI excludes 0; otherwise similar. For tree pairs the own-threshold basis "
    "compares methods at different achieved FPRs and is not used on its own to call a difference clear."
)
CAPTION_S9 = (
    "Table S9. Fetal-profile scenarios ({grid8}; all groups, including the reference, simulated from the "
    "scenario profile; see Methods): G1 (measured), the main analysis; fetal lengths shifted by −10, −5, +5 and "
    r"+10 bp; and a 50% maternal blend. (a) Fetal median length and $g$, $r$, $g\sqrt{{r}}$ of the 40–156 bp window "
    "at FF 2%, 4%, 10%. (b) Conditions (of 96) where B's separation is below A's, where B − A sensitivity at FPR "
    "0.135% is significantly negative (bootstrap), and where B-opt's separation is below A's, with B-opt's largest "
    "sensitivity loss. (c) Detection limits (own threshold) of A, B, B-opt, W (weights from the true profile), "
    "W (G1 weights) and W (FF 10%) (G1 weights at FF 10%); in G1 (measured), W (G1 weights) = W. "
    "(d) W (G1 weights) − A with 95% CIs (test + reference) in the +10 bp and 50% maternal blend scenarios, "
    "own-threshold and matched-FPR bases."
)
CAPTION_S10 = (
    "Table S10. Extra noise on the 40–156 bp window ($k = 1$: main analysis; {grid8}, low and realistic noise; see "
    r"Methods). (a) sep(B)/sep(A) at FF 4%, 1M and 10M, simulated and model ($\sigma_u$), and the model's "
    r"high-depth limit $g\sigma_A/\sigma_B$. (b) Detection limits of A, B and W. (c) The four regimes of each cutoff "
    r"(B 40–$c$: window 40–$c$ bp) by FF and $k$, from $g\sqrt{{r}}$ (low-depth ratio) and $g\sigma_A/\sigma_B$ "
    r"(high-depth ratio), with $\sigma_A^2 = \sigma^2 + \sigma_s^2 r_\mathrm{{win}}^2$ and "
    r"$\sigma_B^2 = \sigma^2 + \sigma_s^2 \rho_c^2$ ($r_\mathrm{{win}}$: fraction in the noisy window; $\rho_c = 1$ "
    r"for $c \le 156$, $r_\mathrm{{win}}/r_c$ above): (i) B always better, (ii) worse at low depth and better at "
    r"high, (iii) better at low depth and worse at high, (iv) always worse; $k_\mathrm{{high}}$: the $k$ where "
    r"$g\sigma_A = \sigma_B$ (regime (iii) begins if $g^2 r \ge 1$); $k^*$: the $k$ above which depth no longer "
    "raises the gain."
)


def grid_order(table, config, by):
    """Sort by config grid order of noise, then the other keys (methods in the Table 2 order)."""
    noise_rank = {n: i for i, n in enumerate(config["grid"]["noise_sigma"])}
    method_rank = {m: i for i, m in enumerate(TABLE2_METHODS)}
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


def ci_version(method):
    return "test" if method in TREES else "test+ref"


# ---------------------------------------------------------------- Table 1 and 2

def table_yang(config):
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


def diff_lookup(diffs, basis="sens"):
    """{(depth, noise, frozenset(pair)): (sign, row)} of the main-version rows of one basis."""
    d = diffs[diffs["primary"].astype(str) == "True"]
    d = d[d["basis"] == basis]
    out = {}
    for r in d.itertuples():
        a, b = split_pair(r.pair)
        out[(r.depth, r.noise, frozenset((a, b)))] = (a, r)
    return out


def split_pair(pair):
    """'B-opt-B' -> ('B-opt', 'B'); 'W_fixedFF-W' -> ('W_fixedFF', 'W')."""
    for m in sorted(MAIN_METHODS + ["W_G1"], key=len, reverse=True):
        if pair.startswith(m + "-") and pair[len(m) + 1:] in MAIN_METHODS + ["W_G1"]:
            return m, pair[len(m) + 1:]
    raise ValueError(f"cannot split pair {pair!r}")


def table_limits(limits, ci, diffs, config):
    """Table 2: the paper's limits (14-point grid) with CIs, the bold flag and the tree flag."""
    t = limits[limits["method"].isin(TABLE2_METHODS)].copy()
    t["main_le_4pct"] = [at_or_below(v, s, COMMERCIAL_MIN_FF) for v, s in zip(t["ff_limit"], t["ff_limit_text"])]
    t["cons_le_4pct"] = [at_or_below(v, s, COMMERCIAL_MIN_FF)
                         for v, s in zip(t["ff_limit_cons"], t["ff_limit_cons_text"])]
    c = ci[(ci["basis"] == "sens")].set_index(["method", "depth", "noise", "version"])
    t["ci_version"] = t["method"].map(ci_version)
    t["ci_lo"] = [c.loc[(m, d, n, v), "ci_lo"] for m, d, n, v in zip(t["method"], t["depth"], t["noise"], t["ci_version"])]
    t["ci_hi"] = [c.loc[(m, d, n, v), "ci_hi"] for m, d, n, v in zip(t["method"], t["depth"], t["noise"], t["ci_version"])]
    t["nobody_lower_end"] = t["lower_nobody_sens"].astype(str).eq("True")
    look = diff_lookup(diffs)
    lowest, bold = {}, []
    for (d, n), g in t.groupby(["depth", "noise"]):
        g = g[np.isfinite(g["ff_limit"])]
        lowest[(d, n)] = g.loc[g["ff_limit"].idxmin(), "method"]
    for r in t.itertuples():
        low = lowest[(r.depth, r.noise)]
        if r.method == low:
            bold.append(True)
            continue
        key = (r.depth, r.noise, frozenset((low, r.method)))
        if key not in look:
            raise ValueError(f"no CI for {low} vs {r.method} at {r.depth} {r.noise}; run limit_ci --refined")
        bold.append(not bool(look[key][1].clear))
    t["lowest_in_setting"] = [lowest[(d, n)] for d, n in zip(t["depth"], t["noise"])]
    t["bold"] = bold
    cols = ["noise", "depth", "method", "ff_limit", "ff_limit_text", "ff_limit_cons", "ff_limit_cons_text",
            "ff_limit_at_fpr", "ff_limit_at_fpr_text", "at_fpr_flag", "main_le_4pct", "cons_le_4pct",
            "ci_version", "ci_lo", "ci_hi", "nobody_lower_end", "lowest_in_setting", "bold"]
    return grid_order(t[cols], config, ["noise", "depth", "method"])


# ---------------------------------------------------------------- Tables S1-S4 (unchanged content)

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


# ---------------------------------------------------------------- Tables S5-S8 (main grid)

def auc_table(results, config):
    """Every method's AUC per condition (wide, methods in the results-table order)."""
    wide = results.pivot_table(index=CONDITION, columns="method", values="auc").reset_index()
    methods = [m for m in method_order(config) if m in wide.columns]
    return grid_order(wide[CONDITION + methods], config, ["noise", "depth", "ff"])


def conservative_fpr_table(limits, ci, config):
    t = limits[limits["method"].isin(TABLE2_METHODS)].copy()
    c = ci[ci["basis"] == "sens_at_fpr"].set_index(["method", "depth", "noise", "version"])
    t["ci_version"] = t["method"].map(ci_version)
    key = list(zip(t["method"], t["depth"], t["noise"], t["ci_version"]))
    t["at_fpr_ci_lo"] = [c.loc[k, "ci_lo"] for k in key]
    t["at_fpr_ci_hi"] = [c.loc[k, "ci_hi"] for k in key]
    cols = ["noise", "depth", "method", "ff_limit_cons", "ff_limit_cons_text", "ff_limit_at_fpr", "ff_limit_at_fpr_text",
            "ci_version", "at_fpr_ci_lo", "at_fpr_ci_hi"]
    return grid_order(t[cols], config, ["noise", "depth", "method"])


def interpolation_table(limits8, limits14, dips, config):
    a = limits8[limits8["method"].isin(MAIN_METHODS)].set_index(["method", "depth", "noise"])
    b = limits14[limits14["method"].isin(MAIN_METHODS)].set_index(["method", "depth", "noise"])
    out = pd.DataFrame({"limit_8": a["ff_limit"], "limit_8_text": a["ff_limit_text"],
                        "limit_14": b["ff_limit"], "limit_14_text": b["ff_limit_text"]})
    out["change_pts"] = 100 * (out["limit_14"] - out["limit_8"])
    flagged = {(r.method, r.depth, r.noise) for r in dips.itertuples()} if len(dips) else set()
    out["dip_flag"] = [k in flagged for k in out.index]
    out["nobody_lower_end_14"] = b["lower_nobody_sens"].astype(str).eq("True")
    out = out.reset_index()
    return grid_order(out[["noise", "depth", "method", "limit_8", "limit_8_text", "limit_14", "limit_14_text",
                           "change_pts", "dip_flag", "nobody_lower_end_14"]], config, ["noise", "depth", "method"])


def differences_table(diffs, config):
    d = diffs[(diffs["primary"].astype(str) == "True") & diffs["pair"].isin(S8_PAIRS)].copy()
    d["pair"] = pd.Categorical(d["pair"], S8_PAIRS, ordered=True)
    d = d.sort_values(["pair", "basis"])
    out = grid_order(d[["noise", "depth", "pair", "version", "basis", "diff", "ci_lo", "ci_hi", "share_numeric",
                        "clear"]], config, ["pair", "noise", "depth", "basis"])
    out["pair"] = out["pair"].astype(str)
    return out


# ---------------------------------------------------------------- Tables S9 and S10 (scenarios, 8-point grid)

def scenario_limits(name):
    path = repo_path("results/tables/detection_limits.csv") if name == "G1" \
        else repo_path(f"results/robustness/{name}/tables/detection_limits.csv")
    return pd.read_csv(path, dtype={"noise": str})


def profile_tables(config):
    tables = repo_path(config["paths"]["results"]) / "tables"
    inputs = pd.read_csv(tables / "robustness_profiles_inputs.csv")
    a = inputs[["profile", "shift_bp", "blend_maternal", "fetal_median_bp"]
               + [f"{v}_ff{f}" for f in (2, 4, 10) for v in ("g", "r", "g_sqrt_r")]]
    summ = pd.read_csv(tables / "robustness_profiles_summary.csv", dtype={"question": str, "depth": str, "noise": str})
    pick = {"n_sep_ratio_B_A_below_1": "n_B_below_A", "n_sens_diff_B_A_sig_negative": "n_B_minus_A_sig_negative",
            "n_sep_ratio_Bopt_A_below_1": "n_Bopt_below_A", "max_sens_at_fpr_loss_Bopt_minus_A_pts":
            "max_Bopt_sens_loss_pts"}
    s = summ[summ["item"].isin(pick)].pivot_table(index="profile", columns="item", values="value", aggfunc="first")
    s = s.rename(columns=pick).reindex(inputs["profile"]).reset_index()
    b = s[["profile"] + list(pick.values())]
    rows = []
    for name in ["G1"] + scenario_names(config, "fetal_profiles"):
        lim = scenario_limits(name)
        if name == "G1":   # W_G1 is W in the main analysis
            lim = pd.concat([lim, lim[lim["method"] == "W"].assign(method="W_G1")])
        lim = lim[lim["method"].isin(PROFILE_METHODS)]
        w = lim.pivot_table(index=["depth", "noise"], columns="method", values="ff_limit_text", aggfunc="first")
        rows.append(w[PROFILE_METHODS].reset_index().assign(profile=name))
    c = pd.concat(rows, ignore_index=True)
    c = grid_order(c[["profile", "noise", "depth"] + PROFILE_METHODS], config, ["noise", "depth"])
    c["profile"] = pd.Categorical(c["profile"], ["G1"] + scenario_names(config, "fetal_profiles"), ordered=True)
    c = c.sort_values("profile", kind="stable").reset_index(drop=True)
    c["profile"] = c["profile"].astype(str)
    diffs = pd.read_csv(tables / "limit_differences_ci.csv", dtype={"noise": str})
    d = diffs[(diffs["pair"] == "W_G1-A") & (diffs["version"] == "test+ref")]
    d = grid_order(d[["grid", "noise", "depth", "basis", "diff", "ci_lo", "ci_hi", "clear"]], config,
                   ["grid", "noise", "depth", "basis"])
    return a, b, c, d


def snoise_tables(config):
    tables = repo_path(config["paths"]["results"]) / "tables"
    names = {"main": 1.0, **{n: float(load_config(scenario=n)["short_noise"]["k"])
                             for n in scenario_names(config, "short_noise")}}
    levels = [n for n, s in zip(config["grid"]["noise_sigma"], config["grid"]["noise_sigma_values"]) if s > 0]
    pred = pd.read_csv(tables / "robustness_snoise_pred.csv", dtype={"noise": str})
    lo_d, hi_d = min(config["grid"]["depth"]), max(config["grid"]["depth"])
    rows, lrows = [], []
    for name, k in names.items():
        res = pd.read_csv((tables if name == "main" else repo_path(f"results/robustness/{name}/tables"))
                          / "results_all.csv", dtype={"noise": str})
        res["sep"] = (res["z_mean_t21"] - res["z_mean_normal"]) / res["z_sd_normal"]
        lim = scenario_limits("G1" if name == "main" else name)
        for noise in levels:
            obs = {}
            for d in (lo_d, hi_d):
                r = res[np.isclose(res["ff"], FIG_FF) & (res["depth"] == d) & (res["noise"] == noise)].set_index("method")
                obs[d] = r.loc["B", "sep"] / r.loc["A", "sep"]
            p = pred[(pred["scenario"] == name) & np.isclose(pred["ff"], FIG_FF) & (pred["noise"] == noise)
                     & (pred["method"] == "B")].set_index("depth")
            rows.append({"k": k, "scenario": name, "noise": noise, "obs_1M": obs[lo_d], "obs_10M": obs[hi_d],
                         "obs_change": obs[hi_d] / obs[lo_d] - 1, "model_1M": p.loc[lo_d, "ratio_pred"],
                         "model_10M": p.loc[hi_d, "ratio_pred"],
                         "model_change": p.loc[hi_d, "ratio_pred"] / p.loc[lo_d, "ratio_pred"] - 1,
                         "sigma_B_over_A": p.loc[lo_d, "sigma_B_over_A"], "high_depth_limit": p.loc[lo_d, "ratio_high_depth"]})
            for d in config["grid"]["depth"]:
                one = lim[(lim["depth"] == d) & (lim["noise"] == noise)].set_index("method")
                lrows.append({"k": k, "scenario": name, "noise": noise, "depth": d,
                              **{m: one.loc[m, "ff_limit_text"] for m in SNOISE_METHODS}})
    reg = pd.read_csv(tables / "snoise_regimes.csv")
    wide = reg.pivot_table(index=["cutoff", "ff"], columns="k", values=["regime", "g_sigmaA_over_sigmaB"],
                           aggfunc="first")
    wide.columns = [f"{v}_k{k:g}" for v, k in wide.columns]
    base = reg[reg["k"] == 1].set_index(["cutoff", "ff"])[["g_sqrt_r", "g2r", "k_high_equal", "k_star"]]
    c = base.join(wide).reset_index()
    ks = sorted(reg["k"].unique())
    c = c[["cutoff", "ff", "g_sqrt_r", "g2r"] + [f"regime_k{k:g}" for k in ks]
          + [f"g_sigmaA_over_sigmaB_k{k:g}" for k in ks if k != 1] + ["k_high_equal", "k_star"]]
    return pd.DataFrame(rows), pd.DataFrame(lrows), c


# ---------------------------------------------------------------- build and captions

def inputs(config):
    tables = repo_path(config["paths"]["results"]) / "tables"
    return {
        "results": pd.read_csv(tables / "results_all.csv"),
        "extra": pd.read_csv(repo_path(REFINED_DIR) / "results_all.csv"),
        "limits8": pd.read_csv(tables / "detection_limits.csv", dtype={"noise": str}),
        "limits": pd.read_csv(tables / "detection_limits_refined.csv", dtype={"noise": str}),
        "ci": pd.read_csv(tables / "detection_limits_ci_refined.csv", dtype={"noise": str}),
        "diffs": pd.read_csv(tables / "limit_differences_ci_refined.csv", dtype={"noise": str}),
        "dips": pd.read_csv(tables / "limits_dip_flags.csv", dtype={"noise": str}),
        "sep": pd.read_csv(tables / "bopt_separation.csv"),
        "imp": pd.read_csv(tables / "c2_feature_importances.csv"),
    }


def build_tables(config, data=None):
    """{file name: table} for every table this module writes."""
    x = inputs(config) if data is None else data
    w_per, w_lim = w_tables(x["results"], x["limits"], config)
    s9a, s9b, s9c, s9d = profile_tables(config)
    s10a, s10b, s10c = snoise_tables(config)
    return {
        "table1_realism_yang.csv": table_yang(config),
        "table2_detection_limits.csv": table_limits(x["limits"], x["ci"], x["diffs"], config),
        "tableS1_bopt_cutoff.csv": bopt_table(x["sep"], x["results"], config),
        "tableS2_c2_importances.csv": c2_table(x["imp"], x["results"], config),
        "tableS3_w_vs_wfixed_sens.csv": w_per,
        "tableS4_w_vs_wfixed_limits.csv": w_lim,
        "tableS5_auc.csv": auc_table(x["results"], config),
        "tableS6_limits_conservative_fpr.csv": conservative_fpr_table(x["limits"], x["ci"], config),
        "tableS7_limits_8_vs_14.csv": interpolation_table(x["limits8"], x["limits"], x["dips"], config),
        "tableS8_limit_differences.csv": differences_table(x["diffs"], config),
        "tableS9a_profile_inputs.csv": s9a, "tableS9b_profile_b_vs_a.csv": s9b,
        "tableS9c_profile_limits.csv": s9c, "tableS9d_wg1_vs_a.csv": s9d,
        "tableS10a_snoise_rise.csv": s10a, "tableS10b_snoise_limits.csv": s10b, "tableS10c_snoise_regimes.csv": s10c,
    }


LABELS = {   # manuscript label -> files (in order)
    "Table 1": ["table1_realism_yang.csv"],
    "Table 2": ["table2_detection_limits.csv"],
    "Table S1": ["tableS1_bopt_cutoff.csv"],
    "Table S2": ["tableS2_c2_importances.csv"],
    "Table S3": ["tableS3_w_vs_wfixed_sens.csv"],
    "Table S4": ["tableS4_w_vs_wfixed_limits.csv"],
    "Table S5": ["tableS5_auc.csv"],
    "Table S6": ["tableS6_limits_conservative_fpr.csv"],
    "Table S7": ["tableS7_limits_8_vs_14.csv"],
    "Table S8": ["tableS8_limit_differences.csv"],
    "Table S9": ["tableS9a_profile_inputs.csv", "tableS9b_profile_b_vs_a.csv", "tableS9c_profile_limits.csv",
                 "tableS9d_wg1_vs_a.csv"],
    "Table S10": ["tableS10a_snoise_rise.csv", "tableS10b_snoise_limits.csv", "tableS10c_snoise_regimes.csv"],
}


def captions(config):
    """{label: (files, caption)} in manuscript order. Captions describe what each table shows; they quote no
    results (those are in the tables and in docs/revision_numbers.md)."""
    ci_short = "test + reference for z-score methods, test for trees"
    fixed_ff = config["methods"]["weight_fixed_ff"]
    caps = {
        "Table 1": CAPTION_TABLE1,
        "Table 2": CAPTION_TABLE2.format(grid=GRID14, ci_text=CI_TEXT),
        "Table S1": CAPTION_BOPT,
        "Table S2": CAPTION_C2.format(),
        "Table S3": CAPTION_W.format(fixed_ff=fixed_ff),
        "Table S4": CAPTION_W_LIMITS.format(fixed_ff=fixed_ff, grid=GRID14),
        "Table S5": CAPTION_AUC.format(grid8=GRID8),
        "Table S6": CAPTION_S6.format(grid=GRID14, ci_short=ci_short),
        "Table S7": CAPTION_S7.format(grid8=GRID8, grid=GRID14),
        "Table S8": CAPTION_S8.format(grid=GRID14),
        "Table S9": CAPTION_S9.format(grid8=GRID8),
        "Table S10": CAPTION_S10.format(grid8=GRID8),
    }
    return {label: (LABELS[label], caps[label]) for label in LABELS}


def main():
    config = load_config()
    tables = repo_path(config["paths"]["results"]) / "tables"
    data = inputs(config)
    out = build_tables(config, data)
    for name, table in out.items():
        table.to_csv(tables / name, index=False, float_format="%.6g")
        print(f"Saved {tables / name} ({len(table)} rows)")
    for label, (files, text) in captions(config).items():
        print(f"\n{label} ({', '.join(files)}):\n{text}")


if __name__ == "__main__":
    main()
