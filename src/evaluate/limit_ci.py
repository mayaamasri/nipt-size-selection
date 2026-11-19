"""Task 09 (9A, 9B, 9D, 9E): bootstrap uncertainty of the detection limits and of their differences (decision #19).

Two paired bootstrap versions, B = uncertainty.n_boot resamples, percentile CI (uncertainty.ci_level):
  test      resample, per FF independently, the test normals (10,000) and test T21 (2,000) with replacement.
            Held fixed: the reference statistics (mean, SD of each statistic), the trained trees and their
            calibrated thresholds, and B-opt's cutoff. All methods: A, B, B-opt, W, W_fixedFF, C1, C2.
  test+ref  the same test resamples, plus the 2,000 reference normals with replacement (new mean and SD per
            statistic, test patients rescored, z > z_threshold). Held fixed: B-opt's cutoff and the threshold.
            z-score methods only (A, B, B-opt, W, W_fixedFF, W_G1 in fetal-profile scenarios); the trees are not
            retrained, so they stay in the test version.
Both versions condition on the trained models and thresholds: training and calibration variability are not
included. Every method is scored on the same resampled patients (paired), and the two versions share their test
resamples, so they are paired with each other too.

Per resample and setting (depth x noise): for every FF, sens (at the method's own threshold) and sens_at_fpr
(src.evaluate.metrics), then the detection limit with detection_limits.crossing (the paper's rule), from sens and
from sens_at_fpr, and the differences of the 9B pairs in the same resample. sens_at_fpr depends only on the
ranking within the test set, and the reference only shifts and scales z (SD > 0), so it is the same in both
versions; it is computed from the original reference's z, and the self-test checks the identity exactly on the
first CHECK_SAME resamples of every setting (all resamples with --check-all).

Which version and basis to read:
  z-score pairs (W-B, W-A, B-A, B-opt-B, W_fixedFF-W, W_G1-*): the main version is test+ref (primary = True);
            test is kept for comparison.
  tree pairs (W-C2, C2-B): test only, reported on both bases. On the sens basis each method sits at its own
            threshold, so the methods are compared at different achieved FPRs (pooled_test_fpr in
            detection_limits_ci.csv; e.g. C2's test FPR at FF 2%, 10M realistic was 0.23%), and a clear sens-basis
            difference can come from the tree's higher FPR rather than better detection. sens_at_fpr compares at
            the same FPR and is the fair comparison, but for the trees its CI is lumpy: their few leaf scores make
            sens_at_fpr jump when the number of resampled normals in the top leaf crosses the target FPR. A tree
            pair is not called clear from the sens basis alone; the summary marks tree pairs whose bases disagree.

Limits outside the grid keep their text ("< 1%", "> 15%"). share_numeric = share of resamples with a numeric
limit (for a difference: both limits numeric); a CI is given only if share_numeric >= uncertainty.min_share_numeric,
from the numeric resamples. A difference is clear when its CI excludes 0.

Seeds: family 5 (families 0-4 and 6 are taken): test resamples SeedSequence(seed, (5, i_grid, i_depth, i_noise, 0)),
reference resamples (5, i_grid, i_depth, i_noise, 1), i_grid = 0 for the main grid, 1 + the scenario's position in
src.config.scenario_names(config). Within a stream the draws go resample by resample and FF by FF, in grid order.

Cache: each condition's per-patient raw statistics (reference, test normals, test T21) and the trees' test
probabilities and exact thresholds are computed once with the functions run_condition uses (generate_group,
build_reference, statistics, zscores, zscore_methods, fit_tree_method) and saved under
data/processed/limit_ci_cache/<grid>/ (gitignored, ~0.75 MB per condition). Every time a condition is loaded, each
method's sens, fpr, auc and sens_at_fpr are recomputed from the cache and compared with results_all.csv as written
(%.10g); any difference raises. The self-test also recomputes the paper's limits with no resampling.

Writes results/tables/detection_limits_ci.csv (grid, method, depth, noise, version, basis, limit, limit_text,
ci_lo, ci_hi, share_numeric, pooled_test_fpr: the method's test FPR pooled over the 8 FF conditions of the
setting, from results_all.csv, not resampled) and limit_differences_ci.csv (grid, pair, depth, noise, version,
basis, diff, ci_lo, ci_hi, share_numeric, clear, primary), and prints per pair and basis how many of the 12 settings are clear, and in which
direction. Run from the repo root:
    python -m src.evaluate.limit_ci                                     everything (~10 min with --jobs 6)
    python -m src.evaluate.limit_ci --setting 10000000 realistic        one main-grid setting, printed only (gate)
    python -m src.evaluate.limit_ci --cache-only                        build and verify the caches
    python -m src.evaluate.limit_ci --refined --jobs 6                  main grid on the 14-point FF grid (Task 09
        follow-up, decision #19): extra FFs from results/refined_grid, limits checked against
        detection_limits_refined.csv; the 8 grid FFs keep their resamples (streams 0/1), the extra FFs use streams
        2/3, and the same resamples on the 8 grid FFs must reproduce the 8-FF CIs exactly (self-test). Writes
        detection_limits_ci_refined.csv (plus lower_nobody_positive: for tree limits on the sens basis, whether the
        lower end of the crossing interval is a nobody_positive condition) and limit_differences_ci_refined.csv.
"""

import argparse
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.config import load_config, repo_path, scenario_names, scenario_noise_levels
from src.evaluate.detection_limits import crossing
from src.evaluate.metrics import evaluate, sens_at_fpr
from src.methods.reference import build_reference, g1_matrices, statistics, zscores
from src.methods.trees import fit_tree_method
from src.methods.zscore import window_name, zscore_methods
from src.simulate.patients import GROUPS, _grid_index, generate_group, origin_matrices

MAIN = "main"
Z_STAT = {"A": "frac_all", "W": "weighted", "W_fixedFF": "weighted_fixedff", "W_G1": "weighted_g1"}  # + B, B-opt
TREES = ("C1", "C2")
PAIRS = {MAIN: [("W", "B"), ("W", "C2"), ("W", "A"), ("B", "A"), ("C2", "B"), ("B-opt", "B"), ("W_fixedFF", "W")],
         "scenario": [("W_G1", "A"), ("W_G1", "B"), ("W_G1", "W")]}
VERSIONS = ("test", "test+ref")
BASES = ("sens", "sens_at_fpr")
CHECK_SAME = 20          # resamples per setting where sens_at_fpr is recomputed from the new reference and compared
COMPARE = ("sens", "fpr", "auc", "sens_at_fpr")
CACHE_DIR = "data/processed/limit_ci_cache"
REFINED_DIR = "results/refined_grid/tables"      # extra-FF conditions of the 14-point main-grid limits


# ---------------------------------------------------------------- grids, methods, seeds

def grid_config(grid):
    return load_config() if grid == MAIN else load_config(scenario=grid)


def grid_index(config, grid):
    return 0 if grid == MAIN else 1 + scenario_names(config).index(grid)


def z_methods(grid):
    return ["A", "B", "B-opt", "W", "W_fixedFF"] + ([] if grid == MAIN else ["W_G1"])


def versions_of(grid):
    return VERSIONS if grid == MAIN else ("test+ref",)


def methods_of(grid, version):
    return z_methods(grid) + (list(TREES) if version == "test" else [])


def pair_versions(grid, pair):
    if grid != MAIN:
        return ("test+ref",)
    return ("test",) if "C2" in pair else VERSIONS


def stream(config, grid, depth, noise, which):
    g = config["grid"]
    key = (5, grid_index(config, grid), _grid_index(g["depth"], depth, "depth"),
           _grid_index(g["noise_sigma"], noise, "noise_sigma"), which)
    return np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=key))


# ---------------------------------------------------------------- cache

def cache_path(grid, ff, depth, noise):
    return repo_path(CACHE_DIR) / grid / f"ff{ff:g}_d{depth}_{noise}.npz"


def saved_rows(results, ff, depth, noise):
    same = (np.isclose(results["ff"].astype(float), ff, rtol=1e-9, atol=0) & (results["depth"].astype(int) == depth)
            & (results["noise"] == str(noise)))
    return results[same].set_index("method")


def stat_names(config, grid, cutoff):
    names = {m: s for m, s in Z_STAT.items() if m in z_methods(grid)}
    return {**names, "B": window_name(config), "B-opt": window_name(config, cutoff)}


def build_cache(grid, ff, depth, noise, cutoff):
    """Regenerate one condition with the grid seeds and save the per-patient statistics (and tree scores)."""
    config = grid_config(grid)
    matrices = origin_matrices(config)
    m_g1 = g1_matrices(config, matrices)
    ref_x = generate_group(config, ff, depth, noise, "reference", matrices)
    reference = build_reference(ref_x, ff, config, matrices, m_g1)
    raw = {"reference": statistics(ref_x, ff, config, matrices, m_g1)}
    groups = [g for g in GROUPS if g != "reference"] if grid == MAIN else ["test_normal", "test_t21"]
    for g in groups:
        raw[g] = statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices, m_g1)
    out = {"cutoff": cutoff}
    for m, name in stat_names(config, grid, cutoff).items():
        for g, key in (("reference", "ref"), ("test_normal", "tn"), ("test_t21", "tt")):
            out[f"{key}_{m}"] = raw[g][name]
    if grid == MAIN:   # trees and B-opt's choice, exactly as in run_condition
        z = {g: zscores(raw[g], reference) for g in groups}
        test_z = {k: np.concatenate([z["test_normal"][k], z["test_t21"][k]]) for k in z["test_normal"]}
        methods, _ = zscore_methods(test_z, z["train_normal"], z["train_t21"], config)
        if methods["B-opt"]["cutoff"] != cutoff:
            raise ValueError(f"{grid} FF {ff} {depth} {noise}: B-opt chose {methods['B-opt']['cutoff']}, "
                             f"results_all.csv has {cutoff}")
        for m in TREES:
            t = fit_tree_method(m, z["train_normal"], z["train_t21"], z["calibration"], test_z, config)
            out[f"score_{m}"], out[f"threshold_{m}"] = t["scores"], t["threshold"]
    path = cache_path(grid, ff, depth, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def load_condition(grid, ff, depth, noise, results, config):
    """Cached condition, verified against results_all.csv: dict with y, per-method original z (or tree scores),
    raw test and reference statistics, the threshold, and the cutoff."""
    rows = saved_rows(results, ff, depth, noise)
    cutoff = int(rows.loc["B-opt", "cutoff"])
    path = cache_path(grid, ff, depth, noise)
    if not path.exists():
        build_cache(grid, ff, depth, noise, cutoff)
    c = dict(np.load(path))
    if int(c["cutoff"]) != cutoff:
        raise ValueError(f"cache {path} has B-opt cutoff {int(c['cutoff'])}, results_all.csv {cutoff}")
    t = config["methods"]["z_threshold"]
    n_n, n_t = len(c["tn_A"]), len(c["tt_A"])
    y = np.r_[np.zeros(n_n, int), np.ones(n_t, int)]
    cond = {"y": y, "n_normal": n_n, "n_t21": n_t, "n_ref": len(c["ref_A"]), "z": {}, "raw": {}, "ref": {}, "thr": {}}
    for m in z_methods(grid):
        ref = c[f"ref_{m}"]
        mean, sd = float(ref.mean()), float(ref.std(ddof=1))      # as build_reference
        x = np.concatenate([c[f"tn_{m}"], c[f"tt_{m}"]])
        cond["raw"][m], cond["ref"][m] = x, ref
        cond["z"][m], cond["thr"][m] = (x - mean) / sd, t        # as zscores; calls are z > t
    if grid == MAIN:
        for m in TREES:
            cond["z"][m], cond["thr"][m] = c[f"score_{m}"], float(c[f"threshold_{m}"])   # calls are p >= t
    # verify every method against results_all.csv as written
    target = config["methods"]["target_fpr"]
    for m, s in cond["z"].items():
        calls = s >= cond["thr"][m] if m in TREES else s > cond["thr"][m]
        got = evaluate(y, s, calls, target)
        for k in COMPARE:
            if f"{got[k]:.10g}" != rows.loc[m, k]:
                raise ValueError(f"cache {path}: {m} {k} = {got[k]:.10g}, results_all.csv has {rows.loc[m, k]}")
    return cond


# ---------------------------------------------------------------- bootstrap

def sens_of(scores_t21, threshold, tree):
    return float(np.mean(scores_t21 >= threshold if tree else scores_t21 > threshold))


def limits_from(ffs, sens_by_ff, target):
    """crossing() per resample: arrays (numeric limit or NaN, text)."""
    out = [crossing(ffs, s, target) for s in sens_by_ff]
    return np.array([v for v, _ in out]), np.array([t for _, t in out], dtype=object)


def bootstrap_setting(args):
    """All resamples of one (grid, depth, noise): per version, method and basis, the limit of every resample.

    refined (main grid only, Task 09 follow-up): the limits use the 14-point FF grid. The 8 grid FFs draw from the
    same streams in the same order as without refined (so they get exactly the same resamples, and limits8 below
    reproduces the 8-FF run); the extra FFs draw from their own streams (5, 0, i_depth, i_noise, 2 / 3)."""
    grid, depth, noise, n_boot, check_all, refined = args
    config = grid_config(grid)
    tables = repo_path(config["paths"]["results"]) / "tables"
    results = pd.read_csv(tables / "results_all.csv", dtype=str, keep_default_na=False)
    ffs8 = sorted(config["grid"]["fetal_fraction"])
    extra_ffs = sorted(config["evaluation"]["limit_extra_ff"]) if refined else []
    ffs = sorted(ffs8 + extra_ffs)
    col = {ff: i for i, ff in enumerate(ffs)}
    target, target_sens = config["methods"]["target_fpr"], config["evaluation"]["sensitivity_target"]
    conds = [load_condition(grid, ff, depth, noise, results, config) for ff in ffs8]
    if refined:
        extra_results = pd.read_csv(repo_path(REFINED_DIR) / "results_all.csv", dtype=str, keep_default_na=False)
        conds_x = [load_condition(grid, ff, depth, noise, extra_results, config) for ff in extra_ffs]
    rng_test, rng_ref = stream(config, grid, depth, noise, 0), stream(config, grid, depth, noise, 1)
    rng_test_x, rng_ref_x = stream(config, grid, depth, noise, 2), stream(config, grid, depth, noise, 3)
    versions = versions_of(grid)
    methods = {v: methods_of(grid, v) for v in versions}
    scored = methods["test" if "test" in versions else "test+ref"]
    # sens[version][method][basis]: array (n_boot, n_ff) in ffs order
    sens = {v: {m: {b: np.empty((n_boot, len(ffs))) for b in BASES} for m in methods[v]} for v in versions}
    identity = {m: [None] * len(ffs) for m in scored}
    n_same = 0
    pairs_ff = [(ff, c) for ff, c in zip(ffs8, conds)] + ([(ff, c) for ff, c in zip(extra_ffs, conds_x)] if refined else [])
    for ff, cond in pairs_ff:   # self-test: no resampling reproduces the paper's inputs
        y = cond["y"]
        for m in scored:
            # rounded as results_all.csv stores them (%.10g), which is what detection_limits read
            identity[m][col[ff]] = (float(f"{sens_of(cond['z'][m][y == 1], cond['thr'][m], m in TREES):.10g}"),
                                    float(f"{sens_at_fpr(y, cond['z'][m], target):.10g}"))

    def score(b, j, cond, rt, rr):
        nonlocal n_same
        idx = np.r_[rt.integers(0, cond["n_normal"], cond["n_normal"]),
                    cond["n_normal"] + rt.integers(0, cond["n_t21"], cond["n_t21"])]
        idx_ref = rr.integers(0, cond["n_ref"], cond["n_ref"]) if "test+ref" in versions else None
        y = cond["y"][idx]
        for m in scored:
            s = cond["z"][m][idx]
            sat = sens_at_fpr(y, s, target)
            if "test" in versions:
                sens["test"][m]["sens"][b, j] = sens_of(s[y == 1], cond["thr"][m], m in TREES)
                sens["test"][m]["sens_at_fpr"][b, j] = sat
            if "test+ref" in versions and m not in TREES:
                ref = cond["ref"][m][idx_ref]
                z = (cond["raw"][m][idx] - float(ref.mean())) / float(ref.std(ddof=1))
                sens["test+ref"][m]["sens"][b, j] = sens_of(z[y == 1], cond["thr"][m], False)
                sens["test+ref"][m]["sens_at_fpr"][b, j] = sat
                if b < CHECK_SAME or check_all:
                    n_same += 1
                    if sens_at_fpr(y, z, target) != sat:
                        raise ValueError(f"self-test: {grid} {depth} {noise} FF {ffs[j]} {m} resample {b}: "
                                         f"sens_at_fpr differs between the original and the resampled reference")

    for b in range(n_boot):
        for ff, cond in zip(ffs8, conds):             # same order and streams as the 8-FF run
            score(b, col[ff], cond, rng_test, rng_ref)
        for ff, cond in (zip(extra_ffs, conds_x) if refined else ()):
            score(b, col[ff], cond, rng_test_x, rng_ref_x)
    limits = {v: {m: {bs: limits_from(ffs, sens[v][m][bs], target_sens) for bs in BASES} for m in methods[v]}
              for v in versions}
    out = {"grid": grid, "depth": depth, "noise": noise, "refined": refined, "limits": limits, "n_same": n_same,
           "identity": {m: {"sens": crossing(ffs, [s for s, _ in identity[m]], target_sens),
                            "sens_at_fpr": crossing(ffs, [a for _, a in identity[m]], target_sens)} for m in identity}}
    if refined:   # the same resamples on the 8 grid FFs only, to check against the 8-FF run
        cols8 = [col[ff] for ff in ffs8]
        out["limits8"] = {v: {m: {bs: limits_from(ffs8, sens[v][m][bs][:, cols8], target_sens) for bs in BASES}
                              for m in methods[v]} for v in versions}
    return out


# ---------------------------------------------------------------- summaries

def ci(values, level, min_share):
    """(share numeric, lo, hi): percentile CI of the numeric values, only if their share >= min_share."""
    finite = values[np.isfinite(values)]
    share = len(finite) / len(values)
    if share < min_share or not len(finite):
        return share, math.nan, math.nan
    a = (1 - level) / 2
    lo, hi = np.percentile(finite, [100 * a, 100 * (1 - a)])
    return share, float(lo), float(hi)


def paper_limits(config, refined=False):
    """The paper's limits of one grid (its detection_limits.csv, or detection_limits_refined.csv on the 14-point
    grid), per basis: columns ff_limit, ff_limit_text."""
    name = "detection_limits_refined.csv" if refined else "detection_limits.csv"
    lim = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / name, dtype={"noise": str})
    lim = lim.set_index(["method", "depth", "noise"])
    return {"sens": lim[["ff_limit", "ff_limit_text"]], "sens_at_fpr":
            lim[["ff_limit_at_fpr", "ff_limit_at_fpr_text"]].set_axis(["ff_limit", "ff_limit_text"], axis=1)}


def check_identity(res, config):
    """Self-test: with no resampling, the limits equal detection_limits.csv (both bases, every method)."""
    paper = paper_limits(config, res["refined"])
    worst = 0
    for m, by in res["identity"].items():
        for basis, (v, text) in by.items():
            row = paper[basis].loc[(m, res["depth"], res["noise"])]
            same = text == row["ff_limit_text"] and (math.isnan(v) and math.isnan(row["ff_limit"])
                                                     or f"{v:.10g}" == f"{row['ff_limit']:.10g}")
            if not same:
                raise ValueError(f"self-test: {res['grid']} {res['depth']} {res['noise']} {m} {basis}: no-resampling "
                                 f"limit {text} != detection_limits.csv {row['ff_limit_text']}")
            worst += 1
    return worst


def is_tree_pair(pair):
    return any(m in TREES for m in pair)


def pooled_fpr(results, depth, noise):
    """method -> test FPR pooled over the setting's FF conditions (fpr x test normals, from results_all.csv)."""
    r = results[(results["depth"].astype(int) == depth) & (results["noise"] == noise)]
    return r.groupby("method")["fpr"].apply(lambda f: float(f.astype(float).mean())).to_dict()   # equal n per FF


def tables_for(res, config):
    u = config["uncertainty"]
    cfg = grid_config(res["grid"])
    paper = paper_limits(cfg, res["refined"])
    results = pd.read_csv(repo_path(cfg["paths"]["results"]) / "tables" / "results_all.csv", dtype=str,
                          keep_default_na=False)
    if res["refined"]:   # pooled FPR over all 14 FFs
        results = pd.concat([results, pd.read_csv(repo_path(REFINED_DIR) / "results_all.csv", dtype=str,
                                                  keep_default_na=False)], ignore_index=True)
    fpr = pooled_fpr(results, res["depth"], res["noise"])
    nobody = (pd.read_csv(repo_path(cfg["paths"]["results"]) / "tables" / "detection_limits_refined.csv",
                          dtype={"noise": str}).set_index(["method", "depth", "noise"])["lower_nobody_sens"]
              if res["refined"] else None)
    rows, drows = [], []
    key = {"grid": res["grid"], "depth": res["depth"], "noise": res["noise"]}
    for v, by_m in res["limits"].items():
        for m, by_b in by_m.items():
            for basis, (vals, _) in by_b.items():
                p = paper[basis].loc[(m, res["depth"], res["noise"])]
                share, lo, hi = ci(vals, u["ci_level"], u["min_share_numeric"])
                rows.append({"grid": res["grid"], "method": m, "depth": res["depth"], "noise": res["noise"],
                             "version": v, "basis": basis, "limit": p["ff_limit"], "limit_text": p["ff_limit_text"],
                             "ci_lo": lo, "ci_hi": hi, "share_numeric": share, "pooled_test_fpr": fpr[m],
                             **({"lower_nobody_positive": nobody.loc[(m, res["depth"], res["noise"])]
                                 if basis == "sens" else pd.NA} if res["refined"] else {})})
    for a, b in PAIRS[MAIN if res["grid"] == MAIN else "scenario"]:
        for v in pair_versions(res["grid"], (a, b)):
            for basis in BASES:
                la, lb = res["limits"][v][a][basis][0], res["limits"][v][b][basis][0]
                share, lo, hi = ci(la - lb, u["ci_level"], u["min_share_numeric"])
                pa = paper[basis].loc[(a, res["depth"], res["noise"]), "ff_limit"]
                pb = paper[basis].loc[(b, res["depth"], res["noise"]), "ff_limit"]
                drows.append({**key, "pair": f"{a}-{b}", "version": v, "basis": basis, "diff": pa - pb,
                              "ci_lo": lo, "ci_hi": hi, "share_numeric": share,
                              "clear": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                              "primary": v == ("test" if is_tree_pair((a, b)) else "test+ref")})
    return rows, drows


LIMIT_COLS = ["grid", "method", "depth", "noise", "version", "basis", "limit", "limit_text", "ci_lo", "ci_hi",
              "share_numeric", "pooled_test_fpr"]
DIFF_COLS = ["grid", "pair", "depth", "noise", "version", "basis", "diff", "ci_lo", "ci_hi", "share_numeric", "clear",
             "primary"]


def check_task09(res, config):
    """Self-test (refined): the same resamples restricted to the 8 grid FFs reproduce the 8-FF CI tables
    (detection_limits_ci.csv, limit_differences_ci.csv) exactly, as written."""
    rows, drows = tables_for({**res, "refined": False, "limits": res["limits8"]}, config)
    tables = repo_path(config["paths"]["results"]) / "tables"
    for got, name, key in ((rows, "detection_limits_ci.csv", ["method", "version", "basis"]),
                           (drows, "limit_differences_ci.csv", ["pair", "version", "basis"])):
        saved = pd.read_csv(tables / name, dtype=str, keep_default_na=False)
        saved = saved[(saved["grid"] == res["grid"]) & (saved["depth"] == str(res["depth"]))
                      & (saved["noise"] == res["noise"])].set_index(key)
        for r in got:
            row = saved.loc[tuple(r[k] for k in key)]
            for c in ("ci_lo", "ci_hi", "share_numeric"):
                want, have = row[c], ("" if pd.isna(r[c]) else f"{r[c]:.6g}")
                if want != have:
                    raise ValueError(f"self-test: refined run on the 8 grid FFs, {name} {r} {c}: {have} != {want}")
    print(f"self-test {res['grid']} {res['depth']:,} {res['noise']}: same resamples on the 8 grid FFs reproduce "
          f"the 8-FF CIs exactly ({len(rows)} limits, {len(drows)} differences)")


def self_test_ci():
    """The share/CI rule on hand-made values."""
    v = np.r_[np.arange(1, 101) / 100.0]
    share, lo, hi = ci(v, 0.95, 0.95)
    assert share == 1 and math.isclose(lo, np.percentile(v, 2.5)) and math.isclose(hi, np.percentile(v, 97.5))
    v[:6] = np.nan   # 94% numeric -> no CI
    share, lo, hi = ci(v, 0.95, 0.95)
    assert math.isclose(share, 0.94) and math.isnan(lo) and math.isnan(hi)
    v[5] = 0.5       # 95% numeric -> CI from the numeric 95
    share, lo, hi = ci(v, 0.95, 0.95)
    assert math.isclose(share, 0.95) and math.isclose(lo, np.percentile(v[np.isfinite(v)], 2.5))
    print("self-test CI rule (100% / 94% / 95% numeric): ok")


def settings(config, grid):
    cfg = grid_config(grid)
    return [(d, str(n)) for n in scenario_noise_levels(cfg) for d in cfg["grid"]["depth"]]


def direction(row):
    return 0 if not row["clear"] else (-1 if row["ci_hi"] < 0 else 1)


def summary(diffs):
    print("\nPer pair and basis: settings with a clear difference (CI excludes 0), by direction "
          "(* = main version: test+ref for z-score pairs, test for tree pairs)")
    for (grid, pair, version, basis), g in diffs.groupby(["grid", "pair", "version", "basis"], sort=False):
        clear = g[g["clear"]]
        neg, pos = int((clear["diff"] < 0).sum()), int((clear["diff"] > 0).sum())
        first = pair.split("-")[0]
        mark = "*" if g["primary"].iloc[0] else " "
        print(f" {mark}{grid:>9} {pair:>12} {version:>8} {basis:>11}: {len(clear):2d}/{len(g)} clear "
              f"({neg} {first} lower, {pos} {first} higher), {int(g['share_numeric'].lt(1).sum())} with non-numeric")
    trees = diffs[diffs["pair"].map(lambda p: is_tree_pair(tuple(p.split("-"))))]
    print("\nTree pairs: settings where the sens and sens_at_fpr bases disagree (clear on one, or opposite "
          "directions). The sens basis compares at different FPRs; do not call a tree pair clear from it alone:")
    n = 0
    for (grid, pair, depth, noise), g in trees.groupby(["grid", "pair", "depth", "noise"], sort=False):
        by = {r["basis"]: r for _, r in g.iterrows()}
        if direction(by["sens"]) != direction(by["sens_at_fpr"]):
            n += 1
            print(f"  {pair:>6} {depth / 1e6:>4g}M {noise:>9}: sens {100 * by['sens']['diff']:+.2f} "
                  f"[{100 * by['sens']['ci_lo']:+.2f}, {100 * by['sens']['ci_hi']:+.2f}] vs sens_at_fpr "
                  f"{100 * by['sens_at_fpr']['diff']:+.2f} [{100 * by['sens_at_fpr']['ci_lo']:+.2f}, "
                  f"{100 * by['sens_at_fpr']['ci_hi']:+.2f}]")
    print(f"  {n} of {trees.groupby(['grid', 'pair', 'depth', 'noise']).ngroups} tree-pair settings disagree")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", nargs=2, metavar=("DEPTH", "NOISE"), help="one main-grid setting, printed only")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--check-all", action="store_true", help="check sens_at_fpr identity on every resample")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=None, help="override uncertainty.n_boot (quick tests only)")
    parser.add_argument("--refined", action="store_true",
                        help="main grid on the 14-point FF grid (writes *_ci_refined.csv; Task 09 follow-up)")
    args = parser.parse_args()
    config = load_config()
    u = config["uncertainty"]
    if args.n_boot is not None:
        u["n_boot"] = args.n_boot
    self_test_ci()
    start = time.time()

    grids = [MAIN] if args.refined else [MAIN] + list(u["scenarios"])
    if args.setting:
        jobs = [(MAIN, int(args.setting[0]), args.setting[1], u["n_boot"], args.check_all, args.refined)]
    else:
        jobs = [(g, d, n, u["n_boot"], args.check_all, args.refined) for g in grids for d, n in settings(config, g)]
    if args.cache_only:
        for g, d, n, _, _, refined in jobs:
            cfg = grid_config(g)
            results = pd.read_csv(repo_path(cfg["paths"]["results"]) / "tables" / "results_all.csv", dtype=str,
                                  keep_default_na=False)
            for ff in cfg["grid"]["fetal_fraction"]:
                load_condition(g, ff, d, n, results, cfg)
            if refined:
                extra = pd.read_csv(repo_path(REFINED_DIR) / "results_all.csv", dtype=str, keep_default_na=False)
                for ff in cfg["evaluation"]["limit_extra_ff"]:
                    load_condition(g, ff, d, n, extra, cfg)
            print(f"cache ok: {g} {d:,} {n} ({time.time() - start:.0f} s)")
        return

    with Pool(args.jobs) as pool:
        out = pool.map(bootstrap_setting, jobs) if args.jobs > 1 else [bootstrap_setting(j) for j in jobs]
    rows, drows = [], []
    for res in out:
        n_ident = check_identity(res, grid_config(res["grid"]))
        print(f"self-test {res['grid']} {res['depth']:,} {res['noise']}: no-resampling limits == "
              f"{'detection_limits_refined.csv' if res['refined'] else 'detection_limits.csv'} "
              f"({n_ident} method x basis) ok; sens_at_fpr identical with the resampled reference in {res['n_same']} "
              f"method x FF x resample checks")
        if res["refined"]:
            check_task09(res, config)
        r, d = tables_for(res, config)
        rows += r
        drows += d
    limit_cols = LIMIT_COLS + (["lower_nobody_positive"] if args.refined else [])
    limits, diffs = pd.DataFrame(rows, columns=limit_cols), pd.DataFrame(drows, columns=DIFF_COLS)
    elapsed = time.time() - start

    if args.setting:
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            show = limits.assign(**{c: 100 * limits[c] for c in ("limit", "ci_lo", "ci_hi")})
            print("\nLimits (FF %) with 95% CI:")
            print(show.drop(columns=["grid", "depth", "noise", "limit"]).to_string(index=False,
                                                                                     float_format=lambda v: f"{v:.2f}"))
            showd = diffs.assign(**{c: 100 * diffs[c] for c in ("diff", "ci_lo", "ci_hi")})
            print("\nDifferences (points) with 95% CI:")
            print(showd.drop(columns=["grid", "depth", "noise"]).to_string(index=False,
                                                                           float_format=lambda v: f"{v:+.3f}"))
        print(f"\nRun time {elapsed:.0f} s ({u['n_boot']} resamples)")
        return
    tables = repo_path(config["paths"]["results"]) / "tables"
    tag = "_refined" if args.refined else ""
    limits.to_csv(tables / f"detection_limits_ci{tag}.csv", index=False, float_format="%.6g")
    diffs.to_csv(tables / f"limit_differences_ci{tag}.csv", index=False, float_format="%.6g")
    summary(diffs)
    print(f"\nSaved detection_limits_ci{tag}.csv ({len(limits)} rows) and limit_differences_ci{tag}.csv "
          f"({len(diffs)} rows); {elapsed:.0f} s")


if __name__ == "__main__":
    main()
