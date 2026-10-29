"""Task 3E/3F: run one condition end to end and write its results.

Generates each group of the condition once (reference, train_normal, train_t21, test_normal, test_t21,
calibration), runs A, B, B-opt (src.methods.zscore) and C1, C2 (src.methods.trees), evaluates them on the
test set (src.evaluate.metrics) and writes, under results/tables/:
  results_all.csv              one row per condition x method
  bopt_separation.csv          B-opt separation per upper cutoff, on the training set
  c2_feature_importances.csv   C2 feature importances
  tree_cv_auc.csv              mean CV AUC per max_depth, C1 and C2
  run_log.csv                  git commit, timestamp and dirty flag of the run behind each condition's rows
Running a condition again replaces its earlier rows in every table (never duplicates them). Tables are
sorted by ff, depth, noise (in config grid order), then their own columns. run_log.csv is written last.
Prints the milestone sanity checks (brief 3F) at the end.

Run from the repo root, e.g. the milestone condition:
    python -m src.evaluate.run_condition --ff 0.04 --depth 5000000 --noise none
"""

import argparse
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config, repo_path
from src.evaluate.metrics import evaluate, g_and_r
from src.methods.features import TREE_FEATURES
from src.methods.reference import build_reference, statistics, zscores
from src.methods.trees import fit_tree_method
from src.methods.zscore import zscore_methods
from src.simulate.patients import GROUPS, generate_group, origin_matrices

CONDITION = ["ff", "depth", "noise"]
METHODS = ["A", "B", "B-opt", "C1", "C2"]
RESULT_COLUMNS = CONDITION + ["method", "sens", "sens_lo", "sens_hi", "fpr", "fpr_lo", "fpr_hi", "auc",
                              "sens_at_fpr", "g", "r", "cutoff", "threshold", "max_depth", "calib_fpr",
                              "nobody_positive"]
# Per table: columns to sort by after the condition, and fixed orders for text columns
SORT = {
    "results_all.csv": (["method"], {"method": METHODS}),
    "bopt_separation.csv": (["upper_bp"], {}),
    "c2_feature_importances.csv": (["feature"], {"feature": TREE_FEATURES["C2"]}),
    "tree_cv_auc.csv": (["method", "max_depth"], {"method": list(TREE_FEATURES)}),
    "run_log.csv": ([], {}),
}


def parse_noise(text):
    """Noise level from the command line: a number if it parses as one, else its name (e.g. none)."""
    try:
        return float(text)
    except ValueError:
        return text


def git_state():
    """Commit hash of HEAD and whether code or config differ from it (changes under results/ are ignored,
    since this script writes there)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)results").strip())
    return commit, dirty


def run_condition(config, ff, depth, noise):
    """All tables of one condition as DataFrames, plus the test normals' zA for the sanity checks."""
    matrices = origin_matrices(config)
    target = config["methods"]["target_fpr"]

    # Each group is generated once; raw counts are dropped after their statistics are computed
    reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config, matrices)
    z = {g: zscores(statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices), reference)
         for g in GROUPS if g != "reference"}
    test_z = {k: np.concatenate([z["test_normal"][k], z["test_t21"][k]]) for k in z["test_normal"]}
    y = np.r_[np.zeros(len(z["test_normal"]["frac_all"]), int), np.ones(len(z["test_t21"]["frac_all"]), int)]

    z_methods, sep = zscore_methods(test_z, z["train_normal"], z["train_t21"], config)
    trees = {m: fit_tree_method(m, z["train_normal"], z["train_t21"], z["calibration"], test_z, config)
             for m in TREE_FEATURES}
    g, r = g_and_r(matrices, ff, config)

    cond = {"ff": ff, "depth": depth, "noise": str(noise)}
    rows = []
    for method, res in {**z_methods, **trees}.items():
        rows.append({**cond, "method": method, **evaluate(y, res["scores"], res["calls"], target),
                     "g": g if method == "B" else np.nan, "r": r if method == "B" else np.nan,
                     "cutoff": res.get("cutoff"), "threshold": res.get("threshold"),
                     "max_depth": res.get("max_depth"), "calib_fpr": res.get("calib_fpr"),
                     "nobody_positive": res.get("nobody_positive")})
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results[["cutoff", "max_depth"]] = results[["cutoff", "max_depth"]].astype("Int64")
    results["nobody_positive"] = results["nobody_positive"].astype("boolean")

    u = z_methods["B-opt"]["cutoff"]
    tables = {
        "results_all.csv": results,
        "bopt_separation.csv": pd.DataFrame([{**cond, "upper_bp": k, "separation": s, "chosen": k == u}
                                             for k, s in sep.items()]),
        "c2_feature_importances.csv": pd.DataFrame([{**cond, "feature": f, "importance": v}
                                                    for f, v in trees["C2"]["importances"].items()]),
        "tree_cv_auc.csv": pd.DataFrame([{**cond, "method": m, "max_depth": d, "cv_auc": a,
                                          "chosen": d == trees[m]["max_depth"]}
                                         for m in TREE_FEATURES for d, a in trees[m]["cv_auc"].items()]),
    }
    return tables, z["test_normal"]["frac_all"]


def sort_table(table, name, config):
    """Sort by ff, depth, noise (config grid order), then the table's own columns (fixed orders for text)."""
    extra, orders = SORT[name]
    orders = {"noise": [str(v) for v in config["grid"]["noise_sigma"]], **orders}

    def key(col):
        if col.name in orders:
            rank = col.map({v: i for i, v in enumerate(orders[col.name])})
            if rank.isna().any():
                raise ValueError(f"{name}: unexpected {col.name} values {sorted(set(col[rank.isna()]))}")
            return rank
        return col
    return table.sort_values(CONDITION + extra, key=key, kind="stable").reset_index(drop=True)


def replace_condition_rows(path, new, config):
    """Write new to path, replacing earlier rows of the same condition, sorted. The write is atomic."""
    ff, depth, noise = new.loc[0, CONDITION]
    if path.exists():
        old = pd.read_csv(path, dtype={"noise": str})
        if list(old.columns) != list(new.columns):
            raise ValueError(f"{path} has columns {list(old.columns)}, expected {list(new.columns)}; "
                             f"move the old file away before writing")
        same = np.isclose(old["ff"], ff, rtol=1e-9, atol=0) & (old["depth"] == depth) & (old["noise"] == noise)
        new = pd.concat([old[~same], new], ignore_index=True)
    new = sort_table(new, path.name, config)
    tmp = path.with_name(path.name + ".tmp")
    new.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)


def sanity_checks(results, test_normal_za, n_reference, target_fpr):
    """Milestone checks (brief 3F). Tolerances are 3 approximate SEs, including the reference's own error."""
    by = results.set_index("method")
    a, c1 = by.loc["A"], by.loc["C1"]

    ok = a.fpr_lo <= target_fpr <= a.fpr_hi
    print(f"1. A test FPR {100 * a.fpr:.3f}% [{100 * a.fpr_lo:.3f}, {100 * a.fpr_hi:.3f}] "
          f"contains {100 * target_fpr:.2f}%: {'PASS' if ok else 'FAIL'}")

    n = len(test_normal_za)
    mean, sd = test_normal_za.mean(), test_normal_za.std(ddof=1)
    mean_tol = 3 * np.sqrt(1 / n + 1 / n_reference)
    sd_tol = 3 * np.sqrt(1 / (2 * (n - 1)) + 1 / (2 * (n_reference - 1)))
    print(f"2. test normals zA: mean {mean:+.4f} (|mean| <= {mean_tol:.3f}: {'PASS' if abs(mean) <= mean_tol else 'FAIL'}), "
          f"SD {sd:.4f} (|SD - 1| <= {sd_tol:.3f}: {'PASS' if abs(sd - 1) <= sd_tol else 'FAIL'})")

    overlap = a.sens_lo <= c1.sens_hi and c1.sens_lo <= a.sens_hi
    print(f"3. C1 vs A: sens {100 * c1.sens:.2f}% vs {100 * a.sens:.2f}% (Wilson CIs overlap: "
          f"{'PASS' if overlap else 'FAIL'}) | FPR {100 * c1.fpr:.3f}% vs {100 * a.fpr:.3f}% | "
          f"sens_at_fpr {100 * c1.sens_at_fpr:.2f}% vs {100 * a.sens_at_fpr:.2f}% | "
          f"AUC {c1.auc:.4f} vs {a.auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run one condition (FF x depth x noise) end to end.")
    parser.add_argument("--ff", type=float, required=True, help="fetal fraction, e.g. 0.04")
    parser.add_argument("--depth", type=int, required=True, help="depth, e.g. 5000000")
    parser.add_argument("--noise", type=parse_noise, required=True, help="noise level as in the config, e.g. none")
    args = parser.parse_args()

    config = load_config()
    commit, dirty = git_state()  # state of the code that runs, taken before anything is written
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tables, test_normal_za = run_condition(config, args.ff, args.depth, args.noise)
    tables["run_log.csv"] = pd.DataFrame([{"ff": args.ff, "depth": args.depth, "noise": str(args.noise),
                                           "commit": commit, "timestamp_utc": started, "dirty": dirty}])

    out_dir = repo_path(config["paths"]["results"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():  # run_log.csv last: it only names a run whose tables were all written
        replace_condition_rows(out_dir / name, table, config)
        print(f"Wrote {len(table)} rows of FF {args.ff}, depth {args.depth:,}, noise {args.noise} to {out_dir / name}")
    if dirty:
        print("Note: code or config had uncommitted changes; run_log.csv records dirty = True")

    results = tables["results_all.csv"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n" + results.drop(columns=CONDITION).to_string(index=False))
    print("\nSanity checks")
    sanity_checks(results, test_normal_za, config["patients_per_condition"]["reference"],
                  config["methods"]["target_fpr"])
    nobody = results.loc[results["nobody_positive"].fillna(False).astype(bool), "method"].tolist()
    if nobody:
        print(f"Note: nobody positive for {', '.join(nobody)} (no leaf meets the target FPR on calibration)")


if __name__ == "__main__":
    main()
