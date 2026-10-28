"""Task 3D: decision-tree methods C1 and C2 (decisions #5, #9, #10).

One tree per condition, trained on that condition's training set (normals label 0, T21 label 1):
  1. Depth: StratifiedKFold(cv_folds, shuffle=True, random_state=seed); for each max_depth in
     tree_max_depth, mean AUC over the validation folds. Highest mean wins (smallest depth if tied).
  2. Fit the final tree at the chosen depth on the full training set.
  3. Threshold: positive if p >= t. Candidates are the final tree's distinct leaf probabilities plus one
     value above their max (nobody positive). t is the smallest candidate with FPR <= target_fpr on the
     condition's calibration normals (a separate group, never used for training).

DecisionTreeClassifier(class_weight="balanced", random_state=seed).

Run from the repo root to fit C1 and C2 on the milestone condition (FF 4%, depth 5M, no noise):
    python -m src.methods.trees
"""

import math

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.config import load_config
from src.methods.features import TREE_FEATURES, feature_matrix
from src.methods.reference import build_reference, statistics, zscores
from src.simulate.patients import generate_group, origin_matrices


def make_tree(max_depth, config):
    return DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=config["seed"])


def cross_validate(X, y, config):
    """Mean validation AUC per depth, on stratified folds."""
    m = config["methods"]
    depths = m["tree_max_depth"]
    folds = StratifiedKFold(n_splits=m["cv_folds"], shuffle=True, random_state=config["seed"])
    aucs = {d: [] for d in depths}
    for train_idx, val_idx in folds.split(X, y):
        for d in depths:
            p = make_tree(d, config).fit(X[train_idx], y[train_idx]).predict_proba(X[val_idx])[:, 1]
            if not np.isfinite(p).all():
                raise ValueError(f"max_depth {d}: {(~np.isfinite(p)).sum()} validation probabilities are not finite")
            aucs[d].append(roc_auc_score(y[val_idx], p))
    cv_auc = {d: float(np.mean(a)) for d, a in aucs.items()}
    for d, a in cv_auc.items():
        if not np.isfinite(a):
            raise ValueError(f"max_depth {d}: CV AUC is {a!r}")
    return cv_auc


def leaf_candidates(tree, X_train):
    """Distinct leaf probabilities of the tree, plus one value above their max (nobody positive).
    Every leaf holds at least one training patient, so predicting the training set reaches all leaves."""
    leaves = np.unique(tree.predict_proba(X_train)[:, 1])
    return np.append(leaves, np.nextafter(leaves[-1], np.inf))


def threshold_at_fpr(calib_scores, candidates, target_fpr):
    """Smallest candidate t with mean(calib_scores >= t) <= target_fpr. Returns (t, fpr reached)."""
    s = np.sort(calib_scores)
    fpr = (len(s) - np.searchsorted(s, candidates, side="left")) / len(s)  # non-increasing in t
    i = int(np.argmax(fpr <= target_fpr))  # the last candidate is above every leaf, so FPR 0
    return float(candidates[i]), float(fpr[i])


def check_threshold(calib_scores, candidates, threshold, calib_fpr, target_fpr):
    """Recount independently: FPR constraint met, calib_fpr exact, and no smaller candidate meets it."""
    if not np.isfinite(calib_scores).all():
        raise ValueError(f"{(~np.isfinite(calib_scores)).sum()} calibration probabilities are not finite")
    n = len(calib_scores)
    allowed = math.floor(target_fpr * n)
    count = int((calib_scores >= threshold).sum())
    if count > allowed:
        raise ValueError(f"threshold {threshold!r}: {count} of {n} calibration normals >= t, "
                         f"more than allowed {allowed}")
    if count / n != calib_fpr:
        raise ValueError(f"threshold {threshold!r}: recounted FPR {count}/{n} = {count / n!r} "
                         f"differs from reported calib_fpr {calib_fpr!r}")
    smaller = candidates[candidates < threshold]
    if len(smaller):
        prev = smaller.max()
        prev_count = int((calib_scores >= prev).sum())
        if prev_count <= allowed:
            raise ValueError(f"threshold {threshold!r} is not minimal: smaller candidate {prev!r} gives "
                             f"{prev_count} of {n} calibration normals, within allowed {allowed}")


def fit_tree_method(method, train_normal_z, train_t21_z, calib_z, test_z, config):
    """Fit C1 or C2, set its threshold on the calibration normals and score the test patients.

    Returns dict: scores (test probabilities), calls (scores >= threshold), threshold, calib_fpr,
    max_depth, cv_auc {depth: mean AUC}, importances {feature: importance}.
    """
    X = np.vstack([feature_matrix(train_normal_z, method), feature_matrix(train_t21_z, method)])
    y = np.concatenate([np.zeros(len(train_normal_z["frac_all"]), int), np.ones(len(train_t21_z["frac_all"]), int)])

    cv_auc = cross_validate(X, y, config)
    depths = list(cv_auc)
    depth = depths[int(np.argmax([cv_auc[d] for d in depths]))]
    tree = make_tree(depth, config).fit(X, y)

    target = config["methods"]["target_fpr"]
    candidates = leaf_candidates(tree, X)
    calib_scores = tree.predict_proba(feature_matrix(calib_z, method))[:, 1]
    threshold, calib_fpr = threshold_at_fpr(calib_scores, candidates, target)
    check_threshold(calib_scores, candidates, threshold, calib_fpr, target)

    scores = tree.predict_proba(feature_matrix(test_z, method))[:, 1]
    return {"scores": scores, "calls": scores >= threshold, "threshold": threshold, "calib_fpr": calib_fpr,
            "max_depth": depth, "cv_auc": cv_auc,
            "importances": dict(zip(TREE_FEATURES[method], tree.feature_importances_.tolist()))}


def main():
    config = load_config()
    ff, depth, noise = 0.04, 5_000_000, "none"  # milestone condition (brief 3F); not a study parameter
    matrices = origin_matrices(config)
    reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config, matrices)
    z = {g: zscores(statistics(generate_group(config, ff, depth, noise, g, matrices), ff, config, matrices), reference)
         for g in ("train_normal", "train_t21", "calibration", "test_normal", "test_t21")}
    test_z = {k: np.concatenate([z["test_normal"][k], z["test_t21"][k]]) for k in z["test_normal"]}
    n_test_normal = len(z["test_normal"]["frac_all"])

    print(f"Trees: FF {ff:.0%}, depth {depth:,}, noise {noise}")
    a_calls = test_z["frac_all"] > config["methods"]["z_threshold"]
    print(f"\nA (zA > {config['methods']['z_threshold']}) for comparison: test FPR "
          f"{100 * a_calls[:n_test_normal].mean():.3f}% | sensitivity {100 * a_calls[n_test_normal:].mean():.2f}%")
    for method in TREE_FEATURES:
        r = fit_tree_method(method, z["train_normal"], z["train_t21"], z["calibration"], test_z, config)
        calls = r["calls"]
        print(f"\n{method} ({', '.join(TREE_FEATURES[method])})")
        print("  CV AUC: " + ", ".join(f"depth {d} {a:.4f}" for d, a in r["cv_auc"].items())
              + f" -> max_depth {r['max_depth']}")
        print(f"  threshold {r['threshold']:.6f}, calibration FPR {100 * r['calib_fpr']:.3f}%")
        print(f"  test: FPR {100 * calls[:n_test_normal].mean():.3f}% | sensitivity {100 * calls[n_test_normal:].mean():.2f}% "
              f"| unique test probabilities {len(np.unique(r['scores']))}")
        print("  importances: " + ", ".join(f"{f} {v:.3f}" for f, v in r["importances"].items()))


if __name__ == "__main__":
    main()
