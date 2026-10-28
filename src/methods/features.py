"""Task 3C: decision-tree features for C1 and C2 (decision #9).

  zA          z-score of the chr21 fraction, all fragments (method A)
  z_short     Yu et al.: (share of fragments <= short_fragment_bp on chr21) - (same share on other
              autosomes), z-scored against the reference normals
  z_weighted  Rabinowitz et al.: chr21 fraction with each length weighted by w(l) = P(fetal | l),
              using the condition's FF and the post-mappability length distributions, z-scored against
              the reference normals

C1 uses zA only; C2 uses all three. The raw statistics and their reference are in src.methods.reference.

Run from the repo root to print the features on the training set of the milestone condition
(FF 4%, depth 5M, no noise):
    python -m src.methods.features
"""

import numpy as np

from src.config import load_config
from src.methods.reference import build_reference, statistics, zscores
from src.simulate.patients import generate_group, origin_matrices

# feature name -> statistic name in src.methods.reference
FEATURES = {"zA": "frac_all", "z_short": "short_excess", "z_weighted": "weighted"}
TREE_FEATURES = {"C1": ["zA"], "C2": ["zA", "z_short", "z_weighted"]}


def feature_matrix(z, method):
    """Features of one tree method as an (n, k) array, columns in TREE_FEATURES[method] order."""
    return np.column_stack([z[FEATURES[f]] for f in TREE_FEATURES[method]])


def main():
    config = load_config()
    ff, depth, noise = 0.04, 5_000_000, "none"  # milestone condition (brief 3F); not a study parameter
    matrices = origin_matrices(config)
    reference = build_reference(generate_group(config, ff, depth, noise, "reference", matrices), ff, config, matrices)
    X = {g: feature_matrix(zscores(statistics(generate_group(config, ff, depth, noise, g, matrices),
                                              ff, config, matrices), reference), "C2")
         for g in ("train_normal", "train_t21")}

    names = TREE_FEATURES["C2"]
    print(f"C2 features on training set: FF {ff:.0%}, depth {depth:,}, noise {noise}")
    for j, name in enumerate(names):
        n, t = X["train_normal"][:, j], X["train_t21"][:, j]
        print(f"  {name:>10}: normals mean {n.mean():+.3f} SD {n.std(ddof=1):.3f} | "
              f"T21 mean {t.mean():+.3f} SD {t.std(ddof=1):.3f} | separation {t.mean() - n.mean():.3f}")
    corr = np.corrcoef(X["train_normal"], rowvar=False)
    print("  correlation among training normals:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"    {names[i]} vs {names[j]}: {corr[i, j]:+.3f}")


if __name__ == "__main__":
    main()
