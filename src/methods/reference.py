"""Task 3A: per-condition reference statistics from the reference normals (decisions #4, #6, #9).

Raw statistics per patient (X has shape (n, 2, 465): row 0 chr21, row 1 other autosomes, column j = 36 + j bp):
  frac_all            chr21 fraction, all fragments                          (method A, feature zA)
  frac_<lo>_<hi>      chr21 fraction, fragments lo..hi bp only               (method B: 40-156; B-opt: 40-u)
  short_excess        Yu: share of <= short_fragment_bp on chr21 minus share on other autosomes  (z_short)
  weighted            chr21 fraction with each length weighted by w(l) = P(fetal | l)            (z_weighted)

w(l) = FF * p_f(l) / (FF * p_f(l) + (1 - FF) * p_m(l)), with p_f, p_m the length distributions after
mappability (normal fetal and maternal matrices from origin_matrices, summed over the two rows).

The reference for a condition is the mean and SD (ddof=1) of each statistic over its reference normals.

Run from the repo root to print the reference for the milestone condition (FF 4%, depth 5M, no noise):
    python -m src.methods.reference
"""

import numpy as np

from src.config import load_config
from src.simulate.patients import CHR21, OTHER, generate_group, origin_matrices


def window_names(config):
    """Names of the windowed chr21 fractions: B's window, then one per B-opt upper cutoff."""
    m = config["methods"]
    lo, hi = m["size_selection_window_bp"]
    names = [f"frac_{lo}_{hi}"]
    names += [f"frac_{lo}_{u}" for u in m["size_opt_upper_grid"] if f"frac_{lo}_{u}" not in names]
    return names


def posterior_weights(ff, matrices):
    """w(l) = P(fetal | length l) for a patient with fetal fraction ff, using the normal fetal matrix."""
    p_f = matrices["fetal"].sum(axis=0)
    p_m = matrices["maternal"].sum(axis=0)
    for name, p in (("fetal", p_f), ("maternal", p_m)):
        if abs(p.sum() - 1) > 1e-9:
            raise ValueError(f"{name} length distribution sums to {p.sum():.12f}, not 1; w(l) needs both normalised")
    return ff * p_f / (ff * p_f + (1 - ff) * p_m)


def statistics(X, ff, config, matrices):
    """Raw per-patient statistics: dict name -> array of length n."""
    lengths = matrices["lengths"]
    chr21, other = X[:, CHR21].astype(float), X[:, OTHER].astype(float)
    out = {"frac_all": chr21.sum(axis=1) / (chr21 + other).sum(axis=1)}

    for name in window_names(config):
        lo, hi = (int(v) for v in name.split("_")[1:])
        win = (lengths >= lo) & (lengths <= hi)
        out[name] = chr21[:, win].sum(axis=1) / (chr21[:, win] + other[:, win]).sum(axis=1)

    short = lengths <= config["methods"]["short_fragment_bp"]
    out["short_excess"] = (chr21[:, short].sum(axis=1) / chr21.sum(axis=1)
                           - other[:, short].sum(axis=1) / other.sum(axis=1))

    w = posterior_weights(ff, matrices)
    out["weighted"] = (chr21 @ w) / ((chr21 + other) @ w)

    for name, v in out.items():
        bad = ~np.isfinite(v)
        if bad.any():
            raise ValueError(f"statistic {name} is not finite for {bad.sum()} of {len(v)} patients "
                             f"(zero denominator?), first at patient {np.flatnonzero(bad)[0]}")
    return out


def build_reference(reference_patients, ff, config, matrices):
    """Mean and SD (ddof=1) of every statistic over the reference normals: dict name -> (mean, sd)."""
    stats = statistics(reference_patients, ff, config, matrices)
    reference = {name: (float(v.mean()), float(v.std(ddof=1))) for name, v in stats.items()}
    for name, (_, sd) in reference.items():
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError(f"reference SD of {name} is {sd!r}; z-scores need a finite SD > 0")
    return reference


def zscores(stats, reference):
    """z-score each raw statistic against the reference: dict name -> array."""
    return {name: (v - reference[name][0]) / reference[name][1] for name, v in stats.items()}


def main():
    config = load_config()
    ff, depth, noise = 0.04, 5_000_000, "none"  # milestone condition (brief 3F); not a study parameter
    matrices = origin_matrices(config)
    ref_patients = generate_group(config, ff, depth, noise, "reference", matrices)
    reference = build_reference(ref_patients, ff, config, matrices)

    q21 = float(matrices["maternal"][CHR21].sum())
    print(f"Reference: FF {ff:.0%}, depth {depth:,}, noise {noise}, {len(ref_patients):,} normals "
          f"(q(chr21) = {q21:.6f})")
    for name, (mean, sd) in reference.items():
        print(f"  {name:>14}: mean {mean:+.6e}  SD {sd:.4e}")

    # Normals: chr21 share is q21 for both origins, so frac_all ~ binomial(N, q21) / N
    mean, sd = reference["frac_all"]
    sd_theory = np.sqrt(q21 * (1 - q21) / depth)
    sd_rel_se = 1 / np.sqrt(2 * (len(ref_patients) - 1))  # approximate relative SE of a sample SD
    print(f"\nfrac_all mean {mean:.6f} vs q(chr21) {q21:.6f}: difference {mean - q21:+.2e} "
          f"(SE {sd / np.sqrt(len(ref_patients)):.2e})")
    print(f"frac_all SD {sd:.4e} vs sqrt(q(1-q)/N) {sd_theory:.4e}: ratio {sd / sd_theory:.4f} "
          f"(sampling SE of the ratio ~{sd_rel_se:.4f})")


if __name__ == "__main__":
    main()
