"""Task 2D/2E: virtual-patient generator (decisions #3, #4, #12) and its QC.

A patient is a 2 x 465 count matrix: row 0 = chr21, row 1 = all other autosomes (chr1-22),
columns = fragment length 36..500 bp. Counts are surviving fragments (after the MAPQ filter), so
depth and FF are both defined after the filter (decision #12).

For origin o:  P_o(g, l) ∝ P_o(l) * S(l) * q(g),  q(chr21) = q[chr21], q(other) = 1 - q[chr21],
normalised over the 2 x 465 cells. Trisomy 21: the fetal chr21 row is multiplied by
grid.trisomy_fetal_chr21_multiplier, then renormalised. A patient with fetal fraction FF and depth N
is rng.multinomial(N, FF * P_fetal + (1 - FF) * P_maternal).

Seeds (all from config seed):
  grid patients  SeedSequence(seed, spawn_key=(1, i_ff, i_depth, i_noise, i_group)), indices = positions
                 in the config grid lists and in GROUPS, so a condition gives the same patients
                 whatever order conditions are run in
  2E QC patients SeedSequence(seed, spawn_key=(2, k)), k fixed per QC run (QC_RUNS)
  (the pool uses spawn_key (0, k), see pool.py)

Run from the repo root to print the 2E QC numbers and save results/tables/patients_qc.csv:
    python -m src.simulate.patients
"""

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.simulate.lengths import load_length_distributions
from src.simulate.mappability import load_mappability

GROUPS = ("reference", "train_normal", "train_t21", "test_normal", "test_t21")
CHR21, OTHER = 0, 1
NO_NOISE = ("none", 0)  # accepted values for the first noise level (index 0)
# 2E QC runs: spawn_key (2, k). Separate from grid patients even where the setting is a grid condition.
QC_RUNS = {
    "normal_ff10_depth1M": 0,
    "t21_ff10_depth1M": 1,
    "normal_ff04_depth1M": 2,
    "timing_normal_ff10_depth10M": 3,
}


def origin_matrices(config, mappability=None):
    """Normalised 2 x 465 probability matrices: "maternal", "fetal", "fetal_t21", plus "lengths"."""
    dist = load_length_distributions(config)
    mapp = load_mappability(config) if mappability is None else mappability
    lengths = dist["length_bp"].to_numpy()
    if not np.array_equal(lengths, mapp["lengths"]):
        raise ValueError("length grid of the length distributions and the mappability table differ")
    q21 = float(mapp["q"][list(mapp["chromosomes"]).index("chr21")])
    q_group = np.array([q21, 1 - q21])

    out = {"lengths": lengths}
    for origin in ("maternal", "fetal"):
        m = q_group[:, None] * (dist[origin].to_numpy() * mapp["survival_length"])[None, :]
        out[origin] = m / m.sum()
    t21 = out["fetal"].copy()
    t21[CHR21] *= config["grid"]["trisomy_fetal_chr21_multiplier"]
    out["fetal_t21"] = t21 / t21.sum()
    return out


@lru_cache(maxsize=1)
def _cached_matrices():
    return origin_matrices(load_config())


def mixture(ff, trisomy, matrices):
    """Per-patient cell probabilities (2 x 465): FF * fetal (normal or T21) + (1 - FF) * maternal."""
    fetal = matrices["fetal_t21"] if trisomy else matrices["fetal"]
    return ff * fetal + (1 - ff) * matrices["maternal"]


def generate_patients(ff, depth, n_patients, trisomy, sigma=0.0, *, rng, matrices=None):
    """Counts of surviving fragments, array (n_patients, 2, 465): [:, 0] chr21, [:, 1] other autosomes."""
    if sigma != 0:
        raise NotImplementedError("noise (sigma > 0) comes in Task 02b, decision #7")
    p = mixture(ff, trisomy, _cached_matrices() if matrices is None else matrices)
    counts = rng.multinomial(depth, p.ravel(), size=n_patients)
    return counts.reshape(n_patients, *p.shape)


def window_stats(matrices, ff, lo, hi):
    """Exact effect of keeping only fragments lo..hi bp (both rows, NORMAL fetal matrix).

    r          = FF * sum_win P_fetal + (1 - FF) * sum_win P_maternal   (fraction of fragments kept)
    ff_window  = FF * sum_win P_fetal / r                               (fetal fraction among kept)
    gain       = ff_window / FF
    """
    win = (matrices["lengths"] >= lo) & (matrices["lengths"] <= hi)
    fetal_in = matrices["fetal"][:, win].sum()
    maternal_in = matrices["maternal"][:, win].sum()
    r = ff * fetal_in + (1 - ff) * maternal_in
    ff_window = ff * fetal_in / r
    return {"ff_window": ff_window, "gain": ff_window / ff, "r": r}


def _grid_index(values, value, name):
    hits = [i for i, v in enumerate(values) if np.isclose(v, value)] if not isinstance(value, str) \
        else [i for i, v in enumerate(values) if v == value]
    if len(hits) != 1:
        raise ValueError(f"{name} {value!r} is not in the config grid {values}")
    return hits[0]


def patient_seed(config, ff, depth, noise, group):
    """SeedSequence for one (FF, depth, noise level, group) of the grid."""
    grid = config["grid"]
    if grid["noise_sigma"][0] not in NO_NOISE:
        raise ValueError(f"the no-noise level must be first in grid.noise_sigma (seeds use its index), "
                         f"got {grid['noise_sigma']}")
    key = (1,
           _grid_index(grid["fetal_fraction"], ff, "fetal_fraction"),
           _grid_index(grid["depth"], depth, "depth"),
           _grid_index(grid["noise_sigma"], noise, "noise_sigma"),
           GROUPS.index(group))
    return np.random.SeedSequence(config["seed"], spawn_key=key)


def generate_group(config, ff, depth, noise, group, matrices=None):
    """All patients of one group of one grid condition (size from patients_per_condition)."""
    if noise not in NO_NOISE:
        raise NotImplementedError("noise levels other than none come in Task 02b")
    rng = np.random.default_rng(patient_seed(config, ff, depth, noise, group))
    return generate_patients(ff, depth, config["patients_per_condition"][group], group.endswith("t21"),
                             rng=rng, matrices=matrices)


def qc_rng(config, run):
    return np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=(2, QC_RUNS[run])))


def chr21_fraction(patients):
    return patients[:, CHR21].sum(axis=1) / patients.sum(axis=(1, 2))


def window_share(patients, lengths, lo, hi):
    win = (lengths >= lo) & (lengths <= hi)
    return patients[:, :, win].sum(axis=(1, 2)) / patients.sum(axis=(1, 2))


def main():
    config = load_config()
    mapp = load_mappability(config)
    n_qc, depth_qc = 2000, 1_000_000  # 2E settings from the brief; not study parameters
    lo, hi = config["methods"]["size_selection_window_bp"]
    rows = []

    def record(quantity, value, reference=np.nan, note=""):
        rows.append({"quantity": quantity, "value": value, "reference": reference, "note": note})

    t = time.perf_counter()
    matrices = origin_matrices(config, mapp)
    t_matrices = time.perf_counter() - t
    lengths = matrices["lengths"]
    q21 = float(matrices["maternal"][CHR21].sum())
    survival = mapp["n_survived_bin"].sum() / mapp["n_fragments_bin"].sum()
    print(f"q(chr21) = {q21:.6f} | overall pool survival {100 * survival:.3f}% | "
          f"matrices built in {t_matrices:.2f} s")
    for name in ("maternal", "fetal", "fetal_t21"):
        print(f"  {name:>9}: sums to {matrices[name].sum():.12f}, chr21 row {matrices[name][CHR21].sum():.6f}")
    record("q_chr21", q21)
    record("pool_survival", survival)

    # Normal patients, FF 10%, depth 1M
    ff = 0.10
    normal = generate_patients(ff, depth_qc, n_qc, False, rng=qc_rng(config, "normal_ff10_depth1M"),
                               matrices=matrices)
    f_n = chr21_fraction(normal)
    sd_theory = np.sqrt(q21 * (1 - q21) / depth_qc)
    se_mean = f_n.std(ddof=1) / np.sqrt(n_qc)
    z_mean = (f_n.mean() - q21) / se_mean
    sd_rel_se = 1 / np.sqrt(2 * (n_qc - 1))  # approximate relative SE of a sample SD
    print(f"\n== Normal, FF {ff:.0%}, depth {depth_qc:,}, {n_qc:,} patients ==")
    print(f"mean chr21 fraction {f_n.mean():.6f} vs q(chr21) {q21:.6f}: difference {f_n.mean() - q21:+.2e}, "
          f"SE {se_mean:.2e}, z = {z_mean:+.2f}")
    print(f"SD of chr21 fraction {f_n.std(ddof=1):.4e} vs sqrt(p(1-p)/N) {sd_theory:.4e}: "
          f"ratio {f_n.std(ddof=1) / sd_theory:.4f} (sampling SE of the ratio ~{sd_rel_se:.4f})")
    record("normal_mean_chr21_fraction", f_n.mean(), q21, f"z = {z_mean:+.2f} using SE {se_mean:.3e}")
    record("normal_sd_chr21_fraction", f_n.std(ddof=1), sd_theory,
           f"ratio {f_n.std(ddof=1) / sd_theory:.4f}, sampling SE ~{sd_rel_se:.4f}")

    # Batch check: one call vs 4 batches of 500 from the same seed
    rng = qc_rng(config, "normal_ff10_depth1M")
    batched = np.concatenate([generate_patients(ff, depth_qc, n_qc // 4, False, rng=rng, matrices=matrices)
                              for _ in range(4)])
    same = bool(np.array_equal(batched, normal))
    print(f"2,000 patients in one call == 4 batches of 500 from the same seed: {same}")
    record("batches_identical_to_one_call", float(same))

    # Trisomy, same setting
    t21 = generate_patients(ff, depth_qc, n_qc, True, rng=qc_rng(config, "t21_ff10_depth1M"), matrices=matrices)
    f_t = chr21_fraction(t21)
    increase = f_t.mean() - f_n.mean()
    se_inc = np.sqrt(f_t.var(ddof=1) / n_qc + f_n.var(ddof=1) / n_qc)
    m = config["grid"]["trisomy_fetal_chr21_multiplier"]
    exact = ff * (m * q21 / (1 + (m - 1) * q21) - q21)
    approx = q21 * ff / 2
    print(f"\n== T21, FF {ff:.0%}, depth {depth_qc:,}, {n_qc:,} patients ==")
    print(f"mean chr21 increase vs normals {increase:.4e} (SE {se_inc:.1e}) | exact model value {exact:.4e} "
          f"(z = {(increase - exact) / se_inc:+.2f}) | p*FF/2 = {approx:.4e}")
    print(f"mean z-score of T21 patients against the normals: {(f_t.mean() - f_n.mean()) / f_n.std(ddof=1):.2f}")
    record("t21_mean_increase_chr21_fraction", increase, exact, f"SE {se_inc:.2e}; p*FF/2 = {approx:.4e}")

    # Realism check: exact window stats, cross-checked against sampled normal patients
    print(f"\n== Size selection {lo}-{hi} bp (exact, normal fetal matrix) ==")
    samples = {0.10: normal,
               0.04: generate_patients(0.04, depth_qc, n_qc, False, rng=qc_rng(config, "normal_ff04_depth1M"),
                                       matrices=matrices)}
    for ff_r in (0.04, 0.10):
        w = window_stats(matrices, ff_r, lo, hi)
        share = window_share(samples[ff_r], lengths, lo, hi)
        se = share.std(ddof=1) / np.sqrt(n_qc)
        z = (share.mean() - w["r"]) / se
        print(f"FF {ff_r:.0%}: FF in window {100 * w['ff_window']:.2f}% | gain {w['gain']:.3f}x | "
              f"kept r = {100 * w['r']:.3f}% | sampled patients kept {100 * share.mean():.3f}% "
              f"(SE {100 * se:.4f}, z = {z:+.2f})")
        record(f"ff{ff_r:.2f}_window_gain", w["gain"], np.nan, "expected roughly 2.2-2.5x")
        record(f"ff{ff_r:.2f}_window_ff", w["ff_window"])
        record(f"ff{ff_r:.2f}_window_kept_r", w["r"], np.nan, "expected roughly 0.20")
        record(f"ff{ff_r:.2f}_sampled_kept", share.mean(), w["r"], f"z = {z:+.2f}")

    # Timing: 2,000 patients at depth 10M
    t = time.perf_counter()
    big = generate_patients(0.10, 10_000_000, n_qc, False, rng=qc_rng(config, "timing_normal_ff10_depth10M"),
                            matrices=matrices)
    seconds = time.perf_counter() - t
    print(f"\n== Timing ==\n{n_qc:,} patients at depth 10,000,000: {seconds:.2f} s "
          f"({1000 * seconds / n_qc:.2f} ms per patient), array {big.nbytes / 1e6:.0f} MB")
    record("seconds_2000_patients_depth10M", seconds)

    out = repo_path(config["paths"]["results"]) / "tables" / "patients_qc.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.8g")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
