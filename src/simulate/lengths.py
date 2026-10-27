"""Task 1A: maternal and fetal fragment-length distributions (decisions #1, #2).

Run from the repo root to print the QC numbers and save the QC plot:
    python -m src.simulate.lengths
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import load_config, repo_path

ORIGINS = ("maternal", "fetal")


def load_length_distributions(config):
    """Return a DataFrame with columns length_bp, maternal, fetal (probabilities).

    Processing (decision #1), in this order:
      1. drop length 0 (no valid insert size)
      2. replace the artifact bin (101 bp) in both columns by the mean of its neighbours
      3. keep fragment_length_min..fragment_length_max, then normalise each column to sum to 1
    """
    sim = config["simulation"]
    counts = pd.read_csv(repo_path(sim["length_counts_csv"]), comment="#")
    counts = counts.rename(columns={"maternal_count": "maternal", "fetal_count": "fetal"})
    counts[list(ORIGINS)] = counts[list(ORIGINS)].astype(float)

    # 1. drop length 0
    counts = counts[counts["length_bp"] != 0]

    # 2. artifact bin -> mean of the bins on either side
    counts = counts.set_index("length_bp")
    bad = sim["length_artifact_bp"]
    counts.loc[bad] = (counts.loc[bad - 1] + counts.loc[bad + 1]) / 2

    # 3. length window, then normalise
    counts = counts.loc[sim["fragment_length_min"]:sim["fragment_length_max"]]
    probs = counts / counts.sum()
    return probs.reset_index()


@lru_cache(maxsize=1)
def _cached_distributions():
    # Loaded once per process, so sample_lengths can be called many times cheaply.
    return load_length_distributions(load_config())


def sample_lengths(origin, n, rng):
    """Draw n fragment lengths (bp, integers) for origin "maternal" or "fetal"."""
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
    dist = _cached_distributions()
    return rng.choice(dist["length_bp"].to_numpy(), size=n, p=dist[origin].to_numpy())


def distribution_median(dist, origin):
    """Smallest length at which the cumulative probability reaches 0.5."""
    cumulative = dist[origin].cumsum().to_numpy()
    return int(dist["length_bp"].iloc[np.searchsorted(cumulative, 0.5)])


def kept_fraction(config):
    """Fraction of raw counts (length > 0) that falls inside the length window."""
    sim = config["simulation"]
    counts = pd.read_csv(repo_path(sim["length_counts_csv"]), comment="#")
    counts = counts[counts["length_bp"] != 0]
    inside = counts["length_bp"].between(sim["fragment_length_min"], sim["fragment_length_max"])
    return {o: counts.loc[inside, f"{o}_count"].sum() / counts[f"{o}_count"].sum() for o in ORIGINS}


def plot_distributions(dist, medians, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical palette (slots 1 and 2), light surface
    colors = {"maternal": "#2a78d6", "fetal": "#eb6834"}
    text, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    for origin in ORIGINS:
        ax.plot(dist["length_bp"], dist[origin], color=colors[origin], linewidth=2,
                label=f"{origin.capitalize()} (median {medians[origin]} bp)")
        ax.axvline(medians[origin], color=colors[origin], linewidth=1, linestyle="--")
    ax.set_xlabel("Fragment length (bp)", color=muted)
    ax.set_ylabel("Probability", color=muted)
    ax.set_title("Fragment-length distributions (Rabinowitz G1, processed)", color=text, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    legend = ax.legend(frameon=False)
    for t in legend.get_texts():
        t.set_color(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    config = load_config()
    dist = load_length_distributions(config)
    rng = np.random.default_rng(config["seed"])
    n_check = 100_000  # QC sample size only; not a study parameter

    print(f"Lengths kept: {dist['length_bp'].min()}-{dist['length_bp'].max()} bp ({len(dist)} bins)")
    kept = kept_fraction(config)
    medians = {}
    for origin in ORIGINS:
        medians[origin] = distribution_median(dist, origin)
        sampled = sample_lengths(origin, n_check, rng)
        print(f"{origin:>8}: sums to {dist[origin].sum():.6f} | "
              f"exact median {medians[origin]} bp | "
              f"median of {n_check:,} samples {np.median(sampled):.0f} bp | "
              f"kept by length window {100 * kept[origin]:.2f}%")

    out_path = repo_path(config["paths"]["results"]) / "figures" / "qc_length_distributions.png"
    plot_distributions(dist, medians, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
