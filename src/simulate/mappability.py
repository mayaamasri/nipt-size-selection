"""Task 2C: mappability table from the mapped pool (decision #3).

From all pool chunks (maternal and fetal together, the genome is the same for both):
  S(l)  probability that a fragment of true length l survives (status == "survived"),
        in survival_bin_bp bins starting at fragment_length_min (36-45, 46-55, ..., 496-500)
  q(c)  share of survivors on chromosome c, by mapped_chrom (what a lab sees), over chr1-22 only

Saved to paths.mappability (.npz) and results/tables/mappability.csv. QC (printed and plotted):
  - S(l) against length (results/figures/qc_survival_by_length.png)
  - chr21 share of survivors per bin with Wilson 95% CIs (results/figures/qc_chr21_share_by_length.png);
    the patient model assumes it does not depend on length
  - maternal vs fetal survival within each bin: origin only changes the length in the simulation,
    so within a bin they should agree up to noise

Run from the repo root:
    python -m src.simulate.mappability
"""

import glob

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config, repo_path
from src.simulate.lengths import ORIGINS
from src.simulate.pool import n_chunks

Z95 = 1.959963984540054


def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion k/n (arrays allowed). Returns (low, high)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def length_bins(config):
    """Bin edges: bin i covers lengths lo[i]..hi[i] (inclusive); the last bin may be shorter."""
    sim = config["simulation"]
    lmin, lmax, width = sim["fragment_length_min"], sim["fragment_length_max"], sim["survival_bin_bp"]
    lo = np.arange(lmin, lmax + 1, width)
    hi = np.minimum(lo + width - 1, lmax)
    return lo, hi


def bin_index(lengths, config):
    sim = config["simulation"]
    return (np.asarray(lengths) - sim["fragment_length_min"]) // sim["survival_bin_bp"]


def load_pool(config):
    files = sorted(glob.glob(str(repo_path(config["paths"]["pool_dir"]) / "chunk_*.parquet")))
    expected = n_chunks(config)
    if len(files) != expected:
        raise FileNotFoundError(f"found {len(files)} pool chunks, expected {expected}; "
                                f"run python -m src.simulate.pool to finish the pool")
    pool = pd.concat([pd.read_parquet(f, columns=["origin", "true_length", "status", "mapped_chrom"])
                      for f in files], ignore_index=True)
    return pool, len(files)


def estimate(pool, config):
    """Counts and estimates for S(l) (per bin and per length) and q(c)."""
    sim = config["simulation"]
    chroms = sim["chromosomes"]
    lo, hi = length_bins(config)
    survived = (pool["status"] == "survived").to_numpy()
    b = bin_index(pool["true_length"].to_numpy(), config)

    n_bin = np.bincount(b, minlength=len(lo))
    k_bin = np.bincount(b[survived], minlength=len(lo))
    lengths = np.arange(sim["fragment_length_min"], sim["fragment_length_max"] + 1)
    survival_bin = k_bin / n_bin
    survival_length = survival_bin[bin_index(lengths, config)]

    surv_chrom = pool.loc[survived, "mapped_chrom"].astype(str)
    in_autosomes = surv_chrom.isin(chroms).to_numpy()
    chrom_counts = surv_chrom[in_autosomes].value_counts().reindex(chroms, fill_value=0).to_numpy()
    return {
        "lengths": lengths,
        "bin_lo": lo,
        "bin_hi": hi,
        "n_fragments_bin": n_bin,
        "n_survived_bin": k_bin,
        "survival_bin": survival_bin,
        "survival_length": survival_length,
        "chromosomes": np.array(chroms),
        "n_survivors_chrom": chrom_counts,
        "q": chrom_counts / chrom_counts.sum(),
        "n_survivors_outside_autosomes": int((~in_autosomes).sum()),
    }


def save(est, config):
    npz_path = repo_path(config["paths"]["mappability"])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **est)

    rows = [("S", f"{l}-{h}", n, k, k / n) for l, h, n, k in
            zip(est["bin_lo"], est["bin_hi"], est["n_fragments_bin"], est["n_survived_bin"])]
    n_auto = est["n_survivors_chrom"].sum()
    rows += [("q", c, n_auto, k, q) for c, k, q in
             zip(est["chromosomes"], est["n_survivors_chrom"], est["q"])]
    table = pd.DataFrame(rows, columns=["quantity", "key", "n", "count", "value"])
    csv_path = repo_path(config["paths"]["results"]) / "tables" / "mappability.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.8g")
    return npz_path, csv_path


def load_mappability(config):
    """S(l) per length (465 values) and q(c) for chr1-22, from the saved .npz."""
    with np.load(repo_path(config["paths"]["mappability"])) as f:
        return {k: f[k] for k in f.files}


def origin_comparison(pool, config):
    """Per bin: maternal vs fetal survival, two-proportion z-test. Also the overall gap predicted by
    length alone: each origin's own length mix weighted by the pooled S(bin)."""
    lo, hi = length_bins(config)
    survived = pool["status"] == "survived"
    b = bin_index(pool["true_length"].to_numpy(), config)
    counts = {}
    for origin in ORIGINS:
        sel = (pool["origin"] == origin).to_numpy()
        counts[origin] = (np.bincount(b[sel], minlength=len(lo)),
                          np.bincount(b[sel & survived.to_numpy()], minlength=len(lo)))
    (nm, km), (nf, kf) = counts["maternal"], counts["fetal"]
    pooled = (km + kf) / np.maximum(nm + nf, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(pooled * (1 - pooled) * (1 / nm + 1 / nf))
        z = (km / nm - kf / nf) / se
    rows = pd.DataFrame({"bin": [f"{l}-{h}" for l, h in zip(lo, hi)], "n_maternal": nm, "S_maternal": km / nm,
                         "n_fetal": nf, "S_fetal": kf / nf, "z": z})

    s_bin = (km + kf) / (nm + nf)
    predicted = {o: float((counts[o][0] * s_bin).sum() / counts[o][0].sum()) for o in ORIGINS}
    observed = {o: float(counts[o][1].sum() / counts[o][0].sum()) for o in ORIGINS}
    return rows, predicted, observed


def chr21_by_bin(pool, config):
    """Per bin: survivors on chr21 / survivors on chr1-22 (mapped_chrom), with Wilson 95% CI."""
    lo, hi = length_bins(config)
    surv = pool[(pool["status"] == "survived") & pool["mapped_chrom"].isin(config["simulation"]["chromosomes"])]
    b = bin_index(surv["true_length"].to_numpy(), config)
    n = np.bincount(b, minlength=len(lo))
    k = np.bincount(b[(surv["mapped_chrom"] == "chr21").to_numpy()], minlength=len(lo))
    low, high = wilson_interval(k, n)
    return pd.DataFrame({"lo": lo, "hi": hi, "n": n, "k": k, "share": k / n, "ci_low": low, "ci_high": high})


def _style(ax, title, xlabel, ylabel):
    text, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    ax.set_title(title, color=text, loc="left")
    ax.set_xlabel(xlabel, color=muted)
    ax.set_ylabel(ylabel, color=muted)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)


def plot_survival(est, comparison, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same palette as lengths.py: maternal slot 1, fetal slot 2; pooled in neutral ink
    colors = {"maternal": "#2a78d6", "fetal": "#eb6834", "pooled": "#0b0b0b"}
    surface = "#fcfcfb"
    mid = (est["bin_lo"] + est["bin_hi"]) / 2
    low, high = wilson_interval(est["n_survived_bin"], est["n_fragments_bin"])

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    ax.fill_between(mid, 100 * low, 100 * high, color=colors["pooled"], alpha=0.12, linewidth=0)
    ax.plot(mid, 100 * est["survival_bin"], color=colors["pooled"], linewidth=2, marker="o", markersize=3,
            label="Pooled S(l), 95% CI")
    for origin in ORIGINS:
        ax.plot(mid, 100 * comparison[f"S_{origin}"], color=colors[origin], linewidth=1, alpha=0.9,
                label=origin.capitalize())
    _style(ax, "Survival (proper pair, both mates MAPQ ≥ 40) by fragment length, 10 bp bins",
           "Fragment length (bp, bin midpoint)", "Survival (%)")
    legend = ax.legend(frameon=False)
    for t in legend.get_texts():
        t.set_color("#0b0b0b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def plot_chr21_share(share, q21, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    mid = (share["lo"] + share["hi"]) / 2
    ok = share["n"] > 0
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    ax.axhline(100 * q21, color="#52514e", linewidth=1, linestyle="--", label=f"q(chr21) = {100 * q21:.3f}%")
    ax.errorbar(mid[ok], 100 * share["share"][ok],
                yerr=[100 * (share["share"] - share["ci_low"])[ok], 100 * (share["ci_high"] - share["share"])[ok]],
                fmt="o", markersize=3, color="#2a78d6", ecolor="#2a78d6", elinewidth=1, capsize=2,
                label="chr21 share of survivors, Wilson 95% CI")
    _style(ax, "chr21 share of surviving fragments by length (chr1–22, mapped chromosome)",
           "Fragment length (bp, bin midpoint)", "chr21 share (%)")
    legend = ax.legend(frameon=False)
    for t in legend.get_texts():
        t.set_color("#0b0b0b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    config = load_config()
    pool, n_files = load_pool(config)
    est = estimate(pool, config)
    npz_path, csv_path = save(est, config)

    n, k = est["n_fragments_bin"].sum(), est["n_survived_bin"].sum()
    q = dict(zip(est["chromosomes"], est["q"]))
    print(f"Pool: {n_files} chunks, {n:,} fragments, {k:,} survived ({100 * k / n:.3f}%)")
    print(f"Survivors mapped outside chr1-22: {est['n_survivors_outside_autosomes']:,} "
          f"({100 * est['n_survivors_outside_autosomes'] / k:.4f}%)")
    n21, n_auto = est["n_survivors_chrom"][list(q).index("chr21")], est["n_survivors_chrom"].sum()
    low, high = wilson_interval(n21, n_auto)
    print(f"q(chr21) = {q['chr21']:.6f} ({n21:,} / {n_auto:,}), 95% CI {low:.6f}-{high:.6f}; "
          f"q sums to {est['q'].sum():.12f}")
    print(f"Saved {npz_path} and {csv_path}")

    print("\n== S(l) per bin ==")
    for l, h, nb, kb in zip(est["bin_lo"], est["bin_hi"], est["n_fragments_bin"], est["n_survived_bin"]):
        print(f"{l:>4}-{h:<4} n={nb:>10,}  S={100 * kb / nb:6.2f}%")

    comparison, predicted, observed = origin_comparison(pool, config)
    print("\n== Maternal vs fetal survival within each bin (two-proportion z) ==")
    print(comparison.to_string(index=False, formatters={
        "n_maternal": "{:,}".format, "n_fetal": "{:,}".format,
        "S_maternal": lambda v: f"{100 * v:.2f}%", "S_fetal": lambda v: f"{100 * v:.2f}%",
        "z": lambda v: f"{v:+.2f}"}))
    valid = comparison["z"].notna()
    chi2 = float((comparison.loc[valid, "z"] ** 2).sum())
    df = int(valid.sum())
    print(f"Sum of z^2 over {df} bins = {chi2:.1f} (chi-square, {df} df: p = {stats.chi2.sf(chi2, df):.3g}); "
          f"bins with |z| > 3: {int((comparison['z'].abs() > 3).sum())}")
    print("Overall survival, observed vs predicted from each origin's length mix and the pooled S(bin):")
    for origin in ORIGINS:
        print(f"  {origin:>8}: observed {100 * observed[origin]:.3f}% | predicted {100 * predicted[origin]:.3f}%")
    print(f"  gap maternal - fetal: observed {100 * (observed['maternal'] - observed['fetal']):+.3f} | "
          f"predicted {100 * (predicted['maternal'] - predicted['fetal']):+.3f} points")

    share = chr21_by_bin(pool, config)
    ok = share["n"] > 0
    expected = share["n"][ok] * q["chr21"]
    chi2_21 = float((((share["k"][ok] - expected) ** 2) / (expected * (1 - q["chr21"]))).sum())
    df21 = int(ok.sum()) - 1
    print("\n== chr21 share of survivors per bin ==")
    for r in share[ok].itertuples():
        print(f"{r.lo:>4}-{r.hi:<4} n={r.n:>10,}  chr21 {100 * r.share:6.3f}% "
              f"[{100 * r.ci_low:.3f}, {100 * r.ci_high:.3f}]")
    print(f"Flatness: chi-square {chi2_21:.1f} on {df21} df, p = {stats.chi2.sf(chi2_21, df21):.3g}")
    surv = pool[(pool["status"] == "survived") & pool["mapped_chrom"].isin(config["simulation"]["chromosomes"])]
    slope = stats.linregress(surv["true_length"].to_numpy(), (surv["mapped_chrom"] == "chr21").to_numpy(float))
    print(f"Linear trend of chr21 indicator on length: {1e4 * slope.slope:+.4f} percentage points per 100 bp, "
          f"p = {slope.pvalue:.3g}")

    figures = repo_path(config["paths"]["results"]) / "figures"
    plot_survival(est, comparison, figures / "qc_survival_by_length.png")
    plot_chr21_share(share, q["chr21"], figures / "qc_chr21_share_by_length.png")
    print(f"\nSaved {figures / 'qc_survival_by_length.png'} and {figures / 'qc_chr21_share_by_length.png'}")


if __name__ == "__main__":
    main()
