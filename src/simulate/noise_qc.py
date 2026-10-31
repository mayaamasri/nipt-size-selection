"""Task 4A: QC of the noise model (decision #7).

Patients: the grid's reference group (2,000 normals) at FF 4%, depths 1M and 10M, every noise level, i.e. the
patients the study uses. f = chr21 fraction of a patient, p0 = q(chr21), N = depth.

  sd_theory        p0 * sqrt(1/(p0 N) + sigma^2)                      (brief 4A)
  sd_theory_exact  p0 * sqrt((1 - p0)/(p0 N) + (1 - p0)^2 sigma^2)   binomial (1 - p0) factor, and the noise
                   on the share shrinks to sigma (1 - p0) because the matrix is renormalised after (1 + eps)
  ratio_se         ~ 1 / sqrt(2 (n - 1)), approximate relative SE of a sample SD

Writes results/tables/noise_qc.csv and results/figures/qc_noise_sd.png. Run from the repo root:
    python -m src.simulate.noise_qc
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.simulate.mappability import _style
from src.simulate.patients import CHR21, chr21_fraction, generate_group, noise_sigma, origin_matrices

QC_FF = 0.04                          # brief 4A settings; not study parameters
QC_DEPTHS = (1_000_000, 10_000_000)
QC_GROUP = "reference"


def sd_theory(p0, depth, sigma):
    return p0 * np.sqrt(1 / (p0 * depth) + sigma ** 2)


def sd_theory_exact(p0, depth, sigma):
    return p0 * np.sqrt((1 - p0) / (p0 * depth) + (1 - p0) ** 2 * sigma ** 2)


def noise_qc(config, matrices):
    p0 = float(matrices["maternal"][CHR21].sum())
    rows = []
    for depth in QC_DEPTHS:
        for noise in config["grid"]["noise_sigma"]:
            sigma = noise_sigma(config, noise)
            f = chr21_fraction(generate_group(config, QC_FF, depth, noise, QC_GROUP, matrices))
            sd = float(f.std(ddof=1))
            rows.append({"ff": QC_FF, "depth": depth, "noise": str(noise), "sigma": sigma, "n": len(f), "p0": p0,
                         "mean_obs": float(f.mean()), "sd_obs": sd,
                         "sd_theory": sd_theory(p0, depth, sigma), "ratio": sd / sd_theory(p0, depth, sigma),
                         "sd_theory_exact": sd_theory_exact(p0, depth, sigma),
                         "ratio_exact": sd / sd_theory_exact(p0, depth, sigma),
                         "ratio_se": 1 / np.sqrt(2 * (len(f) - 1))})
    return pd.DataFrame(rows)


def plot_noise_sd(table, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#0b0b0b"]  # none, low, realistic
    surface = "#fcfcfb"
    p0 = table["p0"].iloc[0]
    depths = np.geomspace(min(QC_DEPTHS) / 1.5, max(QC_DEPTHS) * 1.5, 200)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    for color, (noise, g) in zip(colors, table.groupby("noise", sort=False)):
        sigma = g["sigma"].iloc[0]
        label = f"{noise} (σ = {100 * sigma:.2f}%)"
        ax.plot(depths, sd_theory(p0, depths, sigma), color=color, linewidth=1.5, label=f"{label}: theory (brief)")
        ax.plot(depths, sd_theory_exact(p0, depths, sigma), color=color, linewidth=1, linestyle="--")
        ax.errorbar(g["depth"], g["sd_obs"], yerr=1.96 * g["sd_obs"] * g["ratio_se"], fmt="o", markersize=5,
                    color=color, elinewidth=1, capsize=3)
    ax.plot([], [], color="#52514e", linestyle="--", linewidth=1, label="exact theory (dashed)")
    ax.errorbar([], [], yerr=[], fmt="o", color="#52514e", label=f"observed, {table['n'].iloc[0]:,} normals, ~95% CI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"SD of the chr21 fraction among normals, FF {QC_FF:.0%}", "Depth (surviving fragments)",
           "SD of chr21 fraction")
    legend = ax.legend(frameon=False, fontsize=8)
    for t in legend.get_texts():
        t.set_color("#0b0b0b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    config = load_config()
    table = noise_qc(config, origin_matrices(config))
    print(f"Noise QC: {QC_GROUP} group, FF {QC_FF:.0%}, p0 = q(chr21) = {table['p0'].iloc[0]:.6f}")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.drop(columns=["ff", "p0"]).to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    for r in table.itertuples():
        for col in ("ratio", "ratio_exact"):
            z = (getattr(r, col) - 1) / r.ratio_se
            if abs(z) > 3:
                print(f"Note: depth {r.depth:,} {r.noise}: {col} {getattr(r, col):.4f} is {z:+.1f} SE from 1")

    results = repo_path(config["paths"]["results"])
    out = results / "tables" / "noise_qc.csv"
    table.to_csv(out, index=False, float_format="%.8g")
    print(f"Saved {out}")
    fig = results / "figures" / "qc_noise_sd.png"
    plot_noise_sd(table, fig)
    print(f"Saved {fig}")


if __name__ == "__main__":
    main()
