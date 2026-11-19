"""Task 06, supplementary figure: one counting model for B and W (extends Fig. 4).

FF 4%, one panel per noise level: x = depth (log), y = sep(method) / sep(A), sep = (z_mean_t21 - z_mean_normal) /
z_sd_normal, for B (40-156 bp) and W (weights w*(l) = P(fetal | l) at the true FF). Points: simulation; lines:
the unified model (Task 06, 6A), which is decision #8's formula with g_eff = E[u w*] / (FF E[u]) and
r_eff = E[u]^2 / E[u^2] (both depth-independent), drawn with breakeven_check.predicted_ratio. Dashed line at 1.

Reads results/tables/unified_model_check.csv and unified_model_summary.csv
(python -m src.evaluate.unified_model_check). Writes results/figures/figS3_unified_model.png (300 dpi) and .pdf.
Run from the repo root:
    python -m src.evaluate.figures.figS_unified_model
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
from src.evaluate import labels as L
from src.evaluate.breakeven_check import predicted_ratio
from src.evaluate.figures._style import FIG_WIDTH, MIN_FONT, MUTED, TEXT, new_figure, save_figure, style_axes
from src.evaluate.figures.fig4_breakeven import DEPTH_RANGE, FIG_HEIGHT

FIG_FF = 0.04
# method: (label, colour, marker, line style); B in Fig. 4's 156 bp colour, W in the fetal slot
SERIES = {"B": ("B (40–156 bp)", "#123766", "D", "-"), "W": ("W (weights P(fetal | l))", "#eb6834", "o", "--")}

CAPTION = (
    "Figure S3. One counting model for size selection and length weighting, FF 4%: ratio of standardised "
    "separations sep(method)/sep(A) for B (40–156 bp) and W (each fragment weighted by "
    r"$P(\mathrm{fetal} \mid \mathrm{length})$ at the true FF). Points: simulation; lines: the unified model, in "
    "which every z-score method is a chr21 fraction with length weights $u(l)$, and the break-even formula (see "
    r"Methods) holds with an effective enrichment $g_\mathrm{eff} = E[u w^*]/(\mathrm{FF}\cdot E[u])$ and an "
    r"effective read fraction $r_\mathrm{eff} = E[u]^2/E[u^2]$. Dashed: method = A. Like W, the model uses the "
    "true length distributions."
)


def panel(ax, data, sigma, title):
    depths = np.geomspace(*DEPTH_RANGE, 200)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    for method, (_, color, marker, style) in SERIES.items():
        d = data[data["method"] == method].sort_values("depth")
        g, r, p0 = d[["g_eff", "r_eff", "p0"]].iloc[0]
        ax.plot(depths, predicted_ratio(g, r, p0, depths, sigma), color=color, linestyle=style, linewidth=1.1)
        ax.plot(d["depth"], d["ratio_obs"], color=color, linestyle="none", marker=marker, markersize=3.5,
                markeredgecolor=TEXT, markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlim(*DEPTH_RANGE)
    ax.set_xticks([1e6, 2.5e6, 5e6, 1e7], ["1M", "2.5M", "5M", "10M"])
    ax.minorticks_off()
    style_axes(ax, title, "Depth", "")


def load(config):
    tables = repo_path(config["paths"]["results"]) / "tables"
    return (pd.read_csv(tables / "unified_model_check.csv", dtype={"noise": str}),
            pd.read_csv(tables / "unified_model_summary.csv", dtype={"value": str}))


def caption(config):
    """Caption (describes what is shown; model errors are in unified_model_summary.csv)."""
    return CAPTION


def main():
    config = load_config()
    table, _ = load(config)
    data = table[np.isclose(table["ff"], FIG_FF) & table["method"].isin(SERIES)]
    grid = config["grid"]

    fig, axes = new_figure(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88), wspace=0.04)    # room for the legend above
    for ax, noise, sigma in zip(axes, grid["noise_sigma"], grid["noise_sigma_values"]):
        panel(ax, data[data["noise"] == str(noise)], sigma, L.noise_title(noise, sigma))
    axes[0].set_ylabel("sep(method) / sep(A)", color=MUTED)
    axes[0].set_ylim(0.9, 1.65)

    handles = [Line2D([], [], color=color, marker=marker, linestyle=style, markersize=3.5, markeredgecolor=TEXT,
                      markeredgewidth=0.4, label=label) for label, color, marker, style in SERIES.values()]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                        handlelength=2.6, columnspacing=1.2)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.text(0.98, 0.985, "lines: model · points: simulation", color=MUTED, fontsize=MIN_FONT, ha="right",
             va="top")
    paths = save_figure(fig, config, "figS3_unified_model", tight_layout=False)
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + caption(config))


if __name__ == "__main__":
    main()
