"""Task 5C, Figure 4: does the break-even formula predict the simulation? (research question 1)

FF 4%, one panel per noise level: x = depth (log), y = sep(B_c) / sep(A), sep = (z_mean_t21 - z_mean_normal) /
z_sd_normal, one series per cutoff (130, 140, 150 from B_c; 156 is B). Points: simulation; lines: decision #8,
ratio(N) = g sqrt((1/(p0 N) + sigma^2) / (1/(p0 r N) + sigma^2)) with that cutoff's exact g and r at FF 4%.
Dashed line at 1 (B = A); predicted break-even depths N* inside the plotted range are marked on it.

Reads results/tables/breakeven_check.csv, breakeven_summary.csv (python -m src.evaluate.breakeven_check) and
breakeven_ratio_noise.csv (python -m src.evaluate.bootstrap_ratio, quoted in docs/revision_numbers.md).
Writes results/figures/fig4_breakeven.png (300 dpi) and .pdf. Run from the repo root:
    python -m src.evaluate.figures.fig4_breakeven
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
from src.evaluate import labels as L
from src.evaluate.breakeven_check import predicted_ratio
from src.evaluate.figures._style import FIG_WIDTH, MIN_FONT, MUTED, TEXT, new_figure, save_figure, style_axes

FIG_FF = 0.04
CUTOFFS = (130, 140, 150, 156)
FIG_HEIGHT = 2.6                      # in, at FIG_WIDTH
DEPTH_RANGE = (0.8e6, 11e6)
# Ordered cutoffs: one hue, light to dark, plus marker and line style for greyscale
SERIES = {130: ("#5a8fd3", "o", ":"), 140: ("#2f6fc0", "s", "-."), 150: ("#1f5399", "^", "--"),
          156: ("#123766", "D", "-")}

CAPTION = (
    "Figure 4. Observed and predicted gain of size selection at FF 4%: ratio of standardised separations "
    "sep(B 40–$c$)/sep(A), sep = (mean $z$ of T21 − mean $z$ of normals)/SD of normals, for $c$ = 130, 140, 150 "
    "and 156 bp (B). Points: simulation; lines: break-even formula (see Methods) with the exact $g$ and $r$ of each "
    r"cutoff at FF 4% and $p_0 = q(\mathrm{chr21})$. Dashed: B = A; $N^*$: predicted break-even depth, where a line "
    "crosses B = A."
)


def nstar_label(n):
    return f"{n / 1e6:.1f}M" if n < 1e7 else f"{n / 1e6:.0f}M"


def panel(ax, data, sigma, title):
    depths = np.geomspace(*DEPTH_RANGE, 200)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    marks = []
    for cutoff in CUTOFFS:
        color, marker, style = SERIES[cutoff]
        d = data[data["cutoff"] == cutoff].sort_values("depth")
        g, r, p0 = d[["g", "r", "p0"]].iloc[0]
        ax.plot(depths, predicted_ratio(g, r, p0, depths, sigma), color=color, linestyle=style,
                linewidth=1.4 if style == ":" else 1.1)
        ax.plot(d["depth"], d["ratio_obs"], color=color, linestyle="none", marker=marker, markersize=3.5,
                markeredgecolor=TEXT, markeredgewidth=0.4)
        n = d["breakeven_depth"].iloc[0]
        if np.isfinite(n):
            marks.append((cutoff, n))
            if DEPTH_RANGE[0] <= n <= DEPTH_RANGE[1]:
                ax.plot([n, n], [0.97, 1.03], color=TEXT, linewidth=0.8)
                ax.text(n, 0.955, f"N* {nstar_label(n)}\n({L.cutoff(cutoff)})", color=TEXT, fontsize=MIN_FONT,
                        ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlim(*DEPTH_RANGE)
    ax.set_xticks([1e6, 2.5e6, 5e6, 1e7], ["1M", "2.5M", "5M", "10M"])
    ax.minorticks_off()
    style_axes(ax, title, "Depth", "")
    return marks


def caption(config):
    """Caption (describes what is shown; break-even depths and prediction errors are in breakeven_check.csv and
    breakeven_summary.csv)."""
    return CAPTION


def main():
    config = load_config()
    table = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "breakeven_check.csv")
    data = table[np.isclose(table["ff"], FIG_FF) & table["cutoff"].isin(CUTOFFS)]
    grid = config["grid"]

    fig, axes = new_figure(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88), wspace=0.04)    # room for the legend above
    for ax, noise, sigma in zip(axes, grid["noise_sigma"], grid["noise_sigma_values"]):
        panel(ax, data[data["noise"] == str(noise)], sigma, L.noise_title(noise, sigma))
    axes[0].set_ylabel("sep(B) / sep(A)", color=MUTED)
    axes[0].set_ylim(0.7, 1.65)

    handles = [Line2D([], [], color=SERIES[c][0], marker=SERIES[c][1], linestyle=SERIES[c][2], markersize=3.5,
                      markeredgecolor=TEXT, markeredgewidth=0.4, label=L.cutoff(c))
               for c in CUTOFFS]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
                        handlelength=2.6, columnspacing=1.2)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.text(0.98, 0.985, "lines: formula · points: simulation", color=MUTED, fontsize=MIN_FONT, ha="right",
             va="top")
    paths = save_figure(fig, config, "fig4_breakeven", tight_layout=False)
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + caption(config))


if __name__ == "__main__":
    main()
