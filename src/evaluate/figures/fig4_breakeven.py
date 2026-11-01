"""Task 5C, Figure 4: does the break-even formula predict the simulation? (research question 1)

FF 4%, one panel per noise level: x = depth (log), y = sep(B_c) / sep(A), sep = (z_mean_t21 - z_mean_normal) /
z_sd_normal, one series per cutoff (130, 140, 150 from B_c; 156 is B). Points: simulation; lines: decision #8,
ratio(N) = g sqrt((1/(p0 N) + sigma^2) / (1/(p0 r N) + sigma^2)) with that cutoff's exact g and r at FF 4%.
Dashed line at 1 (B = A); predicted break-even depths N* inside the plotted range are marked on it.

Reads results/tables/breakeven_check.csv and breakeven_summary.csv (python -m src.evaluate.breakeven_check).
Writes results/figures/fig4_breakeven.png (300 dpi) and .pdf. Run from the repo root:
    python -m src.evaluate.figures.fig4_breakeven
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
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
    "sep(B_c)/sep(A), sep = (mean z of T21 − mean z of normals)/SD of normals, for upper cutoffs 130, 140, 150 "
    "and 156 bp (B). Points: simulation; lines: break-even formula (decision #8) with the exact g and r of each "
    "cutoff at FF 4% and p0 = q(chr21). Dashed: B = A. Without extra noise the ratio is g·√r at every depth "
    "(Fig. 2e); noise lifts it towards g as depth grows, so strict cutoffs that lose at low depth win at high "
    "depth. Predicted break-even depths: {nstar_text}. Over all {n_all} FF × depth × noise × cutoff "
    "combinations (breakeven_check.csv), the median |relative error| is {rel_med_all:.1%} (95th percentile "
    "{rel_p95_all:.1%}); without the {n_low} combinations where A's separation is below 1, {rel_med:.1%} "
    "({rel_p95:.1%}). A paired bootstrap of the test patients gives a sampling SD of the observed ratio of "
    "0.9–2.3%, so the typical error is within sampling noise. The mean signed error is {bias:+.1%} "
    "(t = {bias_t:.1f}), with no trend across FF or depth."
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
                ax.text(n, 0.955, f"N* {nstar_label(n)}\n({cutoff} bp)", color=TEXT, fontsize=MIN_FONT,
                        ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlim(*DEPTH_RANGE)
    ax.set_xticks([1e6, 2.5e6, 5e6, 1e7], ["1M", "2.5M", "5M", "10M"])
    ax.minorticks_off()
    style_axes(ax, title, "Depth", "")
    return marks


def main():
    config = load_config()
    tables = repo_path(config["paths"]["results"]) / "tables"
    table = pd.read_csv(tables / "breakeven_check.csv")
    summary = pd.read_csv(tables / "breakeven_summary.csv")
    data = table[np.isclose(table["ff"], FIG_FF) & table["cutoff"].isin(CUTOFFS)]
    grid = config["grid"]

    fig, axes = new_figure(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88), wspace=0.04)    # room for the legend above
    nstar = []
    for ax, noise, sigma in zip(axes, grid["noise_sigma"], grid["noise_sigma_values"]):
        label = "No noise" if sigma == 0 else str(noise).capitalize()
        marks = panel(ax, data[data["noise"] == str(noise)], sigma, f"{label} (σ = {100 * sigma:g}%)")
        nstar += [f"{c} bp at {label.lower()} noise, {nstar_label(n)}" for c, n in marks]
    axes[0].set_ylabel("sep(B) / sep(A)", color=MUTED)
    axes[0].set_ylim(0.7, 1.65)

    handles = [Line2D([], [], color=SERIES[c][0], marker=SERIES[c][1], linestyle=SERIES[c][2], markersize=3.5,
                      markeredgecolor=TEXT, markeredgewidth=0.4, label=f"{c} bp" + (" (B)" if c == 156 else ""))
               for c in CUTOFFS]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
                        handlelength=2.6, columnspacing=1.2)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.text(0.98, 0.985, "lines: formula · points: simulation", color=MUTED, fontsize=MIN_FONT, ha="right",
             va="top")
    paths = save_figure(fig, config, "fig4_breakeven", tight_layout=False)

    row = summary[summary["group"] == "overall"].set_index(["value", "rows"])
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + CAPTION.format(
        nstar_text="; ".join(nstar) + "; none for 150 and 156 bp (g²r ≥ 1)",
        n_all=int(row.loc[("all", "all"), "n"]), n_low=int(row.loc[("all", "all"), "n"]
                                                          - row.loc[("all", "without_low_signal"), "n"]),
        rel_med_all=row.loc[("all", "all"), "rel_median"], rel_p95_all=row.loc[("all", "all"), "rel_p95"],
        rel_med=row.loc[("all", "without_low_signal"), "rel_median"],
        rel_p95=row.loc[("all", "without_low_signal"), "rel_p95"],
        bias=row.loc[("all", "without_low_signal"), "rel_mean_signed"],
        bias_t=row.loc[("all", "without_low_signal"), "rel_mean_t"]))


if __name__ == "__main__":
    main()
