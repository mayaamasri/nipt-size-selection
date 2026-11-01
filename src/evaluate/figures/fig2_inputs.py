"""Task 5A, Figure 2: inputs of the simulation.

(a) maternal and fetal length distributions after mappability, P_o(l) S(l) normalised (origin_matrices),
    with Yang's 40-156 bp window shaded
(b) survival S(l) by length: the binned pooled S the simulation uses for both origins (step function, Wilson
    95% CI), with maternal and fetal survival as a check (results/tables/survival_by_origin.csv)
(c) exact g (FF enrichment), (d) r (fraction kept) and (e) g*sqrt(r) (B / A separation without noise,
    decision #8) for windows 40-c bp, c = 120..170 at every bp, FF 4% (window_stats, decision #14); shared
    x-axis, markers at the B-opt grid, Yang's 156 bp dotted; (e) labels where g*sqrt(r) first reaches 1 and
    its peak

Writes results/figures/fig2_inputs.png (300 dpi) and .pdf. Run from the repo root
(after python -m src.simulate.survival_by_origin):
    python -m src.evaluate.figures.fig2_inputs
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
import matplotlib.pyplot as plt

from src.evaluate.figures._style import (COLORS, FIG_WIDTH, LINESTYLES, MIN_FONT, MUTED, SURFACE, TEXT, save_figure,
                                         style_axes, style_legend)
from src.simulate.lengths import ORIGINS
from src.simulate.patients import origin_matrices, window_stats

FIG_FF = 0.04                       # brief: panel (c) at FF 4%
UPPER = np.arange(120, 171)         # brief: upper cutoff 120-170 bp at every bp, lower = 40
FIG_HEIGHT = 4.4                    # in, at FIG_WIDTH
S_AXIS = (74, 92)                   # panel (b) y-axis, % (quoted in the caption)
LABEL_BOX = {"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5}   # reference lines pass behind labels
SHADE = "#f1f1ef"                   # LABEL_BOX face inside the shaded "B worse" area of (e)

CAPTION = (
    "Figure 2. Inputs of the simulation. (a) Maternal and fetal fragment-length distributions after the "
    "mappability filter (Rabinowitz G1 lengths × survival); shaded: the 40–156 bp window of Yang et al. "
    "(b) Survival (proper pair, both mates MAPQ ≥ 40) in 10 bp bins with Wilson 95% CI; the simulation uses "
    "the pooled value for both origins, and the maternal and fetal curves show that origin does not change "
    "survival within a bin (y-axis starts at {s_axis_min:g}%). (c–e) Exact values for windows 40–c bp at FF 4%, against "
    "the upper cutoff c (dotted: Yang's {yang} bp): (c) FF enrichment g, which peaks at {g_peak_bp} bp "
    "(g = {g_peak:.2f}; {g_yang:.2f} at {yang} bp); (d) fraction of fragments kept r ({r_yang:.1%} at {yang} bp); "
    "(e) g·√r, the ratio of B's to A's "
    "standardised separation without extra noise (decision #8). Below the dashed line at 1 (cutoffs under "
    "{cross_bp} bp) size selection stops paying off when counting noise is the only noise; it peaks at "
    "{gain_peak_bp} bp ({gain_peak:.2f}). "
    "With extra noise the ratio moves towards g; Fig. 4 compares the full prediction with the simulation."
)


def exact_g_r(matrices, ff, upper, lower):
    stats = [window_stats(matrices, ff, lower, c) for c in upper]
    return np.array([s["gain"] for s in stats]), np.array([s["r"] for s in stats])


def panel_lengths(ax, matrices, window):
    lengths = matrices["lengths"]
    ax.axvspan(window[0], window[1], color=MUTED, alpha=0.12, linewidth=0)
    ax.text((window[0] + window[1]) / 2, 0.98, f"{window[0]}–{window[1]} bp", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=MUTED, fontsize=MIN_FONT)
    for origin in ORIGINS:
        ax.plot(lengths, 100 * matrices[origin].sum(axis=0), color=COLORS[origin],
                linestyle=LINESTYLES[origin], linewidth=1.1, label=origin.capitalize())
    style_axes(ax, "(a) Lengths after mappability", "Fragment length (bp)", "Share (% per bp)")
    ax.set_xlim(lengths[0], lengths[-1])
    style_legend(ax, loc="center right")


def panel_survival(ax, surv):
    bins = surv[surv["bin"] != "all"]
    edges = np.append(bins["lo"].to_numpy(), bins["hi"].iloc[-1] + 1)
    low, high = 100 * bins["S_pooled_lo"].to_numpy(), 100 * bins["S_pooled_hi"].to_numpy()
    ax.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]), step="post",
                    color=COLORS["pooled"], alpha=0.15, linewidth=0)
    ax.stairs(100 * bins["S_pooled"], edges, baseline=None, color=COLORS["pooled"], linewidth=1.4,
              label="Pooled (used), 95% CI")
    for origin in ORIGINS:
        ax.stairs(100 * bins[f"S_{origin}"], edges, baseline=None, color=COLORS[origin], linestyle=LINESTYLES[origin],
                  linewidth=0.7, label=f"{origin.capitalize()} (check)")
    style_axes(ax, "(b) Survival S(l), 10 bp bins", "Fragment length (bp)", "Survival (%)")
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_xlim(edges[0] - 4, edges[-1] + 4)
    ax.set_ylim(*S_AXIS)
    ax.set_yticks([75, 80, 85, 90])
    style_legend(ax, loc="lower right")


def panel_cutoff(ax, values, opt_grid, window, style, title, ylabel):
    """One curve against the upper cutoff (every bp), markers at the B-opt grid, Yang's cutoff dotted."""
    marks = np.isin(UPPER, opt_grid)
    ax.plot(UPPER, values, color=TEXT, linestyle=style["linestyle"], linewidth=1.2)
    ax.plot(UPPER[marks], values[marks], color=TEXT, linestyle="none", marker=style["marker"], markersize=3,
            markerfacecolor=style["face"])
    ax.axvline(window[1], color=MUTED, linestyle=":", linewidth=0.8)
    style_axes(ax, title, "Upper cutoff (bp)", ylabel)
    ax.set_xlim(UPPER[0] - 1, UPPER[-1] + 1)
    ax.set_xticks([120, 140, 160])


def label(ax, text, xy, xytext, ha="left", box=None):
    ax.annotate(text, xy, xytext=xytext, color=TEXT, fontsize=MIN_FONT, ha=ha, va="center", bbox=box or LABEL_BOX,
                zorder=5, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})


def cutoff_panels(axes, matrices, window, opt_grid):
    g, r = exact_g_r(matrices, FIG_FF, UPPER, window[0])
    gain = g * np.sqrt(r)
    i_yang = int(np.flatnonzero(UPPER == window[1])[0])
    i_g = int(np.argmax(g))
    i_gain = int(np.argmax(gain))
    i_cross = int(np.flatnonzero(gain >= 1)[0])     # first cutoff (bp) with g*sqrt(r) >= 1

    ax_g, ax_r, ax_e = axes
    panel_cutoff(ax_g, g, opt_grid, window, {"linestyle": "-", "marker": "o", "face": TEXT}, "(c) g", "FF enrichment (×)")
    ax_g.set_ylim(1, 4)
    label(ax_g, f"peak {UPPER[i_g]} bp\ng = {g[i_g]:.2f}", (UPPER[i_g], g[i_g]), (UPPER[i_g] + 9, 3.85))
    label(ax_g, f"{window[1]} bp\ng = {g[i_yang]:.2f}", (window[1], g[i_yang]), (window[1] - 3, 1.6), ha="right")

    panel_cutoff(ax_r, 100 * r, opt_grid, window, {"linestyle": "--", "marker": "s", "face": "white"},
                 "(d) r", "Fragments kept (%)")
    ax_r.set_ylim(0, 50)
    label(ax_r, f"{window[1]} bp\nr = {100 * r[i_yang]:.1f}%", (window[1], 100 * r[i_yang]),
          (window[1] - 4, 35), ha="right")

    panel_cutoff(ax_e, gain, opt_grid, window, {"linestyle": "-", "marker": "D", "face": "white"},
                 "(e) g·√r", "Gain without noise (×)")
    ax_e.set_ylim(0.5, 1.2)
    ax_e.axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    ax_e.axhspan(0.5, 1, color=MUTED, alpha=0.08, linewidth=0)
    ax_e.text(UPPER[-1], 0.52, "B worse than A", color=MUTED, fontsize=MIN_FONT, ha="right", va="bottom",
              bbox={**LABEL_BOX, "facecolor": SHADE}, zorder=5)
    label(ax_e, f"= 1 at\n{UPPER[i_cross]} bp", (UPPER[i_cross], 1), (UPPER[i_cross] + 4, 0.75),
          box={**LABEL_BOX, "facecolor": SHADE})
    ax_e.text(150, 1.12, f"peak {UPPER[i_gain]} bp, {gain[i_gain]:.2f}", color=TEXT, fontsize=MIN_FONT,
              ha="center", va="bottom", bbox=LABEL_BOX, zorder=5)
    return g, r, gain, i_cross


def caption(config):
    """Caption with its numbers computed from the length/mappability model."""
    window = config["methods"]["size_selection_window_bp"]
    g, r = exact_g_r(origin_matrices(config), FIG_FF, UPPER, window[0])
    gain = g * np.sqrt(r)
    i_yang = int(np.flatnonzero(UPPER == window[1])[0])
    i_g, i_gain = int(np.argmax(g)), int(np.argmax(gain))
    return CAPTION.format(s_axis_min=S_AXIS[0], yang=window[1], g_peak_bp=UPPER[i_g], g_peak=g[i_g],
                          g_yang=g[i_yang], r_yang=r[i_yang], cross_bp=UPPER[int(np.flatnonzero(gain >= 1)[0])],
                          gain_peak_bp=UPPER[i_gain], gain_peak=gain[i_gain])


def main():
    config = load_config()
    matrices = origin_matrices(config)
    surv = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "survival_by_origin.csv")
    window = config["methods"]["size_selection_window_bp"]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 6)
    ax_a, ax_b = fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])
    ax_c = fig.add_subplot(grid[1, :2])
    ax_d, ax_e = fig.add_subplot(grid[1, 2:4], sharex=ax_c), fig.add_subplot(grid[1, 4:], sharex=ax_c)
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.set_facecolor(SURFACE)
    panel_lengths(ax_a, matrices, window)
    panel_survival(ax_b, surv)
    g, r, gain, i_cross = cutoff_panels((ax_c, ax_d, ax_e), matrices, window, config["methods"]["size_opt_upper_grid"])
    paths = save_figure(fig, config, "fig2_inputs")

    k = i_cross - 1
    exact = UPPER[k] + (1 - gain[k]) / (gain[k + 1] - gain[k])
    i_g, i_gain = int(np.argmax(g)), int(np.argmax(gain))
    print(f"FF {FIG_FF:.0%}: g peaks at {UPPER[i_g]} bp (g = {g[i_g]:.4f}, r = {r[i_g]:.4f})")
    print(f"g*sqrt(r) first >= 1 at {UPPER[i_cross]} bp ({gain[i_cross]:.4f}; interpolated crossing {exact:.2f} bp); "
          f"peaks at {UPPER[i_gain]} bp ({gain[i_gain]:.4f})")
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + caption(config))


if __name__ == "__main__":
    main()
