"""Task 5A, Figure 2: inputs of the simulation.

(a) maternal and fetal length distributions after mappability, P_o(l) S(l) normalised (origin_matrices),
    with Yang's 40-156 bp window shaded
(b) survival S(l) by length: the binned pooled S the simulation uses for both origins (step function, Wilson
    95% CI), with maternal and fetal survival as a check (results/tables/survival_by_origin.csv)
(c) exact g (FF enrichment) and r (fraction kept) for windows 40-c bp, c = 120..170 at every bp, FF 4%
    (window_stats, decision #14); markers at the B-opt grid, Yang's 156 bp marked

Writes results/figures/fig2_inputs.png (300 dpi) and .pdf. Run from the repo root
(after python -m src.simulate.survival_by_origin):
    python -m src.evaluate.figures.fig2_inputs
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.figures._style import (COLORS, LINESTYLES, MUTED, TEXT, new_figure, save_figure,
                                         style_axes, style_legend)
from src.simulate.lengths import ORIGINS
from src.simulate.patients import origin_matrices, window_stats

FIG_FF = 0.04                       # brief: panel (c) at FF 4%
UPPER = np.arange(120, 171)         # brief: upper cutoff 120-170 bp at every bp, lower = 40

CAPTION = (
    "Figure 2. Inputs of the simulation. (a) Maternal and fetal fragment-length distributions after the "
    "mappability filter (Rabinowitz G1 lengths × survival); shaded: the 40–156 bp window of Yang et al. "
    "(b) Survival (proper pair, both mates MAPQ ≥ 40) in 10 bp bins with Wilson 95% CI; the simulation uses "
    "the pooled value for both origins, and the maternal and fetal curves show that origin does not change "
    "survival within a bin (y-axis starts at 74%). (c) Exact FF enrichment g and fraction of fragments kept r for windows 40–c bp at "
    "FF 4%; g peaks at 132 bp (g = 3.69), and Yang's 156 bp window gives g = 2.43, r = 20.3%."
)


def exact_g_r(matrices, ff, upper, lower):
    stats = [window_stats(matrices, ff, lower, c) for c in upper]
    return np.array([s["gain"] for s in stats]), np.array([s["r"] for s in stats])


def panel_lengths(ax, matrices, window):
    lengths = matrices["lengths"]
    ax.axvspan(window[0], window[1], color=MUTED, alpha=0.12, linewidth=0)
    ax.text((window[0] + window[1]) / 2, 0.98, f"{window[0]}–{window[1]} bp", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=MUTED, fontsize=8)
    for origin in ORIGINS:
        ax.plot(lengths, 100 * matrices[origin].sum(axis=0), color=COLORS[origin],
                linestyle=LINESTYLES[origin], linewidth=1.6, label=origin.capitalize())
    style_axes(ax, "(a) Length distributions after mappability", "Fragment length (bp)", "Share of fragments (% per bp)")
    ax.set_xlim(lengths[0], lengths[-1])
    style_legend(ax, loc="center right")


def panel_survival(ax, surv):
    bins = surv[surv["bin"] != "all"]
    edges = np.append(bins["lo"].to_numpy(), bins["hi"].iloc[-1] + 1)
    low, high = 100 * bins["S_pooled_lo"].to_numpy(), 100 * bins["S_pooled_hi"].to_numpy()
    ax.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]), step="post",
                    color=COLORS["pooled"], alpha=0.15, linewidth=0)
    ax.stairs(100 * bins["S_pooled"], edges, baseline=None, color=COLORS["pooled"], linewidth=2,
              label="Pooled (used), 95% CI")
    for origin in ORIGINS:
        ax.stairs(100 * bins[f"S_{origin}"], edges, baseline=None, color=COLORS[origin], linestyle=LINESTYLES[origin],
                  linewidth=1, label=f"{origin.capitalize()} (check)")
    style_axes(ax, "(b) Survival S(l), 10 bp bins", "Fragment length (bp)", "Survival (%)")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(74, 92)
    style_legend(ax, loc="lower right")


def panel_g_r(ax, matrices, window, opt_grid):
    g, r = exact_g_r(matrices, FIG_FF, UPPER, window[0])
    marks = np.isin(UPPER, opt_grid)
    ax.plot(UPPER, g, color=TEXT, linestyle="-", linewidth=1.6)
    ax.plot(UPPER[marks], g[marks], color=TEXT, linestyle="none", marker="o", markersize=4)
    ax.set_ylim(0, 4.2)
    ax_r = ax.twinx()
    ax_r.plot(UPPER, 100 * r, color=MUTED, linestyle="--", linewidth=1.4)
    ax_r.plot(UPPER[marks], 100 * r[marks], color=MUTED, linestyle="none", marker="s", markersize=4,
              markerfacecolor="white")
    ax_r.set_ylim(0, 50)
    ax_r.set_ylabel("r, fragments kept (%) — dashed, squares", color=MUTED)
    ax_r.tick_params(colors=MUTED)
    for side in ("top", "left", "bottom"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_color("#e4e3df")

    i_peak = int(np.argmax(g))
    ax.annotate(f"peak {UPPER[i_peak]} bp, g = {g[i_peak]:.2f}", (UPPER[i_peak], g[i_peak]),
                xytext=(UPPER[i_peak] + 6, g[i_peak] + 0.25), color=TEXT, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    i_yang = int(np.flatnonzero(UPPER == window[1])[0])
    ax.axvline(window[1], color=MUTED, linestyle=":", linewidth=1)
    ax.text(window[1] - 1, 0.15, f"Yang {window[1]} bp\ng = {g[i_yang]:.2f}, r = {100 * r[i_yang]:.1f}%",
            color=MUTED, ha="right", fontsize=8)
    ax.text(UPPER[-1], g[-1] + 0.2, "g", color=TEXT, ha="right", fontsize=9)
    i_r = int(np.flatnonzero(UPPER == 165)[0])
    ax_r.text(UPPER[i_r] + 1, 100 * r[i_r] - 3, "r", color=MUTED, ha="left", va="top", fontsize=9)
    style_axes(ax, f"(c) Exact g and r, window 40–c bp, FF {100 * FIG_FF:.0f}%", "Upper cutoff c (bp)",
               "g, FF enrichment (×) — solid, circles")
    ax.set_xlim(UPPER[0] - 1, UPPER[-1] + 1)
    return g, r


def main():
    config = load_config()
    matrices = origin_matrices(config)
    surv = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "survival_by_origin.csv")
    window = config["methods"]["size_selection_window_bp"]

    fig, axes = new_figure(1, 3, figsize=(15, 4.4))
    panel_lengths(axes[0], matrices, window)
    panel_survival(axes[1], surv)
    g, r = panel_g_r(axes[2], matrices, window, config["methods"]["size_opt_upper_grid"])
    paths = save_figure(fig, config, "fig2_inputs")

    i_peak = int(np.argmax(g))
    print(f"FF {FIG_FF:.0%}: g peaks at {UPPER[i_peak]} bp (g = {g[i_peak]:.4f}, r = {r[i_peak]:.4f})")
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + CAPTION)


if __name__ == "__main__":
    main()
