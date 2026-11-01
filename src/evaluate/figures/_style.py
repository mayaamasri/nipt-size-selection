"""Shared style for the manuscript figures (Task 05): same palette and axes as the QC plots.

Greyscale: every series also differs by line style and/or marker, never by colour alone.
save_figure writes results/figures/<name>.png (300 dpi) and <name>.pdf.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import repo_path  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Same slots as the QC plots: maternal slot 1, fetal slot 2, pooled/neutral in ink
COLORS = {"maternal": "#2a78d6", "fetal": "#eb6834", "pooled": TEXT}
LINESTYLES = {"maternal": "-", "fetal": "--", "pooled": "-"}
DPI = 300


def new_figure(nrows=1, ncols=1, figsize=(8, 4.5), **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor(SURFACE)
    return fig, axes


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def style_legend(ax, **kwargs):
    legend = ax.legend(frameon=False, **kwargs)
    for t in legend.get_texts():
        t.set_color(TEXT)
    return legend


def save_figure(fig, config, name):
    """Save <name>.png (300 dpi) and <name>.pdf in results/figures/; returns both paths."""
    out_dir = repo_path(config["paths"]["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return paths
