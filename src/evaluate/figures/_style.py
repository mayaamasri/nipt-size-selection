"""Shared style for the manuscript figures (Task 05): same palette and axes as the QC plots.

Figures are made at their final size for the NeurIPS 2026 template (single column, text width 397.485 pt =
5.5 in), so LaTeX includes them at width=\\textwidth with no scaling: FIG_WIDTH wide (or less), no text smaller
than MIN_FONT pt. save_figure checks both and reports the smallest text size.

Greyscale: every series also differs by line style and/or marker, never by colour alone.
save_figure writes results/figures/<name>.png (300 dpi) and <name>.pdf.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from src.config import repo_path  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Same slots as the QC plots: maternal slot 1, fetal slot 2, pooled/neutral in ink
COLORS = {"maternal": "#2a78d6", "fetal": "#eb6834", "pooled": TEXT}
LINESTYLES = {"maternal": "-", "fetal": "--", "pooled": "-"}
DPI = 300

FIG_WIDTH = 5.5        # in: NeurIPS 2026 text width (397.485 pt)
MIN_FONT = 7           # pt at final size
FONT_TITLE = 8
FONT_LABEL = 7.5
plt.rcParams.update({
    "font.size": MIN_FONT,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": MIN_FONT,
    "ytick.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "hatch.linewidth": 0.5,
})


def new_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """figsize defaults to (FIG_WIDTH, 0.6 * FIG_WIDTH)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (FIG_WIDTH, 0.6 * FIG_WIDTH), facecolor=SURFACE,
                             **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor(SURFACE)
    return fig, axes


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
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


def smallest_font(fig):
    """Smallest font size (pt) among the figure's visible, non-empty texts."""
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes)


def save_figure(fig, config, name, tight_layout=True):
    """Save <name>.png (300 dpi) and <name>.pdf in results/figures/; returns both paths.

    Raises if the figure is wider than FIG_WIDTH or has text below MIN_FONT; prints the final size and the
    smallest text. tight_layout=False for figures whose layout tight_layout cannot handle (e.g. a colorbar
    shared by several axes); they keep their set size and use constrained layout instead.
    """
    width, height = fig.get_size_inches()
    if width > FIG_WIDTH + 1e-9:
        raise ValueError(f"{name}: width {width} in exceeds FIG_WIDTH = {FIG_WIDTH} in")
    small = smallest_font(fig)
    if small < MIN_FONT:
        raise ValueError(f"{name}: text at {small} pt, below MIN_FONT = {MIN_FONT} pt")
    out_dir = repo_path(config["paths"]["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    if tight_layout:
        fig.tight_layout(pad=0.6)   # keeps edge tick labels (e.g. "500") clear of the figure edge
    for p in paths:
        fig.savefig(p, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"{name}: {width:.2f} x {height:.2f} in, smallest text {small:g} pt")
    return paths
