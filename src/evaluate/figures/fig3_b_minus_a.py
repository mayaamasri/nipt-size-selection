"""Task 5B, Figure 3: when does size selection help? (research question 1)

Heatmaps of sens_at_fpr(B) - sens_at_fpr(A), in percentage points, over FF (rows) x depth (columns), one
panel per noise level, diverging scale centred on 0 (shared by the panels), signed values in the cells.
Cells whose approximate bootstrap 95% CI includes 0 are lightly hatched with grey italic values; cells where
both methods are saturated (sens_at_fpr >= 99.95%, bootstrap_diff.SATURATED) show "=".

Reads results/tables/fig3_b_minus_a.csv (python -m src.evaluate.bootstrap_diff). Writes
results/figures/fig3_b_minus_a.png (300 dpi) and .pdf. Run from the repo root:
    python -m src.evaluate.figures.fig3_b_minus_a
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from src.config import load_config, repo_path
from src.evaluate.bootstrap_diff import SATURATED
from src.evaluate.figures._style import FIG_WIDTH, MIN_FONT, MUTED, SURFACE, TEXT, new_figure, save_figure

# Diverging pair of the house palette: red (B worse) - neutral grey - blue (B better)
CMAP = LinearSegmentedColormap.from_list("b_minus_a", ["#e34948", "#f0efec", "#2a78d6"])
NOISE_TEXT = "#8a8985"      # within-noise values
HATCH_COLOR = "#c9c8c3"
FIG_HEIGHT = 3.6            # in, at FIG_WIDTH

CAPTION = (
    "Figure 3. Difference in sensitivity between size selection (B, 40–156 bp) and all fragments (A), "
    "sens_at_fpr(B) − sens_at_fpr(A) in percentage points, both read from the test ROC at FPR 0.135%. "
    "Rows: fetal fraction; columns: depth (fragments passing MAPQ ≥ 40); one panel per noise level. "
    "Blue: B better; red: B worse. Hatched cells with grey italic values: the approximate 95% CI of a paired "
    "bootstrap over the test patients (2,000 resamples) includes 0; the CIs are approximate because the "
    "threshold at FPR 0.135% rests on ~13 of the 10,000 test normals. '=': both methods saturated "
    "(sens_at_fpr ≥ {saturated:g}%). Small values at low FF and 1M mean that both methods fail there "
    "(sensitivity near 0), not that they perform equally well. "
    "Bootstrap SEs range from {se_min:.1f} to {se_max:.1f} points (median {se_med:.1f})."
)


def cell_text(value, saturated):
    if saturated:
        return "="
    v = 100 * value
    return "0.0" if round(v, 1) == 0 else f"{v:+.1f}".replace("-", "−")


def text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else TEXT


def panel(ax, table, ffs, depths, norm, title):
    grid = table.pivot(index="ff", columns="depth", values="diff").loc[ffs, depths]
    image = ax.imshow(100 * grid.to_numpy(), cmap=CMAP, norm=norm, origin="lower", aspect="auto")
    lookup = table.set_index(["ff", "depth"])
    for i, ff in enumerate(ffs):
        for j, depth in enumerate(depths):
            row = lookup.loc[(ff, depth)]
            if row["within_noise"]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="/", edgecolor=HATCH_COLOR,
                                       linewidth=0))
                color, style = NOISE_TEXT, "italic"
                # Small box in the cell's own colour so the hatch lines stop short of the digits
                box = {"facecolor": CMAP(norm(100 * row["diff"])), "edgecolor": "none", "pad": 0.8}
            elif row["saturated"]:
                color, style = MUTED, "normal"
            else:
                color, style = text_color(CMAP(norm(100 * row["diff"]))), "normal"
            ax.text(j, i, cell_text(row["diff"], row["saturated"]), ha="center", va="center", fontsize=MIN_FONT,
                    color=color, style=style, bbox=box if row["within_noise"] else None)
    ax.set_xticks(range(len(depths)), [f"{d / 1e6:g}M" for d in depths])
    ax.set_yticks(range(len(ffs)), [f"{100 * f:g}%" for f in ffs])
    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xlabel("Depth", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, len(depths)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ffs)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1)
    ax.tick_params(which="minor", length=0)
    return image


def main():
    config = load_config()
    table = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "fig3_b_minus_a.csv")
    grid = config["grid"]
    ffs, depths = grid["fetal_fraction"], grid["depth"]
    limit = 10 * np.ceil(10 * table["diff"].abs().max())
    norm = Normalize(-limit, limit)

    fig, axes = new_figure(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95), w_pad=0.02, wspace=0.03)   # room for the key below
    for k, (noise, sigma) in enumerate(zip(grid["noise_sigma"], grid["noise_sigma_values"])):
        label = "No noise" if sigma == 0 else str(noise).capitalize()
        title = f"{label} (σ = {100 * sigma:g}%)"
        image = panel(axes[k], table[table["noise"] == str(noise)], ffs, depths, norm, title)
    axes[0].set_ylabel("Fetal fraction", color=MUTED)
    bar = fig.colorbar(image, ax=axes, fraction=0.04, pad=0.02, aspect=30)
    bar.set_label("Sensitivity difference, B − A (points)", color=MUTED)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)

    fig.text(0.01, 0.01, f"Hatched, grey italic: approximate 95% CI includes 0     "
             f"=  both methods ≥ {100 * SATURATED:g}%", color=MUTED, fontsize=MIN_FONT, ha="left", va="bottom")
    paths = save_figure(fig, config, "fig3_b_minus_a", tight_layout=False)   # shared colorbar

    se = 100 * table.loc[~table["saturated"], "se"]
    print(f"Cells: {len(table)}; within noise: {int(table['within_noise'].sum())}; "
          f"saturated: {int(table['saturated'].sum())}")
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + CAPTION.format(saturated=100 * SATURATED, se_min=se.min(), se_max=se.max(), se_med=se.median()))


if __name__ == "__main__":
    main()
