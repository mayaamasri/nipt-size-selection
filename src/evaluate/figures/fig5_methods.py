"""Task 5D, Figure 5: which method is best? (research question 2)

sens_at_fpr against FF, one panel per depth (2 x 2), for A, B, W, C1, C2. Main figure at realistic noise;
the same figure at no and low noise as supplementary. Tree points whose own calibrated threshold flags
nobody (nobody_positive) have hollow markers. Light reference lines: 95% (detection-limit target, Table 2)
and FF 4% (commercial minimum).

Colours: the house categorical order with violet in place of yellow (validated with the dataviz palette
validator, light mode: all checks pass; aqua and magenta are below 3:1 contrast on the surface, so every
series also has its own marker with a dark edge and a line style, z-score methods solid, trees dashed).

Reads results/tables/results_all.csv. Writes results/figures/fig5_methods.{png,pdf} (realistic) and
figS1_methods_none.{png,pdf}, figS2_methods_low.{png,pdf}. Run from the repo root:
    python -m src.evaluate.figures.fig5_methods
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
from src.evaluate.figures._style import FIG_WIDTH, MUTED, TEXT, new_figure, save_figure, style_axes

FIG_HEIGHT = 4.3                      # in, at FIG_WIDTH
MAIN_NOISE = "realistic"
COMMERCIAL_MIN_FF = 0.04
REF = "#9a9995"                       # light grey reference lines
METHODS = {   # colour, marker, line style
    "A": ("#2a78d6", "o", "-"),
    "B": ("#eb6834", "s", "-"),
    "W": ("#1baf7a", "^", "-"),
    "C1": ("#6250d6", "D", "--"),
    "C2": ("#e87ba4", "v", "--"),
}
TREES = ("C1", "C2")

CAPTION_MAIN = (
    "Figure 5. Sensitivity at FPR 0.135% (read from the test ROC) against fetal fraction for all fragments "
    "(A), size selection 40–156 bp (B), length-weighted z-score (W) and the decision trees on the count "
    "z-score (C1) and on the count, short-fragment and length-weighted z-scores (C2), at realistic noise "
    r"($\sigma = 0.35\%$); one panel per depth; 10,000 euploid and 2,000 trisomic test patients per condition. "
    "Hollow markers: the tree's own trained threshold flags nobody in that condition; its actual sensitivity "
    "there is 0%, which is what the detection limits in Table 2 use. W uses the true FF in its weights; W (FF "
    "{fixed_ff:.0%}), with the weights fixed at FF {fixed_ff:.0%}, is compared in Tables S3 and S4. Dashed grey: "
    "95% (detection-limit target); dotted grey: FF 4% (commercial minimum)."
)
CAPTION_SUPP = "Figure {label}. As Fig. 5, at {noise} noise ($\\sigma = {sigma:g}\\%$)."


def panel(ax, data, depth, ffs):
    ax.axhline(95, color=REF, linestyle="--", linewidth=0.6)
    ax.axvline(100 * COMMERCIAL_MIN_FF, color=REF, linestyle=":", linewidth=0.7)
    for method, (color, marker, style) in METHODS.items():
        d = data[data["method"] == method].set_index("ff").loc[ffs]
        x, y = 100 * np.asarray(ffs), 100 * d["sens_at_fpr"].to_numpy()
        ax.plot(x, y, color=color, linestyle=style, linewidth=1.0, zorder=2)
        hollow = d["nobody_positive"].fillna(False).astype(bool).to_numpy() if method in TREES else np.zeros(len(x), bool)
        for mask, face in ((~hollow, color), (hollow, "white")):
            ax.plot(x[mask], y[mask], linestyle="none", marker=marker, markersize=3.5, markerfacecolor=face,
                    markeredgecolor=TEXT if face == color else color, markeredgewidth=0.5 if face == color else 0.9,
                    zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(100 * np.asarray(ffs), [f"{100 * f:g}" for f in ffs])
    ax.minorticks_off()
    ax.set_xlim(100 * ffs[0] / 1.12, 100 * ffs[-1] * 1.12)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axes(ax, f"{depth / 1e6:g}M fragments", "", "")


def legend_handles():
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, linewidth=1.0, markersize=3.5,
                      markeredgecolor=TEXT, markeredgewidth=0.5, label=name)
               for name, (c, m, s) in METHODS.items()]
    handles.append(Line2D([], [], color=MUTED, marker="D", linestyle="none", markersize=3.5,
                          markerfacecolor="white", markeredgewidth=0.9, label="hollow: tree flags nobody"))
    return handles


def make_figure(config, results, noise, name):
    grid = config["grid"]
    ffs, depths = grid["fetal_fraction"], grid["depth"]
    data = results[results["noise"] == str(noise)]
    fig, axes = new_figure(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93), h_pad=0.04, w_pad=0.04)   # room for the legend above
    for ax, depth in zip(axes.flat, depths):
        panel(ax, data[data["depth"] == depth], depth, ffs)
    for ax in axes[1]:
        ax.set_xlabel("Fetal fraction (%)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sensitivity at FPR 0.135% (%)", color=MUTED)
    legend = fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6,
                        frameon=False, handlelength=2.2, columnspacing=1.0, handletextpad=0.4)
    for t in legend.get_texts():
        t.set_color(TEXT)
    return save_figure(fig, config, name, tight_layout=False)


def captions(config):
    """{figure name: caption} (they describe what is shown; the W vs W_fixedFF numbers are in Tables S3, S4)."""
    grid = config["grid"]
    sigma = dict(zip(grid["noise_sigma"], grid["noise_sigma_values"]))
    out = {"fig5_methods": CAPTION_MAIN.format(fixed_ff=config["methods"]["weight_fixed_ff"])}
    for k, noise in enumerate([n for n in grid["noise_sigma"] if n != MAIN_NOISE], start=1):
        out[f"figS{k}_methods_{noise}"] = CAPTION_SUPP.format(label=f"S{k}", noise="no" if sigma[noise] == 0 else noise,
                                                           sigma=100 * sigma[noise])
    return out


def main():
    config = load_config()
    results = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "results_all.csv")
    grid = config["grid"]

    paths = make_figure(config, results, MAIN_NOISE, "fig5_methods")
    for k, noise in enumerate([n for n in grid["noise_sigma"] if n != MAIN_NOISE], start=1):
        paths += make_figure(config, results, noise, f"figS{k}_methods_{noise}")
    print("Saved " + ", ".join(str(p) for p in paths))
    for name, text in captions(config).items():
        print(f"\nCaption ({name}):\n{text}")


if __name__ == "__main__":
    main()
