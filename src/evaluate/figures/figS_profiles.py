"""Task 07, supplementary figure: the gain over A under different fetal length profiles (decision #17).

FF 4%, realistic noise, one panel per profile (G1 and the five scenarios): x = depth (log), y = sep(method) /
sep(A), sep = (z_mean_t21 - z_mean_normal) / z_sd_normal, for B (40-156 bp), W (weights from the patients' true
profile) and W_G1 (weights from the published G1 profile; in G1 it is W, so it is not drawn there). Points:
simulation; lines: the unified model (Task 06) with u = the method's weights and E[.], w* from the patients'
profile, i.e. decision #8's formula with g_eff and r_eff (depth-independent), drawn with
breakeven_check.predicted_ratio. Dashed line at 1.

Reads results/tables/robustness_profiles_pred.csv (python -m src.evaluate.robustness_inputs), each profile's
results_all.csv (results/tables for G1, results/robustness/<NAME>/tables) and robustness_profiles_summary.csv
(python -m src.evaluate.robustness_profiles, for the caption). Writes results/figures/figS_profiles.png (300 dpi)
and .pdf. Run from the repo root:
    python -m src.evaluate.figures.figS_profiles
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
from src.evaluate.breakeven_check import predicted_ratio
from src.evaluate.figures._style import FIG_WIDTH, MIN_FONT, MUTED, TEXT, new_figure, save_figure, style_axes
from src.evaluate.figures.fig4_breakeven import DEPTH_RANGE
from src.evaluate.robustness_profiles import MAIN, POOLED, profiles

FIG_FF = 0.04
NOISE = "realistic"
FIG_HEIGHT = 4.3                      # in, two rows of three panels
# method: (label, colour, marker, line style); B and W as in Fig. S3, W_G1 in the maternal slot
SERIES = {"B": ("B (40–156 bp)", "#123766", "D", "-"),
          "W": ("W (weights from the true profile)", "#eb6834", "o", "--"),
          "W_G1": ("W_G1 (weights from G1)", "#2a78d6", "s", ":")}

CAPTION = (
    "Figure S4. Gain over A under different fetal length profiles, FF 4%, realistic noise (σ = 0.35%): ratio "
    "of standardised separations sep(method)/sep(A) for B (40–156 bp), W (weights P(fetal | length) from the "
    "patients' true profile) and W_G1 (weights from the G1 profile used in the main analysis, as a lab using a "
    "published profile would; in G1 it equals W). Scenarios shift the fetal length distribution by −10 to +10 "
    "bp (fetal median {medians}; maternal 172 bp) or blend it 50:50 with the maternal one (fetal median "
    "{blend_median} bp); all groups, including the reference, are simulated from the scenario profile. Points: "
    "simulation; lines: the unified counting model (Fig. S3) with each method's weights and the scenario's "
    "true length distributions. Over all 96 conditions per scenario, the model's median |relative error| is "
    "{b_lo:.1%}–{b_hi:.1%} for B and {wg_lo:.1%}–{wg_hi:.1%} for W_G1. Dashed: method = A."
)


def observed(results):
    sep = (results["z_mean_t21"] - results["z_mean_normal"]) / results["z_sd_normal"]
    r = results.assign(sep=sep)
    r = r[np.isclose(r["ff"], FIG_FF) & (r["noise"] == NOISE)]
    a = r[r["method"] == "A"].set_index("depth")["sep"]
    return {m: (r[r["method"] == m].set_index("depth")["sep"] / a).sort_index() for m in SERIES}


def panel(ax, name, pred, obs, sigma, title):
    depths = np.geomspace(*DEPTH_RANGE, 200)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    for method, (_, color, marker, style) in SERIES.items():
        if name == MAIN and method == "W_G1":
            continue
        p = pred[pred["method"] == method].iloc[0]
        ax.plot(depths, predicted_ratio(p["g_eff"], p["r_eff"], p["p0"], depths, sigma), color=color,
                linestyle=style, linewidth=1.4 if style == ":" else 1.1)
        o = obs[method]
        ax.plot(o.index, o.to_numpy(), color=color, linestyle="none", marker=marker, markersize=3.5,
                markeredgecolor=TEXT, markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlim(*DEPTH_RANGE)
    ax.set_xticks([1e6, 2.5e6, 5e6, 1e7], ["1M", "2.5M", "5M", "10M"])
    ax.minorticks_off()
    style_axes(ax, title, "", "")


def titles(config):
    inputs = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "robustness_profiles_inputs.csv")
    out = {}
    for r in inputs.itertuples():
        what = ("main analysis" if r.profile == MAIN else f"blend {100 * r.blend_maternal:g}%" if r.blend_maternal
                else f"shift {r.shift_bp:+d} bp".replace("-", "−"))
        out[r.profile] = (f"{r.profile} ({what})", r.fetal_median_bp)
    return out


def caption(config):
    """Caption with its numbers computed from the inputs and the summary table."""
    tables = repo_path(config["paths"]["results"]) / "tables"
    inputs = pd.read_csv(tables / "robustness_profiles_inputs.csv").set_index("profile")
    s = pd.read_csv(tables / "robustness_profiles_summary.csv", dtype={"question": str, "depth": str, "noise": str})
    s = s[(s["question"] == "3") & ~s["profile"].isin([MAIN, POOLED])].set_index(["profile", "item"])["value"]
    b = s.xs("median_abs_rel_error_B_all", level="item")
    wg = s.xs("median_abs_rel_error_W_G1_all", level="item")
    shifts = inputs[inputs["blend_maternal"] == 0].sort_values("shift_bp")
    blend = inputs[inputs["blend_maternal"] > 0]
    return CAPTION.format(
        medians=", ".join(f"{m} bp" for m in shifts["fetal_median_bp"]) + f" for shifts of "
                + ", ".join(f"{v:+d}".replace("-", "−") for v in shifts["shift_bp"]) + " bp",
        blend_median=int(blend["fetal_median_bp"].iloc[0]),
        b_lo=b.min(), b_hi=b.max(), wg_lo=wg.min(), wg_hi=wg.max())


def main():
    config = load_config()
    grid = config["grid"]
    sigma = grid["noise_sigma_values"][grid["noise_sigma"].index(NOISE)]
    pred = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "robustness_profiles_pred.csv",
                       dtype={"noise": str})
    pred = pred[np.isclose(pred["ff"], FIG_FF) & (pred["noise"] == NOISE)]
    head = titles(config)

    fig, axes = new_figure(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93), wspace=0.04, hspace=0.06)   # room for the legend above
    for ax, (name, d) in zip(axes.flat, profiles(config).items()):
        res = pd.read_csv(d / "results_all.csv", dtype={"noise": str})
        title, median = head[name]
        panel(ax, name, pred[pred["profile"] == name], observed(res), sigma, f"{title}\nfetal median {median} bp")
    for ax in axes[1]:
        ax.set_xlabel("Depth", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("sep(method) / sep(A)", color=MUTED)
    axes[0, 0].set_ylim(0.75, 2.3)

    handles = [Line2D([], [], color=color, marker=marker, linestyle=style, markersize=3.5, markeredgecolor=TEXT,
                      markeredgewidth=0.4, label=label) for label, color, marker, style in SERIES.values()]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
                        handlelength=2.6, columnspacing=1.0)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.text(0.98, 0.955, "lines: model · points: simulation", color=MUTED, fontsize=MIN_FONT, ha="right",
             va="top")
    paths = save_figure(fig, config, "figS_profiles", tight_layout=False)
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + caption(config))


if __name__ == "__main__":
    main()
