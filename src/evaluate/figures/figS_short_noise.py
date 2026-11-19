"""Task 08, supplementary figure: the gain over A with extra noise on short fragments (decision #18).

FF 4%, one panel per noise level (low, realistic): x = depth (log), y = sep(method) / sep(A), sep =
(z_mean_t21 - z_mean_normal) / z_sd_normal, for B (40-156 bp) and W at k = 1 (main analysis), 1.5 and 2, where
the chr21 cells of the 40-156 bp window get extra noise so that the window's chr21 fraction has relative SD
~ sigma k. Points: simulation; lines: the unified model with sigma_u (Task 08, src.evaluate.unified_model_check),
i.e. g_eff sqrt((1/(p0 N) + sigma_A^2) / (1/(p0 r_eff N) + sigma_u^2)) drawn with
breakeven_check.predicted_ratio_two_sigma. Dashed line at 1.

Reads results/tables/robustness_snoise_pred.csv (python -m src.evaluate.robustness_snoise_inputs), each
scenario's results_all.csv (results/tables for k = 1, results/robustness/<NAME>/tables) and
robustness_snoise_summary.csv (python -m src.evaluate.robustness_snoise, for the caption). Writes
results/figures/figS_short_noise.png (300 dpi) and .pdf. Run from the repo root:
    python -m src.evaluate.figures.figS_short_noise
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import load_config, repo_path
from src.evaluate.breakeven_check import predicted_ratio_two_sigma
from src.evaluate.figures._style import FIG_WIDTH, MIN_FONT, MUTED, TEXT, new_figure, save_figure, style_axes
from src.evaluate.figures.fig4_breakeven import DEPTH_RANGE
from src.evaluate.robustness_snoise import MAIN, profiles

FIG_FF = 0.04
FIG_HEIGHT = 2.9                      # in, one row of two panels plus a two-line legend
METHODS = {"B": ("B (40–156 bp)", "D", ("#123766", "#4a6fa5", "#8fa8cc")),
           "W": ("W", "o", ("#c4471a", "#eb6834", "#f2a27f"))}     # colour per k: 1, 1.5, 2 (dark to light)
STYLES = ("-", "--", ":")                                            # line style per k

CAPTION = (
    "Figure S5. Gain over A with extra noise on short fragments, FF 4%: ratio of standardised separations "
    "sep(method)/sep(A) for B (40–156 bp) and W, at k = 1 (main analysis), 1.5 and 2. In the k scenarios each "
    "patient's chr21 fragments of 40–156 bp get a second, independent relative noise σ_s = σ·√(k² − 1), so the "
    "window's chr21 fraction has relative SD ≈ σ·k (a step at 156 bp, the worst case for B); all groups, including "
    "the reference, are simulated with it. Points: simulation; lines: the counting model with σ_u² = σ² + σ_s²·ρ_u², "
    "ρ_u the share of a method's weight inside the window (A: {rho_a:.2f}, B: 1, W: {rho_w:.2f} at FF 4%), so "
    "σ_B/σ_A = {sba_15:.2f} (k = 1.5) and {sba_2:.2f} (k = 2). At realistic noise B's ratio rises from {b1_1:.2f} at "
    "1M to {b1_10:.2f} at 10M without extra noise, to {b15_10:.2f} at k = 1.5 and {b2_10:.2f} at k = 2 (simulation); "
    "W, which takes only part of the extra noise, keeps a ratio of {w2_10:.2f} at k = 2. Over the 64 conditions per "
    "scenario the model's median |relative error| is {b_lo:.2%}–{b_hi:.2%} for B and {w_lo:.2%}–{w_hi:.2%} for W. "
    "Dashed: method = A."
)


def observed(results, noise):
    sep = (results["z_mean_t21"] - results["z_mean_normal"]) / results["z_sd_normal"]
    r = results.assign(sep=sep)
    r = r[np.isclose(r["ff"], FIG_FF) & (r["noise"] == noise)]
    a = r[r["method"] == "A"].set_index("depth")["sep"]
    return {m: (r[r["method"] == m].set_index("depth")["sep"] / a).sort_index() for m in METHODS}


def panel(ax, pred, obs_by_k, noise, title):
    depths = np.geomspace(*DEPTH_RANGE, 200)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    p = pred[np.isclose(pred["ff"], FIG_FF) & (pred["noise"] == noise)]
    for i, (name, obs) in enumerate(obs_by_k.items()):
        q = p[p["scenario"] == name].drop_duplicates("method").set_index("method")   # g, r, sigma_u: depth-free
        sigma_a = q.loc["B", "sigma_u"] / q.loc["B", "sigma_B_over_A"]
        for method, (_, marker, colors) in METHODS.items():
            row = q.loc[method]
            ax.plot(depths, predicted_ratio_two_sigma(row["g_eff"], row["r_eff"], row["p0"], depths, sigma_a,
                                                      row["sigma_u"]),
                    color=colors[i], linestyle=STYLES[i], linewidth=1.4 if STYLES[i] == ":" else 1.1)
            o = obs[method]
            ax.plot(o.index, o.to_numpy(), color=colors[i], linestyle="none", marker=marker, markersize=3.5,
                    markeredgecolor=TEXT, markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlim(*DEPTH_RANGE)
    ax.set_xticks([1e6, 2.5e6, 5e6, 1e7], ["1M", "2.5M", "5M", "10M"])
    ax.minorticks_off()
    style_axes(ax, title, "Depth", "")


def caption(config):
    """Caption with its numbers computed from the predictions, the results and the summary."""
    tables = repo_path(config["paths"]["results"]) / "tables"
    pred = pd.read_csv(tables / "robustness_snoise_pred.csv", dtype={"noise": str})
    p = pred[np.isclose(pred["ff"], FIG_FF) & (pred["noise"] == "realistic")]
    sba = p[p["method"] == "B"].groupby("scenario")["sigma_B_over_A"].first()
    names = list(profiles(config))
    rho_a = p[(p["method"] == "B")]["r_eff"].iloc[0]        # rho_A = r of the window = B's r
    rho_w = p[p["method"] == "W"]["rho_u"].iloc[0]
    obs = {n: observed(pd.read_csv(d / "results_all.csv", dtype={"noise": str}), "realistic")
           for n, (_, d) in profiles(config).items()}
    s = pd.read_csv(tables / "robustness_snoise_summary.csv", dtype={"question": str, "depth": str, "noise": str})
    s = s[s["question"] == "3"].set_index(["profile", "item"])["value"]
    b = s.xs("median_abs_rel_error_B_all", level="item").drop(MAIN)
    w = s.xs("median_abs_rel_error_W_all", level="item").drop(MAIN)
    k15, k2 = names[1], names[2]
    return CAPTION.format(rho_a=rho_a, rho_w=rho_w, sba_15=sba[k15], sba_2=sba[k2],
                          b1_1=obs[MAIN]["B"].iloc[0], b1_10=obs[MAIN]["B"].iloc[-1], b15_10=obs[k15]["B"].iloc[-1],
                          b2_10=obs[k2]["B"].iloc[-1], w2_10=obs[k2]["W"].iloc[-1],
                          b_lo=b.min(), b_hi=b.max(), w_lo=w.min(), w_hi=w.max())


def main():
    config = load_config()
    pred = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "robustness_snoise_pred.csv",
                       dtype={"noise": str})
    prof = profiles(config)
    ks = {n: (1.0 if n == MAIN else float(c["short_noise"]["k"])) for n, (c, _) in prof.items()}
    levels = [n for n, s in zip(config["grid"]["noise_sigma"], config["grid"]["noise_sigma_values"]) if s > 0]
    results = {n: pd.read_csv(d / "results_all.csv", dtype={"noise": str}) for n, (_, d) in prof.items()}

    fig, axes = new_figure(1, len(levels), figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.80), wspace=0.04)   # room for the legend above
    for ax, noise in zip(axes, levels):
        sigma = config["grid"]["noise_sigma_values"][config["grid"]["noise_sigma"].index(noise)]
        panel(ax, pred, {n: observed(results[n], noise) for n in prof}, noise,
              f"{str(noise).capitalize()} (σ = {100 * sigma:g}%)")
    axes[0].set_ylabel("sep(method) / sep(A)", color=MUTED)
    axes[0].set_ylim(0.95, 1.65)

    handles = [Line2D([], [], color=colors[i], marker=marker, linestyle=STYLES[i], markersize=3.5,
                      markeredgecolor=TEXT, markeredgewidth=0.4, label=f"{label}, k = {ks[n]:g}")
               for i, n in enumerate(prof) for label, marker, colors in METHODS.values()]   # columns: k; rows: B, W
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
                        handlelength=2.6, columnspacing=1.0)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.text(0.98, 0.83, "lines: model · points: simulation", color=MUTED, fontsize=MIN_FONT, ha="right", va="bottom")
    paths = save_figure(fig, config, "figS_short_noise", tight_layout=False)
    print("Saved " + ", ".join(str(p) for p in paths))
    print("\nCaption:\n" + caption(config))


if __name__ == "__main__":
    main()
