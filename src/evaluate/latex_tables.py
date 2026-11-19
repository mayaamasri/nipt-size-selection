"""Task 10 (10B): every manuscript and supplementary table as a LaTeX file, from the table CSVs.

Reads the CSVs written by src.evaluate.tables and their captions (tables.captions)
and writes results/tables/latex/<name>.tex, one file per table:
  table1_realism_yang.tex, table2_detection_limits.tex, tableS1_bopt_cutoff.tex, tableS2_c2_importances.tex,
  tableS3_w_vs_wfixed_sens.tex, tableS4_w_vs_wfixed_limits.tex, tableS5_auc.tex,
  tableS6_limits_conservative_fpr.tex, tableS7_limits_8_vs_14.tex, tableS8_limit_differences.tex,
  tableS9_fetal_profiles.tex, tableS10_short_noise.tex
Style: booktabs rules; en-dashes (--) for every range (36--500, 40--156, 4.09--4.48); a thin space before bp
(40--156\\,bp); minus signs in math mode. Tables 1 and 2 are written as the bare tabular only (no table
environment, caption or label: main.tex holds those). Supplementary tables carry their own caption and label;
short ones use a table float, tables with one row per condition use longtable. Tables S9 and S10 have parts
(a)-(d) and (a)-(c): every part is a longtable, the first carries the numbered caption (which describes all parts)
and the later ones a plain heading line (no \\caption), so each is one table number. Packages needed in the document:
booktabs, longtable, amsmath. Values are formatted here and nothing else is computed, so a .tex table always shows
what its CSV holds.

Labels: the CSVs and the code keep their identifiers (W_fixedFF, shift_p10, sens_at_fpr, 1000000, ...); the LaTeX
shows reader labels (W (FF 10%), +10 bp, matched FPR, 1M, ...) from src.evaluate.labels (shared with the
figures and captions), as LaTeX through tex_math(). Captions carry their formulas as $...$ math, which caption_tex() passes through
unchanged. check() fails on any raw underscore outside math mode and on any old label (labels.OLD_LABELS).

Table 2 shows each limit with its 95% CI on a second line in a smaller font (two-line cells keep the table within a
single-column text width); bold and the symbols follow table2_detection_limits.csv (bold, main_le_4pct,
nobody_lower_end). Run from the repo root (after python -m src.evaluate.tables):
    python -m src.evaluate.latex_tables
"""

import math
import re

import pandas as pd

from src.config import load_config, repo_path
from src.evaluate import labels as L
from src.evaluate import tables as T

OUT_DIR = "results/tables/latex"
TIGHT = "\\scriptsize\\setlength{\\tabcolsep}{3pt}"   # wide supplementary tables (S3, S5, S10)
NOISE_NAMES = {"none": "No noise", "low": "Low noise", "realistic": "Realistic noise"}
UNICODE = [("–", "--"), ("—", "---"), ("−", "$-$"), ("≥", "$\\geq$"), ("≤", "$\\leq$"), ("×", "$\\times$"),
           ("·", "$\\cdot$"), ("σ", "$\\sigma$"), ("ρ", "$\\rho$"), ("√", "$\\surd$"), ("²", "$^2$"), ("→", "$\\to$"),
           ("†", "$^\\dagger$"), ("‡", "$^\\ddagger$"), ("~", "$\\sim$"), ("✓", "$\\checkmark$")]


# ---------------------------------------------------------------- formatting

def tex(text):
    """Plain text -> LaTeX: escape specials, unicode symbols to commands, thin space before bp."""
    s = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}")):
        s = s.replace(a, b)
    for a, b in UNICODE:
        s = s.replace(a, b)
    s = s.replace(">", "$>$").replace("<", "$<$")
    s = re.sub(r"(\d)\s+bp\b", r"\1\\,bp", s)
    s = s.replace("$$", "")   # adjacent math pieces
    return s


def limit_cell(text):
    """'6.90%' -> '6.90'; '< 1%' -> '$<$1'; '> 15%' -> '$>$15'."""
    t = str(text).replace("%", "").strip()
    if t.startswith("<") or t.startswith(">"):
        return f"${t[0]}${t[1:].strip()}"
    return t


def num(v, fmt="{:.2f}", pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return "--"
    s = fmt.format(100 * v if pct else v)
    return s.replace("-", "$-$") if s.startswith("-") else s


def rng(lo, hi, pct=True):
    if pd.isna(lo) or pd.isna(hi):
        return "--"
    a, b = (100 * lo, 100 * hi) if pct else (lo, hi)
    fa = f"{a:.2f}".replace("-", "$-$")
    fb = f"{b:.2f}".replace("-", "$-$")
    return f"{fa}--{fb}"


def depth_label(d):
    return f"{int(d) / 1e6:g}M"


def ff_label(ff):
    return f"{100 * float(ff):g}"


def tex_math(text):
    """Text that may hold $...$ math -> LaTeX: math segments pass through unchanged, the rest goes through tex()."""
    out = "".join(tex(seg) if i % 2 == 0 else f"${seg}$" for i, seg in enumerate(str(text).split("$"))).replace("$$", "")
    return re.sub(r"\$ bp\b", r"$\\,bp", out)   # '40--$c$ bp' -> '40--$c$\\,bp'


def caption_tex(text):
    """Caption text -> LaTeX, without its leading 'Table X.' label (LaTeX numbers the float itself)."""
    return tex_math(re.sub(r"^Table S?\d+\.\s*", "", text))


# ---------------------------------------------------------------- reader labels (src.evaluate.labels, as LaTeX)

def method(m):
    """'W_fixedFF' -> 'W (FF 10\\%)'; 'B_130' -> 'B 40--130'."""
    return tex_math(L.method(m))


def pair(p):
    """'W_fixedFF-W' -> 'W (FF 10\\%) $-$ W'."""
    a, b = T.split_pair(p)
    return f"{method(a)} $-$ {method(b)}"


def profile(name):
    """'G1' -> 'G1 (measured)'; 'shift_m10' -> '$-$10\\,bp'; 'blend50' -> '50\\% maternal blend'."""
    return tex_math(L.scenario(name))


def noise_name(noise):
    return tex_math(L.noise(noise))


def basis_name(basis, title=False):
    return tex_math(L.basis(basis, title))


FEATURE_NAMES = L.FEATURES
VERSION_NAMES = {k: tex_math(v) for k, v in L.VERSION.items()}


# ---------------------------------------------------------------- table environments

def tabular(colspec, header, rows):
    """The bare booktabs tabular (Tables 1 and 2: main.tex supplies the table environment, caption and label)."""
    body = "\n".join(rows)
    return (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n")


def table_float(label, caption, colspec, header, rows, size="\\small"):
    return (f"\\begin{{table}}[htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\smallskip\n"
            + tabular(colspec, header, rows) + "\\end{table}\n")


def long_table(label, caption, colspec, header, rows, size="\\footnotesize", part=None):
    """longtable with its caption. label=None: a later part of a multi-part table, headed by a plain line (no
    \\caption; longtable steps the table counter for every longtable, so the part takes it back with
    \\addtocounter{table}{-1}). part: a line under a numbered caption naming the first part, e.g. '(a) ...'."""
    n = colspec.count("l") + colspec.count("r") + colspec.count("c") + colspec.count("p{")
    body = "\n".join(rows)
    cap = (f"\\caption{{{caption}}}\\label{{{label}}} \\\\\n" if label
           else f"\\multicolumn{{{n}}}{{@{{}}l}}{{{caption}}} \\\\[2pt]\n")
    if part:
        cap += f"\\multicolumn{{{n}}}{{@{{}}l}}{{{part}}} \\\\\n"
    back = "" if label else "\\addtocounter{table}{-1}\n"
    return (f"{{{size}\n{back}\\begin{{longtable}}{{{colspec}}}\n{cap}"
            f"\\toprule\n{header} \\\\\n\\midrule\n\\endfirsthead\n"
            f"\\multicolumn{{{n}}}{{l}}{{\\emph{{(continued)}}}} \\\\\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
            f"\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}\n}}\n")


def noise_block(noise, ncols):
    return f"\\addlinespace\n\\multicolumn{{{ncols}}}{{l}}{{\\emph{{{NOISE_NAMES.get(noise, noise)}}}}} \\\\"


def stack(*lines):
    """A column header on several lines (keeps wide supplementary tables within the text width)."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def cond(r):
    """Noise, depth and FF cells of a per-condition row."""
    return f"{noise_name(r.noise)} & {depth_label(r.depth)} & {ff_label(r.ff)}"


# ---------------------------------------------------------------- tables

def table1(df, cap):
    def quantity(q):
        return tex(q).replace("(r)", "($r$)").replace("(g)", "($g$)")
    rows = [f"{quantity(r.quantity)} & {tex(r.at)} & {tex(r.ours)} & {tex(r.yang)} \\\\" for r in df.itertuples()]
    return tabular("@{}l l r p{4.6cm}@{}", "Quantity & At & This study & Yang et al.", rows)


def table2(df, cap):
    depths = sorted(df["depth"].unique())
    rows = []
    for noise, g in df.groupby("noise", sort=False):
        rows.append(noise_block(noise, len(depths) + 1))
        for m, gm in g.groupby("method", sort=False):
            cells = []
            for d in depths:
                r = gm[gm["depth"] == d].iloc[0]
                val = limit_cell(r["ff_limit_text"])
                if str(r["bold"]) == "True":
                    val = f"\\textbf{{{val}}}"
                marks = ("$^\\dagger$" if str(r["main_le_4pct"]) == "True" else "") + \
                        ("$^\\ddagger$" if str(r["nobody_lower_end"]) == "True" else "")
                ci = rng(r["ci_lo"], r["ci_hi"])
                cells.append(f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{val}{marks}\\\\[-2pt]{{\\scriptsize {ci}}}\\end{{tabular}}")
            rows.append(f"{method(m)} & " + " & ".join(cells) + " \\\\[2pt]")
    header = "Method & " + " & ".join(depth_label(d) for d in depths)
    return tabular("@{}l" + "c" * len(depths) + "@{}", header, rows)


def tableS1(df, cap):
    rows = [f"{cond(r)} & {int(r.chosen_upper_bp)} & "
            f"{num(r.train_separation)} & {num(r.test_separation_Bopt)} & {num(r.test_separation_B)} & "
            f"{num(r.sens_at_fpr_Bopt, '{:.1f}', True)} & {num(r.sens_at_fpr_B, '{:.1f}', True)} & "
            f"{num(r.diff_sens_at_fpr, '{:+.1f}', True)} \\\\" for r in df.itertuples()]
    header = ("Noise & Depth & FF (\\%) & " + " & ".join(stack(*h) for h in (
        ("Cutoff", "(bp)"), ("Sep.", "train"), ("Sep.", "B-opt"), ("Sep.", "B"), ("Sens.\\ (\\%)", "B-opt"),
        ("Sens.\\ (\\%)", "B"), ("Diff.", "(pts)"))))
    return long_table("tab:s1", cap, "@{}llrrrrrrrr@{}", header, rows)


def tableS2(df, cap):
    rows = [f"{cond(r)} & {num(r.zA, '{:.3f}')} & "
            f"{num(r.z_short, '{:.3f}')} & {num(r.z_weighted, '{:.3f}')} & {int(r.max_depth)} & "
            f"{'yes' if str(r.nobody_positive) == 'True' else 'no'} \\\\" for r in df.itertuples()]
    header = ("Noise & Depth & FF (\\%) & " + " & ".join(FEATURE_NAMES[f] for f in ("zA", "z_short", "z_weighted"))
              + " & Max depth & Flags nobody")
    return long_table("tab:s2", cap, "@{}llrrrrrl@{}", header, rows)


def tableS3(df, cap):
    wf = method("W_fixedFF")
    rows = [f"{cond(r)} & {num(r.W_sens, '{:.1f}', True)} & "
            f"{num(r.Wfixed_sens, '{:.1f}', True)} & {num(r.W_sens_at_fpr, '{:.1f}', True)} & "
            f"{num(r.Wfixed_sens_at_fpr, '{:.1f}', True)} & {num(r.diff_sens_at_fpr, '{:+.1f}', True)} & "
            f"{num(r.W_auc, '{:.4f}')} & {num(r.Wfixed_auc, '{:.4f}')} \\\\" for r in df.itertuples()]
    header = ("Noise & Depth & FF (\\%) & " + " & ".join(stack(*h) for h in (
        ("Sens.", "W"), ("Sens.", wf), ("Matched", "FPR W"), ("Matched", f"FPR {wf}"), ("Diff.", "(pts)"),
        ("AUC", "W"), ("AUC", wf))))
    return long_table("tab:s3", cap, "@{}llrrrrrrrr@{}", header, rows, size=TIGHT)


def tableS4(df, cap):
    wf = method("W_fixedFF")
    rows = [f"{noise_name(r.noise)} & {depth_label(r.depth)} & {limit_cell(r.W_ff_limit_text)} & "
            f"{limit_cell(r.W_fixedFF_ff_limit_text)} & {limit_cell(r.W_ff_limit_cons_text)} & "
            f"{limit_cell(r.W_fixedFF_ff_limit_cons_text)} & {num(r.diff_ff_limit, '{:+.2f}', True)} \\\\"
            for r in df.itertuples()]
    header = f"Noise & Depth & W & {wf} & W (cons.) & {wf} (cons.) & Diff.\\ (pts)"
    return table_float("tab:s4", cap, "@{}llrrrrr@{}", header, rows)


def tableS5(df, cap):
    methods = T.MAIN_METHODS
    rows = [f"{noise_name(r['noise'])} & {depth_label(r['depth'])} & {ff_label(r['ff'])} & "
            + " & ".join(num(r[m], "{:.4f}") for m in methods) + " \\\\" for _, r in df.iterrows()]
    header = "Noise & Depth & FF (\\%) & " + " & ".join(method(m) for m in methods)
    return long_table("tab:s5", cap, "@{}llr" + "r" * len(methods) + "@{}", header, rows, size=TIGHT)


def tableS6(df, cap):
    rows = []
    for noise, g in df.groupby("noise", sort=False):
        rows.append(noise_block(noise, 4))
        for r in g.itertuples():
            rows.append(f"{depth_label(r.depth)} & {method(r.method)} & {limit_cell(r.ff_limit_cons_text)} & "
                        f"{limit_cell(r.ff_limit_at_fpr_text)} [{rng(r.at_fpr_ci_lo, r.at_fpr_ci_hi)}] \\\\")
    header = "Depth & Method & Conservative & Matched FPR [95\\% CI]"
    return long_table("tab:s6", cap, "@{}llrr@{}", header, rows)


def tableS7(df, cap):
    rows = []
    for noise, g in df.groupby("noise", sort=False):
        rows.append(noise_block(noise, 5))
        for r in g.itertuples():
            rows.append(f"{depth_label(r.depth)} & {method(r.method)} & {limit_cell(r.limit_8_text)} & "
                        f"{limit_cell(r.limit_14_text)} & {num(r.change_pts, '{:+.2f}')} \\\\")
    header = "Depth & Method & 8-point & 14-point & Change (pts)"
    return long_table("tab:s7", cap, "@{}llrrr@{}", header, rows)


def tableS8(df, cap):
    rows = []
    for (p, noise, depth), g in df.groupby(["pair", "noise", "depth"], sort=False):
        cells = []
        for basis in ("sens", "sens_at_fpr"):
            r = g[g["basis"] == basis].iloc[0]
            mark = "clear" if str(r["clear"]) == "True" else "similar"
            cells.append(f"{num(r['diff'], '{:+.2f}', True)} [{rng(r['ci_lo'], r['ci_hi'])}] & {mark}")
        rows.append(f"{pair(p)} & {noise_name(noise)} & {depth_label(depth)} & " + " & ".join(cells) + " \\\\")
    header = (f"Pair & Noise & Depth & {basis_name('sens', title=True)} [95\\% CI] & & "
              f"{basis_name('sens_at_fpr', title=True)} [95\\% CI] &")
    return long_table("tab:s8", cap, "@{}lllrlrl@{}", header, rows)


def tableS9(parts, cap):
    a, b, c, d = parts
    out = [f"% Table S9 (fetal-profile scenarios), four parts\n{{\\footnotesize\n"]
    rows = [f"{profile(r.profile)} & {int(r.fetal_median_bp)} & "
            + " & ".join(f"{num(getattr(r, f'g_ff{f}'))} & {num(getattr(r, f'r_ff{f}'), '{:.3f}')} & "
                         f"{num(getattr(r, f'g_sqrt_r_ff{f}'))}" for f in (2, 4, 10)) + " \\\\" for r in a.itertuples()]
    header = "Profile & Median (bp) & " + " & ".join(f"$g$ {f}\\% & $r$ {f}\\% & $g\\sqrt{{r}}$ {f}\\%" for f in (2, 4, 10))
    out.append(long_table("tab:s9", cap, "@{}lr" + "rrr" * 3 + "@{}", header, rows,
                          part="(a) Inputs of each fetal profile."))
    rows = [f"{profile(r.profile)} & {int(r.n_B_below_A)} & {int(r.n_B_minus_A_sig_negative)} & {int(r.n_Bopt_below_A)} & "
            f"{num(r.max_Bopt_sens_loss_pts, '{:.1f}')} \\\\" for r in b.itertuples()]
    out.append(long_table(None, "(b) B and B-opt against A (of 96 conditions).", "@{}lrrrr@{}",
                          "Profile & B $<$ A & B $-$ A sig.\\ $<$ 0 & B-opt $<$ A & B-opt worst loss (pts)", rows))
    rows = []
    for prof, g in c.groupby("profile", sort=False):
        rows.append(f"\\addlinespace\n\\multicolumn{{8}}{{l}}{{\\emph{{{profile(prof)}}}}} \\\\")
        rows += [f"{noise_name(r['noise'])} & {depth_label(r['depth'])} & "
                 + " & ".join(limit_cell(r[m]) for m in T.PROFILE_METHODS) + " \\\\" for _, r in g.iterrows()]
    out.append(long_table(None, "(c) Detection limits (FF, \\%) per scenario.", "@{}llrrrrrr@{}",
                          "Noise & Depth & " + " & ".join(method(m) for m in T.PROFILE_METHODS), rows))
    rows = []
    for (grid, noise, depth), g in d.groupby(["grid", "noise", "depth"], sort=False):
        cells = []
        for basis in ("sens", "sens_at_fpr"):
            r = g[g["basis"] == basis].iloc[0]
            cells.append(f"{num(r['diff'], '{:+.2f}', True)} [{rng(r['ci_lo'], r['ci_hi'])}]"
                         + (" clear" if str(r["clear"]) == "True" else ""))
        rows.append(f"{profile(grid)} & {noise_name(noise)} & {depth_label(depth)} & " + " & ".join(cells) + " \\\\")
    out.append(long_table(None, f"(d) {method('W_G1')} $-$ A (FF points, {VERSION_NAMES['test+ref']}, 95\\% CI).",
                          "@{}lllrr@{}", f"Scenario & Noise & Depth & {basis_name('sens', title=True)} & "
                          f"{basis_name('sens_at_fpr', title=True)}", rows))
    out.append("}\n")
    return "\n".join(out)


def tableS10(parts, cap):
    a, b, c = parts
    out = ["% Table S10 (short-fragment noise), three parts\n{\\footnotesize\n"]
    rows = [f"{num(r.k, '{:g}')} & {noise_name(r.noise)} & {num(r.obs_1M, '{:.3f}')} & {num(r.obs_10M, '{:.3f}')} & "
            f"{num(r.obs_change, '{:+.1f}', True)} & {num(r.model_1M, '{:.3f}')} & {num(r.model_10M, '{:.3f}')} & "
            f"{num(r.model_change, '{:+.1f}', True)} & {num(r.sigma_B_over_A, '{:.3f}')} & "
            f"{num(r.high_depth_limit, '{:.3f}')} \\\\" for r in a.itertuples()]
    out.append(long_table("tab:s10", cap, "@{}llrrrrrrrr@{}",
                          "$k$ & Noise & Sim.\\ 1M & Sim.\\ 10M & Change (\\%) & Model 1M & Model 10M & Change (\\%) & "
                          "$\\sigma_B/\\sigma_A$ & $g\\sigma_A/\\sigma_B$", rows, size=TIGHT,
                          part="(a) sep(B)/sep(A) at FF 4\\%."))
    rows = [f"{num(r.k, '{:g}')} & {noise_name(r.noise)} & {depth_label(r.depth)} & "
            + " & ".join(limit_cell(getattr(r, m)) for m in T.SNOISE_METHODS) + " \\\\" for r in b.itertuples()]
    out.append(long_table(None, "(b) Detection limits (FF, \\%) of A, B and W.", "@{}lllrrr@{}",
                          "$k$ & Noise & Depth & A & B & W", rows))
    regs = [col for col in c.columns if col.startswith("regime_k")]
    highs = [col for col in c.columns if col.startswith("g_sigmaA_over_sigmaB_k")]
    rows = [f"{tex_math(L.cutoff(r['cutoff']))} & {ff_label(r['ff'])} & "
            f"{num(r['g_sqrt_r'])} & {num(r['g2r'])} & "
            + " & ".join(f"({r[col]})" for col in regs) + " & "
            + " & ".join(num(r[col]) for col in highs) + f" & {num(r['k_high_equal'])} & {num(r['k_star'])} \\\\"
            for _, r in c.iterrows()]
    header = ("Method & FF (\\%) & $g\\sqrt{r}$ & $g^2r$ & "
              + " & ".join(f"Regime $k = {col.replace('regime_k', '')}$" for col in regs) + " & "
              + " & ".join(f"$g\\sigma_A/\\sigma_B$, $k = {col.replace('g_sigmaA_over_sigmaB_k', '')}$" for col in highs)
              + " & $k_\\mathrm{high}$ & $k^*$")
    out.append(long_table(None, "(c) Regimes per cutoff, FF and $k$.", "@{}lr" + "r" * (2 + len(regs) + len(highs) + 2)
                          + "@{}", header, rows, size=TIGHT))
    out.append("}\n")
    return "\n".join(out)


BUILDERS = {
    "Table 1": ("table1_realism_yang.tex", table1),
    "Table 2": ("table2_detection_limits.tex", table2),
    "Table S1": ("tableS1_bopt_cutoff.tex", tableS1),
    "Table S2": ("tableS2_c2_importances.tex", tableS2),
    "Table S3": ("tableS3_w_vs_wfixed_sens.tex", tableS3),
    "Table S4": ("tableS4_w_vs_wfixed_limits.tex", tableS4),
    "Table S5": ("tableS5_auc.tex", tableS5),
    "Table S6": ("tableS6_limits_conservative_fpr.tex", tableS6),
    "Table S7": ("tableS7_limits_8_vs_14.tex", tableS7),
    "Table S8": ("tableS8_limit_differences.tex", tableS8),
    "Table S9": ("tableS9_fetal_profiles.tex", tableS9),
    "Table S10": ("tableS10_short_noise.tex", tableS10),
}


MAIN_TABLES = ("Table 1", "Table 2")
outside_math = L.outside_math
old_labels = L.old_labels


def check(text, name, main=False):
    """Self-test of the style rules on the written LaTeX: no unicode dash, minus or symbol left, every range written
    with --, no 'bp' without the thin space; Tables 1 and 2 hold only the tabular (no table environment, caption or
    label); a supplementary table has exactly one caption (later parts are headed by a plain line) and no internal
    references (decision #, Task); no raw underscore outside math mode, no old label (OLD_LABELS) and no hard-coded
    reference number like [7]."""
    body = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
    numbered = len(re.findall(r"\\caption\b", body))
    if main and (numbered or "\\label" in body or "\\begin{table}" in body or "longtable" in body):
        raise ValueError(f"{name}: a main table must be the bare tabular")
    if not main and numbered != 1:
        raise ValueError(f"{name}: {numbered} numbered captions (one table number expected)")
    if re.search(r"decision \\#|\\#\d|Task \d", body):
        raise ValueError(f"{name}: internal reference in the caption")
    L.check_text(body, name, allow_escaped=True)
    plain = outside_math(body)
    if re.search(r"\[\d+(?:[,–-]\d+)*\]", plain):
        raise ValueError(f"{name}: hard-coded reference number")
    bad = [ch for ch in "–—−≥≤×·σρ√²→†‡✓" if ch in text]
    if bad:
        raise ValueError(f"{name}: unconverted characters {bad}")
    if re.search(r"\d bp\b", text) or re.search(r"\d(?<!\\,)bp\b", text.replace("\\,bp", "")):
        raise ValueError(f"{name}: 'bp' without a thin space")
    if re.search(r"\d\.\d+-\d", text):
        raise ValueError(f"{name}: a range written with a hyphen")


def self_test():
    """check() must reject a raw underscore, an old label, a hard-coded reference and a bare table with a caption."""
    ok = "% Table S0. test\n\\begin{table}\\caption{x $\\sigma_A$}\n\\begin{tabular}{l}\nW (FF 10\\%) & 1M \\\\\n\\end{tabular}\n"
    check(ok, "ok")
    for bad, main in ((ok.replace("W (FF 10\\%)", "W\\_fixedFF"), False), (ok.replace("1M", "1000000"), False),
                      (ok.replace("x $", "x_y $"), False), (ok.replace("& 1M", "& none"), False),
                      (ok.replace("x $", "Yang [7] $"), False), (ok, True)):
        try:
            check(bad, "bad", main=main)
        except ValueError:
            continue
        raise AssertionError(f"check() accepted {bad!r}")
    print("self-test check(): ok")


def main():
    self_test()
    config = load_config()
    tables_dir = repo_path(config["paths"]["results"]) / "tables"
    caps = T.captions(config)
    out_dir = repo_path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, (files, caption) in caps.items():
        name, build = BUILDERS[label]
        frames = [pd.read_csv(tables_dir / f, dtype={"noise": str}) for f in files]
        text = build(frames[0] if len(frames) == 1 else frames, caption_tex(caption))
        note = " Bare tabular: the table environment, caption and label are in main.tex." if label in MAIN_TABLES else ""
        text = (f"% {label}. Generated by python -m src.evaluate.latex_tables from {', '.join(files)}; do not edit."
                f"{note}\n" + text)
        check(text, name, main=label in MAIN_TABLES)
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"Saved {out_dir / name}")


if __name__ == "__main__":
    main()
