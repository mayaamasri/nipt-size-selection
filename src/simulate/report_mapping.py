"""Task 1D: mapping verification numbers for Maya, from the extract_mapped table.

Run from the repo root:
    python -m src.simulate.report_mapping --out-prefix data/processed/task01/sim10k
"""

import argparse
import json
import math
import re

import pandas as pd

from src.config import load_config

STATUSES = ("survived", "not_concordant", "low_mapq")
LENGTH_BIN_EDGES = [100, 150, 200]  # report-only bins: [min,100), [100,150), [150,200), [200,max]


def pct(part, whole):
    return 100 * part / whole if whole else float("nan")


def two_proportion_z(k1, n1, k2, n2):
    """Two-sided z-test for p1 == p2 (normal approximation, pooled variance). Returns (z, p)."""
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def survival_tests(table):
    """Maternal vs fetal survival: overall z-test, then survival by length bin."""
    sim = load_config()["simulation"]
    edges = [sim["fragment_length_min"]] + LENGTH_BIN_EDGES + [sim["fragment_length_max"] + 1]
    labels = [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:])]
    table = table.assign(survived=table["status"] == "survived",
                         length_bin=pd.cut(table["true_length"], edges, right=False, labels=labels))

    by = table.groupby("origin")["survived"].agg(["sum", "count"])
    (km, nm), (kf, nf) = by.loc["maternal"], by.loc["fetal"]
    z, p = two_proportion_z(km, nm, kf, nf)
    print("\n== Maternal vs fetal survival (two-proportion z-test) ==")
    print(f"maternal {pct(km, nm):.2f}% vs fetal {pct(kf, nf):.2f}%: "
          f"difference {pct(km, nm) - pct(kf, nf):+.2f} points, z = {z:.2f}, p = {p:.2g}")

    print("\n== Survival by true fragment length (bp) ==")
    grid = table.groupby(["length_bin", "origin"], observed=False)["survived"].agg(["sum", "count"])
    print(f"{'bin':>8}  {'maternal':>20}  {'fetal':>20}  {'all':>20}")
    for b in labels:
        cells = []
        for origin in ("maternal", "fetal"):
            k, n = grid.loc[(b, origin)]
            cells.append(f"{pct(k, n):6.2f}% (n={n:,})")
        k, n = grid.loc[b].sum()
        cells.append(f"{pct(k, n):6.2f}% (n={n:,})")
        print(f"{b:>8}  " + "  ".join(f"{c:>20}" for c in cells))


def bowtie2_timing(log_path):
    """Wall time (s) and peak memory (bytes) from /usr/bin/time -l, plus Bowtie2's overall rate line."""
    text = open(log_path).read()
    wall = re.search(r"([\d.]+) real", text)
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    rate = re.search(r"([\d.]+%) overall alignment rate", text)
    return (float(wall.group(1)) if wall else None,
            int(rss.group(1)) if rss else None,
            rate.group(1) if rate else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    prefix = args.out_prefix

    min_mapq = load_config()["simulation"]["min_mapq"]
    table = pd.read_csv(f"{prefix}_fragments.tsv", sep="\t")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)

    table["correct_position"] = ((table["mapped_chrom"] == table["true_chrom"])
                                 & (table["mapped_start"] == table["true_start"]))
    table["correct_length"] = table["mapped_length"] == table["true_length"]
    surv = table[table["status"] == "survived"]

    print(f"== Survival (both mates proper-paired and MAPQ >= {min_mapq}) ==")
    for origin, grp in table.groupby("origin", sort=False):
        n = len(grp)
        counts = grp["status"].value_counts()
        parts = " | ".join(f"{s} {counts.get(s, 0):,} ({pct(counts.get(s, 0), n):.2f}%)" for s in STATUSES)
        print(f"{origin:>8}: {n:,} fragments | {parts}")
    survival_tests(table)

    print("\n== Accuracy among survivors ==")
    for origin, grp in surv.groupby("origin", sort=False):
        n = len(grp)
        print(f"{origin:>8}: {n:,} survivors | correct chrom+position {pct(grp['correct_position'].sum(), n):.2f}% "
              f"| mapped length = true length {pct(grp['correct_length'].sum(), n):.2f}% "
              f"| R1 on reverse strand {int(grp['r1_reverse'].sum())}")

    wrong = surv[~surv["correct_position"]]
    print(f"\nSurvivors at the wrong place: {len(wrong)}")
    if len(wrong):
        same_chrom = (wrong["mapped_chrom"] == wrong["true_chrom"]).sum()
        print(f"  same chromosome, other position: {same_chrom}; other chromosome/contig: {len(wrong) - same_chrom}")
        print("  mapped to:", wrong["mapped_chrom"].value_counts().to_dict())
        cols = ["id", "origin", "true_chrom", "true_start", "true_length",
                "mapped_chrom", "mapped_start", "mapped_length", "mapq_r1", "mapq_r2"]
        print(wrong[cols].head(20).to_string(index=False))
    bad_len = surv[surv["correct_position"] & ~surv["correct_length"]]
    if len(bad_len):
        print(f"\nCorrect position but wrong length: {len(bad_len)}")

    print("\n== N-rejections in simulation ==")
    for origin in summary["rejections"]:
        print(f"{origin:>8}: {summary['rejections'][origin]:,} ({100 * summary['rejection_rate'][origin]:.2f}% of draws)")
    print(f"letters: {summary['rejected_letters']}")

    wall, rss, rate = bowtie2_timing(f"{prefix}.bowtie2.log")
    print("\n== Bowtie2 ==")
    print(f"pairs: {len(table):,} | wall time {wall} s | peak memory "
          f"{rss / 1e9:.2f} GB ({rss:,} bytes) | Bowtie2 overall alignment rate {rate}")


if __name__ == "__main__":
    main()
