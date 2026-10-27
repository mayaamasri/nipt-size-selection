"""Task 1D: mapping verification numbers for Maya, from the extract_mapped table.

Run from the repo root:
    python -m src.simulate.report_mapping --out-prefix data/processed/task01/sim10k
"""

import argparse
import json
import re

import pandas as pd

from src.config import load_config

STATUSES = ("survived", "not_concordant", "low_mapq")


def pct(part, whole):
    return 100 * part / whole if whole else float("nan")


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
