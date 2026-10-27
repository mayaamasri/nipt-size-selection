"""Task 1B check: verify simulated FASTQ files against the reference genome.

Every fragment is re-fetched from the FASTA using the coordinates in its read name (0-based),
and the reads are compared with an independent reverse complement (pyfaidx's own), so this does
not reuse the simulator's code paths. A second part forces short lengths (down to the read length,
where R1 and R2 cover the same bases) to test the overlapping-reads case explicitly.

Run from the repo root:
    python -m src.simulate.check_fragments --out-prefix data/processed/task01/sim10k
"""

import argparse
import json
import sys
from collections import Counter

import numpy as np
from pyfaidx import Fasta

from src.config import load_config, repo_path
from src.simulate.fragments import (ORIGIN_LABELS, chromosome_weights, draw_fragment, make_reads,
                                    open_reference)

EDGE_REPEATS = 25  # forced-length fragments per edge length; test-only


def read_fastq(path):
    """Yield (name, seq, plus, qual) records."""
    with open(path) as f:
        while True:
            lines = [f.readline().rstrip("\n") for _ in range(4)]
            if not lines[0]:
                return
            yield tuple(lines)


def expected_reads(ref, chrom, start, length, read_length):
    """Truth from the reference: R1 = first bases (+ strand), R2 = last bases, reverse-complemented."""
    frag = ref[chrom][start:start + length]
    r1 = frag[:read_length].seq
    r2 = frag[length - read_length:length].reverse.complement.seq
    return frag.seq, r1, r2


def check_fastq(config, prefix, summary, ref, chrom_len):
    read_length = config["simulation"]["read_length"]
    label_to_origin = {v: k for k, v in ORIGIN_LABELS.items()}
    quality = "I" * read_length

    failures = Counter()
    ids = set()
    n = Counter()
    lengths = {o: [] for o in ORIGIN_LABELS}
    chrom_counts = Counter()
    n_overlap, n_full_overlap = 0, 0

    for rec1, rec2 in zip(read_fastq(f"{prefix}_R1.fastq"), read_fastq(f"{prefix}_R2.fastq"), strict=True):
        name1, r1, plus1, q1 = rec1
        name2, r2, plus2, q2 = rec2
        if name1 != name2:
            failures["R1/R2 names differ"] += 1
            continue
        frag_id, label, chrom, start, length = name1[1:].split("|")
        start, length = int(start), int(length)
        origin = label_to_origin[label]
        n[origin] += 1
        lengths[origin].append(length)
        chrom_counts[chrom] += 1

        if frag_id in ids:
            failures["duplicate id"] += 1
        ids.add(frag_id)
        if plus1 != "+" or plus2 != "+" or q1 != quality or q2 != quality:
            failures["bad '+' line or quality string"] += 1
        if len(r1) != read_length or len(r2) != read_length:
            failures["read length != read_length"] += 1
        if set(r1 + r2) - set("ACGT"):
            failures["read has a non-ACGT / lowercase letter"] += 1
        if start < 0 or start + length > chrom_len[chrom]:
            failures["fragment outside chromosome"] += 1
            continue

        frag, exp_r1, exp_r2 = expected_reads(ref, chrom, start, length, read_length)
        if set(frag) - set("ACGT"):
            failures["reference fragment has a non-ACGT letter"] += 1
        if r1 != exp_r1:
            failures["R1 != reference"] += 1
        if r2 != exp_r2:
            failures["R2 != reference"] += 1
        if length < 2 * read_length:
            n_overlap += 1
        if length == read_length:
            n_full_overlap += 1
            if r1 != frag or r2 != ref[chrom][start:start + length].reverse.complement.seq:
                failures["read-length fragment: reads != whole fragment"] += 1

    print("== Part 1: FASTQ files vs reference ==")
    for origin in ORIGIN_LABELS:
        expected = summary["n_fragments"][origin]
        status = "OK" if n[origin] == expected else "MISMATCH"
        print(f"{origin:>8}: {n[origin]:,} records (summary says {expected:,}) {status} | "
              f"median length {np.median(lengths[origin]):.0f} bp")
        if n[origin] != expected:
            failures["record count != summary"] += 1
    print(f"Fragments checked base-by-base against the reference: {sum(n.values()):,}")
    print(f"  with overlapping reads (length < {2 * read_length} bp): {n_overlap}")
    print(f"  with length exactly {read_length} bp (R1 and R2 cover the same bases): {n_full_overlap}")

    # Chromosome shares vs each chromosome's share of N-free (A/C/G/T) bases.
    # Fragments that touch a non-ACGT base are redrawn, so the expected share is roughly the N-free
    # share, not the total-length share. This is an approximation: a start can sit on A/C/G/T and the
    # fragment still overlap an N further along (and be rejected), which slightly lowers the share of
    # chromosomes with many short N stretches. Good enough to catch a broken chromosome draw.
    chroms = config["simulation"]["chromosomes"]
    raw = open_reference(config)
    acgt = np.array([sum(raw[c][:].count(b) for b in "ACGT") for c in chroms], dtype=float)
    expected = acgt / acgt.sum()
    observed = np.array([chrom_counts[c] for c in chroms], dtype=float)
    total = observed.sum()
    z = (observed / total - expected) / np.sqrt(expected * (1 - expected) / total)
    chi2 = float(np.sum((observed - total * expected) ** 2 / (total * expected)))
    print(f"\nChromosome shares vs N-free share (genome non-ACGT: {100 * (1 - acgt.sum() / sum(chrom_len.values())):.2f}%)")
    print("chrom   N-free%  observed%      z")
    for c, e, o, zc in zip(chroms, expected, observed / total, z):
        print(f"{c:6} {100 * e:8.2f} {100 * o:9.2f} {zc:+6.2f}")
    print(f"chi-square {chi2:.1f} on {len(chroms) - 1} df (expect about {len(chroms) - 1} if the draw is right)")
    return failures


def check_edge_lengths(config, ref, chrom_len):
    """Force short lengths through the simulator and verify against the reference."""
    sim = config["simulation"]
    read_length = sim["read_length"]
    edge_lengths = [read_length, read_length + 1, 2 * read_length - 1, 2 * read_length,
                    2 * read_length + 1, sim["fragment_length_max"]]
    fasta = open_reference(config)
    lengths_arr, probs = chromosome_weights(fasta, sim["chromosomes"])
    rng = np.random.default_rng(config["seed"])

    failures = Counter()
    for length in edge_lengths:
        for _ in range(EDGE_REPEATS):
            chrom, start, seq, _ = draw_fragment(fasta, sim["chromosomes"], lengths_arr, probs,
                                                 length, rng, Counter())
            r1, r2 = make_reads(seq, read_length)
            if len(seq) != length or start + length > chrom_len[chrom]:
                failures[f"{length} bp: wrong fragment length or out of bounds"] += 1
            frag, exp_r1, exp_r2 = expected_reads(ref, chrom, start, length, read_length)
            if seq != frag:
                failures[f"{length} bp: fragment != reference"] += 1
            if r1 != exp_r1 or r2 != exp_r2:
                failures[f"{length} bp: reads != reference"] += 1
            if length == read_length and r1 != ref[chrom][start:start + length].seq:
                failures[f"{length} bp: R1 != whole fragment"] += 1
            if length == read_length and r2 != ref[chrom][start:start + length].reverse.complement.seq:
                failures[f"{length} bp: R2 != revcomp of whole fragment"] += 1

    print("\n== Part 2: forced short lengths (overlapping reads) ==")
    print(f"Lengths tested: {edge_lengths}, {EDGE_REPEATS} fragments each, "
          f"{len(edge_lengths) * EDGE_REPEATS} total")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    config = load_config()
    with open(f"{args.out_prefix}_summary.json") as f:
        summary = json.load(f)
    # Independent reader: pyfaidx Sequence objects (not as_raw), uppercased
    ref = Fasta(str(repo_path(config["paths"]["reference_fasta"])), sequence_always_upper=True)
    chrom_len = {c: len(ref[c]) for c in config["simulation"]["chromosomes"]}

    failures = check_fastq(config, args.out_prefix, summary, ref, chrom_len)
    failures += check_edge_lengths(config, ref, chrom_len)

    print("\n== Rejections (non-ACGT fragments, redrawn) ==")
    for origin in ORIGIN_LABELS:
        print(f"{origin:>8}: {summary['rejections'][origin]:,} rejections in "
              f"{summary['draws'][origin]:,} draws ({100 * summary['rejection_rate'][origin]:.2f}%)")
    print(f"Letters found in rejected draws (draws containing each letter): {summary['rejected_letters'] or 'none'}")

    print("\n== Result ==")
    if failures:
        for what, count in failures.items():
            print(f"FAIL {what}: {count}")
        sys.exit(1)
    print("PASS: all checks passed")


if __name__ == "__main__":
    main()
