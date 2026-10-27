"""Task 1B: simulate cfDNA fragments and write paired-end FASTQ (decisions #1, #2, #11).

Each fragment: origin -> length (1A) -> chromosome (weighted by length) -> uniform 0-based start.
A fragment containing any letter other than A/C/G/T is rejected; its origin and length are kept
and only the chromosome and start are redrawn.

Read names carry the truth, all coordinates 0-based (pyfaidx slicing, pysam reference_start):
    @frag000001|mat|chr7|12345678|163    (id | origin | chrom | start | length)

Run from the repo root:
    python -m src.simulate.fragments --n-maternal 5000 --n-fetal 5000 --out-prefix data/processed/task01/sim10k
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

from src.config import load_config, repo_path
from src.simulate.lengths import sample_lengths

ORIGIN_LABELS = {"maternal": "mat", "fetal": "fet"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
DELETE_ACGT = str.maketrans("", "", "ACGT")  # whatever survives translate() is a non-ACGT letter


def open_reference(config):
    # Soft-masked FASTA: sequence_always_upper turns repeats (lowercase) into uppercase.
    return Fasta(str(repo_path(config["paths"]["reference_fasta"])),
                 as_raw=True, sequence_always_upper=True)


def chromosome_weights(fasta, chromosomes):
    """Chromosome lengths (from the .fai) and the probability of each, proportional to length."""
    lengths = np.array([len(fasta[c]) for c in chromosomes], dtype=np.int64)
    return lengths, lengths / lengths.sum()


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def make_reads(seq, read_length):
    """R1 = first read_length bases; R2 = reverse complement of the last read_length bases."""
    return seq[:read_length], reverse_complement(seq[-read_length:])


def draw_fragment(fasta, chromosomes, chrom_lengths, chrom_probs, length, rng, rejected_letters):
    """Draw chromosome + start for a fragment of fixed length until it is pure A/C/G/T.

    Returns (chrom, start, seq, n_rejections). rejected_letters (a Counter) is updated with the
    non-ACGT letters found in each rejected draw (one count per draw per letter).
    """
    n_rejections = 0
    while True:
        i = rng.choice(len(chromosomes), p=chrom_probs)
        chrom = chromosomes[i]
        # Last valid 0-based start is chrom_length - length (fragment ends exactly at the chromosome end).
        start = int(rng.integers(0, chrom_lengths[i] - length + 1))
        seq = fasta[chrom][start:start + length]
        bad = set(seq.translate(DELETE_ACGT))
        if not bad:
            return chrom, start, seq, n_rejections
        n_rejections += 1
        rejected_letters.update(bad)


def simulate(config, n_by_origin, out_prefix):
    """Simulate fragments, write <out_prefix>_R1.fastq / _R2.fastq / _summary.json; return the summary."""
    sim = config["simulation"]
    read_length = sim["read_length"]
    chromosomes = sim["chromosomes"]
    rng = np.random.default_rng(config["seed"])

    fasta = open_reference(config)
    chrom_lengths, chrom_probs = chromosome_weights(fasta, chromosomes)

    n_total = sum(n_by_origin.values())
    id_width = max(6, len(str(n_total)))
    quality = "I" * read_length

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    r1_path = out_prefix.with_name(out_prefix.name + "_R1.fastq")
    r2_path = out_prefix.with_name(out_prefix.name + "_R2.fastq")

    rejections = {}
    rejected_letters = Counter()
    frag_id = 0
    with open(r1_path, "w") as r1_out, open(r2_path, "w") as r2_out:
        for origin, n in n_by_origin.items():
            rejections[origin] = 0
            for length in sample_lengths(origin, n, rng):
                chrom, start, seq, n_rej = draw_fragment(
                    fasta, chromosomes, chrom_lengths, chrom_probs, int(length), rng, rejected_letters)
                rejections[origin] += n_rej
                frag_id += 1
                name = f"frag{frag_id:0{id_width}d}|{ORIGIN_LABELS[origin]}|{chrom}|{start}|{length}"
                r1, r2 = make_reads(seq, read_length)
                r1_out.write(f"@{name}\n{r1}\n+\n{quality}\n")
                r2_out.write(f"@{name}\n{r2}\n+\n{quality}\n")

    summary = {
        "seed": config["seed"],
        "read_length": read_length,
        "n_fragments": n_by_origin,
        "rejections": rejections,
        "draws": {o: n_by_origin[o] + rejections[o] for o in n_by_origin},
        "rejection_rate": {o: rejections[o] / (n_by_origin[o] + rejections[o]) for o in n_by_origin},
        "rejected_letters": dict(sorted(rejected_letters.items())),
        "r1": str(r1_path),
        "r2": str(r2_path),
    }
    with open(out_prefix.with_name(out_prefix.name + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-maternal", type=int, required=True)
    parser.add_argument("--n-fetal", type=int, required=True)
    parser.add_argument("--out-prefix", required=True, help="e.g. data/processed/task01/sim10k")
    args = parser.parse_args()

    summary = simulate(load_config(), {"maternal": args.n_maternal, "fetal": args.n_fetal}, args.out_prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
