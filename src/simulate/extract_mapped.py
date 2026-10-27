"""Task 1C: one row per simulated fragment, with its true and mapped position.

Reads the outputs of scripts/map_reads.sh:
  <prefix>.mapped.bam  proper pairs, reads with MAPQ >= min_mapq   -> mapped coordinates
  <prefix>.sam         all Bowtie2 output (unfiltered)             -> why a fragment was lost
and the simulated fragment list from <prefix>_R1.fastq.

A fragment survives only if BOTH mates are in the BAM (both proper-paired, both MAPQ >= min_mapq);
samtools -q filters single reads, so one mate can pass while the other is dropped.
Coordinates are 0-based: mapped_start = leftmost base of the pair, mapped_length = |template_length|.

status column:
  survived        both mates in the BAM
  not_concordant  Bowtie2 did not align the pair as a concordant (proper) pair
  low_mapq        aligned as a proper pair, but at least one mate has MAPQ < min_mapq

Run from the repo root:
    python -m src.simulate.extract_mapped --out-prefix data/processed/task01/sim10k
"""

import argparse
import sys
from collections import defaultdict

import pandas as pd
import pysam
from pyfaidx import Faidx

from src.config import load_config, repo_path

LABEL_TO_ORIGIN = {"mat": "maternal", "fet": "fetal"}


def check_reference_names(header, config):
    """Index (BAM header @SQ) and FASTA (.fai) must agree on chr1-22 names and lengths."""
    index_len = dict(zip(header.references, header.lengths))
    fai = Faidx(str(repo_path(config["paths"]["reference_fasta"])))
    problems = []
    for c in config["simulation"]["chromosomes"]:
        fasta_len = fai.index[c].rlen
        if c not in index_len:
            problems.append(f"{c}: missing from Bowtie2 index")
        elif index_len[c] != fasta_len:
            problems.append(f"{c}: index length {index_len[c]:,} != FASTA length {fasta_len:,}")
    if problems:
        sys.exit("Reference mismatch between FASTA and Bowtie2 index:\n  " + "\n  ".join(problems))
    print(f"Reference check: chr1-22 names and lengths identical in FASTA and Bowtie2 index "
          f"(index has {len(index_len)} sequences in total)")


def primary_reads(path, mode):
    with pysam.AlignmentFile(path, mode) as f:
        for read in f:
            if not (read.is_secondary or read.is_supplementary):
                yield read


def simulated_fragments(prefix):
    """Truth table from the read names in <prefix>_R1.fastq."""
    rows = []
    with open(f"{prefix}_R1.fastq") as f:
        for i, line in enumerate(f):
            if i % 4 == 0:
                name = line[1:].rstrip("\n")
                frag_id, label, chrom, start, length = name.split("|")
                rows.append((name, frag_id, LABEL_TO_ORIGIN[label], chrom, int(start), int(length)))
    return pd.DataFrame(rows, columns=["name", "id", "origin", "true_chrom", "true_start", "true_length"])


def loss_status(sam_path, min_mapq):
    """Per fragment name: survived / not_concordant / low_mapq, from the unfiltered SAM."""
    mates = defaultdict(list)
    for read in primary_reads(sam_path, "r"):
        mates[read.query_name].append(read)
    status = {}
    for name, reads in mates.items():
        if len(reads) != 2 or not all(r.is_proper_pair and not r.is_unmapped for r in reads):
            status[name] = "not_concordant"
        elif min(r.mapping_quality for r in reads) < min_mapq:
            status[name] = "low_mapq"
        else:
            status[name] = "survived"
    return status


def mapped_pairs(bam_path):
    """Per fragment name with both mates in the BAM: mapped chrom/start/length and diagnostics."""
    mates = defaultdict(dict)
    for read in primary_reads(bam_path, "rb"):
        mates[read.query_name][1 if read.is_read1 else 2] = read
    rows = {}
    for name, pair in mates.items():
        if len(pair) != 2:
            continue  # one mate dropped by -q: fragment does not survive
        r1, r2 = pair[1], pair[2]
        # Leftmost base of the pair: read 1's own start if it is the left mate (tlen > 0), else its mate's.
        start = r1.reference_start if r1.template_length > 0 else r1.next_reference_start
        rows[name] = {
            "mapped_chrom": r1.reference_name,
            "mapped_start": start,
            "mapped_length": abs(r1.template_length),
            "mapq_r1": r1.mapping_quality,
            "mapq_r2": r2.mapping_quality,
            "r1_reverse": r1.is_reverse,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    prefix = args.out_prefix

    config = load_config()
    min_mapq = config["simulation"]["min_mapq"]

    with pysam.AlignmentFile(f"{prefix}.mapped.bam", "rb") as bam:
        check_reference_names(bam.header, config)

    table = simulated_fragments(prefix)
    status = loss_status(f"{prefix}.sam", min_mapq)
    mapped = mapped_pairs(f"{prefix}.mapped.bam")

    table["status"] = table["name"].map(status).fillna("not_concordant")
    mapped_df = pd.DataFrame.from_dict(mapped, orient="index")
    table = table.join(mapped_df, on="name")

    # Cross-check: survivors from the BAM must be exactly the SAM pairs with both mates proper and >= min_mapq
    from_bam = table["mapped_chrom"].notna()
    from_sam = table["status"] == "survived"
    if not (from_bam == from_sam).all():
        sys.exit(f"Survivor mismatch between BAM and SAM: {(from_bam != from_sam).sum()} fragments")

    for col in ("mapped_start", "mapped_length", "mapq_r1", "mapq_r2"):
        table[col] = table[col].astype("Int64")  # integers that allow empty values
    table = table.drop(columns="name")
    out = f"{prefix}_fragments.tsv"
    table.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}: {len(table):,} fragments, {from_bam.sum():,} survived "
          f"(BAM and SAM agree on which ones)")


if __name__ == "__main__":
    main()
