"""Task 2B: generate and map the mappability pool in chunks (decision #3).

Pool: pool_size_per_origin maternal + pool_size_per_origin fetal fragments, in chunks of
pool_chunk_size. Every chunk is single-origin and they alternate: chunk 0 maternal, 1 fetal,
2 maternal, ... so the pool stays balanced if the run stops early.

Per chunk: fragments.simulate -> scripts/map_reads.sh -> extract_mapped.build_table, then one small
table is kept, data/processed/pool/chunk_KK.parquet:
    origin, true_chrom, true_length, status, mapped_chrom
plus chunk_KK.json (seed, timings, peak memory, survival and accuracy counts, Bowtie2 log).
FASTQ/SAM/BAM are deleted only after the Parquet has been written, read back and checked.
A chunk whose Parquet already exists (with the expected number of rows) is skipped, so a crashed
run can simply be restarted.

Seeds: chunk k uses SeedSequence(config seed, spawn_key=(0,)).spawn(n_chunks)[k], whose spawn_key is
(0, k) whatever n_chunks is, so the seed depends only on the chunk index. (Patients use spawn_key
(1, ...), see patients.py.) Changing pool_chunk_size still changes the fragments: all lengths of a
chunk are drawn before any position, so a chunk of another size shares its first lengths but not
their positions.

Run from the repo root (nipt env active):
    python -m src.simulate.pool --chunks 0      # one chunk
    python -m src.simulate.pool                 # all chunks, skipping finished ones
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config, repo_path
from src.simulate.extract_mapped import build_table
from src.simulate.fragments import simulate
from src.simulate.report_mapping import STATUSES, bowtie2_timing

COLUMNS = ["origin", "true_chrom", "true_length", "status", "mapped_chrom"]


def n_chunks(config):
    sim = config["simulation"]
    per_origin, size = sim["pool_size_per_origin"], sim["pool_chunk_size"]
    if per_origin % size:
        sys.exit(f"pool_size_per_origin ({per_origin:,}) must be a multiple of pool_chunk_size ({size:,})")
    return 2 * per_origin // size


def chunk_origin(k):
    return "maternal" if k % 2 == 0 else "fetal"


def chunk_seed(config, k):
    return np.random.SeedSequence(config["seed"], spawn_key=(0,)).spawn(n_chunks(config))[k]


def peak_rss_bytes():
    """Peak resident memory of this Python process so far (ru_maxrss: bytes on macOS, kbytes on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else 1024 * rss


def run_chunk(config, k):
    chunk_size = config["simulation"]["pool_chunk_size"]
    pool_dir = repo_path(config["paths"]["pool_dir"])
    name = f"chunk_{k:02d}"
    parquet = pool_dir / f"{name}.parquet"
    origin = chunk_origin(k)

    if parquet.exists():
        n_rows = pq.read_metadata(parquet).num_rows
        if n_rows != chunk_size:
            sys.exit(f"{parquet} has {n_rows:,} rows, expected {chunk_size:,}: delete it and rerun")
        print(f"{name} ({origin}): Parquet exists with {n_rows:,} rows, skipped")
        return

    workdir = pool_dir / "tmp" / name
    prefix = workdir / name
    seed = chunk_seed(config, k)
    print(f"== {name}: {chunk_size:,} {origin} fragments, seed spawn_key {seed.spawn_key} ==")
    times = {}
    t_start = time.perf_counter()

    # 1. Generation
    t = time.perf_counter()
    summary = simulate(config, {origin: chunk_size}, prefix, seed=seed)
    times["generation"] = time.perf_counter() - t
    print(f"  generation  {times['generation']:8.1f} s")

    # 2. Mapping (Bowtie2 + samtools); the script's own output is kept only if it fails
    t = time.perf_counter()
    result = subprocess.run(["bash", str(repo_path("scripts/map_reads.sh")), str(prefix)],
                            capture_output=True, text=True)
    times["mapping"] = time.perf_counter() - t
    if result.returncode != 0:
        sys.exit(f"map_reads.sh failed (files kept in {workdir}):\n{result.stdout}\n{result.stderr}")
    bowtie2_wall, bowtie2_rss, bowtie2_rate = bowtie2_timing(f"{prefix}.bowtie2.log")
    print(f"  mapping     {times['mapping']:8.1f} s   (Bowtie2 alone {bowtie2_wall} s)")

    # 3. Extraction to the per-fragment table
    t = time.perf_counter()
    table = build_table(prefix, config)
    surv = table[table["status"] == "survived"]
    accuracy = {
        "survivors_wrong_position": int(((surv["mapped_chrom"] != surv["true_chrom"])
                                         | (surv["mapped_start"] != surv["true_start"])).sum()),
        "survivors_wrong_length": int((surv["mapped_length"] != surv["true_length"]).sum()),
        "survivors_outside_chr1_22": int((~surv["mapped_chrom"].isin(config["simulation"]["chromosomes"])).sum()),
    }
    out = table[COLUMNS].astype({"origin": "category", "true_chrom": "category", "true_length": "int16",
                                 "status": "category", "mapped_chrom": "category"})
    times["extraction"] = time.perf_counter() - t
    print(f"  extraction  {times['extraction']:8.1f} s")

    # 4. Write (to .tmp, then rename: a crash mid-write cannot leave a file that looks complete),
    #    read back, check; only then delete the large files.
    t = time.perf_counter()
    pool_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_name(parquet.name + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    back = pd.read_parquet(parquet)
    problems = []
    if len(back) != chunk_size:
        problems.append(f"{len(back):,} rows, expected {chunk_size:,}")
    if list(back.columns) != COLUMNS:
        problems.append(f"columns {list(back.columns)}")
    if not set(back["status"].unique()) <= set(STATUSES):
        problems.append(f"unknown status values {set(back['status'].unique()) - set(STATUSES)}")
    if set(back["origin"].unique()) != {origin}:
        problems.append(f"origin values {set(back['origin'].unique())}")
    if (back["status"] == "survived").sum() != len(surv):
        problems.append("survivor count differs from the extracted table")
    try:
        pd.testing.assert_frame_equal(back, out.reset_index(drop=True))
    except AssertionError as e:
        problems.append(f"read-back table differs from the written one: {e}")
    if problems:
        sys.exit(f"{parquet} failed its check (large files kept in {workdir}):\n  " + "\n  ".join(problems))
    times["write_and_check"] = time.perf_counter() - t
    print(f"  write+check {times['write_and_check']:8.1f} s")

    status_counts = {s: int((back["status"] == s).sum()) for s in STATUSES}
    times["total"] = time.perf_counter() - t_start
    record = {
        "chunk": k,
        "origin": origin,
        "n_fragments": chunk_size,
        "seed": summary["seed"],
        "seed_spawn_key": summary["seed_spawn_key"],
        "status_counts": status_counts,
        "survival_rate": status_counts["survived"] / chunk_size,
        **accuracy,
        "n_rejections": summary["rejections"][origin],
        "rejected_letters": summary["rejected_letters"],
        "seconds": times,
        "bowtie2_wall_seconds": bowtie2_wall,
        "bowtie2_peak_rss_bytes": bowtie2_rss,
        "bowtie2_overall_alignment_rate": bowtie2_rate,
        "python_peak_rss_bytes": peak_rss_bytes(),
        "bowtie2_log": open(f"{prefix}.bowtie2.log").read(),
    }
    with open(pool_dir / f"{name}.json", "w") as f:
        json.dump(record, f, indent=2)

    shutil.rmtree(workdir)
    print(f"  total       {times['total']:8.1f} s | survived {100 * record['survival_rate']:.2f}% | "
          f"peak memory Bowtie2 {bowtie2_rss / 1e9 if bowtie2_rss else float('nan'):.2f} GB, "
          f"Python {record['python_peak_rss_bytes'] / 1e9:.2f} GB")
    print(f"  wrote {parquet} and {name}.json; deleted {workdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, nargs="+", help="chunk indices to run (default: all, in order)")
    args = parser.parse_args()

    config = load_config()
    total = n_chunks(config)
    chunks = args.chunks if args.chunks is not None else range(total)
    for k in chunks:
        if not 0 <= k < total:
            sys.exit(f"chunk {k} out of range 0..{total - 1}")
    for k in chunks:
        run_chunk(config, k)


if __name__ == "__main__":
    main()
