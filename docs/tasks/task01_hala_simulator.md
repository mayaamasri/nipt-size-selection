# Task 01 (Hala): fragment-length sampler, read simulator, and a mapping speed test

**Goal:** a small, end-to-end, *verified* version of pipeline steps 1–3 (proposal), plus one timing number
that decides how large the fragment pools can be.
**Target:** Wed Oct 28 evening. Paste this file into a GitHub issue and tick items off there.

Decisions this task relies on: `docs/decisions.md` #1, #2, #11.

---

## 1A. Length distributions (`src/simulate/lengths.py`)

Input: `data/raw/rabinowitz_G1_length_counts.csv` (already in the repo; lines starting with `#` are comments).

Processing, in this order:
1. Drop length 0 (fragments with no valid insert size).
2. Replace the 101 bp bin of **both** columns by the mean of the 100 and 102 bp bins
   (read-length artifact in the source data: fetal spike, maternal dip at exactly 101 bp).
3. Keep lengths **36–500 bp** only, then normalise each column to sum to 1.

Provide a function `sample_lengths(origin, n, rng)` returning `n` integer lengths for `"maternal"` or `"fetal"`
(use `rng.choice` with the probabilities; `rng = np.random.default_rng(seed)` from config).

**Check (must pass):** after step 3, the medians should be about **172 bp (maternal)** and **157 bp (fetal)**.
Save a plot of both distributions to `results/figures/qc_length_distributions.png`.

## 1B. Fragment and read simulation (`src/simulate/fragments.py`)

For each fragment:
1. Draw origin and length (1A).
2. Draw a chromosome among chr1–chr22 with probability proportional to chromosome length, then a uniform start position.
3. Fetch the sequence with `pyfaidx`. If it contains any `N`, **reject and redraw** (keep a counter of rejections).
4. Read 1 = first **36** bases of the fragment. Read 2 = reverse complement of the last 36 bases.
5. Write paired FASTQ files (`_R1.fastq`, `_R2.fastq`); quality string = `I` repeated 36 times.

Put the truth in the read name so mapping can be verified later, e.g.
`@frag000001|mat|chr7|12345678|163` (id | origin | true chrom | true 0-based start | true length).

## 1C. Mapping (`scripts/map_reads.sh`)

```bash
bowtie2 -p <nproc> -X 500 --no-mixed --no-discordant \
  -x data/reference/GRCh38_noalt_as/GRCh38_noalt_as \
  -1 R1.fastq -2 R2.fastq \
| samtools view -b -f 2 -q 40 -o mapped.bam -
```
- `-X 500` is Bowtie2's maximum fragment length (the default is also 500; set it explicitly so it's documented).
- `-f 2` keeps properly paired reads only; `-q 40` is the MAPQ ≥ 40 filter from the proposal.

Then extract one row per fragment into a table (`pysam`, read 1 only, `abs(template_length)`):
`id, origin, true_chrom, true_start, true_length, mapped_chrom, mapped_start, mapped_length`.

## 1D. Verification on 10,000 fragments (5,000 maternal, 5,000 fetal)

Report these numbers to Maya:
- % of fragments surviving MAPQ ≥ 40, separately for maternal and fetal
- % of survivors mapped to the correct chromosome and position (should be ~100%)
- % where `mapped_length == true_length` (should be ~100%)
- number of N-rejections

## 1E. Timing test on 100,000 read pairs

Run 1B + 1C on 100,000 pairs and report:
- output of `nproc`
- wall time of the `bowtie2` step (prefix the command with `time`)
- peak memory if easy (`/usr/bin/time -v`, "Maximum resident set size")

**Do not** generate the full pools yet. The timing number decides the pool size (decision #3).

---

### Conventions
- All parameters read from `config/config.yaml`; no hard-coded numbers in code.
- Large outputs go under `data/processed/` (git-ignored). Commit code and the QC plot only.
- Small commits, `git pull` before starting.
