# Task 02 (Hala): mapped pool, mappability table, and virtual-patient generator

**Goal:** build the virtual patients (proposal step 4) with the approach chosen in decision #3:
the mapped pool is used only to measure how the genome behaves (mappability), and every patient is
then generated as an independent random draw from that model.
**Target:** Friday Oct 30 evening. Noise (step 5) is NOT part of this task; it comes in Task 02b once
decision #7 is made, so leave a `sigma` argument that is 0 for now.

Decisions this task relies on: `docs/decisions.md` #3, #4, #12. **Run `git pull` first.**

---

## 2A. Small fixes to the Task 01 code (before running the pool)

The N-rejection in `draw_fragment` already redraws chromosome **and** position, and
`check_fragments.py` confirms the chromosome shares match the N-free shares. Nothing to change there.

Two things are needed before the pool can be generated in chunks:
1. **Seed per chunk.** `simulate()` always seeds with `config["seed"]`, so every chunk would produce
   *identical* fragments. Add a `seed` argument (or pass an `rng`) and give chunk `k` the seed
   `np.random.SeedSequence(config["seed"]).spawn(n_chunks)[k]`.
2. **Portability of `scripts/map_reads.sh`.** `sysctl -n hw.ncpu` and `/usr/bin/time -l` only exist on macOS,
   so Maya can't rerun the script on Linux/WSL. Use
   `THREADS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"`, and pick `/usr/bin/time -v` on Linux / `-l` on macOS
   (`report_mapping.bowtie2_timing` then needs to read either format, or just drop the timing for the pool runs).

## 2B. Map one large pool (no splitting into sub-pools)

- 10,000,000 maternal + 10,000,000 fetal fragments, generated and mapped in **chunks of 1,000,000**
  (each chunk gets its own seed derived from the config seed, so any chunk can be rerun).
- Reuse the Task 01 pipeline per chunk: `fragments.py` → `scripts/map_reads.sh` → `extract_mapped.py`
  (a small driver script, e.g. `scripts/run_pool.sh` or `src/simulate/pool.py`, that loops over chunks).
- Keep only one small table per chunk: `origin, true_chrom, true_length, status, mapped_chrom`
  (Parquet, under `data/processed/pool/`). **Delete the FASTQ, SAM and BAM files after each chunk**:
  the unfiltered SAM alone is several hundred MB per million pairs.
- Note that `extract_mapped.py` holds all reads of a chunk in memory; 1M pairs per chunk should be fine,
  but lower the chunk size if memory gets tight while Bowtie2's index is also loaded.
- **Run one chunk first and report its total time** (generation + mapping + table). If 20 chunks would take
  more than ~4 hours, tell Maya before running the rest; we'll reduce the pool to 10M.

## 2C. Mappability table (`src/simulate/mappability.py`)

From the pool (maternal and fetal together, since the genome is the same for both), estimate:
- `S(l)`: probability that a fragment of length `l` survives (`status == "survived"`), in **10 bp bins** (36–45, 46–55, …).
- `q(c)`: share of surviving fragments on chromosome `c`, using **`mapped_chrom`** (what a lab would see),
  restricted to chr1–22 (22 numbers summing to 1). Report how many survivors mapped outside chr1–22 (should be ~0).

Save both to `data/processed/mappability.npz` (and a small CSV copy for the repo: `results/tables/mappability.csv`).

**QC plots** (`results/figures/`):
1. `S(l)` against length.
2. chr21 share of survivors **per 10 bp length bin**, with 95% error bars. We assume this is flat
   (chr21 share doesn't depend on length). If it clearly trends with length, tell Maya.

## 2D. Patient generator (`src/simulate/patients.py`)

Each patient only needs **two rows of counts by exact length (36–500 bp)**: chr21 and all other autosomes
(a 2 × 465 matrix). This contains everything methods A, B, C1, C2 need.

For one origin `o` (maternal or fetal), the probability of a surviving fragment with chromosome-group `g`
and length `l` is:

```
P_o(g, l)  ∝  P_o(l) × S(l) × q(g)          q(chr21) = q[chr21],  q(other) = 1 − q[chr21]
```
normalised to sum to 1 over the 2 × 465 cells. `P_o(l)` comes from `src.simulate.lengths.load_length_distributions`
(already exactly the 465 lengths 36–500), `S(l)` from 2C (each length gets its bin's value).

- **Trisomy 21:** in the fetal matrix only, multiply the chr21 row by **1.5**, then renormalise.
- **Mixture** for a patient with fetal fraction `FF` and depth `N`:
  `P = FF × P_fetal + (1 − FF) × P_maternal`, then `counts = rng.multinomial(N, P)`.
  Depth and FF are defined on fragments **after** the MAPQ filter (decision #12).
- **Randomness:** use `np.random.SeedSequence(config seed).spawn(...)` so every
  (condition, set) gets its own independent, reproducible stream.

Function to provide:
```python
generate_patients(ff, depth, n_patients, trisomy: bool, sigma=0.0, rng) -> array (n_patients, 2, 465)
```
Don't save all patients to disk (it would be tens of GB). Task 03 will generate them on the fly
and compute the features in the same pass. For now, save only what the QC below needs.

## 2E. QC (report these numbers to Maya)

1. `q(chr21)` and the overall survival rate (should be ~86% like Task 01).
2. **Normal patients, FF 10%, depth 1M, 2,000 patients:**
   - mean chr21 fraction (should equal `q(chr21)`)
   - SD of chr21 fraction across patients, next to the theoretical `sqrt(p(1−p)/N)` (should match)
3. **Trisomy, same setting:** mean increase of chr21 fraction relative to normals, next to the expected
   `p × FF / 2` (approximately).
4. **Realism check (proposal Evaluation item 5):** for FF 4% and 10%, the fetal fraction **within
   fragments 40–156 bp** divided by the overall FF, and the fraction of fragments kept.
   Expected roughly: FF gain ~2.2–2.5×, ~20% kept (Yang: FF ~2×, 75% discarded).
5. Time to generate 2,000 patients at depth 10M.

---

### Config keys to add (`config/config.yaml`)
```yaml
simulation:
  pool_size_per_origin: 10000000   # decision #3
  pool_chunk_size: 1000000
  survival_bin_bp: 10

patients_per_condition:            # decision #4
  reference: 2000                  # normals only
  train_normal: 10000
  train_t21: 2000
  test_normal: 10000
  test_t21: 2000
```
Replace the old `patients_per_condition` block (it has `train: TODO`) with the one above, remove
`pool_split` (no sub-pools any more), and drop the "(TODO: confirm)" comment on `chromosomes` (decision #11 is made).
