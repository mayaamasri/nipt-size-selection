#!/usr/bin/env bash
# Task 1C: map simulated read pairs with Bowtie2, then keep proper pairs with MAPQ >= min_mapq.
# Run from anywhere, with the nipt conda env active:
#   bash scripts/map_reads.sh data/processed/task01/sim10k
# Inputs:  <prefix>_R1.fastq, <prefix>_R2.fastq            (from src/simulate/fragments.py)
# Outputs: <prefix>.sam          all Bowtie2 output, unfiltered (used for loss reasons in 1D)
#          <prefix>.mapped.bam   proper pairs (-f 2), reads with MAPQ >= min_mapq (-q)
#          <prefix>.bowtie2.log  Bowtie2 alignment summary + /usr/bin/time (wall time, peak memory):
#                                macOS -l (memory in bytes), Linux -v (memory in kbytes);
#                                no timing lines if /usr/bin/time is not installed
# Runs on macOS and Linux/WSL.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: bash scripts/map_reads.sh <prefix>   (e.g. data/processed/task01/sim10k)" >&2
  exit 1
fi
PREFIX="$1"

cd "$(dirname "$0")/.."   # repo root, so config paths resolve

# Parameters from config/config.yaml (never hard-coded here)
read -r MIN_MAPQ MAX_LEN INDEX < <(python -c '
import yaml
c = yaml.safe_load(open("config/config.yaml"))
print(c["simulation"]["min_mapq"], c["simulation"]["fragment_length_max"], c["paths"]["bowtie2_index"])
')
THREADS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"   # nproc on Linux, sysctl on macOS

# /usr/bin/time flag differs: -l (BSD, macOS) vs -v (GNU, Linux). Skip timing if it is missing.
case "$(uname -s)" in
  Darwin) TIME_FLAG="-l" ;;
  *)      TIME_FLAG="-v" ;;
esac
if [ -x /usr/bin/time ]; then
  TIME_CMD=(/usr/bin/time "$TIME_FLAG")
else
  TIME_CMD=()
  echo "warning: /usr/bin/time not found, Bowtie2 runs without timing" >&2
fi

for tool in bowtie2 samtools; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found (conda activate nipt?)" >&2; exit 1; }
done
[ -f "${INDEX}.1.bt2" ] || { echo "error: Bowtie2 index not found at ${INDEX}.*.bt2" >&2; exit 1; }
for r in R1 R2; do
  [ -f "${PREFIX}_${r}.fastq" ] || { echo "error: ${PREFIX}_${r}.fastq not found" >&2; exit 1; }
done

echo ">> bowtie2: ${THREADS} threads, -X ${MAX_LEN}, index ${INDEX}"
# Bowtie2 alone, timed: writes SAM to a file so the time excludes samtools.
# -X: max fragment length for a proper pair; --no-mixed / --no-discordant: report pairs only as concordant pairs.
# ${TIME_CMD[@]+...} expands to nothing for an empty array under set -u (macOS bash 3.2).
${TIME_CMD[@]+"${TIME_CMD[@]}"} bowtie2 -p "$THREADS" -X "$MAX_LEN" --no-mixed --no-discordant \
  -x "$INDEX" -1 "${PREFIX}_R1.fastq" -2 "${PREFIX}_R2.fastq" \
  -S "${PREFIX}.sam" 2> "${PREFIX}.bowtie2.log"
cat "${PREFIX}.bowtie2.log"

echo ">> samtools: keep proper pairs (-f 2) and reads with MAPQ >= ${MIN_MAPQ} (-q)"
# -q works per read, not per pair: the extraction step keeps a fragment only if both mates are here.
samtools view -b -f 2 -q "$MIN_MAPQ" -o "${PREFIX}.mapped.bam" "${PREFIX}.sam"
echo ">> wrote ${PREFIX}.sam, ${PREFIX}.mapped.bam, ${PREFIX}.bowtie2.log"
