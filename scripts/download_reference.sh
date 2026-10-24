#!/usr/bin/env bash
# Downloads the GRCh38 reference genome and a prebuilt Bowtie2 index.
# Run from the repository root:   bash scripts/download_reference.sh
# Needs ~15 GB free disk space during download/unpacking.
set -euo pipefail

REF_DIR="data/reference"
mkdir -p "$REF_DIR"
cd "$REF_DIR"

# 1) Reference FASTA - GRCh38 "no-alt analysis set" (UCSC naming: chr1, chr2, ...)
#    Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet/
if [ ! -f hg38.analysisSet.fa ]; then
  echo ">> Downloading reference FASTA (~0.9 GB compressed)..."
  curl -L -C - -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet/hg38.analysisSet.fa.gz
  gunzip hg38.analysisSet.fa.gz
fi

echo ">> Indexing FASTA with samtools..."
samtools faidx hg38.analysisSet.fa

# 2) Prebuilt Bowtie2 index for the same assembly (avoids hours of index building)
#    Source: https://bowtie-bio.sourceforge.net/bowtie2/index.shtml
if [ ! -d GRCh38_noalt_as ]; then
  echo ">> Downloading Bowtie2 index (~3.5 GB)..."
  curl -L -C - -O https://genome-idx.s3.amazonaws.com/bt/GRCh38_noalt_as.zip
  unzip GRCh38_noalt_as.zip -d GRCh38_noalt_as
  rm GRCh38_noalt_as.zip
fi

echo ">> Done. Now run:  python scripts/check_env.py"
