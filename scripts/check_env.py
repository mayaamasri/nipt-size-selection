"""Sanity-check the environment and reference files.

Run from the repository root:   python scripts/check_env.py
Every line should print OK. Fix anything marked FAIL before writing pipeline code.
"""
import importlib
import shutil
import subprocess
from pathlib import Path

OK, FAIL = "  OK  ", " FAIL "
problems = 0


def report(ok, msg):
    global problems
    print(f"[{OK if ok else FAIL}] {msg}")
    problems += 0 if ok else 1


# 1) Python packages
for pkg in ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "pysam",
            "pyfaidx", "matplotlib", "seaborn", "yaml", "tqdm"]:
    try:
        mod = importlib.import_module(pkg)
        report(True, f"python package {pkg} {getattr(mod, '__version__', '')}")
    except ImportError:
        report(False, f"python package {pkg} missing (conda env active?)")

# 2) Command-line tools
for tool in ["bowtie2", "samtools"]:
    path = shutil.which(tool)
    if path:
        out = subprocess.run([tool, "--version"], capture_output=True, text=True).stdout
        report(True, f"{tool}: {out.splitlines()[0] if out else path}")
    else:
        report(False, f"{tool} not found on PATH")

# 3) Reference files
ref = Path("data/reference")
fasta = ref / "hg38.analysisSet.fa"
index_prefix = ref / "GRCh38_noalt_as" / "GRCh38_noalt_as"
report(fasta.exists(), f"reference FASTA at {fasta}")
report((ref / "hg38.analysisSet.fa.fai").exists(), "FASTA index (.fai)")
report(Path(str(index_prefix) + ".1.bt2").exists(), f"Bowtie2 index at {index_prefix}.*")

# 4) Chromosome names must match between FASTA and Bowtie2 index,
#    otherwise simulated fragment origins can't be compared with mapped positions.
if fasta.exists() and Path(str(index_prefix) + ".1.bt2").exists():
    fasta_names = {l.split("\t")[0] for l in open(str(fasta) + ".fai")}
    idx = subprocess.run(["bowtie2-inspect", "-n", str(index_prefix)],
                         capture_output=True, text=True).stdout.split("\n")
    idx_names = {l.split()[0] for l in idx if l.strip()}
    autosomes = {f"chr{i}" for i in range(1, 23)}
    report(autosomes <= fasta_names, "FASTA contains chr1..chr22")
    report(autosomes <= idx_names, "Bowtie2 index contains chr1..chr22 (same naming)")

print("\nAll checks passed." if problems == 0 else f"\n{problems} problem(s) to fix.")
