"""Task 5F: check that rerunning one condition reproduces its saved rows exactly.

Runs run_condition() for one FF x depth x noise in memory (nothing is written) and compares every table it
produces (results_all, bopt_separation, c2_feature_importances, tree_cv_auc) with that condition's rows in
results/tables/, as the text run_condition would write (float_format %.10g, same column order and sorting).
Exits with status 1 if any row differs.

Run from the repo root (default: FF 4%, 5M, realistic noise):
    python -m src.evaluate.check_rerun
    python -m src.evaluate.check_rerun --ff 0.02 --depth 10000000 --noise low
"""

import argparse
import io
import sys
import time

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.evaluate.run_condition import CONDITION, parse_noise, run_condition, sort_table


def as_written(table):
    """Text form of a table as run_condition writes it, read back as strings."""
    buf = io.StringIO()
    table.to_csv(buf, index=False, float_format="%.10g")
    return pd.read_csv(io.StringIO(buf.getvalue()), dtype=str, keep_default_na=False)


def saved_rows(path, ff, depth, noise):
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    same = (np.isclose(saved["ff"].astype(float), ff, rtol=1e-9, atol=0) & (saved["depth"].astype(int) == depth)
            & (saved["noise"] == str(noise)))
    return saved[same].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Rerun one condition and compare with the saved tables.")
    parser.add_argument("--ff", type=float, default=0.04)
    parser.add_argument("--depth", type=int, default=5_000_000)
    parser.add_argument("--noise", type=parse_noise, default="realistic")
    args = parser.parse_args()

    config = load_config()
    out_dir = repo_path(config["paths"]["results"]) / "tables"
    start = time.time()
    tables, _ = run_condition(config, args.ff, args.depth, args.noise)
    print(f"Reran FF {args.ff}, depth {args.depth:,}, noise {args.noise} in {time.time() - start:.1f} s")

    ok = True
    for name, table in tables.items():
        new = as_written(sort_table(table, name, config))
        old = saved_rows(out_dir / name, args.ff, args.depth, args.noise)
        if list(new.columns) != list(old.columns) or new.shape != old.shape:
            print(f"  {name}: DIFFERENT shape/columns (new {new.shape}, saved {old.shape})")
            ok = False
            continue
        diff = (new != old)
        n_cells = int(diff.to_numpy().sum())
        print(f"  {name}: {len(new)} rows x {new.shape[1]} columns, {n_cells} cells differ")
        if n_cells:
            ok = False
            rows, cols = np.nonzero(diff.to_numpy())
            for r, c in list(zip(rows, cols))[:10]:
                print(f"    row {r}, {new.columns[c]}: new {new.iat[r, c]!r} vs saved {old.iat[r, c]!r}")
    print("IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
