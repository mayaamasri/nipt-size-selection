"""Task 5A: survival by origin per length bin, saved so figures and Fig. 1 need not reload the pool.

Same bins as the mappability table (survival_bin_bp, 36-45, ..., 496-500). Per bin: fragments, survivors and
survival with Wilson 95% CI for maternal, fetal and pooled. The simulation uses only the POOLED S(bin), for
both origins (decision #3); the maternal and fetal columns are a check that origin does not change survival
within a bin. The pooled counts are asserted to equal results/tables/mappability.csv exactly. The last row
(bin "all") is the overall survival of each origin over all lengths.

Writes results/tables/survival_by_origin.csv. Run from the repo root:
    python -m src.simulate.survival_by_origin
"""

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.simulate.lengths import ORIGINS
from src.simulate.mappability import bin_index, length_bins, load_pool, wilson_interval

GROUPS = ORIGINS + ("pooled",)


def survival_by_origin(pool, config):
    lo, hi = length_bins(config)
    b = bin_index(pool["true_length"].to_numpy(), config)
    survived = (pool["status"] == "survived").to_numpy()
    table = pd.DataFrame({"bin": [f"{l}-{h}" for l, h in zip(lo, hi)], "lo": lo, "hi": hi})
    for group in GROUPS:
        sel = np.ones(len(pool), bool) if group == "pooled" else (pool["origin"] == group).to_numpy()
        table[f"n_{group}"] = np.bincount(b[sel], minlength=len(lo))
        table[f"survived_{group}"] = np.bincount(b[sel & survived], minlength=len(lo))

    total = {"bin": "all", "lo": lo[0], "hi": hi[-1]}
    for group in GROUPS:
        total[f"n_{group}"] = table[f"n_{group}"].sum()
        total[f"survived_{group}"] = table[f"survived_{group}"].sum()
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)

    for group in GROUPS:
        n, k = table[f"n_{group}"], table[f"survived_{group}"]
        low, high = wilson_interval(k, n)
        table[f"S_{group}"], table[f"S_{group}_lo"], table[f"S_{group}_hi"] = k / n, low, high
    return table


def check_against_mappability(table, config):
    """Pooled per-bin counts must equal the S rows of mappability.csv (what the simulation uses)."""
    mapp = pd.read_csv(repo_path(config["paths"]["results"]) / "tables" / "mappability.csv")
    s = mapp[mapp["quantity"] == "S"].reset_index(drop=True)
    bins = table[table["bin"] != "all"].reset_index(drop=True)
    assert list(bins["bin"]) == list(s["key"]), "bins differ from mappability.csv"
    assert np.array_equal(bins["n_pooled"], s["n"]), "pooled fragment counts differ from mappability.csv"
    assert np.array_equal(bins["survived_pooled"], s["count"]), "pooled survivor counts differ from mappability.csv"


def main():
    config = load_config()
    pool, _ = load_pool(config)
    table = survival_by_origin(pool, config)
    check_against_mappability(table, config)
    out = repo_path(config["paths"]["results"]) / "tables" / "survival_by_origin.csv"
    table.to_csv(out, index=False, float_format="%.8g")
    print(f"Pooled counts match mappability.csv in all {len(table) - 1} bins")
    for group in GROUPS:
        r = table.iloc[-1]
        print(f"  {group:>8}: {r[f'survived_{group}']:,} / {r[f'n_{group}']:,} survived "
              f"({100 * r[f'S_{group}']:.3f}%)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
