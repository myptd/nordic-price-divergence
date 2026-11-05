"""
Transfer matrix summary: rMAE of pooled LightGBM trained on one year and tested on another, 2018-2025, ten zones,
raw / level-only / level-and-scale targets (models/transfer_matrix.py). Used by Fig. 3a and Fig. S4.

Outputs in results/transfer/: transfer_long.csv (all cells), worst_offdiagonal.csv (worst off-diagonal cell per zone
and target) and summary.txt (worst cells and the mean matrices over NO1-NO5).
Run from the repository root:  python analysis/12_transfer_matrix.py
"""
import os

import numpy as np
import pandas as pd

RT = "model_results/transfer"
OUT = "results/transfer"
NO, OTHER = ["NO1", "NO2", "NO3", "NO4", "NO5"], ["SE1", "SE2", "SE3", "SE4", "DE_LU"]
NORMS = ["none", "level", "level_scale"]
YEARS = list(range(2018, 2026))


def worst_offdiagonal(t):
    rows = []
    for z in NO + OTHER:
        for n in NORMS:
            M = t[(t.zone == z) & (t.norm == n)].pivot(index="train_year", columns="test_year", values="rMAE") \
                .reindex(index=YEARS, columns=YEARS).values.copy()
            np.fill_diagonal(M, np.nan)
            i, j = np.unravel_index(np.nanargmax(M), M.shape)
            rows.append({"zone": z, "norm": n, "worst_offdiag": M[i, j], "train": YEARS[i], "test": YEARS[j],
                         "mean_offdiag": np.nanmean(M), "n_offdiag_above_1": int((M > 1).sum())})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.concat([pd.read_csv(f"{RT}/{z}.csv") for z in NO + OTHER], ignore_index=True)
    worst = worst_offdiagonal(t)
    t.to_csv(f"{OUT}/transfer_long.csv", index=False)
    worst.to_csv(f"{OUT}/worst_offdiagonal.csv", index=False)
    pd.set_option("display.width", 220)
    no = t[t.zone.isin(NO)]
    lines = ["Transfer matrix, pooled LightGBM. rMAE relative to the D-7 naive forecast of the test year.", "",
             "Worst off-diagonal rMAE per zone and target", worst.round(3).to_string(index=False), "",
             "Mean over NO1-NO5, by target (rows train year, columns test year)"]
    for n in NORMS:
        lines += [n, no[no.norm == n].pivot_table(index="train_year", columns="test_year", values="rMAE").round(2).to_string(), ""]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
