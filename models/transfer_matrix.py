"""
Year-to-year transfer matrix, pooled LightGBM: how well a model trained on one year forecasts another.

Train on one year i, test on one year j (2018-2025). The number of trees for year i is set by
early stopping on the last 56 days of year i (fitted on the rest of year i). Off-diagonal cells:
that model predicts all of year j. Diagonal: quarter-blocked cross-validation inside year i
with the same number of trees and a 7-day gap on both sides of the test quarter.
Three target versions: raw price ("none"), level only ("level"), level and scale ("level_scale").
rMAE is relative to the D-7 naive forecast on year j.

Saves model_results/transfer/{zone}.csv.
Run from the repository root:
  python models/transfer_matrix.py --zones NO1
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import advanced
import utils

OUT = "model_results/transfer"
YEARS = list(range(2018, 2026))
NORMS = ["none", "level", "level_scale"]
INNER_HOLDOUT_DAYS = 56
PURGE_DAYS = 7


def run_zone(zone: str) -> None:
    print(f"\n=== transfer matrix {zone} ===", flush=True)
    t0 = time.time()
    hourly = utils.load_zone(zone)
    utils.require_commodities(hourly.columns, "The transfer matrix (pooled LightGBM)")
    X, y, mask = utils.build_features(hourly)
    dates = X.index.get_level_values("date")
    naive = X["price_d7"].values
    rows = []
    for norm in NORMS:
        for i in YEARS:
            start, end = utils.year_range(i)
            cut = pd.Timestamp(end) - pd.Timedelta(days=INNER_HOLDOUT_DAYS - 1)
            fit_rows = utils.in_dates(X, start, cut - pd.Timedelta(days=1))
            hold_rows = utils.in_dates(X, cut, end)
            model = advanced.fit_lgbm(X[fit_rows], y[fit_rows], X[hold_rows], y[hold_rows],
                                      n_jobs=utils.N_JOBS, norm=norm)
            n_trees = advanced.best_iteration(model)
            for j in YEARS:
                test_rows = utils.in_dates(X, *utils.year_range(j))
                if j != i:
                    pred = model.predict(X[test_rows])
                else:
                    pred = np.full(test_rows.sum(), np.nan)
                    year_dates = dates[test_rows]
                    for q in range(1, 5):
                        in_q = np.asarray(year_dates.quarter == q)
                        q0, q1 = year_dates[in_q].min(), year_dates[in_q].max()
                        far = (dates < q0 - pd.Timedelta(days=PURGE_DAYS)) | (dates > q1 + pd.Timedelta(days=PURGE_DAYS))
                        tr = utils.in_dates(X, start, end) & np.asarray(far)
                        m = advanced.fit_lgbm(X[tr], y[tr], n_estimators=n_trees, n_jobs=utils.N_JOBS, norm=norm)
                        pred[in_q] = m.predict(X[test_rows][in_q])
                r = utils.evaluate(y[test_rows], pred, naive[test_rows], mask[test_rows])
                rows.append({"zone": zone, "norm": norm, "train_year": i, "test_year": j, "n_trees": n_trees, **r})
            print(f"  {norm:12s} train {i} done ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{zone}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", nargs="+", default=utils.ALL_ZONES)
    args = parser.parse_args()
    for z in args.zones:
        run_zone(z)
