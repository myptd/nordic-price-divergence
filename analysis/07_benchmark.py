"""
Benchmark of 2025, from models/rolling.py (base inputs, expanding window).

Per zone, on the hours all six forecasts cover (the filled 02:00 slot of spring-change days excluded):
  rMAE (relative to the D-7 naive forecast), MAE, RMSE for naive D-1, naive D-7, LEAR, pooled LightGBM,
  per-hour LightGBM (main model) and the ensemble (equal-weight LEAR + per-hour LightGBM);
  multivariate DM tests (Lago et al., 2021) for all 30 ordered pairs, one-sided (column model more accurate than
  row model), Holm-adjusted within each zone.
Also summarises the monthly numbers of trees (and cap hits) and the LEAR penalties.

Outputs: results/benchmark_2025.csv, results/benchmark_2025_dm.csv, results/benchmark_2025_summary.txt.
Run from the repository root:  python analysis/07_benchmark.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

OUT = "results"
FORECASTS = ["naive_d1", "naive_d7", "lear", "lgbm_pooled", "lgbm_hourly", "ensemble"]
NAMES = {"naive_d1": "Naive D-1", "naive_d7": "Naive D-7", "lear": "LEAR", "lgbm_pooled": "LightGBM pooled",
         "lgbm_hourly": "LightGBM per hour", "ensemble": "Ensemble"}


def holm(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def zone_frame(zone):
    parts = [pd.read_parquet(f"{rolling.OUT}/{zone}_{m}_base_expanding.parquet") for m in rolling.MODELS]
    p = pd.concat(parts, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="model", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    for name, k in [("naive_d1", 1), ("naive_d7", 7)]:
        s = P.shift(k).stack(future_stack=True)
        s.index.names = ["date", "hour"]
        W[name] = s.reindex(W.index).values
    W["ensemble"] = W[["lear", "lgbm_hourly"]].mean(axis=1)
    keep = W[FORECASTS].notna().all(axis=1) & scored
    return W.loc[keep, FORECASTS], y[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, tests = [], []
    for zone in utils.ALL_ZONES:
        W, y = zone_frame(zone)
        ref = np.mean(np.abs(y - W["naive_d7"]))
        E = {m: (y - W[m]).unstack("hour").reindex(columns=range(24)).values for m in FORECASTS}
        for m in FORECASTS:
            err = y - W[m]
            rows.append({"zone": zone, "model": m, "n_hours": int(len(y)), "MAE": float(err.abs().mean()),
                         "RMSE": float(np.sqrt((err ** 2).mean())), "rMAE": float(err.abs().mean() / ref)})
        zt = []
        for a in FORECASTS:
            for b in FORECASTS:
                if a != b:
                    dm, p = utils.dm_test_lago(E[a], E[b])
                    zt.append({"zone": zone, "row": a, "column": b, "dm": dm, "p_column_better": p})
        zt = pd.DataFrame(zt)
        zt["p_holm"] = holm(zt.p_column_better.values)
        tests.append(zt)
    t, dm = pd.DataFrame(rows), pd.concat(tests, ignore_index=True)
    t.to_csv(f"{OUT}/benchmark_2025.csv", index=False)
    dm.to_csv(f"{OUT}/benchmark_2025_dm.csv", index=False)

    par = pd.concat([pd.read_csv(f"{rolling.OUT}/{z}_{m}_base_expanding_params.csv") for z in utils.ALL_ZONES for m in rolling.MODELS])
    pd.set_option("display.width", 220)
    wide = lambda col: t.pivot(index="model", columns="zone", values=col).reindex(FORECASTS)[utils.ALL_ZONES].rename(index=NAMES)
    r = wide("rMAE")
    r.insert(5, "mean NO", r[utils.NO_ZONES].mean(axis=1))
    sig = dm[dm.p_holm < 0.05].groupby(["row", "column"]).size().unstack("column").reindex(index=FORECASTS, columns=FORECASTS).fillna(0).astype(int)
    main_vs = dm[(dm.column == "lgbm_hourly") | (dm.row == "lgbm_hourly")]
    lines = ["2025 benchmark, monthly recalibration, expanding window from 2018-01-08.", "",
             "rMAE (relative to naive D-7)", r.round(3).to_string(), "",
             "MAE (EUR/MWh)", wide("MAE").round(2).to_string(), "",
             "RMSE (EUR/MWh)", wide("RMSE").round(2).to_string(), "",
             "Zones (of 10) where the column model is significantly more accurate than the row model (Holm within zone, p < 0.05)",
             sig.rename(index=NAMES, columns=NAMES).to_string(), "",
             "Main model (per-hour LightGBM) against the others: Holm-adjusted p that the column model is more accurate",
             main_vs.pivot_table(index=["row", "column"], columns="zone", values="p_holm")[utils.ALL_ZONES].round(4).to_string(), "",
             "Monthly number of trees (per-hour: median over hours) and share of fits that reached the 1,000-tree cap",
             par[par.model != "lear"].groupby(["model", "zone"]).agg(median_trees=("n_trees", "median"), cap_share=("cap_hit", "mean")).round(3).to_string(), "",
             "LEAR penalty (AIC), median over months and hours",
             par[par.model == "lear"].groupby("zone").alpha.median().to_string()]
    open(f"{OUT}/benchmark_2025_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
