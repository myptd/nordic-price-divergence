"""
Training windows, 2025: per-hour LightGBM with monthly recalibration (models/rolling.py).

Versions (all training data end on the last day before the test month):
  365, 730, 1095          last 365, 730, 1,095 days, level-and-scale target
  expanding               from 2018-01-08, level-and-scale target (the benchmark forecasts)
  expanding_no_crisis     expanding without delivery days in 2021-2022
  average                 equal-weight mean of the 365, 730, 1095 and expanding forecasts (no new fits)
  raw_expanding, raw_365  raw price target (screening setting for the raw per-hour model), expanding and 365 days
Per zone, on the hours all versions cover (the filled 02:00 slot of spring-change days excluded): rMAE (relative to the
D-7 naive forecast) and MAE; multivariate DM tests (Lago et al., 2021) of every version against the expanding window in
both directions (14 one-sided tests), Holm-adjusted within the zone.

Outputs: results/windows_2025.csv, results/windows_2025_summary.txt.
Run from the repository root:  python analysis/13_training_windows.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

FILES = {"365": "365", "730": "730", "1095": "1095", "expanding": "expanding", "expanding_no_crisis": "expanding_no_crisis",
         "raw_expanding": "expanding_none", "raw_365": "365_none"}
ORDER = ["365", "730", "1095", "expanding", "expanding_no_crisis", "average", "raw_365", "raw_expanding"]
NO, OTHER = utils.NO_ZONES, ["SE1", "SE2", "SE3", "SE4", "DE_LU"]


def holm(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def zone_frame(zone):
    parts = []
    for name, f in FILES.items():
        d = pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_base_{f}.parquet")
        parts.append(d.assign(version=name))
    p = pd.concat(parts, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="version", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    W["average"] = W[["365", "730", "1095", "expanding"]].mean(axis=1)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    W["naive_d7"] = s.reindex(W.index).values
    keep = W[ORDER + ["naive_d7"]].notna().all(axis=1) & scored
    return W[keep], y[keep]


def main():
    rows = []
    for zone in utils.ALL_ZONES:
        W, y = zone_frame(zone)
        ref = np.mean(np.abs(y - W["naive_d7"]))
        E = {v: (y - W[v]).unstack("hour").reindex(columns=range(24)).values for v in ORDER}
        zr = []
        for v in ORDER:
            mae = float(np.mean(np.abs(y - W[v])))
            r = {"zone": zone, "version": v, "n_hours": int(len(y)), "MAE": mae, "rMAE": mae / ref}
            if v != "expanding":
                dm, p_win = utils.dm_test_lago(E["expanding"], E[v])          # small p: this version more accurate
                _, p_exp = utils.dm_test_lago(E[v], E["expanding"])           # small p: expanding more accurate
                r.update({"dm_vs_expanding": dm, "p_version_better": p_win, "p_expanding_better": p_exp})
            zr.append(r)
        zr = pd.DataFrame(zr)
        tested = zr.version != "expanding"
        adj = holm(np.concatenate([zr.loc[tested, "p_version_better"].values, zr.loc[tested, "p_expanding_better"].values]))
        n = int(tested.sum())
        zr.loc[tested, "p_version_better_holm"], zr.loc[tested, "p_expanding_better_holm"] = adj[:n], adj[n:]
        rows.append(zr)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv("results/windows_2025.csv", index=False)

    par = []
    for zone in utils.ALL_ZONES:
        for v, f in FILES.items():
            q = pd.read_csv(f"{rolling.OUT}/{zone}_lgbm_hourly_base_{f}_params.csv")
            par.append({"zone": zone, "version": v, "median_trees": q.n_trees.median(), "cap_share": q.cap_hit.mean()})
    par = pd.DataFrame(par)

    pd.set_option("display.width", 230)
    wide = lambda col: t.pivot(index="version", columns="zone", values=col).reindex(ORDER)[utils.ALL_ZONES]
    r = wide("rMAE")
    r.insert(5, "mean NO", r[NO].mean(axis=1))
    sig_win = t[t.p_version_better_holm < 0.05].groupby("version").zone.apply(lambda s: ", ".join(s)).reindex(ORDER).fillna("")
    sig_exp = t[t.p_expanding_better_holm < 0.05].groupby("version").zone.apply(lambda s: ", ".join(s)).reindex(ORDER).fillna("")
    lines = ["Training windows, per-hour LightGBM, monthly recalibration, 2025.", "",
             "rMAE (relative to naive D-7)", r.round(3).to_string(), "",
             "MAE (EUR/MWh)", wide("MAE").round(2).to_string(), "",
             "Zones where the version is significantly more accurate than the expanding window (Holm within zone, p < 0.05)",
             sig_win.drop("expanding").to_string(), "",
             "Zones where the expanding window is significantly more accurate than the version (Holm within zone, p < 0.05)",
             sig_exp.drop("expanding").to_string(), "",
             "Holm-adjusted p that the version is more accurate than expanding",
             wide("p_version_better_holm").drop("expanding").round(4).to_string(), "",
             "Holm-adjusted p that expanding is more accurate than the version",
             wide("p_expanding_better_holm").drop("expanding").round(4).to_string(), "",
             "Median monthly number of trees (per-hour models) and share of fits at the 1,000-tree cap",
             par.pivot(index="version", columns="zone", values="median_trees").round(0).to_string(), "",
             par.pivot(index="version", columns="zone", values="cap_share").round(3).to_string()]
    open("results/windows_2025_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
