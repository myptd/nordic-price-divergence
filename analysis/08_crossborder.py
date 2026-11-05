"""
Previous-day neighbouring prices (DE-LU, DK1, NL) as inputs, 2025: tests of H1 and H2.

Per-hour LightGBM with the "crossborder" input set against the base per-hour LightGBM, both from models/rolling.py
(monthly recalibration, expanding window, frozen settings), nine Norwegian and Swedish zones, on the hours both forecasts
and the D-7 naive cover (the filled 02:00 slot of spring-change days excluded).
Per zone: rMAE (relative to the D-7 naive forecast) and MAE with and without the inputs, delta = rMAE without - rMAE
with, and the multivariate DM test (Lago et al., 2021) in both directions (raw one-sided p-values).
  H1  Spearman rank correlation of delta with the frozen 2024 coupling values, one-sided (positive), p-value from all
      9! orderings, alpha 0.05.
  H2  NO3, NO4, SE1, SE2: one-sided DM p-values that the inputs help, Holm-adjusted over the four zones; H2 holds if no
      adjusted p < 0.05.

Outputs: results/crossborder_2025.csv, results/crossborder_2025_tests.txt.
Run from the repository root:  python analysis/08_crossborder.py
"""
import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

OUT = "results"
ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"]
_ps_spec = importlib.util.spec_from_file_location("paper_style", "analysis/paper_style.py")
_ps = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_ps)
COUPLING = dict(sorted(_ps.COUPLING.items(), key=lambda kv: kv[1]))   # frozen 2024 coupling measure, ascending
BELOW, ABOVE = ["NO3", "NO4", "SE1", "SE2"], ["NO1", "NO2", "SE3", "SE4"]
SPLIT = {**{z: "below median" for z in BELOW}, **{z: "above median" for z in ABOVE}, "NO5": "median"}


def holm(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def zone_frame(zone):
    parts = [pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_{inp}_expanding.parquet") for inp in ["base", "crossborder"]]
    p = pd.concat(parts, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="inputs", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    W["naive_d7"] = s.reindex(W.index).values
    keep = W[["base", "crossborder", "naive_d7"]].notna().all(axis=1) & scored
    return W[keep], y[keep]


def exact_spearman(x, y):
    """One-sided (positive) exact permutation p-value of Spearman's rho."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    perms = np.array(list(itertools.permutations(ry)))
    rhos = np.array([np.corrcoef(rx, q)[0, 1] for q in perms])
    return rho, float(np.mean(rhos >= rho - 1e-12)), len(perms)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for z in ZONES:
        W, y = zone_frame(z)
        ref = np.mean(np.abs(y - W["naive_d7"]))
        mae_b, mae_x = float(np.mean(np.abs(y - W["base"]))), float(np.mean(np.abs(y - W["crossborder"])))
        E = {k: (y - W[k]).unstack("hour").reindex(columns=range(24)).values for k in ["base", "crossborder"]}
        dm_x, p_x = utils.dm_test_lago(E["base"], E["crossborder"])        # small p: with inputs more accurate
        dm_b, p_b = utils.dm_test_lago(E["crossborder"], E["base"])        # small p: without inputs more accurate
        rows.append({"zone": z, "coupling_2024": COUPLING[z], "split": SPLIT[z], "n_hours": int(len(y)),
                     "MAE_without": mae_b, "MAE_with": mae_x, "rMAE_without": mae_b / ref, "rMAE_with": mae_x / ref,
                     "delta_rMAE": (mae_b - mae_x) / ref, "dm_with_better": dm_x, "p_with_better": p_x,
                     "dm_without_better": dm_b, "p_without_better": p_b})
    t = pd.DataFrame(rows)
    low = t.zone.isin(BELOW)
    t["p_with_better_holm_H2"] = np.nan
    t.loc[low, "p_with_better_holm_H2"] = holm(t.loc[low, "p_with_better"].values)
    t.to_csv(f"{OUT}/crossborder_2025.csv", index=False)

    rho, p_h1, n_perm = exact_spearman(t.coupling_2024.values, t.delta_rMAE.values)
    h2 = t[low]
    pd.set_option("display.width", 220)
    lines = ["Cross-border inputs (DE-LU, DK1, NL D-1 prices), per-hour LightGBM, 2025.",
             "delta = rMAE without - rMAE with (positive: the inputs help). DM p-values are one-sided and raw.", "",
             t.drop(columns=["p_with_better_holm_H2"]).round(4).to_string(index=False), "",
             f"H1: Spearman rho(delta, coupling) = {rho:.3f}; exact one-sided p = {p_h1:.4f} ({n_perm:,} orderings); "
             + ("SUPPORTED (p < 0.05)" if p_h1 < 0.05 else "NOT SUPPORTED (p >= 0.05)"), "",
             "H2: NO3, NO4, SE1, SE2, one-sided DM p that the inputs help, Holm-adjusted over the four zones",
             h2[["zone", "delta_rMAE", "dm_with_better", "p_with_better", "p_with_better_holm_H2"]].round(4).to_string(index=False),
             "H2 " + ("holds: no detectable improvement in any of the four zones (all adjusted p >= 0.05)"
                      if (h2.p_with_better_holm_H2 >= 0.05).all() else
                      "does not hold: detectable improvement in " + ", ".join(h2.zone[h2.p_with_better_holm_H2 < 0.05])), "",
             "H3: analysis/10_crossborder_gas.py."]
    open(f"{OUT}/crossborder_2025_tests.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
