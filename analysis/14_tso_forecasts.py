"""
Value of the TSO forecasts, 2025: per-hour LightGBM without the TSO forecasts against the base per-hour LightGBM.

Input sets (models/rolling.py), all with the frozen settings, level-and-scale target, monthly recalibration and the
expanding window:
  base              the 247 per-hour inputs (the benchmark model)
  prices_calendar   without the load and the wind and solar forecasts (96 prices of D-1, D-2, D-3, D-7 and the weekday,
                    plus L and S)
  no_vre            without the wind and solar forecasts only (keeps the load forecasts); isolates the forecasts whose
                    publication time is uncertain (wind and solar forecasts may be published after 11:00 on D-1)
Per zone, on the hours all three cover: rMAE (relative to the D-7 naive forecast) and MAE; delta = rMAE without the
forecasts - rMAE of the base model (positive: the removed forecasts help); multivariate DM tests (Lago et al., 2021) in
both directions, one-sided, Holm across the ten zones within each input set and direction.

Outputs: results/no_forecasts_2025.csv, results/no_forecasts_2025_summary.txt.
Run from the repository root:  python analysis/14_tso_forecasts.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

SETS = ["prices_calendar", "no_vre"]


def holm(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def zone_frame(zone):
    p = pd.concat([pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_{i}_expanding.parquet") for i in ["base"] + SETS], ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="inputs", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    W["naive_d7"] = s.reindex(W.index).values
    keep = W[["base"] + SETS + ["naive_d7"]].notna().all(axis=1) & scored
    return W[keep], y[keep]


def main():
    rows = []
    for zone in utils.ALL_ZONES:
        W, y = zone_frame(zone)
        ref = np.mean(np.abs(y - W["naive_d7"]))
        E = {k: (y - W[k]).unstack("hour").reindex(columns=range(24)).values for k in ["base"] + SETS}
        mae_b = float(np.mean(np.abs(y - W["base"])))
        for inp in SETS:
            mae = float(np.mean(np.abs(y - W[inp])))
            dm, p_base = utils.dm_test_lago(E[inp], E["base"])      # small p: the base model (with forecasts) more accurate
            _, p_set = utils.dm_test_lago(E["base"], E[inp])         # small p: the model without the forecasts more accurate
            rows.append({"zone": zone, "input_set": inp, "n_hours": int(len(y)), "MAE_base": mae_b, "MAE_without": mae,
                         "rMAE_base": mae_b / ref, "rMAE_without": mae / ref, "delta_rMAE": (mae - mae_b) / ref,
                         "dm_base_better": dm, "p_base_better": p_base, "p_without_better": p_set})
    t = pd.DataFrame(rows)
    for inp in SETS:
        m = t.input_set == inp
        t.loc[m, "p_base_better_holm"] = holm(t.loc[m, "p_base_better"].values)
        t.loc[m, "p_without_better_holm"] = holm(t.loc[m, "p_without_better"].values)
    t.to_csv("results/no_forecasts_2025.csv", index=False)

    pd.set_option("display.width", 230)
    lines = ["Per-hour LightGBM without TSO forecasts against the base per-hour model, 2025",
             "(same frozen settings, level-and-scale target, monthly recalibration, expanding window, same hours).",
             "delta = rMAE without the forecasts - rMAE base (positive: the removed forecasts help). DM one-sided; Holm across",
             "the ten zones within each input set and direction.", ""]
    for inp in SETS:
        s = t[t.input_set == inp].drop(columns="input_set")
        no = s[s.zone.isin(utils.NO_ZONES)]
        lines += [f"{inp}:", s.round(4).to_string(index=False),
                  f"  mean over NO1-NO5: rMAE base {no.rMAE_base.mean():.4f}, without {no.rMAE_without.mean():.4f}, delta {no.delta_rMAE.mean():+.4f}",
                  f"  base significantly better (Holm < 0.05): {', '.join(s.zone[s.p_base_better_holm < 0.05]) or 'none'}",
                  f"  without the forecasts significantly better (Holm < 0.05): {', '.join(s.zone[s.p_without_better_holm < 0.05]) or 'none'}", ""]
    open("results/no_forecasts_2025_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
