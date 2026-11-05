"""
Test of H3: in the four coupled zones (NO1, NO2, SE3, SE4), 2025, is the
association of daily errors with the TTF gas price smaller with the neighbouring prices than without them?
  outcome     daily MAE without the neighbouring prices minus daily MAE with them (EUR/MWh), per-hour LightGBM from
              models/rolling.py (base and crossborder, monthly recalibration, expanding window)
  treatment   TTF close of D-2, standardised over 2025
  background  week of year (sine, cosine), weekday, change of the D-1 mean price from D-2, and the other eight
              market conditions of the error diagnostics (total version)
  estimation  partially linear double machine learning (analysis/dml_tools.py), LightGBM nuisance models cross-fitted
              over calendar months, Newey-West errors (7 lags); theta > 0: the gas-related error is larger without
  test        inverse-variance weighted mean of the four zone estimates, one-sided z-test, alpha 0.05

Outputs: results/h3_2025.csv (zone estimates and the pooled estimate), results/h3_2025.txt.
Run from the repository root:  python analysis/10_crossborder_gas.py
"""
import importlib.util

import numpy as np
import pandas as pd
from scipy import stats

spec = importlib.util.spec_from_file_location("dml_tools", "analysis/dml_tools.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)

ZONES = ["NO1", "NO2", "SE3", "SE4"]
BACKGROUND = ["week_sin", "week_cos"] + [f"wd{k}" for k in range(6)] + ["dprice"] + [c for c in dt.CONDITIONS if c != "gas"]


def main():
    rows = []
    for z in ZONES:
        d = dt.daily_conditions(z)
        d["outcome"] = dt.daily_mae(z, inputs="base") - dt.daily_mae(z, inputs="crossborder")
        g = d[d.index.year == 2025].dropna(subset=["outcome", "gas"] + BACKGROUND)
        assert len(g) >= 360, f"{z}: only {len(g)} complete days"
        res = dt.dml(g, "outcome", "gas", BACKGROUND, g.month.values)
        rows.append({"zone": z, "n_days": len(g), "theta": res["theta"], "se": res["se"], "p_two_sided": res["p"],
                     "p_one_sided": float(1 - stats.norm.cdf(res["theta"] / res["se"])), "variance_left": res["variance_left"]})
    t = pd.DataFrame(rows)
    w = 1 / t.se ** 2
    theta, se = float((w * t.theta).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
    z = theta / se
    p = float(1 - stats.norm.cdf(z))
    t = pd.concat([t, pd.DataFrame([{"zone": "pooled (inverse-variance)", "n_days": int(t.n_days.sum()), "theta": theta, "se": se,
                                     "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(z)))), "p_one_sided": p}])], ignore_index=True)
    t.to_csv("results/h3_2025.csv", index=False)
    lines = ["H3: outcome = daily MAE without - with the cross-border inputs (EUR/MWh), treatment = TTF",
             "close of D-2 (per standard deviation), total version, cross-fitting over calendar months, Newey-West (7 lags).",
             "theta > 0: the gas-related error is larger without the inputs.", "",
             t.round(4).to_string(index=False), "",
             f"Pooled theta = {theta:.4f} EUR/MWh per SD (s.e. {se:.4f}), z = {z:.3f}, one-sided p = {p:.4f}: "
             + ("SUPPORTED (p < 0.05)" if p < 0.05 else "NOT SUPPORTED (p >= 0.05)")]
    open("results/h3_2025.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
