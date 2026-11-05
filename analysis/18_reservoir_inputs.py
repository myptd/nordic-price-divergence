"""
Reservoir inputs (ENTSO-E stored energy and its one- and four-week changes) added to per-hour LightGBM, 2025, nine
Norwegian and Swedish zones, against the base per-hour LightGBM on the same hours.

Per zone: delta rMAE = rMAE without - rMAE with (positive: the reservoir inputs help), relative to naive D-7; one-sided
multivariate DM tests (Lago et al., 2021) in both directions, Holm across the nine zones within each direction; 95%
day-bootstrap interval of delta (days resampled with replacement, each day's 24 hours together, 2,000 resamples, fixed
seed, percentile interval). Spearman rank correlation of delta with the 2024 hydro share (results/market_types_2024.csv)
and with the 2024 coupling measure.

Reads model_results/rolling/{zone}_lgbm_hourly_{base,reservoir}_expanding.parquet.
Outputs: results/reservoir_2025.csv, results/reservoir_2025_summary.txt.
Run from the repository root:  python analysis/18_reservoir_inputs.py
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = load("ps", "analysis/paper_style.py")
nf = load("nf", "analysis/14_tso_forecasts.py")                     # holm()
B, SEED = 2000, 20250929
ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"]


def zone_frame(zone):
    p = pd.concat([pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_{i}_expanding.parquet") for i in ["base", "reservoir"]], ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="inputs", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    W["naive_d7"] = s.reindex(W.index).values
    keep = W[["base", "reservoir", "naive_d7"]].notna().all(axis=1) & scored
    return W[keep], y[keep]


def main():
    rng = np.random.default_rng(SEED)
    mix = pd.read_csv("results/market_types_2024.csv").set_index("zone")
    rows = []
    for z in ZONES:
        W, y = zone_frame(z)
        ref = np.mean(np.abs(y - W["naive_d7"]))
        mb, mr = float(np.mean(np.abs(y - W["base"]))), float(np.mean(np.abs(y - W["reservoir"])))
        E = {k: (y - W[k]).unstack("hour").reindex(columns=range(24)).values for k in ["base", "reservoir"]}
        dm_r, p_r = utils.dm_test_lago(E["base"], E["reservoir"])           # small p: with reservoir inputs more accurate
        dm_b, p_b = utils.dm_test_lago(E["reservoir"], E["base"])           # small p: without them more accurate
        v = pd.DataFrame({k: (y - W[k]).abs() for k in ["base", "reservoir", "naive_d7"]}).groupby(level="date").sum()[["base", "reservoir", "naive_d7"]].values
        s = v[rng.integers(0, len(v), size=(B, len(v)))].sum(axis=1)
        reps = (s[:, 0] - s[:, 1]) / s[:, 2]
        rows.append({"zone": z, "hydro_share_2024": mix.loc[z, "hydro"], "coupling_2024": ps.COUPLING[z], "n_hours": int(len(y)),
                     "MAE_without": mb, "MAE_with": mr, "rMAE_without": mb / ref, "rMAE_with": mr / ref, "delta_rMAE": (mb - mr) / ref,
                     "pct_MAE": 100 * (mb - mr) / mb, "ci_low": np.percentile(reps, 2.5), "ci_high": np.percentile(reps, 97.5),
                     "dm_with_better": dm_r, "p_with_better": p_r, "p_without_better": p_b})
    t = pd.DataFrame(rows)
    t["p_with_better_holm"] = nf.holm(t.p_with_better.values)
    t["p_without_better_holm"] = nf.holm(t.p_without_better.values)
    t.to_csv("results/reservoir_2025.csv", index=False)
    r_h = spearmanr(t.hydro_share_2024, t.delta_rMAE)
    r_c = spearmanr(t.coupling_2024, t.delta_rMAE)
    pd.set_option("display.width", 250)
    lines = ["Reservoir inputs (ENTSO-E stored energy, 1- and 4-week change) in per-hour LightGBM, 2025.",
             "delta = rMAE without - rMAE with (positive: the inputs help); DM one-sided, Holm across the nine zones per direction.", "",
             t.round(4).to_string(index=False), "",
             f"Mean delta over NO1-NO5: {t[t.zone.str.startswith('NO')].delta_rMAE.mean():.4f}; over SE1-SE4: {t[t.zone.str.startswith('SE')].delta_rMAE.mean():.4f}",
             "Significant gain (Holm < 0.05): " + (", ".join(t.zone[t.p_with_better_holm < 0.05]) or "none"),
             "Significant loss (Holm < 0.05): " + (", ".join(t.zone[t.p_without_better_holm < 0.05]) or "none"),
             f"Descriptive: Spearman rho(delta, hydro share 2024) = {r_h.statistic:.3f} (p = {r_h.pvalue:.3f}); "
             f"rho(delta, coupling 2024) = {r_c.statistic:.3f} (p = {r_c.pvalue:.3f})"]
    open("results/reservoir_2025_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
