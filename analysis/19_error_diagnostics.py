"""
Error diagnostics, 2025: association of the daily errors of the main model with nine market conditions.

Main model: base per-hour LightGBM (models/rolling.py). Outcome: daily MAE divided by the zone's 2025 MAE of the D-7
naive forecast on the same hours. Nine conditions, each as treatment in turn with the others in the background
(total version; the direct version adds L and S of D-1). Partially linear double machine learning (analysis/dml_tools.py)
with 5 cross-fitting folds by ISO week mod 5 and a one-day buffer around every test day; theta_z with Newey-West errors
(7 lags). Pooled across the nine Norwegian and Swedish zones: outcome residuals on treatment residuals and treatment
residuals x centred 2024 coupling, no intercept, intervals and p-values from a bootstrap over days (all zones of a day
together, 2,000 resamples, fixed seed), Holm across the nine conditions for the interaction within each version, BH
q-values, and a leave-one-zone-out range. DE-LU is estimated per zone as a reference only.

Outputs: results/diagnostics_2025_zone.csv, results/diagnostics_2025_pooled.csv, results/diagnostics_2025_summary.txt.
Run from the repository root:  python analysis/19_error_diagnostics.py
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("dml_tools", "analysis/dml_tools.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import utils  # noqa: E402

NORDIC = ["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"]
REFERENCE = "DE_LU"
_ps_spec = importlib.util.spec_from_file_location("paper_style", "analysis/paper_style.py")
_ps = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_ps)
COUPLING = dict(sorted(_ps.COUPLING.items(), key=lambda kv: kv[1]))   # frozen 2024 coupling measure, ascending
C_MEAN = float(np.mean(list(COUPLING.values())))                                      # 0.441
BASE_BG = ["week_sin", "week_cos"] + [f"wd{k}" for k in range(6)] + ["holiday", "dprice"]
VERSIONS = {"total": [], "direct": ["level", "spread"]}
B, SEED = 2000, 20250928

HOLIDAYS = {   # 2025 (official lists with sources; plus the eves named there for NO and SE)
    "NO": ["2025-01-01", "2025-04-17", "2025-04-18", "2025-04-20", "2025-04-21", "2025-05-01", "2025-05-17", "2025-05-29",
           "2025-06-08", "2025-06-09", "2025-12-25", "2025-12-26", "2025-12-24", "2025-12-31"],
    "SE": ["2025-01-01", "2025-01-06", "2025-04-18", "2025-04-20", "2025-04-21", "2025-05-01", "2025-05-29", "2025-06-06",
           "2025-06-08", "2025-06-21", "2025-11-01", "2025-12-25", "2025-12-26", "2025-06-20", "2025-12-24", "2025-12-31"],
    "DE": ["2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29", "2025-06-09", "2025-10-03", "2025-12-25",
           "2025-12-26"]}


def holm(p):
    p = np.asarray(p)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def bh(p):
    p = np.asarray(p)
    m, order, q, prev = len(p), np.argsort(p), np.empty(len(p)), 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def zone_frame(zone):
    """2025 days with the scaled daily MAE of the base per-hour LightGBM, the conditions and the background."""
    p = pd.read_parquet(f"{dt.rolling.OUT}/{zone}_lgbm_hourly_base_expanding.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p = p[p.scored.astype(bool)].set_index(["date", "hour"])
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    naive = s.reindex(p.index)
    ok = naive.notna().values
    scale = float(np.mean(np.abs(p.y_true.values[ok] - naive.values[ok])))              # 2025 MAE of naive D-7
    mae = (p.y_true - p.y_pred)[ok].abs().groupby(level="date").mean()
    d = dt.daily_conditions(zone)
    d["y"] = mae / scale
    d["holiday"] = d.index.isin(pd.to_datetime(HOLIDAYS[zone[:2] if zone != "DE_LU" else "DE"])).astype(float)
    need = ["y", "level", "spread", "dprice"] + [c for c in dt.CONDITIONS if not (zone == "DE_LU" and c == "res")]
    g = d[d.index.year == 2025].dropna(subset=need)
    return g, scale


def main():
    zone_rows, resid = [], {}
    for zone in NORDIC + [REFERENCE]:
        g, scale = zone_frame(zone)
        assert len(g) >= 360, f"{zone}: {len(g)} complete days"
        usable = [c for c in dt.CONDITIONS if c in g.columns and g[c].notna().all() and g[c].std() > 1e-9]
        folds = (g.iso_week % 5).values
        for version, extra in VERSIONS.items():
            for c in usable:
                bg = BASE_BG + extra + [o for o in usable if o != c]
                r = dt.dml(g, "y", c, bg, folds, buffer_days=1)
                zone_rows.append({"zone": zone, "reference": zone == REFERENCE, "coupling_2024": COUPLING.get(zone, np.nan),
                                  "version": version, "condition": c, "theta": r["theta"], "se": r["se"], "p": r["p"],
                                  "variance_left": r["variance_left"], "n_days": len(g), "naive_d7_mae_2025": scale,
                                  "holidays_in_sample": int(g.holiday.sum())})
                if zone in NORDIC:
                    resid[(version, c, zone)] = pd.DataFrame({"e": r["e"], "r": r["r"]}, index=g.index)
        print(f"  {zone}: {len(g)} days, conditions {usable}", flush=True)
    zt = pd.DataFrame(zone_rows)
    zt.to_csv("results/diagnostics_2025_zone.csv", index=False)

    rng = np.random.default_rng(SEED)
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    draws = rng.integers(0, len(days), size=(B, len(days)))
    counts = np.zeros((B, len(days)))
    for b in range(B):
        counts[b] = np.bincount(draws[b], minlength=len(days))
    pooled = []
    for version in VERSIONS:
        for c in dt.CONDITIONS:
            zones = [z for z in NORDIC if (version, c, z) in resid]
            per_zone = {}
            for z in zones:
                f = resid[(version, c, z)]
                x1, x2 = f.r.values, f.r.values * (COUPLING[z] - C_MEAN)
                S = pd.DataFrame({"a11": x1 * x1, "a12": x1 * x2, "a22": x2 * x2, "b1": x1 * f.e.values, "b2": x2 * f.e.values}, index=f.index)
                per_zone[z] = S.reindex(days).fillna(0.0)
            total = sum(per_zone.values())

            def solve(M):
                A = np.array([[M["a11"], M["a12"]], [M["a12"], M["a22"]]])
                return np.linalg.solve(A, np.array([M["b1"], M["b2"]]))

            est = solve(total.sum())
            V = counts @ total[["a11", "a12", "a22", "b1", "b2"]].values                 # B x 5 weighted sums
            det = V[:, 0] * V[:, 2] - V[:, 1] ** 2
            boot = np.column_stack([(V[:, 2] * V[:, 3] - V[:, 1] * V[:, 4]) / det, (V[:, 0] * V[:, 4] - V[:, 1] * V[:, 3]) / det])
            lozo = np.array([solve((total - per_zone[z]).sum())[1] for z in zones])
            for k, term in enumerate(["main", "interaction"]):
                pooled.append({"version": version, "condition": c, "term": term, "estimate": float(est[k]), "n_zones": len(zones),
                               "ci_low": float(np.percentile(boot[:, k], 2.5)), "ci_high": float(np.percentile(boot[:, k], 97.5)),
                               "p_boot": float(min(1.0, 2 * min(np.mean(boot[:, k] <= 0), np.mean(boot[:, k] >= 0)))),
                               "lozo_min": float(lozo.min()) if term == "interaction" else np.nan,
                               "lozo_max": float(lozo.max()) if term == "interaction" else np.nan})
    pt = pd.DataFrame(pooled)
    pt["p_holm"], pt["q_bh"] = np.nan, np.nan
    for version in VERSIONS:
        m = (pt.version == version) & (pt.term == "interaction")
        pt.loc[m, "p_holm"] = holm(pt.loc[m, "p_boot"].values)
        pt.loc[m, "q_bh"] = bh(pt.loc[m, "p_boot"].values)
    pt.to_csv("results/diagnostics_2025_pooled.csv", index=False)

    pd.set_option("display.width", 230)
    lines = ["Error diagnostics 2025. Outcome: daily MAE of the base per-hour LightGBM",
             "divided by the zone's 2025 MAE of naive D-7; theta per standard deviation of the condition.", ""]
    for version in VERSIONS:
        z = zt[zt.version == version].pivot(index="condition", columns="zone", values="theta").reindex(dt.CONDITIONS)[NORDIC + [REFERENCE]]
        lines += [f"theta_z ({version}); DE_LU is a reference, not pooled", z.round(4).to_string(), ""]
        lines += [f"Pooled ({version}): main = association at average coupling (0.441); interaction = change per unit of coupling",
                  pt[pt.version == version].drop(columns="version").round(4).to_string(index=False), ""]
    open("results/diagnostics_2025_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
