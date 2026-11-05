"""
Daily gain from the TSO wind and solar forecasts against the reservoir level, and the 2025 gain in EUR/MWh (Fig. S9).

Hypothesis: in the Norwegian zones the daily gain is larger when reservoir filling is
below its seasonal norm (slope < 0). Uses the 2025 per-hour LightGBM forecasts base and no_vre (analysis/14).
- Per zone-day: gain = MAE(no_vre) - MAE(base) in EUR/MWh over the day's scored hours (positive: the forecasts help).
- Reservoir: res_stored_twh at 11:00 on D-1 (features_day_ahead.day_level_features), as deviation from the 2018-2024
  median for the ISO week of D (TWh).
- Per zone: OLS gain ~ deviation + month-of-D fixed effects, Newey-West SE with 7 lags; one-sided p for slope < 0;
  Holm across NO1-NO4 (NO5 has no wind and solar forecast). SE1-SE4 as contrast (not in the Holm family).
- Pooled NO1-NO4: one slope, zone-by-month fixed effects, Driscoll-Kraay SE (Newey-West, 7 lags, daily sums).
- 2025 gain in EUR/MWh (MAE without - MAE base) for all ten zones, no_vre and prices_calendar, with the 95%
  day-bootstrap interval (2,000 resamples, fixed seed).

Reads model_results/rolling/{zone}_lgbm_hourly_{base,no_vre,prices_calendar}_expanding.parquet and data/cleaned/.
Outputs: results/tso_gain_reservoir.csv, results/tso_gain_eur_2025.csv, results/tso_gain_reservoir_summary.txt,
paper_outputs/tables/tab_s_tso_gain_reservoir.tex, paper_outputs/tables/tab_s_tso_gain_eur.tex,
paper_outputs/figures/figS9_tso_gain_reservoir.pdf/.jpg.
Run from the repository root:  python analysis/16_tso_gain_reservoir.py
"""
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
NW_LAGS, B, SEED = 7, 2000, 20251101
NO, SE = ["NO1", "NO2", "NO3", "NO4", "NO5"], ["SE1", "SE2", "SE3", "SE4"]
NORM_YEARS = (2018, 2024)
SETS = ["no_vre", "prices_calendar"]


def zone_hours(zone):
    """Hourly forecasts (base and the sets without TSO forecasts) and actuals on the scored hours of 2025."""
    p = pd.concat([pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_{i}_expanding.parquet") for i in ["base"] + SETS],
                  ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="inputs", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    keep = W[["base"] + SETS].notna().all(axis=1) & scored
    return W[keep], y[keep]


def reservoir_deviation(zone):
    """Stored energy known at 11:00 on D-1, minus its 2018-2024 median for the ISO week of D (TWh), indexed by D."""
    df = fd.prepare_hourly(utils.load_zone(zone))
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(df.index.date))), name="date")
    r = fd.day_level_features(df, days)["res_stored_twh"].to_frame("stored")
    r["week"] = r.index.isocalendar().week.astype(int).values
    ref = r[(r.index.year >= NORM_YEARS[0]) & (r.index.year <= NORM_YEARS[1])]
    r["deviation"] = r["stored"] - r["week"].map(ref.groupby("week")["stored"].median())
    return r


def nve_check(zone, dev):
    """Plausibility check only (NVE is never a model input): correlation of the 2025 ENTSO-E deviation with NVE's
    fill anomaly against its 20-year weekly median, both at 11:00 on D-1."""
    h = utils.load_zone(zone)
    a = fd.at_hour(h, ["nve_fill_anomaly_vs_median"], fd.ORIGIN_HOUR)["nve_fill_anomaly_vs_median"]
    a.index = a.index + pd.Timedelta(days=1)
    j = pd.concat([dev, a], axis=1, join="inner").dropna()
    j = j[j.index.year == 2025]
    return float(j.corr().iloc[0, 1]), 100 * float(j.iloc[:, 1].mean())     # NVE stores the anomaly as a fraction


def zone_days(zone):
    W, y = zone_hours(zone)
    a = pd.DataFrame({k: (y - W[k]).abs() for k in ["base", "no_vre"]}).groupby(level="date").mean()
    d = pd.DataFrame({"gain": a["no_vre"] - a["base"]})
    r = reservoir_deviation(zone)
    d = d.join(r[["stored", "deviation"]])
    d["month"] = d.index.month
    d["zone"] = zone
    return d.dropna()


def one_sided(b, p_two):
    """p for H1: slope < 0, from the two-sided (normal, HAC) p."""
    return float(p_two / 2 if b < 0 else 1 - p_two / 2)


def fit_zone(d):
    X = pd.get_dummies(d["month"], prefix="m", drop_first=True, dtype=float)
    X.insert(0, "deviation", d["deviation"])
    X = sm.add_constant(X)
    return sm.OLS(d["gain"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


def fit_pooled(d):
    fe = pd.get_dummies(d["zone"] + "_" + d["month"].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["deviation"]], fe], axis=1))
    t = pd.factorize(d.index)[0]                                        # day index for the Driscoll-Kraay sums
    return sm.OLS(d["gain"].values, X.values).fit(cov_type="hac-groupsum", cov_kwds={"time": t, "maxlags": NW_LAGS}), X


def eur_gains(rng):
    rows = []
    for z in utils.ALL_ZONES:
        W, y = zone_hours(z)
        v = pd.DataFrame({k: (y - W[k]).abs() for k in ["base"] + SETS}).groupby(level="date").agg(["sum", "count"])
        for s in SETS:
            mb, ms = float(np.mean(np.abs(y - W["base"]))), float(np.mean(np.abs(y - W[s])))
            idx = rng.integers(0, len(v), size=(B, len(v)))
            n = v[("base", "count")].values[idx].sum(axis=1)
            reps = (v[(s, "sum")].values[idx].sum(axis=1) - v[("base", "sum")].values[idx].sum(axis=1)) / n
            rows.append({"zone": z, "input_set": s, "n_hours": int(len(y)), "MAE_base": mb, "MAE_without": ms,
                         "gain_eur": ms - mb, "ci_low": np.percentile(reps, 2.5), "ci_high": np.percentile(reps, 97.5),
                         "gain_pct_MAE_base": 100 * (ms - mb) / mb})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    days, rows = {}, []
    for z in NO + SE:
        d = zone_days(z)
        days[z] = d
        row = {"zone": z, "group": "main" if z in NO else "contrast", "n_days": len(d), "mean_gain_eur": d.gain.mean(),
               "sd_deviation_twh": d.deviation.std(), "mean_deviation_twh": d.deviation.mean()}
        if d.gain.abs().max() == 0:
            row["note"] = "gain identically 0 (no wind and solar forecast); undefined"
        else:
            f = fit_zone(d)
            b, se = f.params["deviation"], f.bse["deviation"]
            row.update({"slope_eur_per_twh": b, "se": se, "t": b / se, "p_one_sided": one_sided(b, f.pvalues["deviation"]),
                        "p_two_sided": float(f.pvalues["deviation"]), "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                        "slope_per_sd": b * d.deviation.std(), "se_per_sd": se * d.deviation.std(), "note": ""})
        rows.append(row)
    t = pd.DataFrame(rows)
    fam = (t.group == "main") & t.slope_eur_per_twh.notna()
    t.loc[fam, "p_one_sided_holm"] = nf.holm(t.loc[fam, "p_one_sided"].values)

    P = pd.concat([days[z] for z in ["NO1", "NO2", "NO3", "NO4"]]).sort_index(kind="stable")
    fp, X = fit_pooled(P)
    b, se = fp.params[1], fp.bse[1]
    sd = P.groupby("zone").deviation.std().mean()
    pooled = {"zone": "NO1-NO4 pooled", "group": "main", "n_days": len(P), "mean_gain_eur": P.gain.mean(),
              "slope_eur_per_twh": b, "se": se, "t": b / se, "p_one_sided": one_sided(b, fp.pvalues[1]),
              "p_two_sided": float(fp.pvalues[1]), "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
              "slope_per_sd": b * sd, "se_per_sd": se * sd, "sd_deviation_twh": sd,
              "note": "zone-by-month fixed effects; Driscoll-Kraay SE; per SD = mean of the zone SDs"}
    t = pd.concat([t, pd.DataFrame([pooled])], ignore_index=True)
    t.to_csv("results/tso_gain_reservoir.csv", index=False)

    nve = {z: nve_check(z, days[z]["deviation"]) for z in NO}
    g = eur_gains(rng)
    g.to_csv("results/tso_gain_eur_2025.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["zone", "n_days", "mean_gain_eur", "mean_deviation_twh", "sd_deviation_twh", "slope_eur_per_twh", "se",
            "p_one_sided", "p_one_sided_holm", "p_two_sided", "slope_per_sd", "se_per_sd"]
    sig = t[(t.p_one_sided_holm < 0.05)].zone.tolist()
    lines = ["Hypothesis: in NO zones the daily gain from the TSO wind and solar forecasts is larger when reservoir filling is below",
             "its seasonal norm (slope < 0). gain = MAE(no_vre) - MAE(base) per day, EUR/MWh, 2025 per-hour LightGBM forecasts.",
             "Deviation = stored energy known at 11:00 on D-1 minus its 2018-2024 median for the ISO week of D (TWh).",
             f"Per zone: OLS with month fixed effects, Newey-West SE ({NW_LAGS} lags); one-sided p for slope < 0; Holm across NO1-NO4",
             "(NO5 undefined: gain identically 0). SE1-SE4 are the contrast, not in the Holm family.",
             f"Pooled NO1-NO4: zone-by-month fixed effects, Driscoll-Kraay SE ({NW_LAGS} lags).", "",
             t[cols].round(4).to_string(index=False), "",
             "NO zones with slope < 0 at Holm 0.05: " + (", ".join(z for z in sig if z.startswith("NO")) or "none"),
             f"Pooled NO1-NO4: slope {b:.3f} EUR/MWh per TWh (s.e. {se:.3f}), one-sided p {pooled['p_one_sided']:.3f}; "
             f"per SD {pooled['slope_per_sd']:.3f} EUR/MWh", "",
             "Plausibility check, 2025 (NVE not a model input): correlation of the ENTSO-E deviation with the NVE fill anomaly",
             "(percentage points vs NVE's 20-year weekly median) and the mean NVE anomaly:",
             *[f"  {z}: r = {nve[z][0]:.3f}, mean NVE anomaly {nve[z][1]:+.1f} pp" for z in NO], "",
             "TSO-forecast gain in EUR/MWh, 2025 (MAE without - MAE base; 95% day-bootstrap interval):",
             g.round(4).to_string(index=False)]
    open("results/tso_gain_reservoir_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Supplement tables (Section S10): the reservoir test, and the gain in EUR/MWh for all ten zones
    def f3(x):
        return "--" if pd.isna(x) else f"{x:.3f}"
    body = []
    for _, r in t.iterrows():
        z = r.zone if r.zone.endswith("pooled") else ps.label(r.zone)
        body.append(f"{z} & {r.n_days:.0f} & {r.mean_gain_eur:.2f} & {r.sd_deviation_twh:.2f} & {f3(r.slope_eur_per_twh)} & "
                    f"{f3(r.se)} & {f3(r.slope_per_sd)} & {f3(r.p_one_sided)} & {f3(r.get('p_one_sided_holm'))} \\\\")
    body.insert(5, "\\midrule")
    body.insert(10, "\\midrule")
    ps.save_table("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
                  " & & Mean daily & SD of & \\multicolumn{2}{c}{Slope (EUR/MWh per TWh)} & Slope & \\multicolumn{2}{c}{One-sided $p$} \\\\\n"
                  "\\cmidrule(lr){5-6} \\cmidrule(lr){8-9}\n"
                  "Zone & Days & gain (EUR/MWh) & deviation (TWh) & estimate & s.e. & per SD & raw & Holm \\\\\n\\midrule\n"
                  + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n", "tab_s_tso_gain_reservoir")

    G = g.pivot(index="zone", columns="input_set").loc[ps.ZONES]
    body = []
    for z, r in G.iterrows():
        cells = [f"{r[('MAE_base', 'no_vre')]:.2f}"]
        for s in SETS:
            cells += [f"{r[('gain_eur', s)]:.2f} ({r[('ci_low', s)]:.2f}, {r[('ci_high', s)]:.2f})", f"{r[('gain_pct_MAE_base', s)]:.1f}"]
        body.append(f"{ps.label(z)} & " + " & ".join(cells) + " \\\\")
    ps.save_table("\\begin{tabular}{lrrrrr}\n\\toprule\n"
                  " & MAE with & \\multicolumn{2}{c}{Without wind and solar forecasts} & \\multicolumn{2}{c}{Without load, wind and solar forecasts} \\\\\n"
                  "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
                  "Zone & forecasts & Gain (95\\% interval) & \\% of MAE & Gain (95\\% interval) & \\% of MAE \\\\\n\\midrule\n"
                  + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n", "tab_s_tso_gain_eur")

    # Figure: (a) slope per SD with 95% interval; (b) binned partial relation in the pooled NO model
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    order = [z for z in ps.NORDIC] + ["NO1-NO4 pooled"]
    for i, z in enumerate(order):
        r = t.set_index("zone").loc[z]
        c = ps.INK if z.endswith("pooled") else ps.color(z)
        if pd.isna(r.slope_per_sd):
            ax.text(0, i, "undefined", ha="center", va="center", fontsize=7.5, color=ps.MUTED)
            continue
        ax.errorbar(r.slope_per_sd, i, xerr=1.96 * r.se_per_sd, fmt="o" if not z.endswith("pooled") else "D", color=c,
                    markersize=5.5, capsize=2.5, linewidth=1.3)
    ax.axvline(0, color=ps.AXIS, linewidth=1)
    ax.set_yticks(range(len(order)), [ps.label(z) if not z.endswith("pooled") else "NO1-NO4\npooled" for z in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in daily gain per SD of reservoir\ndeviation (EUR/MWh, 95% interval)", fontsize=8.6, color=ps.INK)
    ps.clean_axes(ax, grid="x")
    ax = axes[1]
    Xd = X.drop(columns="deviation").values
    res = lambda v: v - Xd @ np.linalg.lstsq(Xd, v, rcond=None)[0]      # noqa: E731  (partial out the fixed effects)
    ex, ey = res(P["deviation"].values), res(P["gain"].values)
    bins = pd.qcut(ex, 10, labels=False)
    bx, by = pd.Series(ex).groupby(bins).mean(), pd.Series(ey).groupby(bins).mean()
    ax.scatter(ex, ey, s=4, color=ps.GRID, zorder=1)
    ax.plot(bx, by, "o", color=ps.INK, markersize=5, zorder=3)
    xs = np.linspace(ex.min(), ex.max(), 2)
    ax.plot(xs, b * xs, color=ps.COLORS["coupled"], linewidth=1.5, zorder=2)
    ax.set_ylim(np.percentile(ey, 1), np.percentile(ey, 99))
    ax.set_xlabel("Reservoir deviation from weekly norm, TWh\n(after zone-by-month fixed effects)", fontsize=8.6, color=ps.INK)
    ax.set_ylabel("Daily gain, EUR/MWh (same adjustment)", fontsize=8.6, color=ps.INK)
    ps.clean_axes(ax, grid="both")
    for k, a in enumerate(axes):
        a.text(0.0, 1.03, f"({'ab'[k]})", transform=a.transAxes, fontsize=9.5, fontweight="bold", color=ps.INK)
    fig.tight_layout()
    ps.save(fig, "figS9_tso_gain_reservoir")


if __name__ == "__main__":
    main()
