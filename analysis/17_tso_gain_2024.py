"""
Gain from the TSO wind and solar forecasts (per-hour LightGBM, base against no_vre) in 2024 and 2025, and per zone the
2024 generation mix next to the 2025 gain.

1. For 2024 (the screening year, in which the model settings were chosen) and 2025, ten zones:
   gain = MAE(no_vre) - MAE(base) in EUR/MWh and delta rMAE = gain / MAE(naive D-7), on the hours scored by both
   forecasts, with 95% day-bootstrap intervals (days resampled with replacement, all hours of a day together, 2,000
   resamples, fixed seed, percentile interval).
2. Per zone: 2024 wind, solar and hydro shares of generation (results/market_types_2024.csv, analysis/05) and the
   highest hourly wind generation in 2024 (onshore + offshore, ENTSO-E actual generation averaged to hours; a lower
   bound for installed wind capacity), next to the 2025 gain.

Reads model_results/rolling/{zone}_lgbm_hourly_{base,no_vre}_expanding[_2024].parquet, data/cleaned/,
data/entsoe/generation/ and results/market_types_2024.csv.
Outputs: results/tso_gain_2024.csv, results/tso_gain_mix.csv, results/tso_gain_2024_summary.txt,
paper_outputs/tables/tab_s_tso_gain_2024.tex, paper_outputs/tables/tab_s_tso_gain_mix.tex.
Run from the repository root:  python analysis/17_tso_gain_2024.py
"""
import glob
import importlib.util
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
B, SEED = 2000, 20241101
SUFFIX = {2024: "_2024", 2025: ""}


def zone_frame(zone, year):
    p = pd.concat([pd.read_parquet(f"{rolling.OUT}/{zone}_lgbm_hourly_{i}_expanding{SUFFIX[year]}.parquet") for i in ["base", "no_vre"]],
                  ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    W = p.pivot_table(index=["date", "hour"], columns="inputs", values="y_pred")
    y = p.groupby(["date", "hour"])["y_true"].mean()
    scored = p.groupby(["date", "hour"])["scored"].min().astype(bool)
    P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = P.shift(7).stack(future_stack=True)
    s.index.names = ["date", "hour"]
    W["naive_d7"] = s.reindex(W.index).values
    keep = W[["base", "no_vre", "naive_d7"]].notna().all(axis=1) & scored
    W, y = W[keep], y[keep]
    assert (W.index.get_level_values("date").year == year).all()
    return W, y


def gains(zone, year, rng):
    W, y = zone_frame(zone, year)
    a = pd.DataFrame({k: (y - W[k]).abs() for k in ["base", "no_vre", "naive_d7"]})
    v = a.groupby(level="date").agg(["sum", "count"])
    sums = {k: v[(k, "sum")].values for k in ["base", "no_vre", "naive_d7"]}
    n = v[("base", "count")].values
    idx = rng.integers(0, len(n), size=(B, len(n)))
    S = {k: sums[k][idx].sum(axis=1) for k in sums}
    eur = (S["no_vre"] - S["base"]) / n[idx].sum(axis=1)
    rel = (S["no_vre"] - S["base"]) / S["naive_d7"]
    mb, mw, mn = a["base"].mean(), a["no_vre"].mean(), a["naive_d7"].mean()
    return {"zone": zone, "year": year, "n_hours": int(len(y)), "MAE_base": mb, "MAE_no_vre": mw, "MAE_naive_d7": mn,
            "gain_eur": mw - mb, "gain_eur_low": np.percentile(eur, 2.5), "gain_eur_high": np.percentile(eur, 97.5),
            "delta_rMAE": (mw - mb) / mn, "delta_rMAE_low": np.percentile(rel, 2.5), "delta_rMAE_high": np.percentile(rel, 97.5),
            "gain_pct_MAE_base": 100 * (mw - mb) / mb}


def peak_wind_mw(zone):
    area = zone if zone == "DE_LU" else f"{zone[:2]}_{zone[2]}"
    (path,) = glob.glob(f"data/entsoe/generation/{area}_[0-9]*.csv")
    g = pd.read_csv(path, index_col=0)
    g.index = pd.to_datetime(g.index, utc=True)
    cols = [c for c in ["Wind Onshore", "Wind Offshore"] if c in g.columns]
    w = g[cols].sum(axis=1, min_count=1)
    w = w[w.index.year == 2024]
    return float(w.resample("1h").mean().max()) if cols else 0.0


def main():
    rng = np.random.default_rng(SEED)
    t = pd.DataFrame([gains(z, yr, rng) for yr in [2024, 2025] for z in ps.ZONES])
    # Use the 2025 EUR/MWh intervals of analysis/16 (Table S6) so that the supplementary tables report one interval.
    e = pd.read_csv("results/tso_gain_eur_2025.csv").query("input_set == 'no_vre'").set_index("zone")
    i25 = t.year == 2025
    t.loc[i25, "gain_eur_low"] = t.loc[i25, "zone"].map(e.ci_low).values
    t.loc[i25, "gain_eur_high"] = t.loc[i25, "zone"].map(e.ci_high).values
    t.to_csv("results/tso_gain_2024.csv", index=False)

    mix = pd.read_csv("results/market_types_2024.csv").set_index("zone")
    g25 = t[t.year == 2025].set_index("zone")
    m = pd.DataFrame({"wind_share_2024": mix.wind, "solar_share_2024": mix.solar, "hydro_share_2024": mix.hydro,
                      "peak_wind_mw_2024": {z: peak_wind_mw(z) for z in ps.ZONES},
                      "gain_eur_2025": g25.gain_eur, "gain_eur_2025_low": g25.gain_eur_low, "gain_eur_2025_high": g25.gain_eur_high,
                      "delta_rMAE_2025": g25.delta_rMAE}).loc[ps.ZONES]
    m.index.name = "zone"
    m.to_csv("results/tso_gain_mix.csv")

    pd.set_option("display.width", 250)
    wide = t.pivot(index="zone", columns="year", values=["gain_eur", "delta_rMAE"]).loc[ps.ZONES]
    lines = ["Gain from the TSO wind and solar forecasts, per-hour LightGBM (base against no_vre), frozen settings, monthly",
             "recalibration, expanding window. 2024 is a robustness year: it is the screening year in which the settings were chosen.",
             "gain = MAE without - MAE with (EUR/MWh); delta rMAE = gain / MAE of naive D-7; 95% day-bootstrap intervals "
             f"({B} resamples).", "", t.round(4).to_string(index=False), "",
             "Side by side (gain EUR/MWh, delta rMAE):", wide.round(4).to_string(), "",
             f"Spearman rank correlation of the zone gains 2024 vs 2025: EUR/MWh {wide['gain_eur'][2024].corr(wide['gain_eur'][2025], method='spearman'):.3f}, "
             f"rMAE {wide['delta_rMAE'][2024].corr(wide['delta_rMAE'][2025], method='spearman'):.3f} (ten zones)", "",
             "2024 generation mix and the highest hourly wind generation in 2024 (a lower bound for installed wind capacity)",
             "next to the 2025 gain:", m.round(3).to_string()]
    open("results/tso_gain_2024_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    body = []
    for z in ps.ZONES:
        cells = []
        for yr in [2024, 2025]:
            r = t[(t.zone == z) & (t.year == yr)].iloc[0]
            cells += [f"{r.delta_rMAE:.3f} ({r.delta_rMAE_low:.3f}, {r.delta_rMAE_high:.3f})", f"{r.gain_eur:.2f} ({r.gain_eur_low:.2f}, {r.gain_eur_high:.2f})"]
        body.append(f"{ps.label(z)} & " + " & ".join(cells) + " \\\\")
    ps.save_table("\\begin{tabular}{lrrrr}\n\\toprule\n"
                  " & \\multicolumn{2}{c}{2024 (robustness year)} & \\multicolumn{2}{c}{2025 (test year)} \\\\\n"
                  "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
                  "Zone & $\\Delta$rMAE (95\\% interval) & EUR/MWh (95\\% interval) & $\\Delta$rMAE (95\\% interval) & EUR/MWh (95\\% interval) \\\\\n\\midrule\n"
                  + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n", "tab_s_tso_gain_2024")

    body = [f"{ps.label(z)} & {r.wind_share_2024:.1f} & {r.solar_share_2024:.1f} & {r.hydro_share_2024:.1f} & {r.peak_wind_mw_2024:,.0f} & "
            f"{r.gain_eur_2025:.2f} ({r.gain_eur_2025_low:.2f}, {r.gain_eur_2025_high:.2f}) & {r.delta_rMAE_2025:.3f} \\\\"
            for z, r in m.iterrows()]
    ps.save_table("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                  " & \\multicolumn{3}{c}{Generation share 2024 (\\%)} & Peak hourly wind & \\multicolumn{2}{c}{Gain 2025} \\\\\n"
                  "\\cmidrule(lr){2-4} \\cmidrule(lr){6-7}\n"
                  "Zone & Wind & Solar & Hydro & generation 2024 (MW) & EUR/MWh (95\\% interval) & $\\Delta$rMAE \\\\\n\\midrule\n"
                  + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n", "tab_s_tso_gain_mix")


if __name__ == "__main__":
    main()
