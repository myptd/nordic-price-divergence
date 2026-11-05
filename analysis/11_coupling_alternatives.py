"""
Alternative coupling measures and the precision of the neighbouring-price result.

1. Alternative coupling measures, 2024, nine Norwegian and Swedish zones (hourly prices from data/cleaned):
   (a) price convergence with DE-LU: share of hours with |p_zone - p_DE-LU| < 1 EUR/MWh, and < 0.01 EUR/MWh as a check;
   (b) correlation with DE-LU of hour-to-hour price changes (first differences of the hourly prices).
   Spearman rank correlation of each with the frozen intraday-profile measure (paper_style.COUPLING).
2. Precision of the neighbouring-price result, 2025:
   (a) largest upper 95% bound of the gain (delta rMAE, without minus with) across the nine zones, from
       results/crossborder_2025_bootstrap.csv (analysis/09), and the upper bound as a percentage of MAE without the
       inputs (delta rMAE x MAE of naive D-7 / MAE without, per bootstrap replicate);
   (b) 95% interval of mean gain (NO1, NO2, SE3, SE4) minus mean gain (NO3, NO4, SE1, SE2) from a day bootstrap that
       keeps all zones of a day together (2,000 resamples, fixed seed; percentile interval).

Outputs: results/coupling_alternatives_2024.csv, paper_outputs/tables/tab_s_coupling_alt.tex,
results/crossborder_precision.txt, results/crossborder_2025_bootstrap_pct.csv.
Run from the repository root:  python analysis/11_coupling_alternatives.py
"""
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = load("ps", "analysis/paper_style.py")
xb = load("xb", "analysis/08_crossborder.py")
B, SEED = 2000, 20250928
ABOVE, BELOW = ["NO1", "NO2", "SE3", "SE4"], ["NO3", "NO4", "SE1", "SE2"]


def price_2024(zone):
    p = pd.read_parquet(f"data/cleaned/{zone}_hourly.parquet", columns=["price_eur_mwh"]).price_eur_mwh
    return p[p.index.year == 2024]


def coupling_alternatives():
    de = price_2024("DE_LU")
    rows = []
    for z in ps.NORDIC:
        p = price_2024(z)
        d = (p - de).abs()
        conv = d < 1.0
        low = conv & (p < 5.0) & (de < 5.0)
        rows.append({"zone": z, "intraday_profile": ps.COUPLING[z], "share_within_1": float(conv.mean()),
                     "conv_share_both_below_5": float(low.sum() / conv.sum()), "conv_median_price": float(p[conv].median()),
                     "share_within_0.01": float((d < 0.01).mean()), "corr_changes": float(p.diff().corr(de.diff())), "n_hours": len(p)})
    t = pd.DataFrame(rows)
    t.to_csv("results/coupling_alternatives_2024.csv", index=False)
    rho = {c: spearmanr(t.intraday_profile, t[c]).statistic for c in ["share_within_1", "share_within_0.01", "corr_changes"]}
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & Intraday profile & \\multicolumn{2}{c}{Share of hours near DE-LU} & Hour-to-hour & \\multicolumn{2}{c}{Hours within 1 EUR/MWh} \\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){6-7}",
             "Zone & (tests) & $<1$ EUR/MWh & $<0.01$ EUR/MWh & changes & both $<5$ EUR/MWh & median price \\\\", "\\midrule"]
    lines += [f"{ps.label(r['zone'])} & {r['intraday_profile']:.3f} & {r['share_within_1']:.3f} & {r['share_within_0.01']:.3f} & "
              f"{r['corr_changes']:.3f} & {r['conv_share_both_below_5']:.2f} & {r['conv_median_price']:.1f} \\\\" for _, r in t.iterrows()]
    lines += ["\\midrule", "Spearman $\\rho$ with the intraday profile & & " + " & ".join(f"{rho[c]:.2f}" for c in ["share_within_1", "share_within_0.01", "corr_changes"]) + " & & \\\\",
              "\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_s_coupling_alt")
    return t, rho


def daily_errors(zone):
    W, y = xb.zone_frame(zone)
    return pd.DataFrame({k: (y - W[k]).abs() for k in ["base", "crossborder", "naive_d7"]}).groupby(level="date").sum()


def precision():
    bs = pd.read_csv("results/crossborder_2025_bootstrap.csv").set_index("zone")
    rng = np.random.default_rng(SEED)
    D = {z: daily_errors(z) for z in xb.ZONES}
    dates = sorted(set.intersection(*[set(d.index) for d in D.values()]))
    D = {z: d.loc[dates] for z, d in D.items()}
    idx = rng.integers(0, len(dates), size=(B, len(dates)))                 # the same days for all zones
    gain, pct = {}, {}
    for z, d in D.items():
        v = d[["base", "crossborder", "naive_d7"]].values
        s = v[idx].sum(axis=1)                                                  # B x 3
        gain[z] = (s[:, 0] - s[:, 1]) / s[:, 2]
        pct[z] = 100 * (s[:, 0] - s[:, 1]) / s[:, 0]
    diff = np.mean([gain[z] for z in ABOVE], axis=0) - np.mean([gain[z] for z in BELOW], axis=0)
    point = {z: (D[z].base.sum() - D[z].crossborder.sum()) / D[z].naive_d7.sum() for z in xb.ZONES}
    point_diff = np.mean([point[z] for z in ABOVE]) - np.mean([point[z] for z in BELOW])
    t = pd.DataFrame({"zone": xb.ZONES, "gain_point": [point[z] for z in xb.ZONES],
                      "gain_hi_joint": [np.percentile(gain[z], 97.5) for z in xb.ZONES],
                      "gain_hi_bootstrap": [bs.loc[z, "ci_high"] for z in xb.ZONES],
                      "pct_point": [100 * (D[z].base.sum() - D[z].crossborder.sum()) / D[z].base.sum() for z in xb.ZONES],
                      "pct_hi": [np.percentile(pct[z], 97.5) for z in xb.ZONES]})
    t.to_csv("results/crossborder_2025_bootstrap_pct.csv", index=False)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return t, (point_diff, lo, hi), len(dates)


def main():
    c, rho = coupling_alternatives()
    t, (pdiff, lo, hi), n = precision()
    lines = ["Alternative coupling measures and precision of the neighbouring-price result.", "",
             "1. Alternative coupling measures, 2024", c.round(4).to_string(index=False), "",
             "Spearman rho with the intraday-profile measure: " + ", ".join(f"{k} {v:.3f}" for k, v in rho.items()), "",
             f"2. Neighbouring prices, 2025 ({n} common days, {B} day resamples, all zones of a day together)",
             t.round(4).to_string(index=False), "",
             f"Largest upper 95% bound of the gain (analysis/09, per-zone bootstrap): {t.gain_hi_bootstrap.max():.4f} rMAE "
             f"({t.zone[t.gain_hi_bootstrap.idxmax()]})",
             f"Largest upper 95% bound of the gain (joint bootstrap): {t.gain_hi_joint.max():.4f} rMAE ({t.zone[t.gain_hi_joint.idxmax()]})",
             f"Largest upper 95% bound as % of MAE without the inputs: {t.pct_hi.max():.2f}% ({t.zone[t.pct_hi.idxmax()]})",
             f"Mean gain above-median minus below-median zones: {pdiff:.4f}, 95% interval {lo:.4f} to {hi:.4f}"]
    open("results/crossborder_precision.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
