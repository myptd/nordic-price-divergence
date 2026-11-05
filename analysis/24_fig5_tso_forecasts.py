"""
Fig. 5: loss of accuracy in 2025 when per-hour LightGBM is fitted without the TSO forecasts (rMAE without minus rMAE
of the base model; positive = the removed forecasts help). Two input sets (analysis/14): without the wind and solar
forecasts (load forecasts kept), and without load, wind and solar forecasts (prices and calendar only). Zones grouped by country, ordered by 2024 coupling
within country, coloured by market type; 95% intervals from a bootstrap over days (each day's 24 hours together;
2,000 resamples, fixed seed; delta recomputed as (sum of absolute errors without - base) / (sum of absolute errors of
naive D-7)). NO5 has no wind or solar forecast, so its "without wind and solar" forecasts equal the base forecasts.

Reads model_results/rolling/*_lgbm_hourly_{base,prices_calendar,no_vre}_expanding.parquet via analysis/14.
Outputs: paper_outputs/figures/fig5_tso_forecasts.pdf/.jpg, results/no_forecasts_2025_bootstrap.csv.
Run from the repository root:  python analysis/24_fig5_tso_forecasts.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = load("ps", "analysis/paper_style.py")
nf = load("nf", "analysis/14_tso_forecasts.py")
B, SEED = 2000, 20250928
SETS = [("no_vre", "Without wind and solar forecasts", "o"), ("prices_calendar", "Without load, wind and solar forecasts", "D")]


def bootstrap(zone, rng):
    W, y = nf.zone_frame(zone)
    cols = ["base"] + [s for s, _, _ in SETS] + ["naive_d7"]
    v = pd.DataFrame({k: (y - W[k]).abs() for k in cols}).groupby(level="date").sum()[cols].values
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = v[idx].sum(axis=1)
    out = []
    for j, (name, _, _) in enumerate(SETS, start=1):
        reps = (s[:, j] - s[:, 0]) / s[:, -1]
        out.append({"zone": zone, "input_set": name, "n_days": len(v), "delta_rMAE": (v[:, j].sum() - v[:, 0].sum()) / v[:, -1].sum(),
                    "ci_low": np.percentile(reps, 2.5), "ci_high": np.percentile(reps, 97.5)})
    return out


def main():
    rng = np.random.default_rng(SEED)
    t = pd.DataFrame([r for z in ps.ZONES for r in bootstrap(z, rng)])
    t.to_csv("results/no_forecasts_2025_bootstrap.csv", index=False)
    print(t.round(4).to_string(index=False))

    xs, ticks, x = {}, [], 0.0
    for country, zones in ps.BY_COUNTRY.items():
        for z in zones:
            xs[z] = x
            x += 1
        ticks.append((np.mean([xs[z] for z in zones]), country))
        x += 0.7
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color=ps.AXIS, linewidth=1.1, zorder=1)
    for k, (name, lab, mk) in enumerate(SETS):
        for r in t[t.input_set == name].itertuples():
            xx, c = xs[r.zone] + (k - 0.5) * 0.3, ps.color(r.zone)
            ax.plot([xx, xx], [r.ci_low, r.ci_high], color=c, linewidth=1.8, alpha=0.8, zorder=2)
            ax.scatter(xx, r.delta_rMAE, marker=mk, s=40, color=c if k == 0 else "white", edgecolor=c, linewidth=1.4, zorder=3)
    ax.set_xticks([xs[z] for z in ps.ZONES])
    ax.set_xticklabels([ps.label(z) for z in ps.ZONES], fontsize=8.5, color=ps.INK)
    for xc, country in ticks:
        ax.text(xc, -0.13, country, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=ps.MUTED)
    ax.set_ylabel("Loss of accuracy without the forecasts\n(change in rMAE)", fontsize=9, color=ps.INK)
    ps.clean_axes(ax)
    handles = [plt.Line2D([], [], marker=mk, linestyle="", markersize=6.5, markerfacecolor=ps.MUTED if k == 0 else "white",
                          markeredgecolor=ps.MUTED, label=lab) for k, (_, lab, mk) in enumerate(SETS)]
    handles += [plt.Line2D([], [], color=ps.COLORS[c], linewidth=2.4, label=ps.TYPE_NAMES[c]) for c in ["coupled", "intermediate", "isolated", "reference"]]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, loc="upper left", ncol=2)
    fig.tight_layout()
    ps.save(fig, "fig5_tso_forecasts")


if __name__ == "__main__":
    main()
