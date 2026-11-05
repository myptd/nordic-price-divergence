"""
Fig. 6: association of daily forecast errors (per-hour LightGBM, 2025, daily MAE relative to the zone's naive-D-7
MAE) with market conditions, against 2024 coupling. Total version.
Large panel: wind and solar forecast; small panels: reservoir, temperature, load. Points: zone estimates theta_z (per
standard deviation of the condition) with 95% intervals (+-1.96 Newey-West s.e.); line: pooled fit
main + interaction * (coupling - mean coupling) over the Norwegian and Swedish zones. DE-LU (reference, not pooled) is
a dotted horizontal line. NO5 has no wind or solar forecast; DE-LU has no reservoir series.

Reads results/diagnostics_2025_zone.csv and results/diagnostics_2025_pooled.csv (analysis/19).
Output: paper_outputs/figures/fig6_a_wind_solar_b_reservoir_c_temperature_d_load.pdf/.jpg.
Run from the repository root:  python analysis/25_fig6_diagnostics.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

PANELS = [("wsf", "(a) Wind and solar forecast"), ("res", "(b) Reservoir"), ("temp", "(c) Temperature"), ("load", "(d) Load forecast")]
DODGE = {"NO4": -0.005, "SE2": 0.005, "SE3": -0.005, "NO1": 0.005}


def panel(ax, z, p, cond, title, big):
    ax.axhline(0, color=ps.AXIS, linewidth=1.0, zorder=1)
    d = z[(z.condition == cond) & (z.zone != "DE_LU")].dropna(subset=["theta"])
    for r in d.itertuples():
        x = ps.COUPLING[r.zone] + DODGE.get(r.zone, 0.0)
        ax.plot([x, x], [r.theta - 1.96 * r.se, r.theta + 1.96 * r.se], color=ps.color(r.zone), linewidth=1.6, alpha=0.8, zorder=2)
        ax.scatter(x, r.theta, s=48 if big else 26, color=ps.color(r.zone), edgecolor="white", linewidth=1.0, zorder=3)
        if big:                                                        # small panels share the x positions of panel a
            left = r.zone in ("NO4",)
            ax.annotate(ps.label(r.zone), (x, r.theta), xytext=(-6 if left else 5, 2), textcoords="offset points", fontsize=8,
                        color=ps.INK, ha="right" if left else "left")
    q = p[p.condition == cond].set_index("term").estimate
    c = np.linspace(0.23, 0.73, 50)
    ax.plot(c, q["main"] + q["interaction"] * (c - ps.C_MEAN), color=ps.INK, linewidth=1.3, zorder=2)
    de = z[(z.condition == cond) & (z.zone == "DE_LU")].theta
    if len(de) and np.isfinite(de.iloc[0]):
        ax.axhline(de.iloc[0], color=ps.COLORS["reference"], linestyle=":", linewidth=1.4, zorder=1)
        ax.text(0.745, de.iloc[0], "DE-LU", fontsize=7.5 if big else 6.8, color=ps.INK, va="top", ha="right")
    ax.set_xlim(0.22, 0.75)
    ax.set_title(title, fontsize=9.5 if big else 8.5, color=ps.INK, loc="left")
    ps.clean_axes(ax, grid="both")


def main():
    z = pd.read_csv("results/diagnostics_2025_zone.csv")
    p = pd.read_csv("results/diagnostics_2025_pooled.csv")
    z, p = z[z.version == "total"], p[p.version == "total"]
    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], hspace=0.55, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    panel(ax, z, p, *PANELS[0], big=True)
    ax.set_xlabel("Coupling in 2024", fontsize=9, color=ps.INK)
    ax.set_ylabel("Change in relative daily error per SD of the condition", fontsize=9, color=ps.INK)
    for i, (cond, title) in enumerate(PANELS[1:]):
        a = fig.add_subplot(gs[i, 1])
        panel(a, z, p, cond, title, big=False)
        if i == 2:
            a.set_xlabel("Coupling in 2024", fontsize=8.5, color=ps.INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=ps.COLORS[k], label=ps.TYPE_NAMES[k])
               for k in ["coupled", "intermediate", "isolated"]]
    handles += [plt.Line2D([], [], color=ps.INK, linewidth=1.3, label="Pooled fit"),
                plt.Line2D([], [], color=ps.COLORS["reference"], linestyle=":", linewidth=1.4, label="DE-LU (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, -0.03))
    ps.save(fig, "fig6_a_wind_solar_b_reservoir_c_temperature_d_load")


if __name__ == "__main__":
    main()
