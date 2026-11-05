"""
Fig. 4: change in 2025 rMAE of per-hour LightGBM with the previous-day neighbouring prices (DE-LU, DK1, NL; without
minus with, positive = better) against the frozen 2024 coupling measure, with 95% day-bootstrap intervals. Colours by
market type; dashed line at the median (NO5) that defines the H2/H3 groups.

Reads results/crossborder_2025_bootstrap.csv (analysis/09).
Output: paper_outputs/figures/fig4_crossborder_vs_coupling.pdf/.jpg.
Run from the repository root:  python analysis/23_fig4_crossborder.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

DODGE = {"NO4": -0.004, "SE2": 0.004, "SE3": -0.004, "NO1": 0.004}   # display only: almost equal coupling
LEFT = {"NO4", "SE3"}


def main():
    t = pd.read_csv("results/crossborder_2025_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhline(0, color=ps.AXIS, linewidth=1.1, zorder=1)
    ax.axvline(ps.COUPLING["NO5"], color=ps.MUTED, linewidth=0.9, linestyle="--", zorder=1)
    ax.text(ps.COUPLING["NO5"] + 0.006, 0.99, "Median", transform=ax.get_xaxis_transform(), fontsize=8, color=ps.MUTED, va="top")
    for r in t.itertuples():
        c, x = ps.color(r.zone), r.coupling_2024 + DODGE.get(r.zone, 0.0)
        ax.plot([x, x], [r.ci_low, r.ci_high], color=c, linewidth=2, alpha=0.8, zorder=2)
        ax.scatter(x, r.delta_rMAE, s=64, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(ps.label(r.zone), (x, r.delta_rMAE), xytext=(-7 if r.zone in LEFT else 6, -11 if r.zone == "NO4" else 3), textcoords="offset points",
                    fontsize=8.5, color=ps.INK, ha="right" if r.zone in LEFT else "left")
    ax.set_xlabel("Coupling in 2024", fontsize=9, color=ps.INK)
    ax.set_ylabel("Change in rMAE with neighbouring prices\n(positive = more accurate)", fontsize=9, color=ps.INK)
    ps.clean_axes(ax, grid="both")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=ps.COLORS[k], label=ps.TYPE_NAMES[k])
               for k in ["coupled", "intermediate", "isolated"]]
    handles.append(plt.Line2D([], [], color=ps.MUTED, linewidth=2, label="95% interval"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    ps.save(fig, "fig4_crossborder_vs_coupling")


if __name__ == "__main__":
    main()
