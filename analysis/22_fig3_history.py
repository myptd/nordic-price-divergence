"""
Fig. 3: the break in the price history and how to bridge it.
Top: transfer matrices (pooled LightGBM trained on one year, tested on another; mean rMAE over NO1-NO5) with the raw
target (left) and the level-and-scale target (right), on a diverging logarithmic colour scale centred at 1 (the D-7
naive forecast). Bottom: 2025 rMAE of per-hour LightGBM by training window, Norway (left) and Sweden with DE-LU (right);
lines connect 365 days to expanding only; expanding without 2021-2022 and the window average are separate points; the
raw target (365 days and expanding) is shown with open squares.

Reads results/transfer/transfer_long.csv (analysis/12) and results/windows_2025.csv (analysis/13).
Output: paper_outputs/figures/fig3_a_transfer_b_windows.pdf/.jpg.
Run from the repository root:  python analysis/22_fig3_history.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

YEARS = list(range(2018, 2026))
NORWAY = ["NO1", "NO2", "NO3", "NO4", "NO5"]
LINE = ["365", "730", "1095", "expanding"]
POINTS = ["expanding_no_crisis", "average"]
X = {"365": 0, "730": 1, "1095": 2, "expanding": 3, "expanding_no_crisis": 4, "average": 5}
XLAB = ["365\ndays", "730\ndays", "1,095\ndays", "Expanding", "Without\n2021-22", "Window\naverage"]


def spread(y, gap):
    order = np.argsort(y)
    pos = np.array(y, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos - np.array(y, float)[order]).mean()
    out = np.empty(len(y))
    out[order] = pos
    return out


def main():
    t = pd.read_csv("results/transfer/transfer_long.csv")
    w = pd.read_csv("results/windows_2025.csv")
    fig = plt.figure(figsize=(12, 9.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 0.95], hspace=0.32, wspace=0.18)
    norm = TwoSlopeNorm(vmin=np.log10(0.3), vcenter=0.0, vmax=np.log10(30.0))
    for k, target in enumerate(["none", "level_scale"]):
        ax = fig.add_subplot(gs[0, k])
        M = t[(t.zone.isin(NORWAY)) & (t.norm == target)].pivot_table(index="train_year", columns="test_year", values="rMAE").reindex(index=YEARS, columns=YEARS)
        im = ax.imshow(np.log10(M.values.clip(0.3, 30.0)), cmap="RdBu_r", norm=norm)
        for i in range(len(YEARS)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ps.INK, linewidth=1.0))
        ax.set_xticks(range(len(YEARS)))
        ax.set_yticks(range(len(YEARS)))
        ax.set_xticklabels(YEARS, rotation=90, fontsize=8, color=ps.MUTED)
        ax.set_yticklabels(YEARS, fontsize=8, color=ps.MUTED)
        ax.set_xlabel("Test year", fontsize=9, color=ps.INK)
        if k == 0:
            ax.set_ylabel("Training year", fontsize=9, color=ps.INK)
        ax.set_title("(a) Raw price target" if target == "none" else "Level-and-scale target", fontsize=9.5, color=ps.INK, loc="left")
    cax = fig.add_subplot(gs[0, 2])
    ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    cb = fig.colorbar(im, cax=cax, ticks=np.log10(ticks))
    cb.ax.set_yticklabels([str(v) for v in ticks], fontsize=8)
    cb.set_label("rMAE (log scale; 1 = D-7 naive)", fontsize=8.5)

    for k, (zones, title) in enumerate([(ps.BY_COUNTRY["Norway"], "(b) Norway"), (ps.BY_COUNTRY["Sweden"] + ["DE_LU"], "Sweden and DE-LU")]):
        ax = fig.add_subplot(gs[1, k])
        ax.axvspan(X["expanding"] - 0.45, X["expanding"] + 0.45, color="#eceff2", zorder=0)
        for z in zones:
            s = w[w.zone == z].set_index("version").rMAE
            ax.plot([X[v] for v in LINE], s.reindex(LINE).values, color=ps.color(z), linestyle=ps.style(z), linewidth=1.8, marker="o", markersize=4.5, zorder=3)
            ax.scatter([X[v] for v in POINTS], s.reindex(POINTS).values, color=ps.color(z), s=26, zorder=3)
            ax.scatter([X["365"], X["expanding"]], [s["raw_365"], s["raw_expanding"]], marker="s", s=30, facecolor="white",
                       edgecolor=ps.color(z), linewidth=1.3, zorder=4)
        first = [w[(w.zone == z) & (w.version == "365")].rMAE.iloc[0] for z in zones]
        lo, hi = ax.get_ylim()
        for z, yl, yv in zip(zones, spread(first, 0.035 * (hi - lo)), first):
            ax.plot([X["365"] - 0.08, X["365"] - 0.22], [yv, yl], color=ps.GRID, linewidth=0.8)
            ax.text(X["365"] - 0.26, yl, ps.label(z), ha="right", va="center", fontsize=7.5, color=ps.INK)
        ax.set_xticks(range(len(XLAB)))
        ax.set_xticklabels(XLAB, fontsize=8, color=ps.INK)
        ax.set_xlim(-1.0, 5.4)
        ax.set_title(title, fontsize=9.5, color=ps.INK, loc="left")
        if k == 0:
            ax.set_ylabel("rMAE in 2025", fontsize=9, color=ps.INK)
        ps.clean_axes(ax)
    handles = [plt.Line2D([], [], color=ps.COLORS[c], linewidth=2.4, label=ps.TYPE_NAMES[c]) for c in ["coupled", "intermediate", "isolated", "reference"]]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", markersize=6, markerfacecolor="white", markeredgecolor=ps.MUTED, label="Raw price target"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.47, 0.0))
    ps.save(fig, "fig3_a_transfer_b_windows")


if __name__ == "__main__":
    main()
