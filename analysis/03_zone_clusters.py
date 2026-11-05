"""
Hierarchical clustering of the 13 zones by the correlation of hourly prices, one dendrogram per year 2018-2025.

Distance 1 - correlation, average linkage, optimal leaf ordering. Leaf labels are coloured by market type
(analysis/paper_style.py) and price-only zones. The dashed line at distance 0.3 (correlation 0.7) is a reading aid only.

Reads results/comovement/corr_{year}.csv (analysis/01).
Outputs: paper_outputs/figures/figS1_dendrograms_by_year.pdf/.jpg (Fig. S1), results/comovement/merge_heights_by_year.txt.
Run from the repository root:  python analysis/03_zone_clusters.py
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

DIR = "results/comovement"
PAPER = "paper_outputs/figures"
ORDER = ["NO4", "NO3", "NO5", "NO1", "NO2", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "NL", "DE_LU"]
YEARS = range(2018, 2026)
_ps = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(_ps)
_ps.loader.exec_module(ps)                         # market types (2024 coupling measure)
PRICE_ONLY = "#9aa3ad"
GROUP = {**ps.TYPE, **{z: "price_only" for z in ["DK1", "DK2", "NL"]}}
COLORS = {**ps.COLORS, "price_only": PRICE_ONLY}
NAMES = {**ps.TYPE_NAMES, "price_only": "Price-only zone"}
LEGEND_ORDER = ["coupled", "intermediate", "isolated", "reference", "price_only"]
LINE, INK, MUTED = "#8a95a3", "#16202a", "#5a6573"
READING_CUT = 0.3


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.6), sharex=True)
    lines = ["Hierarchical clustering of zones by hourly price correlation (distance 1 - r, average linkage).",
             "Per year: leaf order (top to bottom) and the distance at which the last two clusters merge.", ""]
    links = {}
    for year in YEARS:
        c = pd.read_csv(f"{DIR}/corr_{year}.csv", index_col=0).loc[ORDER, ORDER]
        links[year] = linkage(squareform(1 - c.values, checks=False), "average", optimal_ordering=True)
    top = max(Z[-1, 2] for Z in links.values()) * 1.05          # common axis that shows every root
    for ax, year in zip(axes.flat, YEARS):
        Z = links[year]
        d = dendrogram(Z, labels=[z.replace("_", "-") for z in ORDER], orientation="right", ax=ax,
                       color_threshold=0, above_threshold_color=LINE, leaf_font_size=9)
        for lab in ax.get_yticklabels():
            lab.set_color(COLORS[GROUP[lab.get_text().replace("-", "_")]])
            lab.set_fontweight("bold")
        ax.axvline(READING_CUT, color=MUTED, linewidth=0.9, linestyle="--")
        ax.set_xlim(0, top)
        ax.set_title(str(year), fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, colors=MUTED)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        lines.append(f"{year}: root merge at {Z[-1, 2]:.3f}; leaves {', '.join(reversed(d['ivl']))}")
    for ax in axes[1]:
        ax.set_xlabel("Distance (1 - correlation of hourly prices)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=COLORS[g], label=NAMES[g]) for g in LEGEND_ORDER]
    handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=0.9, label="Correlation 0.7 (reading aid)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(PAPER, exist_ok=True)
    fig.savefig(f"{PAPER}/figS1_dendrograms_by_year.pdf")
    fig.savefig(f"{PAPER}/figS1_dendrograms_by_year.jpg", dpi=300, pil_kwargs={"quality": 95})
    open(f"{DIR}/merge_heights_by_year.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
