"""
Cluster membership of the 13 zones by year 2018-2025, with the zones in fixed rows so that years can be compared.

For each year the zones are clustered with distance 1 - correlation of hourly prices and average linkage, and the tree
is cut into k clusters (k = 2, 3, 4; no distance threshold). Clusters are named by an anchor zone so that a colour means
the same in every year: the cluster with DE-LU, the cluster with NO4 (if without DE-LU), the cluster with NO2 (if
without both), and other.

Reads results/comovement/corr_{year}.csv (analysis/01).
Outputs: results/comovement/membership_by_year.csv (k = 2, 3, 4), paper_outputs/figures/figS3_membership_k3.pdf/.jpg
(Fig. S3, k = 3).
Run from the repository root:  python analysis/04_zone_membership.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DIR = "results/comovement"
ORDER = ["NO4", "NO3", "NO5", "NO1", "NO2", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "NL", "DE_LU"]
YEARS = list(range(2018, 2026))
KS = [2, 3, 4]
LABELS = ["with DE-LU", "with NO4", "with NO2", "other"]
COLORS = ["#4a3aa7", "#eb6834", "#2a78d6", "#d9dee4"]      # hues validated as CVD-safe; "other" is a neutral grey
NAMES = {"with DE-LU": "Cluster with DE-LU (continental)", "with NO4": "Cluster with NO4 (northern)",
         "with NO2": "Cluster with NO2 (southern Norway, apart from DE-LU)", "other": "Other cluster"}
INK, MUTED, LINE = "#16202a", "#5a6573", "#ffffff"
_ps = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(_ps)
_ps.loader.exec_module(ps)                         # row labels coloured by market type
LABEL_COLORS = {**{z: ps.color(z) for z in ps.ZONES}, **{z: "#9aa3ad" for z in ["DK1", "DK2", "NL"]}}


def membership(year, k):
    c = pd.read_csv(f"{DIR}/corr_{year}.csv", index_col=0).loc[ORDER, ORDER]
    cut = pd.Series(fcluster(linkage(squareform(1 - c.values, checks=False), "average", optimal_ordering=True), t=k, criterion="maxclust"), index=ORDER)
    names = {}
    for anchor, name in [("DE_LU", "with DE-LU"), ("NO4", "with NO4"), ("NO2", "with NO2")]:
        if cut[anchor] not in names:
            names[cut[anchor]] = name
    return cut.map(lambda cl: names.get(cl, "other"))


def grid(ax, M, cmap, title):
    ax.imshow(M.map(LABELS.index).values, cmap=cmap, vmin=-0.5, vmax=len(LABELS) - 0.5, aspect="auto")
    ax.set_xticks(np.arange(len(YEARS)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(ORDER)) + 0.5, minor=True)
    ax.grid(which="minor", color=LINE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for yv in [4.5, 8.5]:
        ax.axhline(yv, color=INK, linewidth=1.2)
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS, fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([z.replace("_", "-") for z in ORDER], fontsize=9, fontweight="bold")
    for lab in ax.get_yticklabels():
        lab.set_color(LABEL_COLORS[lab.get_text().replace("-", "_")])
    ax.set_title(title, fontsize=11, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    rows = [{"k": k, "year": y, "zone": z, "cluster": m} for k in KS for y in YEARS for z, m in membership(y, k).items()]
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{DIR}/membership_by_year.csv", index=False)

    cmap = ListedColormap(COLORS)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=c, label=NAMES[l]) for l, c in zip(LABELS, COLORS)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    grid(ax, tab[tab.k == 3].pivot(index="zone", columns="year", values="cluster").reindex(ORDER)[YEARS], cmap, "")
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, title="Cell colour: cluster", title_fontsize=9)
    lab_handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=c, label=n) for c, n in
                   [(ps.COLORS["coupled"], "Coupled"), (ps.COLORS["intermediate"], "Intermediate (NO5)"), (ps.COLORS["isolated"], "Isolated"),
                    (ps.COLORS["reference"], "DE-LU (reference)"), ("#9aa3ad", "Price-only zone")]]
    fig.legend(handles=lab_handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.115),
               title="Label colour: market type", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    ps.save(fig, "figS3_membership_k3")
    plt.close(fig)

    for k in KS:
        print(f"\nk = {k}")
        print(tab[tab.k == k].pivot(index="zone", columns="year", values="cluster").reindex(ORDER).to_string())


if __name__ == "__main__":
    main()
