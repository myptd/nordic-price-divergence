"""
Loss from removing the wind and solar forecasts against the generation mix, ten zones (Fig. S8).

no_vre delta rMAE in 2025 (analysis/14) against the 2024 wind-plus-solar share and the 2024 hydro share of generation
(analysis/05). Spearman rank correlations with exact two-sided permutation p-values (all 10! orderings).

Reads results/no_forecasts_2025.csv and results/market_types_2024.csv.
Outputs: results/vre_mix.txt, paper_outputs/figures/figS8_vre_mix.pdf/.jpg.
Run from the repository root:  python analysis/15_vre_mix.py
"""
import importlib.util
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def spearman_exact(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum()), ry - ry.mean()
    rho = float(rx @ ry / np.sqrt((ry ** 2).sum()))
    perms = np.array(list(itertools.permutations(range(len(ry)))), dtype=np.int8)     # 3,628,800 x 10
    stats = (ry[perms] @ rx) / np.sqrt((ry ** 2).sum())
    return rho, float(np.mean(np.abs(stats) >= abs(rho) - 1e-12))


def main():
    d = pd.read_csv("results/no_forecasts_2025.csv")
    d = d[d.input_set == "no_vre"].set_index("zone").delta_rMAE
    m = pd.read_csv("results/market_types_2024.csv").set_index("zone")
    t = pd.DataFrame({"delta_no_vre": d, "wind_solar_share": m.wind + m.solar, "hydro_share": m.hydro}).loc[ps.ZONES]
    lines = ["no_vre delta rMAE (2025) against 2024 generation shares, ten zones; Spearman, exact two-sided p (10! orderings).", "",
             t.round(4).to_string(), ""]
    res = {}
    for col in ["wind_solar_share", "hydro_share"]:
        res[col] = spearman_exact(t[col].values, t.delta_no_vre.values)
        lines.append(f"{col}: rho = {res[col][0]:.3f}, exact two-sided p = {res[col][1]:.4f}")
    open("results/vre_mix.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for k, (ax, col, xl) in enumerate(zip(axes, ["wind_solar_share", "hydro_share"],
                                          ["Wind and solar share of generation, 2024 (%)", "Hydro share of generation, 2024 (%)"])):
        for z, r in t.iterrows():
            ax.scatter(r[col], r.delta_no_vre, s=46, color=ps.color(z), edgecolor="white", linewidth=1.0, zorder=3)
            ax.annotate(ps.label(z), (r[col], r.delta_no_vre), xytext=(5, 2), textcoords="offset points", fontsize=7.8, color=ps.INK)
        ax.set_xlabel(xl, fontsize=8.8, color=ps.INK)
        ax.text(0.0, 1.03, f"({'ab'[k]})", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=ps.INK)
        ps.clean_axes(ax, grid="both")
    axes[0].set_ylabel("Loss without wind and solar\nforecasts (change in rMAE)", fontsize=8.8, color=ps.INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=ps.COLORS[c], label=ps.TYPE_NAMES[c])
               for c in ["coupled", "intermediate", "isolated", "reference"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ps.save(fig, "figS8_vre_mix")


if __name__ == "__main__":
    main()
