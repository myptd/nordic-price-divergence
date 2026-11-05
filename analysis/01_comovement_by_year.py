"""
Co-movement of hourly day-ahead prices between all 13 zones, by year 2018-2025.

  by year       correlation of each zone with DE-LU and NL, and its share of hours with the same price
                (|difference| < 0.01 EUR/MWh) as DE-LU, DK1, NO2 and SE3
  matrices      correlation between all pairs of zones, one matrix per year (used by 03 and 04)

Outputs: results/comovement/by_year.csv, results/comovement/corr_{year}.csv,
paper_outputs/figures/figS2_corr_matrix_by_year.pdf/.jpg (Fig. S2).
Run from the repository root:  python analysis/01_comovement_by_year.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/comovement"
PAPER = "paper_outputs/figures"
ORDER = ["NO4", "NO3", "NO5", "NO1", "NO2", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "NL", "DE_LU"]   # north to south
EQUAL_TOL = 0.01
LABEL = {z: z.replace("_", "-") for z in ORDER}


def load_prices():
    p = pd.concat({z: pd.read_parquet(f"data/cleaned/{z}_hourly.parquet")["price_eur_mwh"] for z in ORDER}, axis=1)
    return p[(p.index.year >= 2018) & (p.index.year <= 2025)]


def equal_share(p):
    v = p.values
    m = np.zeros((len(ORDER), len(ORDER)))
    for i in range(len(ORDER)):
        m[i] = (np.abs(v - v[:, [i]]) < EQUAL_TOL).mean(axis=0)
    return pd.DataFrame(m, index=ORDER, columns=ORDER)


def heat(ax, M, title, fmt, vmin, vmax, cmap, fs=6.3, ls=8):
    im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = M.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "#16202a")
    ax.set_xticks(range(len(ORDER)))
    ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[z] for z in ORDER], rotation=90, fontsize=ls)
    ax.set_yticklabels([LABEL[z] for z in ORDER], fontsize=ls)
    for k in [4.5, 8.5]:                                  # NO | SE | continent and Denmark
        ax.axhline(k, color="white", linewidth=2)
        ax.axvline(k, color="white", linewidth=2)
    ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PAPER, exist_ok=True)
    p = load_prices()
    rows = []
    for year, g in p.groupby(p.index.year):
        c, e = g.corr(), equal_share(g)
        for z in ORDER:
            rows.append({"year": year, "zone": z, "corr_DE_LU": c.loc[z, "DE_LU"], "corr_NL": c.loc[z, "NL"],
                         "equal_DE_LU": e.loc[z, "DE_LU"], "equal_DK1": e.loc[z, "DK1"], "equal_NO2": e.loc[z, "NO2"],
                         "equal_SE3": e.loc[z, "SE3"]})
    by_year = pd.DataFrame(rows)
    by_year.to_csv(f"{OUT}/by_year.csv", index=False)

    # Fig. S2: correlation matrices, one panel per year
    fig, axes = plt.subplots(2, 4, figsize=(24, 12.5))
    for ax, year in zip(axes.flat, sorted(set(p.index.year))):
        M = p[p.index.year == year].corr()
        M.to_csv(f"{OUT}/corr_{year}.csv")
        im = heat(ax, M, str(year), "{:.2f}", 0, 1, "YlGnBu", fs=5.2, ls=7.5)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, format=None).set_label("Correlation of hourly prices", fontsize=11)
    fig.savefig(f"{PAPER}/figS2_corr_matrix_by_year.pdf", bbox_inches="tight")
    fig.savefig(f"{PAPER}/figS2_corr_matrix_by_year.jpg", dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(by_year.pivot(index="zone", columns="year", values="corr_DE_LU").reindex(ORDER).round(2).to_string())


if __name__ == "__main__":
    main()
