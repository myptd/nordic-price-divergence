"""
Fig. 2: correlation with DE-LU by zone and year, 2018-2025, at three time scales (stacked panels): hourly prices, the
intraday profile (hourly price minus its daily mean) and day-to-day movement (daily mean minus its monthly mean). Nine
Norwegian and Swedish zones, coloured by market type, line style by country; crisis years shaded; event ticks
(NordLink, North Sea Link, Nordic flow-based market coupling) on the bottom axis.

Reads results/comovement/by_year.csv and robustness_by_year.csv (analysis/01 and 02).
Output: paper_outputs/figures/fig2_a_hourly_b_intraday_c_day_to_day.pdf/.jpg.
Run from the repository root:  python analysis/21_fig2_divergence.py
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

DIR = "results/comovement"
EVENTS = [("2020-12-09", "NordLink"), ("2021-10-01", "North Sea Link"), ("2024-10-29", "Flow-based\nmarket coupling")]
YEARS = list(range(2018, 2026))


def event_x(date):
    d = pd.Timestamp(date)
    return d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365) - 0.5      # annual values sit at mid-year


def spread(y, gap):
    order = np.argsort(y)
    pos = np.array(y, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos - np.array(y, float)[order]).mean()
    out = np.empty(len(y))
    out[order] = pos
    return out


def panel(ax, M, ylim, events, tag):
    ax.axvspan(2020.5, 2022.5, color="#eceff2", zorder=0)
    for z in ps.NORDIC:
        ax.plot(YEARS, M.loc[YEARS, z], color=ps.color(z), linestyle=ps.style(z), linewidth=1.8, marker="o", markersize=3.5, zorder=3)
    last = M.loc[2025, ps.NORDIC].values
    for z, yl, yv in zip(ps.NORDIC, spread(last, 0.045 * (ylim[1] - ylim[0])), last):
        ax.plot([2025.06, 2025.2], [yv, yl], color=ps.GRID, linewidth=0.8)
        ax.text(2025.26, yl, ps.label(z), va="center", fontsize=8, color=ps.INK)
    if events:
        for date, name in EVENTS:
            x = event_x(date)
            ax.plot(x, ylim[0], marker="^", markersize=6.5, color=ps.INK, clip_on=False, zorder=4)
            ax.text(x, ylim[0] + 0.03 * (ylim[1] - ylim[0]), name, ha="center", va="bottom", fontsize=7.5, color=ps.MUTED, linespacing=0.95)
    ax.text(0.01, 0.98, tag, transform=ax.transAxes, fontsize=9.5, color=ps.INK, va="top", fontweight="bold")
    ax.set_xlim(2017.8, 2025.8)
    ax.set_ylim(*ylim)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Correlation with DE-LU", fontsize=9, color=ps.INK)
    ps.clean_axes(ax)


def main():
    raw = pd.read_csv(f"{DIR}/by_year.csv").pivot(index="year", columns="zone", values="corr_DE_LU")
    rob = pd.read_csv(f"{DIR}/robustness_by_year.csv", index_col=[0, 1])
    intraday, daily = rob.loc["hourly_minus_daily_mean"], rob.loc["daily_minus_monthly_mean"]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 10.2), sharex=True)
    for ax, M, ylim, ev, tag in [(axes[0], raw, (-0.12, 1.08), False, "(a) Hourly prices"),
                                 (axes[1], intraday, (-0.02, 1.08), False, "(b) Intraday profile"),
                                 (axes[2], daily, (-0.32, 1.08), True, "(c) Day-to-day movement")]:
        panel(ax, M, ylim, ev, tag)
    handles = [plt.Line2D([], [], color=ps.COLORS[t], linewidth=2.4, label=ps.TYPE_NAMES[t]) for t in ["coupled", "intermediate", "isolated"]]
    handles += [plt.Line2D([], [], color=ps.MUTED, linestyle=s, linewidth=1.6, label=l) for s, l in [("-", "Norway"), ("--", "Sweden")]]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ps.save(fig, "fig2_a_hourly_b_intraday_c_day_to_day")


if __name__ == "__main__":
    main()
