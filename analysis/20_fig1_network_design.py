"""
Fig. 1: (a) the bidding zones as a network on a Natural Earth base map; (b) the study design.

(a) Nodes are the ten modelled zones (coloured by market type), the price-only zones DK1, DK2 and NL (grey) and Great
Britain (open, no data, end point of North Sea Link). Node positions are set by hand at approximate zone locations.
Edges are the main links between these areas, all drawn with the same width; the four interconnectors from NO2 are
named. Links to Finland, the Baltic states and Poland are not shown. Base map: Natural Earth 1:50m admin-0 countries
(public domain), drawn from the GeoJSON with matplotlib only (equirectangular, x scaled by cos 60 degrees).
(b) Schematic of the study design (boxes and arrows).

Reads data/naturalearth/ne_50m_admin_0_countries.geojson.
Output: paper_outputs/figures/fig1_a_network_b_design.pdf/.jpg.
Run from the repository root:  python analysis/20_fig1_network_design.py
"""
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

MAP = "data/naturalearth/ne_50m_admin_0_countries.geojson"
KX = np.cos(np.deg2rad(60.0))
EXTENT = (-4.0, 27.0, 50.3, 71.3)                     # lon min, lon max, lat min, lat max
NODES = {"NO1": (10.9, 60.3), "NO2": (7.6, 58.9), "NO5": (6.3, 60.7), "NO3": (10.8, 63.5), "NO4": (17.2, 68.6),
         "SE1": (21.0, 66.6), "SE2": (16.4, 63.4), "SE3": (15.2, 59.4), "SE4": (13.9, 56.3),
         "DK1": (9.1, 56.1), "DK2": (12.1, 55.4), "NL": (5.4, 52.4), "DE_LU": (10.2, 51.6), "GB": (-1.4, 55.0)}
NAMED = {("NO2", "DE_LU"): "NordLink", ("NO2", "GB"): "North Sea Link", ("NO2", "NL"): "NorNed", ("NO2", "DK1"): "Skagerrak"}
LINKS = [("NO1", "NO2"), ("NO1", "NO3"), ("NO1", "NO5"), ("NO2", "NO5"), ("NO3", "NO5"), ("NO3", "NO4"),
         ("NO1", "SE3"), ("NO3", "SE2"), ("NO4", "SE1"), ("NO4", "SE2"), ("SE1", "SE2"), ("SE2", "SE3"), ("SE3", "SE4"),
         ("SE3", "DK1"), ("SE4", "DK2"), ("SE4", "DE_LU"), ("DK1", "DK2"), ("DK1", "DE_LU"), ("DK2", "DE_LU"),
         ("DK1", "NL"), ("NL", "DE_LU")] + list(NAMED)
LABEL_POS = {"NordLink": (13.6, 53.4), "North Sea Link": (1.4, 56.3), "NorNed": (4.2, 55.6), "Skagerrak": (10.9, 58.0)}
BEND = {("NO2", "DE_LU"): 0.3}                      # curved in the drawing only, so that NordLink does not run through DK1
LABEL_OFF = {"DK2": (0.35, -0.75), "DE_LU": (0.5, -0.45), "GB": (-0.3, 0.5), "NL": (-1.2, 0.35), "DK1": (-1.3, 0.35)}
PRICE_ONLY = "#9aa3ad"


def xy(lon, lat):
    return lon * KX, lat


def draw_base(ax):
    g = json.load(open(MAP))
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            if ring[:, 0].max() < EXTENT[0] - 5 or ring[:, 0].min() > EXTENT[1] + 5 or ring[:, 1].max() < EXTENT[2] - 5:
                continue
            x, y = xy(ring[:, 0], ring[:, 1])
            ax.fill(x, y, facecolor="#f1f3f5", edgecolor="#c9cfd6", linewidth=0.5, zorder=0)


def node_style(z):
    if z in ps.TYPE:
        return dict(facecolor=ps.color(z), edgecolor="white")
    if z == "GB":
        return dict(facecolor="white", edgecolor=PRICE_ONLY)
    return dict(facecolor=PRICE_ONLY, edgecolor="white")


def panel_map(ax):
    draw_base(ax)
    for a, b in LINKS:
        (x0, y0), (x1, y1) = xy(*NODES[a]), xy(*NODES[b])
        named = (a, b) in NAMED
        t = np.linspace(0, 1, 50)
        bx, by = (x0 + x1) / 2 - BEND.get((a, b), 0.0) * (y1 - y0), (y0 + y1) / 2 + BEND.get((a, b), 0.0) * (x1 - x0)
        px, py = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * bx + t ** 2 * x1, (1 - t) ** 2 * y0 + 2 * t * (1 - t) * by + t ** 2 * y1
        ax.plot(px, py, color=ps.INK if named else "#8a949f", linewidth=1.4, zorder=1, solid_capstyle="round")
    for (a, b), name in NAMED.items():
        ax.text(*xy(*LABEL_POS[name]), name, fontsize=7.5, color=ps.INK, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    for z, (lon, lat) in NODES.items():
        x, y = xy(lon, lat)
        ax.scatter(x, y, s=150, linewidth=1.3, zorder=3, **node_style(z))
        dx, dy = LABEL_OFF.get(z, (0.42, 0.28))
        ax.text(x + dx, y + dy, ps.label(z), fontsize=8, color=ps.INK, zorder=4, fontweight="bold" if z in ps.TYPE else None)
    ax.set_xlim(EXTENT[0] * KX, EXTENT[1] * KX)
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=ps.INK, va="top")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=ps.COLORS[t], markeredgecolor="white",
                          label=ps.TYPE_NAMES[t]) for t in ["coupled", "intermediate", "isolated", "reference"]]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=PRICE_ONLY, markeredgecolor="white",
                           label="Price-only zone"),
                plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=PRICE_ONLY,
                           label="Not in the data"),
                plt.Line2D([], [], color=ps.INK, linewidth=1.4, label="Interconnector from NO2"),
                plt.Line2D([], [], color="#8a949f", linewidth=1.4, label="Other link")]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95, edgecolor="none", fontsize=7.3,
              bbox_to_anchor=(0.0, 0.95))


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor="white",
                                edgecolor=color, linewidth=1.4))
    ax.text(x + 0.02, y + h - 0.022, title, fontsize=8.2, fontweight="bold", color=ps.INK, va="top")
    ax.text(x + 0.02, y + h - 0.062, body, fontsize=7.3, color=ps.MUTED, va="top", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=ps.MUTED, linewidth=1.1))


def panel_design(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=ps.INK, va="top")
    w = 0.9
    steps = [(0.84, "Data, 2018-2025", "Prices of 13 zones; TSO load, wind and solar\nforecasts; weather; reservoirs; commodities"),
             (0.66, "Market types", "Coupling: co-movement with DE-LU, 2024\nGeneration mix: shares by type, 2024"),
             (0.48, "Screening on 2024 (no 2025 data)", "Target, settings and main model chosen,\nthen frozen; expectations recorded"),
             (0.30, "Rolling test on 2025", "Monthly recalibration, expanding window,\nsame hours and tests in every zone")]
    for y, t, b in steps:
        box(ax, 0.05, y, w, 0.13, t, b, ps.AXIS)
    ax.text(0.05 + w - 0.02, 0.66 + 0.13 - 0.022, "4.1", fontsize=7, color=ps.MUTED, ha="right", va="top")
    for (y0, _, _), (y1, _, _) in zip(steps[:-1], steps[1:]):
        arrow(ax, 0.5, y0 - 0.003, 0.5, y1 + 0.133)
    exps = ["Price\nhistory", "Bench-\nmark", "Neigh-\nbouring\nprices", "TSO\nforecasts", "Sources\nof error"]
    secs = ["4.2", "4.2", "4.3", "4.4", "4.5"]
    bw, gap = 0.164, 0.02
    for i, (e, s) in enumerate(zip(exps, secs)):
        x = 0.05 + i * (bw + gap) - 0.012
        dashed = i == 4
        ax.add_patch(FancyBboxPatch((x, 0.03), bw, 0.19, boxstyle="round,pad=0.006,rounding_size=0.015", facecolor="white",
                                    edgecolor=ps.MUTED, linewidth=1.2, linestyle="--" if dashed else "-"))
        ax.text(x + bw / 2, 0.155, e, fontsize=7.3, color=ps.INK, ha="center", va="center", linespacing=1.15)
        ax.text(x + bw / 2, 0.05, s, fontsize=7, color=ps.MUTED, ha="center", va="bottom")
        arrow(ax, 0.5, 0.297, x + bw / 2, 0.225)
    ax.text(0.5, -0.01, "Dashed box: mechanism analyses", fontsize=7, color=ps.MUTED, ha="center", va="top")


def main():
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.04)
    panel_map(fig.add_subplot(gs[0]))
    panel_design(fig.add_subplot(gs[1]))
    ps.save(fig, "fig1_a_network_b_design")


if __name__ == "__main__":
    main()
