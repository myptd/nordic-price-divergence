"""
Graphical abstract: the network of Fig. 1a (zones coloured by market type) next to the three main messages.
Elsevier asks for at least 531 x 1328 pixels (h x w); this file is 11 x 4.4 inches at 300 dpi (3300 x 1320 pixels).

Reuses analysis/20_fig1_network_design.py (panel_map).
Output: paper_outputs/figures/graphical_abstract.pdf/.jpg.
Run from the repository root:  python analysis/30_graphical_abstract.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = load("ps", "analysis/paper_style.py")
f1 = load("f1", "analysis/20_fig1_network_design.py")

MESSAGES = [
    ("Wind and solar forecasts improve price forecasts\nlittle in Norway, much in Sweden and Germany",
     "Loss in relative error without them: 0.017 Norway, 0.036-0.073 Sweden, 0.130 Germany-Luxembourg"),
    ("Neighbouring prices add little beyond\neach zone's own price history",
     "Small gain in every zone, not larger in coupled zones"),
    ("One model serves all market types",
     "LightGBM fitted per hour cut Nordic errors by 24-37% against repeating the previous day's prices"),
]


def main():
    fig = plt.figure(figsize=(11, 4.4))
    ax_map = fig.add_axes([0.0, 0.0, 0.36, 1.0])
    f1.panel_map(ax_map)
    for t in list(ax_map.texts):
        if t.get_text() == "(a)":
            t.remove()
    ax = fig.add_axes([0.37, 0.0, 0.63, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "Diverging Nordic electricity markets:\nwhat day-ahead forecasts reveal about price formation",
            fontsize=12, fontweight="bold", color=ps.INK, va="top", linespacing=1.25)
    ax.text(0.02, 0.8, "Co-movement with Germany-Luxembourg split in Sweden in 2020 and in Norway in 2021; NO5 now lies between the two groups",
            fontsize=8.6, color=ps.MUTED, va="top")
    for i, (head, body) in enumerate(MESSAGES):
        y = 0.52 - i * 0.24
        ax.add_patch(FancyBboxPatch((0.02, y), 0.95, 0.19, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="white", edgecolor=ps.AXIS, linewidth=1.3))
        ax.text(0.045, y + 0.1, str(i + 1), fontsize=18, fontweight="bold", color=ps.COLORS["coupled"], va="center")
        ax.text(0.1, y + 0.123, head, fontsize=9.4, fontweight="bold", color=ps.INK, va="center", linespacing=1.2)
        ax.text(0.1, y + 0.035, body, fontsize=8.4, color=ps.MUTED, va="center")
    ps.save(fig, "graphical_abstract")


if __name__ == "__main__":
    main()
