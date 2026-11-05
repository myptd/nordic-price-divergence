"""
Shared style for the paper figures and tables.

- Zones are always ordered by the frozen 2024 coupling measure, most coupled first, with
  DE-LU last.
- One colour-blind-safe palette for the market types: coupled blue, isolated orange, intermediate (NO5) green, DE-LU
  reference purple. Every mark also carries a direct zone label, since green has low contrast on white.
- Line style by country: solid Norway, dashed Sweden, dotted DE-LU.
- Paper figures carry no titles or numbers; the captions in the manuscript do. Files go to paper_outputs/figures
  as .pdf and 300-dpi .jpg; tables go to paper_outputs/tables as LaTeX.
"""
import os

COUPLING = {"SE4": 0.710, "NO2": 0.668, "NO1": 0.551, "SE3": 0.538, "NO5": 0.424,
            "NO3": 0.304, "SE1": 0.277, "SE2": 0.249, "NO4": 0.248}
NORDIC = sorted(COUPLING, key=lambda z: -COUPLING[z])                  # SE4, NO2, NO1, SE3, NO5, NO3, SE1, SE2, NO4
ZONES = NORDIC + ["DE_LU"]
BY_COUNTRY = {"Norway": [z for z in NORDIC if z.startswith("NO")], "Sweden": [z for z in NORDIC if z.startswith("SE")],
              "DE-LU": ["DE_LU"]}
TYPE = {**{z: "coupled" for z in ["SE4", "NO2", "NO1", "SE3"]}, "NO5": "intermediate",
        **{z: "isolated" for z in ["NO3", "SE1", "SE2", "NO4"]}, "DE_LU": "reference"}
TYPE_NAMES = {"coupled": "Coupled", "intermediate": "Intermediate (NO5)", "isolated": "Isolated", "reference": "DE-LU (reference)"}
COLORS = {"coupled": "#2a78d6", "isolated": "#eb6834", "intermediate": "#1baf7a", "reference": "#4a3aa7"}
STYLE = {"NO": "-", "SE": "--", "DE": ":"}
C_MEAN = sum(COUPLING.values()) / len(COUPLING)                         # 0.441
INK, MUTED, GRID, AXIS = "#16202a", "#5a6573", "#e3e7eb", "#c5ccd4"
FIG, TAB = "paper_outputs/figures", "paper_outputs/tables"


def label(z):
    return z.replace("_", "-")


def color(z):
    return COLORS[TYPE[z]]


def style(z):
    return STYLE[z[:2]]


def clean_axes(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(AXIS)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.jpg", dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})


def save_table(text, name):
    os.makedirs(TAB, exist_ok=True)
    open(f"{TAB}/{name}.tex", "w").write(text)
