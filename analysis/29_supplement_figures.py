"""
Supplementary screening table and figures:
  tab_s_screening.tex          screening on 2024 (rMAE relative to naive D-7, chosen settings)
  figS4_transfer_zones         transfer matrices: (a) mean over NO1-NO5 for the three targets, (b) every zone and
                               target; diverging log colour scale centred at 1, as in Fig. 3
  figS5_dm_benchmark           multivariate DM tests of the 2025 benchmark, Holm-adjusted p-values per zone
  figS6/figS7_diagnostics_*    error diagnostics, all nine conditions, total and direct versions

Reads results/screening_2024.csv (analysis/06), results/transfer/transfer_long.csv (analysis/12),
results/benchmark_2025_dm.csv (analysis/07), results/diagnostics_2025_{zone,pooled}.csv (analysis/19).
Outputs: paper_outputs/tables/tab_s_screening.tex, paper_outputs/figures/figS4-S7 (.pdf/.jpg).
Run from the repository root:  python analysis/29_supplement_figures.py
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = load("ps", "analysis/paper_style.py")
f6 = load("f6", "analysis/25_fig6_diagnostics.py")
YEARS = list(range(2018, 2026))
TARGETS = [("none", "Raw"), ("level", "Level only"), ("level_scale", "Level and scale")]
MODELS = [("naive_d1", "Naive D-1"), ("naive_d7", "Naive D-7"), ("lear", "LEAR"), ("lgbm_pooled", "LightGBM pooled"),
          ("lgbm_hourly", "LightGBM per hour"), ("ensemble", "Ensemble")]
CONDITIONS = [("gas", "(a) TTF gas"), ("carbon", "(b) EUA carbon (Dec. future)"), ("coal", "(c) API2 coal"), ("brent", "(d) Brent oil"),
              ("eurnok", "(e) EUR/NOK"), ("res", "(f) Reservoir"), ("wsf", "(g) Wind and solar forecast"),
              ("load", "(h) Load forecast"), ("temp", "(i) Temperature")]


def table_screening():
    s = pd.read_csv("results/screening_2024.csv")
    rows = [("naive_d1", "-", "Naive D-1"), ("naive_d7", "-", "Naive D-7"), ("lear", "standard", "LEAR, standard"),
            ("lear", "level_scale", "LEAR, level and scale$^{*}$"), ("lgbm_pooled", "none", "LightGBM pooled, raw"),
            ("lgbm_pooled", "level", "LightGBM pooled, level only"), ("lgbm_pooled", "level_scale", "LightGBM pooled, level and scale$^{*}$"),
            ("lgbm_hourly", "none", "LightGBM per hour, raw"), ("lgbm_hourly", "level", "LightGBM per hour, level only"),
            ("lgbm_hourly", "level_scale", "LightGBM per hour, level and scale$^{*}$"),
            ("ensemble", "level_scale + level_scale", "Ensemble"), ("lstm", "level", "LSTM"), ("tcn", "level", "TCN"),
            ("transformer", "level", "Transformer")]
    zones = ps.ZONES
    lines = [f"\\begin{{tabular}}{{l{'r' * len(zones)}}}", "\\toprule", "Model and target & " + " & ".join(ps.label(z) for z in zones) + " \\\\", "\\midrule"]
    for m, t, name in rows:
        d = s[(s.model == m) & (s.target == t)]
        if m.startswith("lgbm"):
            d = d[d.chosen_setting]
        v = d.set_index("zone").rMAE
        lines.append(f"{name} & " + " & ".join("--" if z not in v.index else f"{v[z]:.3f}" for z in zones) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_s_screening")


def matrix(ax, M, norm, title, small):
    im = ax.imshow(np.log10(M.clip(0.3, 30.0)), cmap="RdBu_r", norm=norm)
    for i in range(len(YEARS)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ps.INK, linewidth=0.7))
    ax.set_xticks(range(len(YEARS)))
    ax.set_yticks(range(len(YEARS)))
    lab = [str(y)[2:] for y in YEARS] if small else YEARS
    ax.set_xticklabels(lab, fontsize=6 if small else 7.5, color=ps.MUTED, rotation=0 if small else 90)
    ax.set_yticklabels(lab, fontsize=6 if small else 7.5, color=ps.MUTED)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=7.5 if small else 9, color=ps.INK, loc="left")
    return im


def fig_transfer():
    t = pd.read_csv("results/transfer/transfer_long.csv")
    norm = TwoSlopeNorm(vmin=np.log10(0.3), vcenter=0.0, vmax=np.log10(30.0))
    fig = plt.figure(figsize=(10, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.4], hspace=0.12)
    top = gs[0].subgridspec(1, 3, wspace=0.25)
    for k, (tg, name) in enumerate(TARGETS):
        M = t[t.zone.str.startswith("NO") & (t.norm == tg)].pivot_table(index="train_year", columns="test_year", values="rMAE").reindex(index=YEARS, columns=YEARS)
        im = matrix(fig.add_subplot(top[k]), M.values, norm, ("(a) " if k == 0 else "") + f"Mean NO1-NO5, {name.lower()}", False)
    bottom = gs[1].subgridspec(5, 6, wspace=0.15, hspace=0.45)
    for i, z in enumerate(ps.ZONES):
        for k, (tg, name) in enumerate(TARGETS):
            M = t[(t.zone == z) & (t.norm == tg)].pivot(index="train_year", columns="test_year", values="rMAE").reindex(index=YEARS, columns=YEARS)
            title = ("(b) " if i == 0 and k == 0 else "") + f"{ps.label(z)}, {name.lower()}"
            matrix(fig.add_subplot(bottom[i // 2, (i % 2) * 3 + k]), M.values, norm, title, True)
    cax = fig.add_axes([0.93, 0.35, 0.012, 0.3])
    ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    cb = fig.colorbar(im, cax=cax, ticks=np.log10(ticks))
    cb.ax.set_yticklabels([str(v) for v in ticks], fontsize=8)
    cb.set_label("rMAE (log scale; 1 = D-7 naive)", fontsize=8.5)
    ps.save(fig, "figS4_transfer_zones")


def fig_dm():
    dm = pd.read_csv("results/benchmark_2025_dm.csv")
    keys = [m for m, _ in MODELS]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6))
    cmap = plt.get_cmap("Greens_r").copy()
    cmap.set_over("#eceff2")
    for ax, z in zip(axes.flat, ps.ZONES):
        P = dm[dm.zone == z].pivot(index="row", columns="column", values="p_holm").reindex(index=keys, columns=keys)
        im = ax.imshow(P.values, cmap=cmap, vmin=0, vmax=0.05)
        for i in range(len(keys)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="none"))
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        short = ["D-1", "D-7", "LEAR", "Pooled", "Per hour", "Ens."]
        ax.set_xticklabels(short, rotation=90, fontsize=7, color=ps.INK)
        ax.set_yticklabels(short, fontsize=7, color=ps.INK)
        ax.tick_params(length=0)
        ax.set_title(ps.label(z), fontsize=9, color=ps.INK, loc="left")
    fig.subplots_adjust(wspace=0.55, hspace=0.45)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, extend="max")
    cb.set_label("Holm-adjusted p (column more accurate than row)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ps.save(fig, "figS5_dm_benchmark")


def fig_diagnostics(version, name):
    z = pd.read_csv("results/diagnostics_2025_zone.csv")
    p = pd.read_csv("results/diagnostics_2025_pooled.csv")
    z, p = z[z.version == version], p[p.version == version]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.6))
    for ax, (cond, title) in zip(axes.flat, CONDITIONS):
        f6.panel(ax, z, p, cond, title, big=False)
    for ax in axes[-1]:
        ax.set_xlabel("Coupling in 2024", fontsize=8.5, color=ps.INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Change in relative daily\nerror per SD", fontsize=8.5, color=ps.INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=ps.COLORS[k], label=ps.TYPE_NAMES[k])
               for k in ["coupled", "intermediate", "isolated"]]
    handles += [plt.Line2D([], [], color=ps.INK, linewidth=1.3, label="Pooled fit"),
                plt.Line2D([], [], color=ps.COLORS["reference"], linestyle=":", linewidth=1.4, label="DE-LU (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ps.save(fig, name)


def main():
    table_screening()
    fig_transfer()
    fig_dm()
    fig_diagnostics("total", "figS6_diagnostics_total")
    fig_diagnostics("direct", "figS7_diagnostics_direct")


if __name__ == "__main__":
    main()
