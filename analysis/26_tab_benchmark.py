"""
Benchmark tables: 2025 rMAE by market type (main text) and MAE/RMSE by zone (supplement).

Main table: rMAE (relative to naive D-7) of the six forecasts, zones ordered by 2024 coupling and grouped by market
type, DE-LU last. Best forecast per zone (lowest unrounded MAE) in bold. A dagger marks a forecast that per-hour
LightGBM beats significantly (multivariate DM, Holm within zone, p < 0.05). Bottom rows: MAE improvement of per-hour
LightGBM over naive D-1 (per cent), its MAE (EUR/MWh) and the 2025 mean price (results/market_types_2024.csv).

Reads results/benchmark_2025.csv and results/benchmark_2025_dm.csv (analysis/07).
Outputs: paper_outputs/tables/tab_benchmark.tex, paper_outputs/tables/tab_s_benchmark_mae_rmse.tex.
Run from the repository root:  python analysis/26_tab_benchmark.py
"""
import importlib.util

import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

MODELS = [("naive_d1", "Naive D-1"), ("naive_d7", "Naive D-7"), ("lear", "LEAR"), ("lgbm_pooled", "LightGBM pooled"),
          ("lgbm_hourly", "LightGBM per hour"), ("ensemble", "Ensemble")]
MAIN = "lgbm_hourly"
GROUPS = [("Coupled", [z for z in ps.NORDIC if ps.TYPE[z] == "coupled"]), ("", ["NO5"]),
          ("Isolated", [z for z in ps.NORDIC if ps.TYPE[z] == "isolated"]), ("", ["DE_LU"])]
ZONES = [z for _, zs in GROUPS for z in zs]


def header():
    spec_cols = "l" + "".join("|" + "r" * len(zs) for _, zs in GROUPS)
    top = " & " + " & ".join(f"\\multicolumn{{{len(zs)}}}{{c}}{{{g}}}" if g else "\\multicolumn{1}{c}{}" for g, zs in GROUPS) + " \\\\"
    rules = " ".join(f"\\cmidrule(lr){{{2 + sum(len(x) for _, x in GROUPS[:i])}-{1 + sum(len(x) for _, x in GROUPS[:i + 1])}}}"
                     for i, (g, _) in enumerate(GROUPS) if g)
    names = "Forecast & " + " & ".join(ps.label(z) for z in ZONES) + " \\\\"
    return spec_cols.replace("|", ""), top, rules, names


def main():
    b = pd.read_csv("results/benchmark_2025.csv")
    dm = pd.read_csv("results/benchmark_2025_dm.csv")
    r = b.pivot(index="model", columns="zone", values="rMAE")
    mae = b.pivot(index="model", columns="zone", values="MAE")
    best = mae.loc[[m for m, _ in MODELS]].idxmin()
    beaten = dm[(dm.column == MAIN) & (dm.p_holm < 0.05)].groupby("row").zone.apply(set).to_dict()
    cols, top, rules, names = header()
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", top, rules, names, "\\midrule"]
    for m, name in MODELS:
        cells = []
        for z in ZONES:
            s = f"{r.loc[m, z]:.3f}"
            if best[z] == m:
                s = f"\\textbf{{{s}}}"
            if z in beaten.get(m, set()):
                s += "$^{\\dagger}$"
            cells.append(s)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    imp = 100 * (1 - mae.loc[MAIN] / mae.loc["naive_d1"])
    price = pd.read_csv("results/market_types_2024.csv").set_index("zone").mean_price_2025
    lines += ["\\midrule", "Improvement over naive D-1 (\\%) & " + " & ".join(f"{imp[z]:.1f}" for z in ZONES) + " \\\\",
              "MAE of LightGBM per hour (EUR/MWh) & " + " & ".join(f"{mae.loc[MAIN, z]:.1f}" for z in ZONES) + " \\\\",
              "Mean price 2025 (EUR/MWh) & " + " & ".join(f"{price[z]:.1f}" for z in ZONES) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_benchmark")

    rm = b.pivot(index="model", columns="zone", values="RMSE")
    lines = [f"\\begin{{tabular}}{{l{'r' * len(ZONES)}}}", "\\toprule", "Forecast & " + " & ".join(ps.label(z) for z in ZONES) + " \\\\"]
    for title, M in [("MAE (EUR/MWh)", mae), ("RMSE (EUR/MWh)", rm)]:
        lines += ["\\midrule", f"\\multicolumn{{{1 + len(ZONES)}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        lines += [f"{name} & " + " & ".join(f"{M.loc[m, z]:.2f}" for z in ZONES) + " \\\\" for m, name in MODELS]
    lines += ["\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_s_benchmark_mae_rmse")
    print("best by MAE:", best.to_dict())
    print("improvement over naive D-1 (%):", imp.round(1).to_dict())
    print("beaten by per-hour LightGBM:", {k: sorted(v) for k, v in beaten.items()})


if __name__ == "__main__":
    main()
