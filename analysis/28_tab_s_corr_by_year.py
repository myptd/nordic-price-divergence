"""
Supplementary table: correlation with DE-LU by zone and year, 2018-2025, at the three time scales of Fig. 2 (hourly
prices, intraday profile, day-to-day movement). Nine Norwegian and Swedish zones ordered by 2024 coupling.

Reads results/comovement/by_year.csv and robustness_by_year.csv (analysis/01 and 02).
Output: paper_outputs/tables/tab_s_corr_by_year.tex.
Run from the repository root:  python analysis/28_tab_s_corr_by_year.py
"""
import importlib.util

import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

DIR = "results/comovement"
YEARS = list(range(2018, 2026))


def main():
    raw = pd.read_csv(f"{DIR}/by_year.csv").pivot(index="year", columns="zone", values="corr_DE_LU")
    rob = pd.read_csv(f"{DIR}/robustness_by_year.csv", index_col=[0, 1])
    blocks = [("Hourly prices", raw), ("Intraday profile", rob.loc["hourly_minus_daily_mean"]),
              ("Day-to-day movement", rob.loc["daily_minus_monthly_mean"])]
    lines = [f"\\begin{{tabular}}{{l{'r' * len(YEARS)}}}", "\\toprule", "Zone & " + " & ".join(map(str, YEARS)) + " \\\\"]
    for title, M in blocks:
        lines += ["\\midrule", f"\\multicolumn{{{1 + len(YEARS)}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        lines += [f"{ps.label(z)} & " + " & ".join(f"{M.loc[y, z]:.2f}" for y in YEARS) + " \\\\" for z in ps.NORDIC]
    lines += ["\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_s_corr_by_year")


if __name__ == "__main__":
    main()
