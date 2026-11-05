"""
Table of the three hypotheses on the neighbouring prices with their tests and 2025 outcomes.

Reads results/crossborder_2025.csv, results/crossborder_2025_tests.txt (analysis/08) and results/h3_2025.csv
(analysis/10). H1's rho and exact p are recomputed with the same exact permutation as analysis/08 and checked against
the text file.
Output: paper_outputs/tables/tab_hypotheses.tex.
Run from the repository root:  python analysis/27_tab_hypotheses.py
"""
import importlib.util
import itertools
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def spearman_exact(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    perms = np.array([np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)])
    return rho, float(np.mean(perms >= rho - 1e-12))


def main():
    x = pd.read_csv("results/crossborder_2025.csv")
    rho, p1 = spearman_exact(x.coupling_2024, x.delta_rMAE)
    txt = open("results/crossborder_2025_tests.txt").read()
    m = re.search(r"rho\(delta, coupling\) = (-?[0-9.]+); exact one-sided p = ([0-9.]+)", txt)
    assert abs(rho - float(m.group(1))) < 5e-4 and abs(p1 - float(m.group(2))) < 5e-4, (rho, p1, m.groups())
    h2 = x[x.split == "below median"].set_index("zone").reindex(["NO3", "SE1", "SE2", "NO4"])
    h3 = pd.read_csv("results/h3_2025.csv").set_index("zone")
    pooled = h3.loc["pooled (inverse-variance)"]
    zones3 = ["SE4", "NO2", "NO1", "SE3"]

    def pz(v):
        return "<0.001" if v < 0.001 else f"{v:.3f}"

    h2cells = "; ".join(f"{z} {h2.loc[z, 'p_with_better_holm_H2']:.3f}" for z in h2.index)
    h3cells = "; ".join(f"{z} {h3.loc[z, 'theta']:+.2f} ({h3.loc[z, 'se']:.2f})" for z in zones3)
    rows = [
        ("H1", "The gain from neighbouring prices increases with coupling (nine Nordic zones)",
         "Spearman rank correlation, exact one-sided permutation test",
         f"$\\rho = {rho:.2f}$, $p = {p1:.3f}$", "Not supported"),
        ("H2", "No significant gain in the four zones below the median (NO3, SE1, SE2, NO4)",
         "One-sided multivariate DM per zone, Holm over the four zones",
         f"Holm $p$: {h2cells}", "Holds (no detectable gain)"),
        ("H3", "In the four zones above the median, neighbouring prices reduce the gas-related error",
         "Double machine learning, pooled inverse-variance estimate, one-sided $z$-test",
         f"Pooled $\\theta = {pooled.theta:.2f}$ (s.e.\\ {pooled.se:.2f}), $p = {pz(pooled.p_one_sided)}$; zones: {h3cells}",
         "Not supported"),
    ]
    lines = ["\\begin{tabular}{@{}p{0.6cm}p{3.9cm}p{3.6cm}p{5.2cm}p{2.1cm}@{}}", "\\toprule",
             "& Expectation & Test & Result (2025) & Outcome \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\[3pt]" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_hypotheses")
    print(f"H1 rho {rho:.3f} p {p1:.4f}; H3 pooled {pooled.theta:.4f} se {pooled.se:.4f} p {pooled.p_one_sided:.4f}")


if __name__ == "__main__":
    main()
