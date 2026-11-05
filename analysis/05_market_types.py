"""
Market-type table: per zone, country, 2024 generation shares, 2024 coupling under both decompositions and the
2025 mean day-ahead price. Zones ordered by 2024 coupling, grouped by market type, DE-LU last.

Generation shares: ENTSO-E actual generation per production type, 2024 (bin/collect_generation_entsoe.py), share of
the total over the year (consumption columns, e.g. pumped-storage pumping, excluded). Groups:
  hydro   = run-of-river and pondage + water reservoir + pumped-storage generation
  wind    = onshore + offshore;  solar;  nuclear
  thermal = fossil (all fuels) + biomass + waste
  other   = other + other renewable + geothermal (ENTSO-E "Other" in Sweden is mostly combined heat and power not
            reported by fuel)
Coupling: correlation with DE-LU in 2024 of hourly prices minus their daily mean (the frozen measure) and of daily mean
prices minus their monthly mean, from results/comovement/robustness_by_year.csv (analysis/02).
2025 mean price: mean of the hourly day-ahead prices in 2025 (Europe/Oslo calendar year), data/cleaned.

Outputs: paper_outputs/tables/tab_market_types.tex, results/market_types_2024.csv.
Run from the repository root:  python analysis/05_market_types.py
"""
import glob
import importlib.util

import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "analysis/paper_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

GROUPS = {"hydro": ["Hydro Run-of-river and pondage", "Hydro Water Reservoir", "Hydro Pumped Storage"],
          "wind": ["Wind Onshore", "Wind Offshore"], "solar": ["Solar"], "nuclear": ["Nuclear"],
          "thermal": ["Fossil Brown coal/Lignite", "Fossil Coal-derived gas", "Fossil Gas", "Fossil Hard coal", "Fossil Oil", "Biomass", "Waste"],
          "other": ["Other", "Other renewable", "Geothermal"]}
COUNTRY = {"NO": "Norway", "SE": "Sweden", "DE": "Germany--Luxembourg"}
ROB = "results/comovement/robustness_by_year.csv"


def shares(zone):
    area = zone if zone == "DE_LU" else f"{zone[:2]}_{zone[2]}"
    (path,) = glob.glob(f"data/entsoe/generation/{area}_[0-9]*.csv")
    g = pd.read_csv(path, index_col=0)
    gen = g[[c for c in g.columns if "|" not in c]].fillna(0.0)
    unknown = set(gen.columns) - {c for v in GROUPS.values() for c in v}
    assert not unknown, (zone, unknown)
    tot = gen.sum().sum()
    return {k: 100 * gen[[c for c in v if c in gen.columns]].sum().sum() / tot for k, v in GROUPS.items()}


def main():
    rob = pd.read_csv(ROB, index_col=[0, 1])
    rows = []
    for z in ps.ZONES:
        p = pd.read_parquet(f"data/cleaned/{z}_hourly.parquet", columns=["price_eur_mwh"])
        r = {"zone": z, "type": ps.TYPE[z], "country": COUNTRY[z[:2]], **shares(z),
             "coupling_intraday": rob.loc[("hourly_minus_daily_mean", 2024), z] if z != "DE_LU" else float("nan"),
             "coupling_day_to_day": rob.loc[("daily_minus_monthly_mean", 2024), z] if z != "DE_LU" else float("nan"),
             "mean_price_2025": p.loc[p.index.year == 2025, "price_eur_mwh"].mean()}
        rows.append(r)
    t = pd.DataFrame(rows)
    for z in ps.NORDIC:
        assert abs(t.set_index("zone").loc[z, "coupling_intraday"] - ps.COUPLING[z]) < 5e-4, z
    t.to_csv("results/market_types_2024.csv", index=False)
    print(t.round(3).to_string(index=False))

    def f(v, d=0):
        return "--" if pd.isna(v) else f"{v:.{d}f}"

    lines = ["\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
             " & & \\multicolumn{6}{c}{Generation share 2024 (\\%)} & \\multicolumn{2}{c}{Coupling 2024} & Mean price \\\\",
             "\\cmidrule(lr){3-8} \\cmidrule(lr){9-10}",
             "Zone & Country & Hydro & Wind & Solar & Nuclear & Thermal & Other & Intraday & Day-to-day & 2025 (EUR/MWh) \\\\"]
    for typ in ["coupled", "intermediate", "isolated", "reference"]:
        lines += ["\\midrule", f"\\multicolumn{{11}}{{l}}{{\\textit{{{ps.TYPE_NAMES[typ]}}}}} \\\\"]
        for r in t[t.type == typ].itertuples():
            lines.append(" & ".join([ps.label(r.zone), r.country, f(r.hydro), f(r.wind), f(r.solar), f(r.nuclear), f(r.thermal), f(r.other),
                                     f(r.coupling_intraday, 3), f(r.coupling_day_to_day, 3), f(r.mean_price_2025, 1)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    ps.save_table("\n".join(lines) + "\n", "tab_market_types")


if __name__ == "__main__":
    main()
