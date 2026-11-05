"""
Correlation with DE-LU by year after removing seasonal and intraday patterns, and descriptive checks of 2024.

Correlation with DE-LU by year of
  raw_hourly                 hourly prices
  hourly_minus_daily_mean    hourly prices minus their daily mean (intraday profile only); the 2024 value is the
                             frozen coupling measure that defines the market types
  daily_minus_monthly_mean   daily mean prices minus their calendar-month mean (day-to-day movement)
Checks of 2024 (text only): monthly correlation with DE-LU 2023-2025; 2024 before and after Nordic flow-based market
coupling (first delivery day 30 October 2024); NVE reservoir anomaly by zone and year; price equality and spreads
inside Norway; mean within-day standard deviation and negative-price hours.

Outputs: results/comovement/robustness_by_year.csv, results/comovement/diag_2024.txt.
Run from the repository root:  python analysis/02_comovement_robustness.py
"""
import os

import pandas as pd

DIR = "results/comovement"
ZONES = ["NO1", "NO2", "NO5", "NO3", "NO4", "SE3", "SE4", "SE1", "SE2", "DK1", "DK2", "NL"]
ALL = ZONES + ["DE_LU"]
NO = ["NO1", "NO2", "NO3", "NO4", "NO5"]
FB = pd.Timestamp("2024-10-30", tz="Europe/Oslo")          # first delivery day under flow-based market coupling


def load():
    p = pd.concat({z: pd.read_parquet(f"data/cleaned/{z}_hourly.parquet")["price_eur_mwh"] for z in ALL}, axis=1)
    return p[(p.index.year >= 2018) & (p.index.year <= 2025)]


def corr_de(frame, years):
    return pd.DataFrame({y: frame[frame.index.year == y].corr()["DE_LU"] for y in years}).T[ZONES]


def main():
    p = load()
    years = list(range(2018, 2026))
    day = p.index.normalize()
    intraday = p - p.groupby(day).transform("mean")
    daily = p.groupby(day).mean()
    daily.index = pd.DatetimeIndex(daily.index)
    within_month = daily - daily.groupby([daily.index.year, daily.index.month]).transform("mean")
    A, B, raw = corr_de(intraday, years), corr_de(within_month, years), corr_de(p, years)
    os.makedirs(DIR, exist_ok=True)
    pd.concat({"raw_hourly": raw, "hourly_minus_daily_mean": A, "daily_minus_monthly_mean": B}).to_csv(f"{DIR}/robustness_by_year.csv")

    pd.set_option("display.width", 220)
    lines = ["Correlation with DE-LU by year: raw hourly prices", raw.round(2).to_string(), "",
             "(a) hourly prices minus their daily mean", A.round(2).to_string(), "",
             "(b) daily mean prices minus their calendar-month mean", B.round(2).to_string(), ""]
    m = p[(p.index.year >= 2023)]
    monthly = {f"{k[0]}-{k[1]:02d}": g.corr()["DE_LU"][NO] for k, g in m.groupby([m.index.year, m.index.month])}
    lines += ["Monthly correlation of hourly prices with DE-LU, Norwegian zones, 2023-2025", pd.DataFrame(monthly).T.round(2).to_string(), ""]
    y24 = p[p.index.year == 2024]
    lines += ["2024 before and after flow-based market coupling (first delivery day 30 October 2024)",
              pd.DataFrame({"2024-01-01..10-29": y24[y24.index < FB].corr()["DE_LU"], "2024-10-30..12-31": y24[y24.index >= FB].corr()["DE_LU"]})
              .loc[NO + ["SE1", "SE2", "SE3", "SE4"]].round(2).to_string(), ""]
    nve = pd.concat({z: pd.read_parquet(f"data/cleaned/{z}_hourly.parquet")["nve_fill_anomaly_vs_median"] for z in NO}, axis=1)
    nve = nve[(nve.index.year >= 2018) & (nve.index.year <= 2025)]
    lines += ["NVE reservoir fill anomaly vs the 20-year weekly median (fraction of capacity, 0.10 = 10 percentage points)",
              "annual mean", nve.groupby(nve.index.year).mean().round(3).to_string(),
              "monthly mean, 2023-2025", nve[nve.index.year >= 2023].groupby([nve.index[nve.index.year >= 2023].year,
              nve.index[nve.index.year >= 2023].month]).mean().round(3).to_string(), ""]
    pairs = [("NO1", "NO2"), ("NO2", "NO5"), ("NO1", "NO5"), ("NO1", "NO3"), ("NO5", "NO3"), ("NO3", "NO4")]
    eq = pd.DataFrame({f"{a}={b}": ((p[a] - p[b]).abs() < 0.01).groupby(p.index.year).mean() for a, b in pairs})
    spread = pd.DataFrame({f"{a}-{b}": (p[a] - p[b]).abs().groupby(p.index.year).mean() for a, b in pairs})
    lines += ["Share of hours with equal prices inside Norway", eq.round(2).to_string(), "",
              "Mean absolute price difference inside Norway (EUR/MWh)", spread.round(1).to_string(), ""]
    dstd = p.groupby(day).std().groupby(lambda t: t.year).mean()[["DE_LU", "NO1", "NO2", "NO5", "NO3", "NO4"]]
    neg = (p[["DE_LU", "NO2", "NO5"]] < 0).groupby(p.index.year).sum()
    lines += ["Mean within-day standard deviation of hourly prices (EUR/MWh)", dstd.round(1).to_string(), "",
              "Negative-price hours per year", neg.to_string()]
    open(f"{DIR}/diag_2024.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
