#!/usr/bin/env python3
"""
Audit the cleaned data (data/cleaned/*_hourly.parquet), after bin/data_preprocess.py.

Checks:
  Files        every zone in bin/zones.py has a file; price-only zones have only price and calendar columns
  Index        Europe/Oslo, sorted, unique, every local hour from START to END present, 23/25-hour DST days
  Prices       no missing hours, range, negative and extreme hours, annual means
  15-minute    from 2025-10-01 the cleaned hourly price equals the mean of the raw 15-minute prices
  Neighbours   correlation and share of equal hours with the connected zones (sanity of the new series)
Prints [PASS]/[FAIL]/[INFO] lines and a final verdict.

Run from the repository root:
  python bin/audit_clean_data.py
"""
import glob
import os

import pandas as pd

from zones import END, START, ZONES

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TZ = "Europe/Oslo"
CLEAN = "data/cleaned"
CALENDAR = ["hour", "day_of_week", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
NEIGHBOURS = [("DK1", "NO2"), ("DK1", "SE3"), ("DK1", "DE_LU"), ("DK1", "DK2"), ("DK2", "SE4"), ("DK2", "DE_LU"),
              ("NL", "NO2"), ("NL", "DE_LU")]
failures = []


def report(ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    if not ok:
        failures.append(text)


def info(text):
    print(f"  [INFO] {text}")


frames = {}
print("\nFiles and columns")
for zone, z in ZONES.items():
    path = f"{CLEAN}/{zone}_hourly.parquet"
    report(os.path.exists(path), f"{zone}: {path} exists")
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    frames[zone] = df
    if z["role"] == "price_only":
        expected = ["zone", "country", "role", "price_eur_mwh"] + CALENDAR
        report(list(df.columns) == expected, f"{zone}: columns are zone, country, role, price and calendar only ({df.shape[1]} columns)")

print("\nIndex")
expected_hours = pd.date_range(pd.Timestamp(START, tz=TZ), pd.Timestamp(END, tz=TZ), freq="1h", inclusive="left")
for zone, df in frames.items():
    idx = df.index
    ok_tz = str(idx.tz) == TZ
    ok_order = idx.is_monotonic_increasing and idx.is_unique
    missing = expected_hours.difference(idx)
    extra = idx.difference(expected_hours)
    per_day = pd.Series(1, index=idx).groupby(idx.date).size()
    short, long_ = (per_day == 23).sum(), (per_day == 25).sum()
    report(ok_tz and ok_order, f"{zone}: tz {idx.tz}, sorted and unique")
    report(len(missing) == 0 and len(extra) == 0, f"{zone}: {len(idx):,} hours, missing {len(missing)}, outside the grid {len(extra)}")
    report(set(per_day.unique()) <= {23, 24, 25} and short == long_ == 8,
           f"{zone}: hours per local day in {sorted(per_day.unique())}; 23-hour days {short}, 25-hour days {long_} (expected 8 each, 2018-2025)")

print("\nPrices")
rows = []
for zone, df in frames.items():
    p = df["price_eur_mwh"]
    report(p.notna().all(), f"{zone}: no missing prices ({p.isna().sum()} missing)")
    report(p.between(-500, 4000).all(), f"{zone}: prices within the harmonised limits [-500, 4000] EUR/MWh (min {p.min():.1f}, max {p.max():.1f})")
    info(f"{zone}: negative hours {int((p < 0).sum())}, hours above 500 EUR/MWh {int((p > 500).sum())}")
    rows.append(p.groupby(p.index.year).mean().rename(zone))
annual = pd.concat(rows, axis=1).round(1)
print("\n  Annual mean price (EUR/MWh)")
print("  " + annual.to_string().replace("\n", "\n  "))

print("\n15-minute prices from 2025-10-01")
for zone, z in ZONES.items():
    if z["role"] != "price_only" or zone not in frames:
        continue
    area = z["price_areas"][0][0]
    files = sorted(glob.glob(f"data/entsoe/prices/{area}_[0-9]*.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files])
    raw.index = pd.to_datetime(raw.pop("datetime_utc"), utc=True).dt.tz_convert(TZ)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index().iloc[:, 0]
    after = raw[raw.index >= pd.Timestamp("2025-10-01", tz=TZ)]
    step = after.index.to_series().diff().dropna().mode().iloc[0]
    hourly = after.resample("1h").mean()
    clean = frames[zone]["price_eur_mwh"].reindex(hourly.index)
    diff = (hourly - clean).abs().max()
    report(step == pd.Timedelta(minutes=15), f"{zone}: raw resolution from 2025-10-01 is {step}")
    report(diff < 1e-6, f"{zone}: cleaned hourly price equals the mean of the raw 15-minute prices (max difference {diff:.2e})")

print("\nNeighbours (2018-2025, sanity of the new series)")
for a, b in NEIGHBOURS:
    if a in frames and b in frames:
        j = pd.concat([frames[a]["price_eur_mwh"].rename("a"), frames[b]["price_eur_mwh"].rename("b")], axis=1).dropna()
        info(f"{a}-{b}: correlation {j.a.corr(j.b):.3f}, equal price in {(abs(j.a - j.b) < 0.01).mean():.1%} of hours")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
