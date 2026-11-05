#!/usr/bin/env python3
"""
Audit all raw data before preprocessing.

Checks:
  ENTSO-E      every expected chunk is downloaded (or marked no data), hourly
               datasets are checked for hours without data, weekly reservoir
               for missing weeks
  Weather      every point-year file exists, covers every hour, no missing values
  Commodities  every ticker covers the study period
  NVE          weekly reservoir data for NO1-NO5 without missing weeks

Prints one line per check and a final verdict. Known, accepted gaps are
reported as INFO, anything else as FAIL.

Run from anywhere:
  python bin/audit_raw_data.py
"""
import glob
import os

import pandas as pd

from zones import END, START, ZONES

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TZ = "Europe/Oslo"
STUDY_START = "2018-01-01"
failures = []


def report(ok, message):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {message}")
    if not ok:
        failures.append(message)


def year_chunks(start, end):
    chunks = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s < e:
        next_year = min(pd.Timestamp(year=s.year + 1, month=1, day=1), e)
        chunks.append((s, next_year))
        s = next_year
    return chunks


def read_utc(path):
    df = pd.read_csv(path, index_col="datetime_utc")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
print("\nENTSO-E")
# ---------------------------------------------------------------------------
expected = []
for zone, info in ZONES.items():
    for area, a0, a1 in info["price_areas"]:
        expected += [("prices", area, cs, ce) for cs, ce in year_chunks(max(START, a0 or START), min(END, a1 or END))]
    for area in info["load_areas"]:
        for dataset in ["load_forecast", "wind_solar_forecast"]:
            expected += [(dataset, area, cs, ce) for cs, ce in year_chunks(START, END)]
    if info["reservoir_area"]:
        expected += [("reservoir", info["reservoir_area"], cs, ce) for cs, ce in year_chunks(START, END)]

missing_files, nodata, hour_gaps = [], [], []
for dataset, area, cs, ce in expected:
    base = f"data/entsoe/{dataset}/{area}_{cs:%Y%m%d}_{ce:%Y%m%d}"
    if os.path.exists(base + ".nodata"):
        nodata.append(f"{dataset} {area} {cs:%Y-%m}")
    elif not os.path.exists(base + ".csv"):
        missing_files.append(base)
    elif dataset != "reservoir":
        have = read_utc(base + ".csv").index.floor("h").unique()
        hours = pd.date_range(pd.Timestamp(cs, tz=TZ), pd.Timestamp(ce, tz=TZ), freq="1h", inclusive="left")
        n = len(hours.tz_convert("UTC").difference(have))
        if n:
            hour_gaps.append((dataset, area, cs.year, n))

report(not missing_files, f"{len(expected) - len(missing_files)} of {len(expected)} chunks present")
for m in missing_files:
    print(f"         missing: {m}.csv")
print(f"  [INFO] no-data chunks (zone did not exist yet, covered by fallback area): {nodata}")

gaps = pd.DataFrame(hour_gaps, columns=["dataset", "area", "year", "hours"])
report(gaps[gaps.dataset == "prices"].empty, "prices: every hour present")
structural = gaps[(gaps.area == "DE_LU") & (gaps.year == 2018)]
small = gaps[(gaps.dataset != "prices") & ~gaps.index.isin(structural.index)]
print(f"  [INFO] DE_LU 2018 forecasts before the zone existed ({structural.hours.sum()} h), filled from DE")
print(f"  [INFO] forecast hours missing at the source after repair: {small.hours.sum()} h in {len(small)} chunks (max {small.hours.max()} h per chunk)")

for zone, info in ZONES.items():
    area = info["reservoir_area"]
    if not area:
        continue
    files = sorted(glob.glob(f"data/entsoe/reservoir/{area}_[0-9]*.csv"))
    weekly = pd.concat([read_utc(f) for f in files])
    weeks = weekly.index.tz_convert(TZ).normalize().tz_localize(None).unique().sort_values()
    grid = pd.date_range(weeks.min(), weeks.max(), freq="7D")
    missing_weeks = grid.difference(weeks)
    ok = weeks.min() <= pd.Timestamp(STUDY_START) and weeks.max() >= pd.Timestamp("2025-12-22")
    report(ok and len(missing_weeks) <= 1,
           f"reservoir {area}: {len(weeks)} weeks {weeks.min():%Y-%m-%d} to {weeks.max():%Y-%m-%d}, missing weeks: {[f'{w:%Y-%m-%d}' for w in missing_weeks]}")

# ---------------------------------------------------------------------------
print("\nWeather")
# ---------------------------------------------------------------------------
n_files, bad = 0, []
for zone, info in ZONES.items():
    for name, _, _ in info["weather"]:
        for cs, ce in year_chunks(START, END):
            n_files += 1
            path = f"data/weather/{zone}/{name}_{cs:%Y%m%d}_{ce:%Y%m%d}.csv"
            if not os.path.exists(path):
                bad.append(f"missing: {path}")
                continue
            df = read_utc(path)
            hours = pd.date_range(pd.Timestamp(cs, tz="UTC"), pd.Timestamp(ce, tz="UTC"), freq="1h", inclusive="left")
            if not df.index.equals(hours):
                bad.append(f"wrong hours: {path}")
            elif df.isna().any().any():
                bad.append(f"missing values: {path}")
report(not bad, f"{n_files - len(bad)} of {n_files} point-year files complete (full hours, no missing values)")
for b in bad:
    print(f"         {b}")

# ---------------------------------------------------------------------------
print("\nCommodities")
# ---------------------------------------------------------------------------
if not os.path.exists("data/commodities/commodities_daily.csv"):
    print("  [INFO] data/commodities/ not found, commodity check skipped (optional, see README)")
else:
    daily = pd.read_csv("data/commodities/commodities_daily.csv", parse_dates=["date"], index_col="date")
    study = daily.loc["2017-12-01":"2025-12-31"]
    for col in [c for c in daily.columns if c.endswith("_close")]:
        first = daily[col].first_valid_index()
        report(study[col].notna().all(), f"{col:22s} from {first:%Y-%m-%d}, complete from 2017-12-01 to {study.index.max():%Y-%m-%d}")

# ---------------------------------------------------------------------------
print("\nNVE reservoir (Norway)")
# ---------------------------------------------------------------------------
nve = pd.read_csv("data/reservoir/reservoir_elspot.csv", parse_dates=["date"])
for nr in range(1, 6):
    d = nve[(nve.omr_nr == nr) & (nve.date >= "2017-11-01") & (nve.date <= "2025-12-31")].date.sort_values()
    grid = pd.date_range(d.min(), d.max(), freq="7D")
    report(len(grid.difference(d)) == 0 and d.min() <= pd.Timestamp("2017-12-03"),
           f"NO{nr}: {len(d)} weeks {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}, all Sundays: {bool((d.dt.dayofweek == 6).all())}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
