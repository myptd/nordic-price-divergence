#!/usr/bin/env python3
"""
Collect ENTSO-E data for all zones (NO1-NO5, SE1-SE4, DE_LU, and the
price-only zones DK1, DK2, NL).

Only what the models can use is downloaded:
  - day-ahead prices (target), all zones
  - day-ahead load forecast (not for price-only zones)
  - day-ahead wind and solar forecast (not for price-only zones)
  - weekly reservoir filling (16.1.D), Norway and Sweden

Data are saved per area and per year, e.g.
data/entsoe/prices/NO_1_20190101_20200101.csv, so an interrupted run can be
restarted: files that already exist are skipped. Days with missing hours are
downloaded again in short requests (full-year requests lose 00:00 on 31
December). At the end, every expected chunk is checked and a summary is printed.

Run from anywhere (the script moves to the repository root itself):
  export ENTSOE_API_KEY="..."
  python bin/collect_entsoe.py
"""
import os
import time

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import InvalidBusinessParameterError, NoMatchingDataError

from zones import END, START, ZONES

# Work from the repository root, so data/ paths are right wherever the script is started.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_KEY = os.environ.get("ENTSOE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ENTSOE_API_KEY, e.g. export ENTSOE_API_KEY='...'")

OUTDIR = "data/entsoe"
DELAY = 1.5    # seconds between calls
TIMEOUT = 120  # seconds; without it a dropped connection (e.g. laptop sleep) hangs forever
MAX_REPAIR_DAYS = 31  # larger gaps are structural (e.g. DE_LU before 2018-10-01) and are not re-queried
TZ = "Europe/Oslo"

client = EntsoePandasClient(api_key=API_KEY, timeout=TIMEOUT)


def year_chunks(start, end):
    """Split [start, end) into calendar years."""
    chunks = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s < e:
        next_year = min(pd.Timestamp(year=s.year + 1, month=1, day=1), e)
        chunks.append((s, next_year))
        s = next_year
    return chunks


def query(dataset, area, start, end):
    if dataset == "prices":
        return client.query_day_ahead_prices(area, start=start, end=end)
    if dataset == "load_forecast":
        return client.query_load_forecast(area, start=start, end=end)
    if dataset == "wind_solar_forecast":
        return client.query_wind_and_solar_forecast(area, start=start, end=end)
    if dataset == "reservoir":
        return client.query_aggregate_water_reservoirs_and_hydro_storage(area, start=start, end=end)
    raise ValueError(dataset)


def to_frame(result):
    """ENTSO-E series/frame with UTC timestamps (no DST ambiguity)."""
    df = result.to_frame("value") if isinstance(result, pd.Series) else result.copy()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"
    return df


def save(df, path):
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Write to a temporary file first, so an interrupted run never leaves a half-written file.
    df.to_csv(path + ".tmp")
    os.replace(path + ".tmp", path)


def read_saved(path):
    df = pd.read_csv(path, index_col="datetime_utc")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def missing_hours(path, cs, ce):
    """Local hours in [cs, ce) with no data in a saved file (15-min data count per hour)."""
    have = read_saved(path).index.floor("h").unique()  # floor in UTC (local DST hours are ambiguous)
    expected = pd.date_range(pd.Timestamp(cs, tz=TZ), pd.Timestamp(ce, tz=TZ), freq="1h", inclusive="left")
    return expected.tz_convert("UTC").difference(have).tz_convert(TZ)


def expected_chunks():
    """All (dataset, area, chunk_start, chunk_end) that should be downloaded."""
    chunks = []
    for zone, info in ZONES.items():
        # Prices: each area only for its own period (Germany is spliced).
        for area, area_start, area_end in info["price_areas"]:
            s = max(START, area_start or START)
            e = min(END, area_end or END)
            chunks += [("prices", area, cs, ce) for cs, ce in year_chunks(s, e)]
        # Load and wind/solar forecasts: all listed areas for the full period.
        for area in info["load_areas"]:
            for dataset in ["load_forecast", "wind_solar_forecast"]:
                chunks += [(dataset, area, cs, ce) for cs, ce in year_chunks(START, END)]
        # Weekly reservoir filling (Norway and Sweden).
        if info["reservoir_area"]:
            chunks += [("reservoir", info["reservoir_area"], cs, ce) for cs, ce in year_chunks(START, END)]
    return chunks


def chunk_path(dataset, area, cs, ce):
    return f"{OUTDIR}/{dataset}/{area}_{cs:%Y%m%d}_{ce:%Y%m%d}"


def fetch(dataset, area, cs, ce):
    """Download one area for one year chunk [cs, ce)."""
    os.makedirs(f"{OUTDIR}/{dataset}", exist_ok=True)
    base = chunk_path(dataset, area, cs, ce)
    label = f"{dataset:20s} {os.path.basename(base)}"
    if os.path.exists(base + ".csv") or os.path.exists(base + ".nodata"):
        print(f"  {label}: already downloaded")
        return

    # Ask for one extra day: entsoe-py can drop the last hour at the end of a
    # request (otherwise 23:00 on every 31 December is missing). The overlap is
    # removed in bin/data_preprocess.py.
    query_start = pd.Timestamp(cs, tz=TZ)
    query_end = pd.Timestamp(ce, tz=TZ) + pd.Timedelta(days=1)

    for attempt in range(1, 5):
        try:
            result = query(dataset, area, query_start, query_end)
            save(to_frame(result), base + ".csv")
            print(f"  {label}: OK ({len(result):,} rows)")
            break
        except (NoMatchingDataError, InvalidBusinessParameterError) as e:
            # Remember empty chunks so reruns don't ask again.
            with open(base + ".nodata", "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")
            print(f"  {label}: no data")
            break
        except Exception as e:
            print(f"  {label}: attempt {attempt} failed ({e}), retrying ...")
            time.sleep(10 * attempt)
    time.sleep(DELAY)


def repair(dataset, area, cs, ce):
    """
    Re-download days with missing hours, one short request per day, and merge
    them into the saved file. Full-year requests lose 00:00 on 31 December
    (seen in earlier downloads); short requests return it.
    Runs once per file; a .repaired note records the result.
    """
    base = chunk_path(dataset, area, cs, ce)
    if dataset == "reservoir" or not os.path.exists(base + ".csv") or os.path.exists(base + ".repaired"):
        return
    gaps = missing_hours(base + ".csv", cs, ce)
    days = sorted(set(gaps.normalize()))
    if len(days) > MAX_REPAIR_DAYS:
        print(f"  {dataset:20s} {os.path.basename(base)}: {len(days)} days without data, not re-queried (structural gap)")
        days = []
    if days:
        parts = [read_saved(base + ".csv")]
        for day in days:
            try:
                parts.append(to_frame(query(dataset, area, day, day + pd.Timedelta(days=1))))
            except (NoMatchingDataError, InvalidBusinessParameterError):
                pass
            time.sleep(DELAY)
        save(pd.concat(parts), base + ".csv")  # existing values win over re-downloaded ones
    still = missing_hours(base + ".csv", cs, ce)
    with open(base + ".repaired", "w") as f:
        f.write(f"missing hours before repair: {len(gaps)}, after: {len(still)}\n")
        f.writelines(f"{t}\n" for t in still)
    if days:
        print(f"  {dataset:20s} {os.path.basename(base)}: repaired {len(gaps) - len(still)} of {len(gaps)} missing hours")


def audit(chunks):
    """Check every expected chunk: downloaded, marked as no data, or missing.
    Hourly files are also checked for hours without data."""
    status, problems, gaps = [], [], []
    for dataset, area, cs, ce in chunks:
        base = chunk_path(dataset, area, cs, ce)
        if os.path.exists(base + ".csv"):
            status.append((dataset, "ok"))
            if dataset != "reservoir":  # reservoir is weekly
                n = len(missing_hours(base + ".csv", cs, ce))
                if n:
                    gaps.append((dataset, area, cs.year, n))
        elif os.path.exists(base + ".nodata"):
            status.append((dataset, "no data"))
        else:
            status.append((dataset, "missing"))
            problems.append(f"missing: {base}.csv")

    table = pd.DataFrame(status, columns=["dataset", "status"])
    print("\nAudit:")
    print(pd.crosstab(table["dataset"], table["status"]).to_string())
    for p in problems:
        print("  " + p)
    # Hours ENTSO-E does not provide even in short requests. These are real gaps
    # in the source; preprocessing leaves them as missing values.
    if gaps:
        print("\nHours without data inside downloaded chunks (after repair):")
        print(pd.DataFrame(gaps, columns=["dataset", "area", "year", "missing_hours"]).to_string(index=False))
    return not problems


print("=" * 60)
print(f"ENTSO-E collection, {START} to {END}")
print("=" * 60)

chunks = expected_chunks()
for dataset, area, cs, ce in chunks:
    fetch(dataset, area, cs, ce)
    repair(dataset, area, cs, ce)

if audit(chunks):
    print("All ENTSO-E data complete.")
else:
    print("Some chunks are missing or incomplete; rerun this script.")
