#!/usr/bin/env python3
"""
Collect hourly historical weather for all zones with weather points (not the price-only zones) from the Open-Meteo archive
(ERA5 reanalysis). No API key required.

Each weather point in bin/zones.py is downloaded per year, e.g.
data/weather/NO1/oslo_20190101_20200101.csv. Existing files are skipped, so an
interrupted run can be restarted. At the end, every expected file is checked
again (full hourly coverage, no missing values) and a summary is printed.
The preprocessing script averages the points of a zone into one series.

Free tier limits: 600 calls/min, 5000/hour, 10000/day. A one-year request
counts as ~50 calls, so the full download (153 requests) hits the hourly limit
once; the script then waits and continues by itself (~2 hours in total).

Run from anywhere (the script moves to the repository root itself):
  python bin/collect_weather_openmeteo.py
"""
import os
import time

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from zones import END, START, ZONES

# Work from the repository root, so data/ paths are right wherever the script is started.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = "data/weather"
DELAY = 10         # seconds between requests
LIMIT_WAIT = 600   # seconds to wait when the hourly API limit is reached
API_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "pressure_msl", "cloud_cover",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
]

session = requests_cache.CachedSession(".cache_openmeteo", expire_after=-1)
om = openmeteo_requests.Client(session=retry(session, retries=5, backoff_factor=1.0))


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


def fetch_point(lat, lon, start, end):
    """Hourly weather for [start, end) at one point, UTC timestamps."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start:%Y-%m-%d}",
        "end_date": f"{end - pd.Timedelta(days=1):%Y-%m-%d}",  # Open-Meteo end date is inclusive
        "hourly": HOURLY_VARIABLES,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    hourly = om.weather_api(API_URL, params=params)[0].Hourly()
    index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
        name="datetime_utc",
    )
    data = {var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(HOURLY_VARIABLES)}
    df = pd.DataFrame(data, index=index)

    # Check the response before it is saved, so a bad chunk is never skipped later.
    expected = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1h", inclusive="left")
    if not df.index.equals(expected):
        raise ValueError(f"expected {len(expected)} hours from {start:%Y-%m-%d}, got {len(df)}")
    if df.isna().any().any():
        raise ValueError(f"missing values in {df.columns[df.isna().any()].tolist()}")
    return df


def audit():
    """Re-check every expected file: present, full hourly coverage, no missing values."""
    n_ok, problems = 0, []
    for zone, info in ZONES.items():
        for name, _, _ in info["weather"]:
            for cs, ce in year_chunks(START, END):
                path = f"{OUTDIR}/{zone}/{name}_{cs:%Y%m%d}_{ce:%Y%m%d}.csv"
                if not os.path.exists(path):
                    problems.append(f"missing: {path}")
                    continue
                df = pd.read_csv(path, index_col="datetime_utc")
                df.index = pd.to_datetime(df.index, utc=True)
                expected = pd.date_range(pd.Timestamp(cs, tz="UTC"), pd.Timestamp(ce, tz="UTC"), freq="1h", inclusive="left")
                if not df.index.equals(expected):
                    problems.append(f"wrong hours ({len(df)} of {len(expected)}): {path}")
                elif df.isna().any().any():
                    problems.append(f"missing values: {path}")
                else:
                    n_ok += 1
    print(f"\nAudit: {n_ok} of {n_ok + len(problems)} files complete.")
    for p in problems:
        print("  " + p)
    return not problems


print("=" * 60)
print(f"Open-Meteo weather collection, {START} to {END}")
print("=" * 60)

for zone, info in ZONES.items():
    os.makedirs(f"{OUTDIR}/{zone}", exist_ok=True)
    for name, lat, lon in info["weather"]:
        for cs, ce in year_chunks(START, END):
            path = f"{OUTDIR}/{zone}/{name}_{cs:%Y%m%d}_{ce:%Y%m%d}.csv"
            if os.path.exists(path):
                print(f"  {zone} {name} {cs:%Y}: already downloaded")
                continue
            while True:
                try:
                    df = fetch_point(lat, lon, cs, ce)
                    break
                except Exception as e:
                    # A one-year request counts as ~50 API calls, so the hourly
                    # limit is hit after ~100 requests. Wait and try again.
                    # (The daily limit gives the same message; the script then
                    # keeps waiting until the next day. Stop it with Ctrl+C and rerun later.)
                    if "limit exceeded" not in str(e):
                        raise
                    print(f"  rate limit reached, waiting {LIMIT_WAIT // 60} min ...", flush=True)
                    time.sleep(LIMIT_WAIT)
            # Write to a temporary file first, so an interrupted run never leaves a half-written file.
            df.to_csv(path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"  {zone} {name} {cs:%Y-%m-%d} -> {ce:%Y-%m-%d}: OK ({len(df):,} rows)")
            time.sleep(DELAY)

if audit():
    print("All weather data complete.")
else:
    print("Some files are missing or incomplete; rerun this script.")
