#!/usr/bin/env python3
"""
Collect ENTSO-E actual generation per production type (data item 16.1.B&C) for 2024, for the ten modelling zones
(NO1-NO5, SE1-SE4, DE-LU). Used only for the generation shares of the market-type table and the peak wind generation in analysis/17;
never a model input.

One request per zone (calendar year 2024, local time), saved to data/entsoe/generation/{AREA}_20240101_20250101.csv
with UTC timestamps and one column per production type (and, where ENTSO-E reports it, the consumption of that type,
e.g. pumped storage, as "<type> | Actual Consumption"). Existing files are skipped, so an interrupted run can be
restarted. About 15 MB in total.

Run from anywhere (the script moves to the repository root itself), on a machine with internet access:
  export ENTSOE_API_KEY="..."
  python bin/collect_generation_entsoe.py
"""
import os
import time

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_KEY = os.environ.get("ENTSOE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ENTSOE_API_KEY, e.g. export ENTSOE_API_KEY='...'")

AREAS = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5", "SE_1", "SE_2", "SE_3", "SE_4", "DE_LU"]
START, END = pd.Timestamp("2024-01-01", tz="Europe/Oslo"), pd.Timestamp("2025-01-01", tz="Europe/Oslo")
OUTDIR = "data/entsoe/generation"
client = EntsoePandasClient(api_key=API_KEY, timeout=120)


def flatten(df):
    """One column per production type; the consumption part of a type (if any) gets its own column."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [a if b == "Actual Aggregated" else f"{a} | {b}" for a, b in df.columns]
    df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"
    return df


os.makedirs(OUTDIR, exist_ok=True)
for area in AREAS:
    path = f"{OUTDIR}/{area}_{START:%Y%m%d}_{END:%Y%m%d}.csv"
    if os.path.exists(path):
        print(f"  {area}: already downloaded")
        continue
    for attempt in range(1, 5):
        try:
            df = flatten(client.query_generation(area, start=START, end=END))
            df.to_csv(path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"  {area}: OK ({len(df):,} rows, {df.shape[1]} columns: {', '.join(df.columns)})")
            break
        except NoMatchingDataError:
            print(f"  {area}: no data")
            break
        except Exception as e:
            print(f"  {area}: attempt {attempt} failed ({e}), retrying ...")
            time.sleep(10 * attempt)
    time.sleep(1.5)
print("Done.")
