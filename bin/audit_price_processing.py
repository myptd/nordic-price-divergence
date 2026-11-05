#!/usr/bin/env python3
"""
Check that the day-ahead prices of all 13 zones went through exactly the same processing, with the price-only zones
(DK1, DK2, NL) compared with the others. Re-derives the hourly prices from the raw ENTSO-E files independently of
bin/data_preprocess.py and compares:

  resolution      modal time step of the raw prices before and from 2025-10-01
  averaging       cleaned hourly price = mean of the raw values in that hour (max absolute difference, all hours);
                  hours from 2025-10-01 built from fewer than four quarter-hours
  gaps            cleaned hours without any raw value (would mean filling), missing prices
  daylight saving the 24-slot mapping of models/features_day_ahead.daily_matrix: filled 02:00 slots only on the
                  spring-change days (value = mean of 01:00 and 03:00), 25-hour days averaged at 02:00
  code path       both zone types use build_prices(); price-only zones skip the joins (stated, see data_preprocess.py)
Prints one line per zone and check, then a verdict. Run from the repository root:
  python bin/audit_price_processing.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from zones import END, START, ZONES

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402

TZ = "Europe/Oslo"
QH = pd.Timestamp("2025-10-01", tz=TZ)
failures = []


def report(ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    if not ok:
        failures.append(text)


def raw_prices(info):
    parts = []
    for area, a0, a1 in info["price_areas"]:
        files = sorted(glob.glob(f"data/entsoe/prices/{area}_[0-9]*.csv"))
        r = pd.concat([pd.read_csv(f) for f in files])
        r.index = pd.to_datetime(r.pop("datetime_utc"), utc=True).dt.tz_convert(TZ)
        r = r[~r.index.duplicated(keep="last")].sort_index().iloc[:, 0]
        if a0:
            r = r[r.index >= pd.Timestamp(a0, tz=TZ)]
        if a1:
            r = r[r.index < pd.Timestamp(a1, tz=TZ)]
        parts.append(r)
    r = pd.concat(parts).sort_index()
    return r[(r.index >= pd.Timestamp(START, tz=TZ)) & (r.index < pd.Timestamp(END, tz=TZ))]


spring_days = None
rows = []
for zone, info in ZONES.items():
    clean = pd.read_parquet(f"data/cleaned/{zone}_hourly.parquet")["price_eur_mwh"]
    raw = raw_prices(info)
    step = raw.index.to_series().diff()
    before, after = step[raw.index < QH].mode().iloc[0], step[raw.index >= QH].mode().iloc[0]
    hourly = raw.resample("1h").mean()
    count = raw.resample("1h").count()
    both = clean.index.intersection(hourly.index)
    diff = float((clean.loc[both] - hourly.loc[both]).abs().max())
    no_raw = int((count.reindex(clean.index).fillna(0) == 0).sum())
    partial = int(((count > 0) & (count < 4) & (count.index >= QH)).sum())
    P, mask = fd.daily_matrix(clean)
    filled = (~mask).any(axis=1)
    filled_days = sorted(P.index[filled].date)
    springs = [d for d in P.index if mask.loc[d].sum() == 23]
    ok_fill = all(np.isclose(P.loc[d, 2], (P.loc[d, 1] + P.loc[d, 3]) / 2) for d in springs)
    per_day = pd.Series(1, index=clean.index).groupby(clean.index.date).size()
    long_days = list(per_day.index[per_day == 25])
    ok_long = all(np.isclose(P.loc[pd.Timestamp(d), 2], clean[(clean.index.date == d) & (clean.index.hour == 2)].mean()) for d in long_days)
    spring_days = spring_days or filled_days
    rows.append({"zone": zone, "role": info["role"], "raw_step_before": str(before), "raw_step_from_2025_10": str(after),
                 "max_abs_diff": diff, "hours_without_raw": no_raw, "partial_hours": partial, "missing_prices": int(clean.isna().sum()),
                 "filled_slots": int((~mask).sum().sum()), "filled_days": len(filled_days), "same_days_as_NO1": filled_days == spring_days,
                 "spring_fill_ok": ok_fill, "long_days": len(long_days), "long_avg_ok": ok_long})

t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
print()
for r in t.itertuples():
    report(r.raw_step_before == "0 days 01:00:00" and r.raw_step_from_2025_10 == "0 days 00:15:00",
           f"{r.zone}: raw resolution hourly before 2025-10-01, 15 minutes from then")
    report(r.max_abs_diff < 1e-9 and r.hours_without_raw == 0 and r.missing_prices == 0,
           f"{r.zone}: cleaned price = mean of raw values in every hour (max diff {r.max_abs_diff:.1e}), no filled or missing hours")
    report(r.filled_slots == 8 and r.same_days_as_NO1 and r.spring_fill_ok,
           f"{r.zone}: DST mapping fills 02:00 on the 8 spring-change days only, with the mean of 01:00 and 03:00")
    report(r.long_days == 8 and r.long_avg_ok, f"{r.zone}: 8 autumn-change days, 02:00 slot = mean of the two 02:00 prices")
    if r.partial_hours:
        print(f"  [INFO] {r.zone}: {r.partial_hours} hours from 2025-10-01 averaged over fewer than four quarter-hours")
print("\n  [INFO] code path: every zone's prices come from data_preprocess.build_prices() (read UTC files, drop duplicate "
      "stamps keeping the last, convert to Europe/Oslo, resample to 1 h by mean, cut to START-END); price-only zones then "
      "return after add_calendar(), the other zones join forecasts, weather, reservoir and commodities first.")
print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
