#!/usr/bin/env python3
"""
Download the ECB euro reference rate (US dollars per euro, daily) used to convert API2 coal to EUR/t.

Saves data/commodities/fx_usdeur_ecb.csv (data/commodities/ is not part of the repository; each user downloads it).
Run from the repository root:  python bin/collect_fx_ecb.py
"""
import io
import os
import urllib.request

import pandas as pd

OUTDIR = "data/commodities"
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
           "?format=csvdata&startPeriod=2017-01-01&endPeriod=2025-12-31")


def ecb_usd_eur():
    req = urllib.request.Request(ECB_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"], parse_dates=["TIME_PERIOD"])
    return df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "usd_per_eur"}).set_index("date").sort_index()


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    fx = ecb_usd_eur()
    fx.to_csv(f"{OUTDIR}/fx_usdeur_ecb.csv")
    print(f"  fx_usdeur_ecb  {len(fx):5,} rows  {fx.index[0]:%Y-%m-%d} to {fx.index[-1]:%Y-%m-%d}")
