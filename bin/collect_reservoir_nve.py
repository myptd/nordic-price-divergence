#!/usr/bin/env python3
"""
Collect weekly Norwegian reservoir statistics from NVE Magasinstatistikk.

NVE returns the full history (1995 onwards) in one call. Norway only.
Source: https://biapi.nve.no/magasinstatistikk/swagger/index.html (NLOD licence).
Published every Wednesday; each row is dated by its measurement week.

Outputs in data/reservoir/:
  reservoir_raw.csv              all areas as returned by NVE (columns renamed)
  reservoir_min_max_median.csv   20-year weekly min/median/max per area
  reservoir_elspot.csv           NO1-NO5 with engineered features

Run from the repository root:
  python bin/collect_reservoir_nve.py
"""
import os

import numpy as np
import pandas as pd
import requests

OUTDIR = "data/reservoir"
BASE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

RENAME = {
    "dato_Id": "date", "omrType": "omr_type", "omrnr": "omr_nr",
    "iso_aar": "iso_year", "iso_uke": "iso_week",
    "fyllingsgrad": "fill_pct", "kapasitet_TWh": "capacity_twh", "fylling_TWh": "fill_twh",
    "neste_Publiseringsdato": "next_publish_date",
    "fyllingsgrad_forrige_uke": "fill_pct_prev", "endring_fyllingsgrad": "fill_change_pct",
}
RENAME_MMM = {
    "omrType": "omr_type", "omrnr": "omr_nr", "iso_uke": "iso_week",
    "minFyllingsgrad": "min_fill_pct", "medianFyllingsGrad": "median_fill_pct",
    "maxFyllingsgrad": "max_fill_pct",
}


def fetch(endpoint):
    r = requests.get(f"{BASE_URL}/{endpoint}", timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def add_features(el, mmm):
    """Engineered reservoir features per Elspot area."""
    parts = []
    for _, g in el.groupby("omr_nr"):
        g = g.sort_values("date").copy()
        g["fill_pct_100"] = (g["fill_pct"] * 100).round(2)
        g["fill_pct_yoy"] = g["fill_pct"] - g["fill_pct"].shift(52)
        g["fill_pct_4w_avg"] = g["fill_pct"].rolling(4, min_periods=1).mean()
        g["fill_trend_4w"] = g["fill_pct"].diff(4)
        g["empty_twh"] = g["capacity_twh"] - g["fill_twh"]
        g["fill_acceleration"] = g["fill_change_pct"].diff()
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    # Anomaly against the 20-year weekly median, and position between min and max.
    df = df.merge(mmm[["omr_type", "omr_nr", "iso_week", "min_fill_pct", "median_fill_pct", "max_fill_pct"]],
                  on=["omr_type", "omr_nr", "iso_week"], how="left")
    df["fill_anomaly_vs_median"] = df["fill_pct"] - df["median_fill_pct"]
    rng = df["max_fill_pct"] - df["min_fill_pct"]
    df["fill_normalized_pos"] = np.where(rng > 0, (df["fill_pct"] - df["min_fill_pct"]) / rng, 0.5)
    return df


os.makedirs(OUTDIR, exist_ok=True)

raw = fetch("HentOffentligData").rename(columns=RENAME)
raw["date"] = pd.to_datetime(raw["date"])
raw.to_csv(f"{OUTDIR}/reservoir_raw.csv", index=False)
print(f"Raw: {len(raw):,} rows, {raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}")

mmm = fetch("HentOffentligDataMinMaxMedian").rename(columns=RENAME_MMM)
mmm.to_csv(f"{OUTDIR}/reservoir_min_max_median.csv", index=False)

el = raw[raw["omr_type"] == "EL"].copy()
el = add_features(el, mmm).sort_values(["omr_nr", "date"])
el.to_csv(f"{OUTDIR}/reservoir_elspot.csv", index=False)
print(f"Elspot NO1-NO5: {len(el):,} rows -> {OUTDIR}/reservoir_elspot.csv")
