#!/usr/bin/env python3
"""
Fill in the observed prices (y_true) of the included forecast files.

The forecast files in model_results/ are shipped without the observed day-ahead prices, which come from the ENTSO-E
Transparency Platform and are not redistributed here. After downloading the data and building data/cleaned/
(sh_runs/01_collect_data.sh, sh_runs/02_preprocess.sh), this script adds y_true back to every forecast file, using the same 24-slot
daily matrix as the models (models/features_day_ahead.daily_matrix). Files that already have y_true are left as they are.

Run from the repository root:
  python bin/restore_actuals.py
"""
import glob
import os
import sys

import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import utils  # noqa: E402

FILES = sorted(glob.glob("model_results/rolling/*.parquet") + glob.glob("model_results/screening*/*.parquet"))


def actuals(zone):
    """Observed price per (date, hour slot) for one zone."""
    m, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(zone))[fd.TARGET])
    s = m.stack(future_stack=True).rename("y_true")
    s.index.names = ["date", "hour"]
    return s.reset_index()


def main():
    cache = {}
    for f in FILES:
        df = pd.read_parquet(f)
        if "y_true" in df.columns:
            continue
        zone = df["zone"].iloc[0]
        if zone not in cache:
            cache[zone] = actuals(zone)
        a = cache[zone]
        out = df.assign(date=pd.to_datetime(df["date"])).merge(a, on=["date", "hour"], how="left")
        out["date"] = df["date"].values
        if out["y_true"].isna().any():
            raise RuntimeError(f"{f}: observed prices missing for {out['y_true'].isna().sum()} rows")
        cols = list(df.columns)
        cols.insert(cols.index("y_pred"), "y_true")          # original column order
        out[cols].to_parquet(f, index=False)
        print(f"{f}: y_true added")


if __name__ == "__main__":
    main()
