#!/usr/bin/env python3
"""
Collect daily fuel, carbon and FX prices from Yahoo Finance (yfinance).

TTF gas on Yahoo starts 2017-10-23, which is why the study starts in 2018.
Closes are dated by their trading day. The day-ahead features use the close
of D-2 (models/features_day_ahead.py).

Outputs in data/commodities/:
  {name}_daily.csv         one file per ticker (OHLCV)
  commodities_daily.csv    closes on a calendar-day index (weekends carry the
                           last close) plus engineered features; read by
                           bin/audit_raw_data.py only. The model inputs are built by
                           bin/data_preprocess.py from the {name}_daily.csv files,
                           Ember carbon and the ECB rate.

Run from the repository root:
  python bin/collect_commodities_yfinance.py
"""
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

OUTDIR = "data/commodities"
START = "2017-01-01"  # early start gives warm-up for 20-day features
END = "2026-01-01"    # exclusive

TICKERS = {
    "gas_ttf":      "TTF=F",     # Dutch TTF gas front-month futures, EUR/MWh
    "carbon_eua":   "CARB.L",    # WisdomTree Carbon ETC tracking ICE EUA futures, USD per unit; audited only, carbon input from Ember
    "coal_api2":    "MTF=F",     # Rotterdam coal (API2) front-month futures, USD/t
    "oil_brent":    "BZ=F",      # Brent crude futures, USD/bbl
    "gas_henryhub": "NG=F",      # Henry Hub gas futures, USD/MMBtu
    "fx_eurnok":    "EURNOK=X",  # EUR/NOK
}


def fetch(ticker):
    df = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"{ticker} returned no data; Yahoo may be rate-limiting, try again later.")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def add_features(df):
    """Returns, moving averages, volatility, logs and fuel ratios ."""
    for name in TICKERS:
        s = df[f"{name}_close"]
        df[f"{name}_return"] = s.pct_change()
        df[f"{name}_ma5"] = s.rolling(5, min_periods=1).mean()
        df[f"{name}_ma20"] = s.rolling(20, min_periods=1).mean()
        df[f"{name}_vol20"] = s.pct_change().rolling(20, min_periods=1).std()
        df[f"{name}_log"] = np.log(s.clip(lower=0.01))
    df["ratio_gas_carbon"] = df["gas_ttf_close"] / df["carbon_eua_close"].clip(lower=0.01)
    df["ratio_gas_oil"] = df["gas_ttf_close"] / df["oil_brent_close"].clip(lower=0.01)
    df["ratio_gas_coal"] = df["gas_ttf_close"] / df["coal_api2_close"].clip(lower=0.01)
    df["gas_momentum"] = df["gas_ttf_ma5"] - df["gas_ttf_ma20"]
    df["carbon_momentum"] = df["carbon_eua_ma5"] - df["carbon_eua_ma20"]
    return df


os.makedirs(OUTDIR, exist_ok=True)

closes = {}
for name, ticker in TICKERS.items():
    df = fetch(ticker)
    df.to_csv(f"{OUTDIR}/{name}_daily.csv")
    closes[f"{name}_close"] = df["Close"]
    print(f"  {name:13s} {ticker:9s} {len(df):5,} rows  {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
    time.sleep(2)

# Features are computed on trading days, then carried over
# weekends/holidays so every calendar day has the latest close on or before it.
trading = add_features(pd.DataFrame(closes).sort_index().ffill())
calendar = pd.date_range(trading.index.min(), pd.Timestamp(END) - pd.Timedelta(days=1), freq="D", name="date")
daily = trading.reindex(calendar).ffill()
daily.to_csv(f"{OUTDIR}/commodities_daily.csv")
print(f"Combined: {daily.shape} -> {OUTDIR}/commodities_daily.csv")
