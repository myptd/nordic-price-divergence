#!/usr/bin/env python3
"""
Merge the raw data into one hourly dataset per zone.

Price-only zones (DK1, DK2, NL; role "price_only" in bin/zones.py) get steps 1
and 6 only: hourly prices (15-minute prices averaged, same local time and DST
handling as all other zones) and calendar columns.

For each zone:
  1. Prices define the hourly index. Germany is spliced from DE-AT-LU
     (before 2018-10-01) and DE-LU. Since 2025-10-01 prices are 15-minute,
     so all ENTSO-E series are averaged to hourly.
  2. Load and wind/solar forecasts are joined (later areas fill gaps).
  3. Weather points of the zone are averaged and joined.
  4. Reservoir, two sources with different roles:
     res_*  model features for Norway and Sweden, from ENTSO-E (16.1.D):
            stored energy and its 1- and 4-week change. ENTSO-E stamps a
            week at its Monday; the value is the level at the end of that
            week, so it is usable from the next Monday (stamp + 7 days).
     nve_*  Norwegian regime analysis only (never model features), from NVE:
            fill share and anomaly vs NVE's 20-year weekly median. NVE dates
            a week at its Sunday and measures on Sunday at 24:00 (NVE
            handbook 2023, p. 8), so it is usable from Monday (date + 1 day).
  5. Commodities: each hour gets the close of its local calendar day
     (weekends carry the last close). Built here from the raw files: Yahoo
     gas, coal, oil and FX, Ember carbon (EUR/t), coal in EUR/t at the ECB
     rate, roll-adjusted returns for TTF and coal (load_commodities).
  6. Calendar features.

Values are only aligned to the time their measurement is complete; no
information-set lagging happens here. The day-ahead information set (what is known at
D-1 11:00) is applied when models build their features (models/utils.py).

Outputs in data/cleaned/:
  {ZONE}_hourly.parquet   one file per zone
  panel_hourly.parquet    all zones stacked
  coverage.csv            share of non-missing values per zone, column and year

Run from the repository root:
  python bin/data_preprocess.py            # all zones
  python bin/data_preprocess.py NO1 SE3    # only some zones (panel/coverage cover these only)
  python bin/data_preprocess.py --check    # data quality guard only (raw files and data/cleaned/), writes nothing

Data quality guard (runs first and stops the run on failure):
  - daily commodity closes (Yahoo, Ember carbon, ECB USD/EUR): fail on more than 5 weekdays of identical values or
    of forward fill; list every run of identical closes longer than 2 weekdays; zero volume per series and year
    ("volume not reported" when at least 95% of rows are zero, otherwise warn above 10%);
  - ENTSO-E weekly reservoir: fail on more than one missing week; identical values for up to 4 weeks only warn,
    longer runs fail (SE_4: warning only);
  - hourly columns that reach the models (load forecast, wind/solar forecasts missing on at most 5% of hours,
    weather): fail on more than 120 hours of an identical non-zero value or of interior gaps; snow depth, soil
    moisture, soil temperature and cloud cover at 0 or 100% only warn.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from zones import END, START, ZONES

TZ = "Europe/Oslo"  # NO, SE, DE, DK and NL share CET/CEST
OUTDIR = "data/cleaned"

NVE_COLS = ["fill_pct", "fill_anomaly_vs_median"]  # regime analysis only


def read_utc_csv(path):
    """Read a raw file with a datetime_utc column and index it in local time."""
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop("datetime_utc"), utc=True).dt.tz_convert(TZ)
    df.index.name = "datetime"
    return df


def load_entsoe(dataset, area):
    """All yearly files of one ENTSO-E dataset and area, averaged to hourly."""
    # "[0-9]" keeps e.g. DE_* from also matching DE_LU_* files.
    files = sorted(glob.glob(f"data/entsoe/{dataset}/{area}_[0-9]*.csv"))
    if not files:
        return None
    df = pd.concat([read_utc_csv(f) for f in files])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.resample("1h").mean()


def slug(text):
    return "_".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def build_prices(info):
    parts = []
    for area, area_start, area_end in info["price_areas"]:
        df = load_entsoe("prices", area)
        if df is None:
            raise FileNotFoundError(f"No price files for {area}; run bin/collect_entsoe.py first")
        if area_start:
            df = df[df.index >= pd.Timestamp(area_start, tz=TZ)]
        if area_end:
            df = df[df.index < pd.Timestamp(area_end, tz=TZ)]
        parts.append(df)
    prices = pd.concat(parts).sort_index()
    prices.columns = ["price_eur_mwh"]
    return prices.dropna()


def build_forecasts(info):
    """Load forecast and wind/solar forecast, combining areas in priority order."""
    load, wsf = None, None
    for area in info["load_areas"]:
        lf = load_entsoe("load_forecast", area)
        if lf is not None:
            lf = lf.iloc[:, [0]].set_axis(["load_forecast_mw"], axis=1)
            load = lf if load is None else load.combine_first(lf)
        ws = load_entsoe("wind_solar_forecast", area)
        if ws is not None:
            ws.columns = ["wsf_" + slug(c) for c in ws.columns]
            wsf = ws if wsf is None else wsf.combine_first(ws)
    return load, wsf


def build_weather(zone, info):
    """Average all weather points of a zone."""
    points = []
    for name, _, _ in info["weather"]:
        files = sorted(glob.glob(f"data/weather/{zone}/{name}_*.csv"))
        if files:
            points.append(pd.concat([read_utc_csv(f) for f in files]))
    if not points:
        return None
    weather = pd.concat(points).groupby(level=0).mean()
    weather.columns = ["wx_" + c for c in weather.columns]
    return weather


def to_hours(hourly_index, daily):
    """Give each hour the latest daily row dated on or before its local date."""
    dates = pd.DatetimeIndex(hourly_index.tz_convert(TZ).date)
    calendar = pd.date_range(min(daily.index.min(), dates.min()), dates.max(), freq="D")
    daily = daily.reindex(daily.index.union(calendar)).sort_index().ffill()
    out = daily.reindex(dates)
    out.index = hourly_index
    return out


def load_nve(nr):
    """NVE weekly fill share and anomaly, dated at the day they become usable (Monday)."""
    df = pd.read_csv("data/reservoir/reservoir_elspot.csv", parse_dates=["date"])
    df = df[df["omr_nr"] == nr].set_index("date").sort_index()[NVE_COLS]
    df.index = df.index + pd.Timedelta(days=1)   # measured Sunday 24:00
    df.columns = ["nve_" + c for c in df.columns]
    return df


def load_entsoe_reservoir(area):
    """ENTSO-E weekly stored energy and its changes, dated at the day they become usable."""
    files = sorted(glob.glob(f"data/entsoe/reservoir/{area}_[0-9]*.csv"))
    if not files:
        return None
    weekly = pd.concat([read_utc_csv(f) for f in files])
    weekly = weekly[~weekly.index.duplicated(keep="last")].sort_index().iloc[:, 0]
    weekly.index = pd.DatetimeIndex(weekly.index.date)            # Monday of each week
    # Regular weekly grid; a single missing week (SE, 2020 week 53) carries the previous value.
    grid = pd.date_range(weekly.index.min(), weekly.index.max(), freq="7D")
    stored = weekly.reindex(grid).ffill() / 1e6                     # MWh -> TWh
    df = pd.DataFrame({
        "res_stored_twh": stored,
        "res_change_1w_twh": stored.diff(1),
        "res_change_4w_twh": stored.diff(4),
    })
    df.index = df.index + pd.Timedelta(days=7)                    # level at the end of the week
    return df


# Commodity inputs.
# Yahoo Finance closes (data/commodities/{name}_daily.csv) for gas, coal, oil and FX; carbon from Ember (EU ETS front-December, EUR/t), because the Yahoo
# carbon ETC (CARB.L) has a 15-month frozen run and many stale closes; ECB USD/EUR to express coal in EUR/t.
CARBON_FILE = "data/commodities/carbon_eua_ember.csv"     # EU ETS front-December price, EUR/t (see README)
USDEUR_FILE = "data/commodities/fx_usdeur_ecb.csv"        # bin/collect_fx_ecb.py
COMMODITY_ORDER = ["gas_ttf", "carbon_eua", "coal_api2", "oil_brent", "gas_henryhub", "fx_eurnok"]
ROLLED = ["gas_ttf", "coal_api2"]                          # continuous front-month futures that roll every month

# Data quality guard. The main test is runs of identical values: Yahoo returned flat
# placeholder rows for CARB.L for 15 months, which a limit on forward filling alone would not catch.
MAX_STALE_WEEKDAYS = 5      # daily series: fail above this many weekdays of identical values or of forward fill
REPORT_RUN_WEEKDAYS = 2     # daily series: list every run of identical closes longer than this
MAX_STALE_HOURS = 5 * 24    # hourly inputs: fail above this many hours of an identical non-zero value or of gaps
NO_VOLUME_SHARE = 0.95      # zero volume on at least this share of a year's rows: volume not reported (no warning)
ZERO_VOLUME_WARN = 0.10     # otherwise warn above this share of zero-volume rows
RESERVOIR_WARN_WEEKS = 4    # weekly reservoir: identical runs up to this many weeks only warn, longer runs fail
RESERVOIR_WARN_ONLY = {"SE_4"}   # small reservoir reported on a 1,000 MWh grid: identical runs only warn
GUARD_START = pd.Timestamp(START) - pd.Timedelta(days=31)   # 20-day features of the first delivery days
WSF_MAX_MISSING = 0.05      # as MAX_MISSING in models/features_day_ahead.py: sparser wind/solar columns never reach the models
WX_WARN_ONLY = {"wx_snow_depth", "wx_soil_moisture_0_to_7cm", "wx_soil_temperature_0_to_7cm"}   # slow weather states


def runs(values):
    """Start position and length of each run of identical consecutive values (NaN never matches)."""
    v = pd.Series(values)
    new = v.ne(v.shift()) | v.isna()
    ids = new.cumsum()
    lengths = ids.map(ids.value_counts())
    first = new.to_numpy()
    return [(i, int(lengths.iloc[i])) for i in np.flatnonzero(first)]


def check_daily(name, raw):
    """Identical closes and forward-fill length on the weekday grid, and zero volume per year (if the series has a
    volume). Returns failures."""
    raw = raw[(raw.index >= GUARD_START) & (raw.index < pd.Timestamp(END))]
    weekdays = pd.bdate_range(raw.index.min(), raw.index.max())
    close = raw["Close"].reindex(weekdays)
    fails = []
    gaps = [(i, n) for i, n in runs(close.isna().map({True: 1, False: np.nan})) if close.iloc[i:i + n].isna().all()]
    for i, n in gaps:
        if n > MAX_STALE_WEEKDAYS:
            fails.append(f"{name}: forward fill would cover {n} weekdays from {weekdays[i]:%Y-%m-%d}")
    filled = close.ffill()
    for i, n in runs(filled.round(6)):
        if n > REPORT_RUN_WEEKDAYS:
            line = f"{name}: identical close {filled.iloc[i]:.4f} for {n} weekdays, {weekdays[i]:%Y-%m-%d} to {weekdays[i + n - 1]:%Y-%m-%d}"
            print(("  FAIL  " if n > MAX_STALE_WEEKDAYS else "  run   ") + line)
            if n > MAX_STALE_WEEKDAYS:
                fails.append(line)
    if "Volume" in raw:
        zero = (raw["Volume"] == 0).groupby(raw.index.year).agg(["mean", "size"])
        for year, (share, rows) in zero.iterrows():
            if share >= NO_VOLUME_SHARE:
                kind = "volume not reported"
            elif share > 0:
                kind = "partial zero volume" + ("  WARN" if share > ZERO_VOLUME_WARN else "")
            else:
                kind = "volume reported"
            print(f"  volume {name:13s} {year}  zero on {share:6.1%} of {rows} rows  {kind}")
    return fails


def commodity_files_present():
    """True when every raw commodity file (Yahoo Finance, carbon, ECB rate) is in data/commodities/."""
    files = [CARBON_FILE, USDEUR_FILE] + [f"data/commodities/{n}_daily.csv" for n in COMMODITY_ORDER if n != "carbon_eua"]
    return all(os.path.exists(f) for f in files)


def read_daily(name):
    """Raw daily closes of one commodity input (Yahoo OHLCV file, Ember carbon, or the ECB rate as `Close`)."""
    if name == "carbon_eua":
        return pd.read_csv(CARBON_FILE, parse_dates=["date"], index_col="date").sort_index()
    if name == "usd_eur":
        return pd.read_csv(USDEUR_FILE, parse_dates=["date"], index_col="date").rename(columns={"usd_per_eur": "Close"})
    return pd.read_csv(f"data/commodities/{name}_daily.csv", parse_dates=["date"], index_col="date").sort_index()


def check_commodities():
    """Guard on the raw commodity files actually used (Yahoo, Ember carbon, ECB rate)."""
    fails = []
    for name in COMMODITY_ORDER + ["usd_eur"]:
        fails += check_daily(name, read_daily(name))
    return fails


def check_reservoir():
    """ENTSO-E weekly stored energy: fail on more than one missing week; identical runs counted in weeks warn up to
    RESERVOIR_WARN_WEEKS and fail above it, except in RESERVOIR_WARN_ONLY, where they only warn."""
    fails = []
    for area in sorted({i["reservoir_area"] for i in ZONES.values() if i.get("reservoir_area")}):
        files = sorted(glob.glob(f"data/entsoe/reservoir/{area}_[0-9]*.csv"))
        weekly = pd.concat([read_utc_csv(f) for f in files])
        weekly = weekly[~weekly.index.duplicated(keep="last")].sort_index().iloc[:, 0]
        weekly.index = pd.DatetimeIndex(weekly.index.date)
        grid = pd.date_range(weekly.index.min(), weekly.index.max(), freq="7D")
        w = weekly.reindex(grid)
        for i, n in runs(w.isna().map({True: 1, False: np.nan})):
            if w.iloc[i:i + n].isna().all() and n > 1:
                fails.append(f"reservoir {area}: {n} missing weeks from {grid[i]:%Y-%m-%d}")
        for i, n in runs(w.round(3)):
            if n > 1 and pd.notna(w.iloc[i]):
                line = f"reservoir {area}: identical value for {n} weeks from {grid[i]:%Y-%m-%d}"
                if n > RESERVOIR_WARN_WEEKS and area not in RESERVOIR_WARN_ONLY:
                    fails.append(line)
                else:
                    print("  WARN  " + line)
    return fails


def check_hourly(zone, df):
    """Hourly columns that reach the models (load forecast, wind/solar forecasts kept by
    models/features_day_ahead.prepare_hourly, weather): identical non-zero values or interior gaps longer than
    MAX_STALE_HOURS fail. Runs of zeros are allowed (no solar at night or in winter, NO5 wind). Slow or bounded
    weather variables (WX_WARN_ONLY, and cloud cover at 0 or 100%) can stay constant for days in the reanalysis, so
    their runs are listed as warnings."""
    fails = []
    cols = ["load_forecast_mw"] + [c for c in df.columns if c.startswith("wx_")]
    cols += [c for c in df.columns if c.startswith("wsf_") and df[c].isna().mean() <= WSF_MAX_MISSING]
    for c in [c for c in cols if c in df.columns]:
        s = df[c]
        inside = s.loc[s.first_valid_index():s.last_valid_index()] if s.notna().any() else s.iloc[:0]
        for i, n in runs(inside.isna().map({True: 1, False: np.nan})):
            if n > MAX_STALE_HOURS and inside.iloc[i:i + n].isna().all():
                fails.append(f"{zone} {c}: gap of {n} hours from {inside.index[i]:%Y-%m-%d %H:%M}")
        for i, n in runs(inside.round(6)):
            v = inside.iloc[i]
            if n > MAX_STALE_HOURS and pd.notna(v) and v != 0:
                line = f"{zone} {c}: identical value {v:.3f} for {n} hours from {inside.index[i]:%Y-%m-%d %H:%M}"
                if c in WX_WARN_ONLY or (c == "wx_cloud_cover" and v in (0, 100)):
                    print("  WARN  " + line)
                else:
                    fails.append(line)
    return fails


def guard(fails, what):
    if fails:
        print(f"\nDATA QUALITY GUARD FAILED ({what}):")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print(f"Data quality guard passed ({what})")


def last_weekday(year, month, weekday):
    """Last given weekday (0 = Monday) of a month."""
    d = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.dayofweek - weekday) % 7)


def expiries(name):
    """Contract expiry dates of the continuous front-month series: ICE Endex TTF monthly futures stop trading two
    business days before the delivery month; ICE Rotterdam coal (API2) monthly futures on the last Friday of the
    contract month. The first trading day after an expiry quotes the next contract."""
    months = [(y, m) for y in range(2016, 2027) for m in range(1, 13)]
    if name == "gas_ttf":
        return pd.DatetimeIndex([pd.Timestamp(y, m, 1) - pd.offsets.BDay(2) for y, m in months])
    if name == "coal_api2":
        return pd.DatetimeIndex([last_weekday(y, m, 4) for y, m in months])
    raise ValueError(name)


def roll_days(index, name):
    """First trading day of the series after each expiry."""
    pos = np.unique(index.searchsorted(expiries(name), side="right"))
    return index[pos[(pos > 0) & (pos < len(index))]]


def load_commodities():
    """Daily commodity inputs on a calendar-day index (weekends and holidays carry the last close).

    Level features (close, ma5, ma20, log, ratios) use the raw front-month price; coal is converted to EUR/t with the
    ECB rate of the same day (the previous rate on days without one). Return, volatility and momentum features use a
    ratio roll-adjusted series for TTF and coal: the return on the first trading day after each expiry is set to 0,
    so the gap between the expiring and the next contract does not show as a price move (the genuine move of that day
    is lost too, one day per month). Momentum is ma5 / ma20 - 1 of the roll-adjusted index, which needs no rescaling
    of past prices. Carbon (front December, one roll a year) and the other series are not roll-adjusted.
    """
    usd_eur = read_daily("usd_eur").Close
    raw = {name: read_daily(name).Close.dropna() for name in COMMODITY_ORDER}
    # Trading days are those of the Yahoo series (as before); Ember lists every weekday, holidays included, so its
    # closes are aligned to these days (latest Ember value on or before each day).
    days = pd.DatetimeIndex(sorted(set().union(*[raw[n].index for n in COMMODITY_ORDER if n != "carbon_eua"])))
    c = raw["carbon_eua"]
    raw["carbon_eua"] = c.reindex(c.index.union(days)).ffill().reindex(days).dropna()
    closes, rets = {}, {}
    for name in COMMODITY_ORDER:
        s = raw[name]
        if name == "coal_api2":
            s = s / usd_eur.reindex(s.index.union(usd_eur.index)).ffill().reindex(s.index)   # USD/t -> EUR/t
        r = s.pct_change()
        if name in ROLLED:
            r[roll_days(s.index, name)] = 0.0
        closes[name], rets[name] = s, r

    trading = pd.DataFrame(closes).sort_index()
    ret = pd.DataFrame(rets).reindex(trading.index).fillna(0.0)      # no new close on that day: return 0
    trading = trading.ffill()
    adj = (1 + ret).cumprod()                                        # roll-adjusted index, scale-free

    df = pd.DataFrame({f"{n}_close": trading[n] for n in COMMODITY_ORDER}, index=trading.index)
    for n in COMMODITY_ORDER:
        s = trading[n]
        df[f"{n}_return"] = ret[n].where(s.notna())
        df[f"{n}_ma5"] = s.rolling(5, min_periods=1).mean()
        df[f"{n}_ma20"] = s.rolling(20, min_periods=1).mean()
        df[f"{n}_vol20"] = ret[n].where(s.notna()).rolling(20, min_periods=1).std()
        df[f"{n}_log"] = np.log(s.clip(lower=0.01))
    df["ratio_gas_carbon"] = trading["gas_ttf"] / trading["carbon_eua"].clip(lower=0.01)   # EUR/MWh per EUR/t
    df["ratio_gas_oil"] = trading["gas_ttf"] / trading["oil_brent"].clip(lower=0.01)
    df["ratio_gas_coal"] = trading["gas_ttf"] / trading["coal_api2"].clip(lower=0.01)      # EUR/MWh per EUR/t
    for n, col in [("gas_ttf", "gas_momentum"), ("carbon_eua", "carbon_momentum")]:
        a = adj[n].where(trading[n].notna())
        df[col] = a.rolling(5, min_periods=1).mean() / a.rolling(20, min_periods=1).mean() - 1

    calendar = pd.date_range(df.index.min(), pd.Timestamp(END) - pd.Timedelta(days=1), freq="D", name="date")
    df = df.reindex(calendar).ffill()
    df.columns = ["com_" + c for c in df.columns]
    return df


def add_calendar(df):
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return df


def build_zone(zone, info, commodities):
    print(f"\n--- {zone} ---")
    df = build_prices(info)
    df = df[(df.index >= pd.Timestamp(START, tz=TZ)) & (df.index < pd.Timestamp(END, tz=TZ))]
    print(f"  prices: {len(df):,} hours, {df.index[0]} to {df.index[-1]}")

    if info["role"] == "price_only":
        df = add_calendar(df)
        df.insert(0, "zone", zone)
        df.insert(1, "country", info["country"])
        df.insert(2, "role", info["role"])
        print("  price-only zone: prices and calendar only")
        return df

    load, wsf = build_forecasts(info)
    weather = build_weather(zone, info)
    for name, part in [("load forecast", load), ("wind/solar forecast", wsf), ("weather", weather)]:
        if part is None:
            print(f"  {name}: MISSING")
        else:
            df = df.join(part, how="left")
            print(f"  {name}: +{part.shape[1]} cols")

    if info["reservoir_area"]:
        res = load_entsoe_reservoir(info["reservoir_area"])
        if res is None:
            print("  reservoir (ENTSO-E): MISSING")
        else:
            df = df.join(to_hours(df.index, res))
            print(f"  reservoir (ENTSO-E): +{res.shape[1]} cols")

    if info["reservoir_nr"] is not None:
        nve = load_nve(info["reservoir_nr"])
        df = df.join(to_hours(df.index, nve))
        print(f"  NVE (regime analysis only): +{nve.shape[1]} cols")

    if commodities is not None:
        df = df.join(to_hours(df.index, commodities))
        print(f"  commodities: +{commodities.shape[1]} cols")

    df = add_calendar(df)
    df.insert(0, "zone", zone)
    df.insert(1, "country", info["country"])
    df.insert(2, "role", info["role"])
    return df


def coverage(df, zone):
    """Share of non-missing values per column and year."""
    num = df.select_dtypes("number")
    cov = num.notna().groupby(num.index.year).mean().T
    cov = cov.reset_index(names="column").melt(id_vars="column", var_name="year", value_name="share_present")
    cov.insert(0, "zone", zone)
    return cov


if __name__ == "__main__":
    if "--check" in sys.argv:
        # Guard only, on the raw files and the existing data/cleaned/ files; writes nothing.
        fails = (check_commodities() if commodity_files_present() else []) + check_reservoir()
        for zone in [z for z in ZONES if ZONES[z]["role"] != "price_only"]:
            fails += check_hourly(zone, pd.read_parquet(f"{OUTDIR}/{zone}_hourly.parquet"))
        guard(fails, "check only")
        sys.exit(0)

    has_com = commodity_files_present()
    if not has_com:
        print("Commodity files not found in data/commodities/: data/cleaned/ is built without the com_* columns (see README)")
    guard((check_commodities() if has_com else []) + check_reservoir(), "commodities and reservoir")
    os.makedirs(OUTDIR, exist_ok=True)
    commodities = load_commodities() if has_com else None

    zones = sys.argv[1:] or list(ZONES)

    frames, cov = [], []
    for zone in zones:
        df = build_zone(zone, ZONES[zone], commodities)
        if ZONES[zone]["role"] != "price_only":
            guard(check_hourly(zone, df), zone)
        df.to_parquet(f"{OUTDIR}/{zone}_hourly.parquet")

        c = coverage(df, zone)
        for _, row in c[c["share_present"] < 0.95].iterrows():
            print(f"  low coverage: {row['column']:35s} {row['year']}  {row['share_present']:.1%}")
        frames.append(df)
        cov.append(c)

    pd.concat(frames).to_parquet(f"{OUTDIR}/panel_hourly.parquet")
    pd.concat(cov).to_csv(f"{OUTDIR}/coverage.csv", index=False)
    print(f"\nSaved {len(frames)} zones to {OUTDIR}/")
