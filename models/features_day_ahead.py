"""
Features for the day-ahead task.

A forecast for delivery day D is made at 11:00 on D-1 (Europe/Oslo), one hour
before Nord Pool gate closure at 12:00. All 24 prices of D are predicted at once.

Information set at 11:00 on D-1
-------------------------------
Each variable uses the latest value whose event or measurement happened
before 11:00 on D-1. Publication delay is assumed to be zero.

  price_eur_mwh     all 24 hours of D-1 and earlier (set by the D-2 auction)
  load_forecast_mw  TSO forecast for D
  wsf_*             TSO wind/solar forecast for D
  wx_*              observation at 10:00 on D-1, and the mean temperature
                    over 00:00-10:00 on D-1 (reanalysis, so D is not allowed)
  res_*             value at 11:00 on D-1 (ENTSO-E, Norway and Sweden)
  com_*             close of D-2 (D-1 closes happen after 11:00)
  calendar          known for D

nve_* columns (NVE fill share and anomaly) are never features; they are only
used to define reservoir regimes for Norway in the analysis.

Two layouts:
  build_features  pooled: one row per (delivery date, hour), for LightGBM
  build_wide      one row per delivery date, for LEAR and the neural network.
                  LEAR uses only the Lago et al. (2021) columns (lear_columns).

DST days
--------
Days are mapped to 24 slots (epftoolbox convention). On 23-hour days the
missing 02:00 is the mean of 01:00 and 03:00 and is not scored (mask = False).
On 25-hour days the two 02:00 prices are averaged.
"""
import numpy as np
import pandas as pd

TARGET = "price_eur_mwh"
ORIGIN_HOUR = 11                 # forecast time on D-1
WEATHER_HOUR = 10                # last weather observation before 11:00
PRICE_LAG_DAYS = [1, 2, 3, 7]    # D-1, D-2, D-3 and D-7
COMMODITY_LAG_DAYS = 2           # close of D-2
MAX_MISSING = 0.05               # forecast columns missing more often than this are dropped
NON_FEATURES = ["zone", "country", "role", "hour", "day_of_week", "month",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]


def prepare_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Drop sparse forecast columns and NVE columns, fill short gaps with the last known value."""
    df = hourly.drop(columns=[c for c in hourly.columns if c.startswith("nve_")])
    for c in [c for c in df.columns if c.startswith("wsf_")]:
        if df[c].isna().mean() > MAX_MISSING:
            df = df.drop(columns=c)        # e.g. offshore wind in Norway, solar in Sweden
    wsf = [c for c in df.columns if c.startswith("wsf_")]
    df["wsf_total"] = df[wsf].sum(axis=1, min_count=1) if wsf else np.nan
    cols = [c for c in df.columns if c not in NON_FEATURES and c != TARGET]
    df[cols] = df[cols].ffill()            # past values only
    return df


def daily_matrix(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dates x 24) matrix of an hourly series, with the DST convention, and a mask of real slots."""
    idx = series.index
    key = pd.DataFrame({"date": pd.to_datetime(idx.date), "hour": idx.hour, "v": series.values})
    m = key.groupby(["date", "hour"])["v"].mean().unstack("hour")   # 25-hour days: 02:00 averaged
    m = m.reindex(columns=range(24))
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="D", name="date"))
    mask = m.notna()
    spring = m[2].isna() & m[1].notna() & m[3].notna()                # 23-hour days
    m.loc[spring, 2] = (m.loc[spring, 1] + m.loc[spring, 3]) / 2
    return m, mask


def at_hour(df: pd.DataFrame, cols: list[str], hour: int) -> pd.DataFrame:
    """One row per date with the values of *cols* at local *hour*."""
    rows = df[df.index.hour == hour][cols]
    rows = rows[~pd.Index(rows.index.date).duplicated()]
    rows.index = pd.DatetimeIndex(pd.to_datetime(rows.index.date), name="date")
    return rows


def day_level_features(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Features shared by all hours of D: weather, reservoir, commodities, calendar of D."""
    wx = [c for c in df.columns if c.startswith("wx_")]
    res = [c for c in df.columns if c.startswith("res_")]
    com = [c for c in df.columns if c.startswith("com_")]
    parts = []
    if wx:
        w = at_hour(df, wx, WEATHER_HOUR).reindex(dates - pd.Timedelta(days=1))
        morning = df[df.index.hour <= WEATHER_HOUR]["wx_temperature_2m"]
        tmean = morning.groupby(pd.to_datetime(morning.index.date)).mean()
        w["wx_temperature_2m_mean_0010"] = tmean.reindex(dates - pd.Timedelta(days=1)).values
        w.index = dates
        parts.append(w)
    if res:
        r = at_hour(df, res, ORIGIN_HOUR).reindex(dates - pd.Timedelta(days=1))
        r.index = dates
        parts.append(r)
    if com:
        c = at_hour(df, com, 0).reindex(dates - pd.Timedelta(days=COMMODITY_LAG_DAYS))
        c.index = dates
        parts.append(c)
    cal = pd.DataFrame(index=dates)
    cal["day_of_week"] = dates.dayofweek
    cal["month"] = dates.month
    cal["dow_sin"] = np.sin(2 * np.pi * dates.dayofweek / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * dates.dayofweek / 7)
    cal["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    parts.append(cal)
    return pd.concat(parts, axis=1)


def build_features(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Pooled layout. Return (X, y, mask) indexed by (date, hour) of delivery.
    mask is False for filled spring DST slots (not scored).
    """
    df = prepare_hourly(hourly)
    P, mask = daily_matrix(df[TARGET])
    dates = P.index

    day = day_level_features(df, dates)
    d1 = P.shift(1)
    day["price_d1_mean"] = d1.mean(axis=1)
    day["price_d1_min"] = d1.min(axis=1)
    day["price_d1_max"] = d1.max(axis=1)
    day["price_d1_std"] = d1.std(axis=1)
    day["price_d1_h23"] = d1[23]
    L, _ = daily_matrix(df["load_forecast_mw"])
    day["load_forecast_daymean"] = L.mean(axis=1)

    # Hour-level columns: (dates x 24) matrices stacked to (date, hour).
    hour_level = {f"price_d{k}": P.shift(k) for k in PRICE_LAG_DAYS}
    hour_level["load_forecast_mw"] = L
    for c in [c for c in df.columns if c.startswith("wsf_")]:
        hour_level[c] = daily_matrix(df[c])[0]
    X = pd.concat({name: m.stack(future_stack=True) for name, m in hour_level.items()}, axis=1)
    X.index.names = ["date", "hour"]
    X = X.join(day, on="date")
    hours = X.index.get_level_values("hour")
    X.insert(0, "hour", hours)
    X.insert(1, "hour_sin", np.sin(2 * np.pi * hours / 24))
    X.insert(2, "hour_cos", np.cos(2 * np.pi * hours / 24))

    y = P.stack(future_stack=True).rename(TARGET)
    y.index.names = ["date", "hour"]
    m = mask.stack(future_stack=True).rename("scored")
    m.index.names = ["date", "hour"]

    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep], m[keep]


def build_wide(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wide layout. Return (X, Y, mask) with one row per delivery date; Y and mask are (dates x 24).
    X holds the 247 LEAR inputs of Lago et al. (2021, eq. 2) plus the day-level features.
    """
    df = prepare_hourly(hourly)
    P, mask = daily_matrix(df[TARGET])
    dates = P.index
    L, _ = daily_matrix(df["load_forecast_mw"])
    R, _ = daily_matrix(df["wsf_total"])

    blocks = {}
    for k in PRICE_LAG_DAYS:
        blocks[f"lear_price_d{k}"] = P.shift(k)
    for k in [0, 1, 7]:
        blocks[f"lear_load_d{k}"] = L.shift(k)
        blocks[f"lear_wsf_d{k}"] = R.shift(k)
    X = pd.concat([m.add_prefix(f"{name}_h") for name, m in blocks.items()], axis=1)
    for d in range(7):
        X[f"lear_dow_{d}"] = (dates.dayofweek == d).astype(float)
    X = X.join(day_level_features(df, dates).drop(columns=["day_of_week"]))

    keep = X.notna().all(axis=1) & P.notna().all(axis=1)
    return X[keep], P[keep], mask[keep]


def lear_columns(X: pd.DataFrame) -> list[str]:
    """The 247 Lago et al. (2021) LEAR inputs in a wide X."""
    return [c for c in X.columns if c.startswith("lear_")]
