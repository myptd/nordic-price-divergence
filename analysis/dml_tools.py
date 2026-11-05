"""
Shared tools for the double machine learning analyses of the 2025 forecast errors (analysis/10 and 19).

- daily_conditions(zone): one row per delivery day with the nine market conditions, all known at 11:00 on D-1,
  and the background variables:
    gas, carbon, coal, brent, eurnok   closes of D-2 (TTF, EUA, API2, Brent, EUR/NOK)
    res                                reservoir anomaly at 11:00 on D-1: Norway NVE fill share minus its 20-year weekly
                                       median; Sweden ENTSO-E stored energy minus its same-week mean over 2018-2023;
                                       not defined for DE-LU
    wsf, load                          daily means of the wind and solar and of the load forecast for D
    temp                               temperature at 10:00 on D-1
    dprice                             change of the mean price of D-1 from D-2
    level, spread                      mean and standard deviation of the prices of D-1
    week_sin, week_cos, wd0..wd5, month, iso_week
- daily_mae(zone, model, inputs): daily MAE of a rolling forecast (models/rolling.py) over the scored hours of each day.
- crossfit / dml: partially linear model (Robinson 1988; Chernozhukov et al. 2018) with LightGBM nuisance models;
  folds are given by the caller (calendar months for H3, ISO week mod 5 for the diagnostics), optionally with a buffer
  of days around every test block that is dropped from the training folds.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import rolling  # noqa: E402
import utils  # noqa: E402

NUISANCE = dict(objective="regression", n_estimators=200, learning_rate=0.05, num_leaves=7,
                min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=utils.SEED, verbose=-1, n_jobs=2)
CONDITIONS = ["gas", "carbon", "coal", "brent", "eurnok", "res", "wsf", "load", "temp"]
COM = {"gas": "com_gas_ttf_close", "carbon": "com_carbon_eua_close", "coal": "com_coal_api2_close",
       "brent": "com_oil_brent_close", "eurnok": "com_fx_eurnok_close"}
RES_NORM_YEARS = (2018, 2023)          # Swedish reservoir norm: same-week mean over 2018-2023


def daily_conditions(zone):
    raw = utils.load_zone(zone)
    hourly = fd.prepare_hourly(raw)
    P, _ = fd.daily_matrix(hourly[fd.TARGET])
    days = P.index
    d = pd.DataFrame(index=days)
    for name, col in COM.items():
        d[name] = fd.at_hour(hourly, [col], 0).reindex(days - pd.Timedelta(days=2)).values[:, 0]
    d["temp"] = fd.at_hour(hourly, ["wx_temperature_2m"], fd.WEATHER_HOUR).reindex(days - pd.Timedelta(days=1)).values[:, 0]
    d["load"] = fd.daily_matrix(hourly["load_forecast_mw"])[0].mean(axis=1)
    d["wsf"] = fd.daily_matrix(hourly["wsf_total"])[0].mean(axis=1)
    if "nve_fill_anomaly_vs_median" in raw.columns:
        nve = raw[["nve_fill_anomaly_vs_median"]].ffill()
        d["res"] = fd.at_hour(nve, ["nve_fill_anomaly_vs_median"], fd.ORIGIN_HOUR).reindex(days - pd.Timedelta(days=1)).values[:, 0]
    elif "res_stored_twh" in hourly.columns:
        stored = pd.Series(fd.at_hour(hourly, ["res_stored_twh"], fd.ORIGIN_HOUR).reindex(days - pd.Timedelta(days=1)).values[:, 0], index=days)
        week = pd.Series((days - pd.Timedelta(days=1)).isocalendar().week.values, index=days)
        base = (days >= pd.Timestamp(utils.STUDY_START)) & (days.year >= RES_NORM_YEARS[0]) & (days.year <= RES_NORM_YEARS[1])
        norm = stored[base].groupby(week[base]).mean()
        d["res"] = stored - week.map(norm)
    else:
        d["res"] = np.nan                                                     # DE-LU: no reservoir condition
    d["level"] = P.shift(1).mean(axis=1)
    d["spread"] = P.shift(1).std(axis=1)
    d["dprice"] = P.shift(1).mean(axis=1) - P.shift(2).mean(axis=1)
    iso = days.isocalendar()
    d["iso_week"] = iso.week.values.astype(int)
    d["week_sin"], d["week_cos"] = np.sin(2 * np.pi * d.iso_week / 52.18), np.cos(2 * np.pi * d.iso_week / 52.18)
    for k in range(6):
        d[f"wd{k}"] = (days.dayofweek == k).astype(float)
    d["month"] = days.month
    return d.rename_axis("date")


def daily_mae(zone, model="lgbm_hourly", inputs="base", window="expanding"):
    p = pd.read_parquet(f"{rolling.OUT}/{zone}_{model}_{inputs}_{window}.parquet")
    p = p[p.scored.astype(bool)]
    p["date"] = pd.to_datetime(p["date"])
    return (p.y_true - p.y_pred).abs().groupby(p.date).mean().rename_axis("date")


def crossfit(X, v, folds, dates=None, buffer_days=0):
    """Out-of-fold predictions of v from X. With buffer_days > 0, days within buffer_days of a test day are dropped
    from the training folds."""
    pred = np.full(len(v), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if buffer_days:
            test_days = pd.DatetimeIndex(dates[te])
            near = test_days.union(pd.DatetimeIndex([t + pd.Timedelta(days=k) for t in test_days
                                                     for k in range(-buffer_days, buffer_days + 1)]))
            tr &= ~pd.DatetimeIndex(dates).isin(near)
        m = lgb.LGBMRegressor(**NUISANCE).fit(X[tr], v[tr])
        pred[te] = m.predict(X[te])
    return pred


def dml(g, outcome, treatment, background, folds, buffer_days=0):
    """theta (change in the outcome per standard deviation of the treatment), Newey-West s.e. (7 lags), p (two-sided),
    share of the treatment variance left after the background, and the residuals (outcome, treatment)."""
    y = g[outcome].values
    t = ((g[treatment] - g[treatment].mean()) / g[treatment].std()).values
    X = g[background].values
    dates = g.index.values
    e = y - crossfit(X, y, folds, dates, buffer_days)
    r = t - crossfit(X, t, folds, dates, buffer_days)
    fit = sm.OLS(e, r).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return {"theta": float(fit.params[0]), "se": float(fit.bse[0]), "p": float(fit.pvalues[0]),
            "variance_left": float(np.var(r) / np.var(t)), "e": e, "r": r}
