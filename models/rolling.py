"""
Monthly rolling recalibration over a test year: the forecasts of the benchmark and
of every input-set and training-window experiment.

rolling_forecast(zone, model, inputs="base", window="expanding")
  model    "lear" | "lgbm_pooled" | "lgbm_hourly"   (naive forecasts and the ensemble are formed in the analysis)
  inputs   name of an input set in INPUT_SETS: "base", "crossborder" (neighbouring prices, section 3.2),
           "prices_calendar", "no_vre" (section 8) or "reservoir" (section 11); the extra sets are used with lgbm_hourly
  window   "expanding" (from 2018-01-08), "expanding_no_crisis" (expanding without delivery days in 2021-2022),
           or a length in days (365, 730, 1095, ...)
  target   "level_scale" (default, frozen) or "none" (raw price; per-hour LightGBM only, with the raw setting chosen
           in the screening); files of a non-default target get the suffix _{target}
For each test month m (first day s):
  training data end on s - 1 day (asserted);
  LightGBM: early stopping on the last `es` months before s (es = min(12, quarter of the window), fitted on the
            window up to the start of that block), then refit on the whole window with the selected number of trees;
  LEAR:     penalty per hour by the AIC (LassoLarsIC) on the whole window.
Frozen versions from the screening: level-and-scale target; per-hour LightGBM 127 leaves,
learning rate 0.05, min_data_in_leaf 50; pooled 63, 0.05, 50; at most 1,000 trees.

Saves model_results/rolling/{zone}_{model}_{inputs}_{window}.parquet (hourly forecasts of the test year) and
..._params.csv (per month and hour: number of trees and whether the cap was reached, or the LEAR penalty).
Run from the repository root:
  python models/rolling.py --zones NO1                         # benchmark: all three models, base inputs, expanding
  python models/rolling.py --zones NO1 --months 1              # quick check (January only, file suffix _check)
  python models/rolling.py --zones NO1 --models lgbm_hourly --year 2024   # test year 2024 (file suffix _2024)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import advanced
import baseline
import features_day_ahead as fd
import utils

OUT = "model_results/rolling"
MONTHS = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
NORM = "level_scale"                    # default target (frozen in the screening)
SETTINGS = {("lgbm_hourly", "level_scale"): {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 50},
            ("lgbm_pooled", "level_scale"): {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50},
            ("lgbm_hourly", "none"): {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20}}   # raw target, screening choice
CRISIS = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"))    # dropped by window "expanding_no_crisis"
CAP = advanced.LGBM_PARAMS["n_estimators"]
MODELS = ["lear", "lgbm_pooled", "lgbm_hourly"]


# ---------------------------------------------------------------------------
# Input sets: each returns the pooled frame (X, y, mask) and the wide LEAR-layout frame (Xl, Yw) for one zone.
# ---------------------------------------------------------------------------

def base_inputs(zone, hourly):
    X, y, mask = utils.build_features(hourly)
    Xw, Yw, _ = fd.build_wide(hourly)
    return {"pooled": (X, y, mask), "wide": (Xw[fd.lear_columns(Xw)], Yw)}


NEIGHBOURS = ["DE_LU", "DK1", "NL"]


def crossborder_inputs(zone, hourly):
    """
    Neighbouring prices: base inputs + the 24 D-1 prices of DE-LU, DK1 and NL, each normalised with
    the neighbour's own L and S, (p - L_nb) / S_nb, + one level gap per neighbour, (L_nb - L_zone) / S_zone.
    The neighbours' D-1 prices use the same 24-slot daylight-saving mapping (daily_matrix, shifted one day) as the
    zone's own, and L and S come from the same functions (advanced.price_level / price_spread: mean and standard
    deviation of the 24 D-1 prices, S floored at 1 EUR/MWh). The new columns are named xb_*, so the level-and-scale
    step (advanced.to_relative, which acts on price_* and lear_price_* columns) leaves them as they are.
    Known at 11:00 on D-1: prices of D-1 are set by the auction on D-2.
    """
    data = base_inputs(zone, hourly)
    Xl, Yw = data["wide"]
    L_zone, S_zone = advanced.price_level(Xl), advanced.price_spread(Xl)
    days = pd.DatetimeIndex(pd.to_datetime(Xl.index))
    extra = {}
    for nb in NEIGHBOURS:
        P, _ = fd.daily_matrix(fd.prepare_hourly(utils.load_zone(nb))[fd.TARGET])
        d1 = P.shift(1).reindex(days)                                       # prices of D-1 for delivery day D
        as_lear = pd.DataFrame(d1.values, columns=[f"lear_price_d1_h{h}" for h in range(24)])
        L_nb, S_nb = advanced.price_level(as_lear), advanced.price_spread(as_lear)
        for h in range(24):
            extra[f"xb_{nb.lower()}_h{h}"] = (d1[h].values - L_nb) / S_nb
        extra[f"xb_gap_{nb.lower()}"] = (L_nb - L_zone) / S_zone
    X_extra = pd.DataFrame(extra, index=Xl.index)
    assert X_extra.shape[1] == 75 and not X_extra.isna().any().any(), "neighbouring-price inputs incomplete"
    assert not any(c.startswith(("price_", "lear_price_")) for c in X_extra.columns)
    data["wide"] = (pd.concat([Xl, X_extra], axis=1), Yw)
    return data


def drop_inputs(prefixes):
    """Input set = base inputs without the per-hour (LEAR-layout) columns that start with any of the prefixes."""
    def inputs(zone, hourly):
        data = base_inputs(zone, hourly)
        Xl, Yw = data["wide"]
        keep = [c for c in Xl.columns if not c.startswith(prefixes)]
        assert len(keep) < Xl.shape[1], "nothing dropped"
        data["wide"] = (Xl[keep], Yw)
        return data
    return inputs


RESERVOIR = ["res_stored_twh", "res_change_1w_twh", "res_change_4w_twh"]


def reservoir_inputs(zone, hourly):
    """
    Base inputs + the reservoir inputs of the pooled model (ENTSO-E stored energy,
    latest weekly value complete at 11:00 on D-1, and its change over one and four weeks; Norway and Sweden). The
    columns are day-level (one value per delivery day) and not price columns, so the level-and-scale step leaves them as
    they are. The NVE columns (nve_*) are never inputs: NVE revises past values and its norm includes later years.
    """
    data = base_inputs(zone, hourly)
    Xl, Yw = data["wide"]
    Xw, _, _ = fd.build_wide(hourly)
    X_extra = Xw.loc[Xl.index, RESERVOIR]
    assert not X_extra.isna().any().any(), "reservoir inputs incomplete"
    data["wide"] = (pd.concat([Xl, X_extra], axis=1), Yw)
    return data


INPUT_SETS = {"base": base_inputs, "crossborder": crossborder_inputs, "reservoir": reservoir_inputs,
              "prices_calendar": drop_inputs(("lear_load_", "lear_wsf_")),   # prices and weekday only (+ L, S)
              "no_vre": drop_inputs(("lear_wsf_",))}                         # without the wind and solar forecasts


# ---------------------------------------------------------------------------

def es_months(window):
    """Early-stopping block: min(12 months, one quarter of the window)."""
    return 12 if str(window).startswith("expanding") else min(12, int(round(window / 365.25 * 12 / 4)))


def periods(origin, window):
    """(window start, start of the early-stopping block, last training day) for the model used from `origin`."""
    last = origin - pd.Timedelta(days=1)
    start = pd.Timestamp(utils.STUDY_START) if str(window).startswith("expanding") else max(pd.Timestamp(utils.STUDY_START), origin - pd.Timedelta(days=window))
    return start, origin - pd.DateOffset(months=es_months(window)), last


def fit_lgbm(Xf, yf, Xe, ye, Xa, ya, params, target=NORM):
    m = advanced.fit_lgbm(Xf, yf, Xe, ye, n_jobs=utils.N_JOBS, norm=target, params=params)
    n, ran = advanced.best_iteration(m), m.model.booster_.current_iteration()
    return advanced.fit_lgbm(Xa, ya, n_estimators=n, n_jobs=utils.N_JOBS, norm=target, params=params), n, ran >= CAP


def rolling_forecast(zone, model, inputs="base", window="expanding", months=MONTHS, save=True, suffix="", target=NORM):
    t0 = time.time()
    hourly = utils.load_zone(zone)
    if model == "lgbm_pooled":
        utils.require_commodities(hourly.columns, "Pooled LightGBM")
    data = INPUT_SETS[inputs](zone, hourly)
    X, y, mask = data["pooled"]
    Xl, Yw = data["wide"]
    pdates = pd.DatetimeIndex(pd.to_datetime(X.index.get_level_values("date")))
    wdates = pd.DatetimeIndex(pd.to_datetime(Xl.index))
    parts, params = [], []
    if model == "lear" and target != NORM:
        raise ValueError("LEAR runs with the frozen level-and-scale target only")
    keep_p = ~((pdates >= CRISIS[0]) & (pdates <= CRISIS[1])) if window == "expanding_no_crisis" else np.ones(len(pdates), bool)
    keep_w = ~((wdates >= CRISIS[0]) & (wdates <= CRISIS[1])) if window == "expanding_no_crisis" else np.ones(len(wdates), bool)
    for origin in months:
        end = origin + pd.offsets.MonthEnd(0)
        start, es_start, last = periods(origin, window)
        p_all, p_fit, p_es = [(pdates >= a) & (pdates <= b) & keep_p for a, b in [(start, last), (start, es_start - pd.Timedelta(days=1)), (es_start, last)]]
        w_all, w_fit, w_es = [(wdates >= a) & (wdates <= b) & keep_w for a, b in [(start, last), (start, es_start - pd.Timedelta(days=1)), (es_start, last)]]
        p_test, w_test = (pdates >= origin) & (pdates <= end), (wdates >= origin) & (wdates <= end)
        assert pdates[p_all].max() < origin and wdates[w_all].max() < origin, "training data must end before the test month"

        if model == "lear":
            fit = baseline.fit_linear_aic(Xl[w_all], Yw[w_all], norm=target)
            pred = baseline.predict_linear(fit, Xl[w_test])
            params += [{"month": origin, "hour": h, "alpha": m["alpha"]} for h, m in enumerate(fit["models"])]
        elif model == "lgbm_hourly":
            cols = []
            for h in range(24):
                m, n, cap = fit_lgbm(Xl[w_fit], Yw[w_fit][h], Xl[w_es], Yw[w_es][h], Xl[w_all], Yw[w_all][h], SETTINGS[(model, target)], target)
                cols.append(m.predict(Xl[w_test]))
                params.append({"month": origin, "hour": h, "n_trees": n, "cap_hit": cap})
            pred = np.column_stack(cols)
        elif model == "lgbm_pooled":
            m, n, cap = fit_lgbm(X[p_fit], y[p_fit], X[p_es], y[p_es], X[p_all], y[p_all], SETTINGS[(model, target)], target)
            params.append({"month": origin, "hour": -1, "n_trees": n, "cap_hit": cap})
        else:
            raise ValueError(model)

        rows = np.asarray(p_test)
        if model == "lgbm_pooled":
            values = m.predict(X[rows])
        else:
            s = pd.DataFrame(pred, index=Xl.index[w_test], columns=range(24)).stack(future_stack=True)
            s.index.names = ["date", "hour"]
            values = s.reindex(X.index[rows]).values
        parts.append(pd.DataFrame({"date": pdates[rows], "hour": X.index.get_level_values("hour")[rows], "zone": zone, "model": model,
                                   "inputs": inputs, "window": str(window), "target": target, "refit_month": origin, "y_true": y.values[rows],
                                   "y_pred": values, "scored": mask.values[rows]}))
        print(f"  {zone} {model:11s} {inputs} {window} {target}  {origin:%Y-%m}: train {start:%Y-%m-%d}..{last:%Y-%m-%d} "
              f"({int(w_all.sum())} days), early stopping from {es_start:%Y-%m-%d} ({time.time() - t0:.0f} s)", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    par = pd.DataFrame(params).assign(zone=zone, model=model, inputs=inputs, window=str(window), target=target)
    if save:
        os.makedirs(OUT, exist_ok=True)
        name = f"{OUT}/{zone}_{model}_{inputs}_{window}{'' if target == NORM else '_' + target}{suffix}"
        pred.to_parquet(f"{name}.parquet", index=False)
        par.to_csv(f"{name}_params.csv", index=False)
    return pred, par


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", nargs="+", default=utils.ALL_ZONES)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--inputs", default="base", choices=list(INPUT_SETS))
    parser.add_argument("--window", default="expanding", help='"expanding", "expanding_no_crisis" or a length in days')
    parser.add_argument("--target", default=NORM, choices=["level_scale", "none"])
    parser.add_argument("--months", type=int, default=len(MONTHS), help="first n months of the test year (quick check)")
    parser.add_argument("--year", type=int, default=2025, help="test year; other than 2025 adds the suffix _{year}")
    args = parser.parse_args()
    months = MONTHS if args.year == 2025 else pd.date_range(f"{args.year}-01-01", f"{args.year}-12-01", freq="MS")
    year_suffix = "" if args.year == 2025 else f"_{args.year}"
    window = args.window if args.window.startswith("expanding") else int(args.window)
    for z in args.zones:
        for m in args.models:
            if m == "lgbm_pooled" and args.models == MODELS and \
                    not any(c.startswith("com_") for c in pd.read_parquet(f"data/cleaned/{z}_hourly.parquet").columns):
                print(f"{z}: pooled LightGBM skipped, no commodity inputs in data/cleaned/ (see README)", flush=True)
                continue
            rolling_forecast(z, m, args.inputs, window, months[:args.months],
                             suffix=year_suffix + ("" if args.months == len(months) else "_check"), target=args.target)
