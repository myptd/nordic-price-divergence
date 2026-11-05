"""
Screening of LEAR and the LightGBM grid on 2024 (no 2025 data), which fixed the target, the settings and the main model.

Periods (delivery days):  fit 2018-01-08..2022-12-31, early stopping 2023, refit 2018-01-08..2023-12-31, evaluate 2024.
The cleaned data are cut at 2024-12-31 23:00 before any feature is built, so no 2025 value can enter.

  LEAR           standard and normalised (level and scale) target; penalty per hour by the AIC (LassoLarsIC) on
                 2018-2023, no validation set (baseline.fit_linear_aic)
  LightGBM       pooled and per-hour layouts x targets raw ("none"), level-only ("level"), level-and-scale
                 ("level_scale") x grid num_leaves {31, 63, 127} x learning_rate {0.05, 0.1} x min_data_in_leaf {20, 50};
                 at most 1,000 trees; number of trees by early stopping on 2023 (50 rounds), then refit on 2018-2023
The naive forecasts, the ensemble and the selection are done in analysis/06_screening.py.

Saves model_results/screening/{zone}_classic.parquet (2024 forecasts, long format) and
model_results/screening/{zone}_trees.csv (best iteration and whether the 1,000-tree cap was reached, per fit).
Run from the repository root:
  python models/screening.py --zones NO1
  python models/screening.py --zones NO1 --quick        # one grid setting, for a check
"""
import argparse
import itertools
import os
import time

import pandas as pd

import advanced
import baseline
import features_day_ahead as fd
import utils

OUT = os.environ.get("SCREENING_OUT", "model_results/screening")   # output folder (sh_runs/03, 04 set it)
CUTOFF = pd.Timestamp("2025-01-01", tz=utils.TZ)          # nothing from 2025 is loaded
FIT = (utils.STUDY_START, "2022-12-31")
ES = ("2023-01-01", "2023-12-31")
REFIT = (utils.STUDY_START, "2023-12-31")
EVAL = ("2024-01-01", "2024-12-31")
TARGETS = ["none", "level", "level_scale"]
GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": md}
        for nl, lr, md in itertools.product([31, 63, 127], [0.05, 0.1], [20, 50])]
CAP = advanced.LGBM_PARAMS["n_estimators"]                # 1,000


def load_until_2024(zone):
    hourly = utils.load_zone(zone)
    hourly = hourly[hourly.index < CUTOFF]
    assert hourly.index.max() < CUTOFF, "2025 data must not be used in the screening"
    return hourly


def setting_name(p):
    return f"nl{p['num_leaves']}_lr{p['learning_rate']}_md{p['min_child_samples']}"


def run_zone(zone, grid):
    print(f"\n=== screening {zone} ===", flush=True)
    t0 = time.time()
    hourly = load_until_2024(zone)
    utils.require_commodities(hourly.columns, "The 2024 screening (pooled LightGBM)")
    X, y, mask = utils.build_features(hourly)
    Xw, Yw, _ = fd.build_wide(hourly)
    cols = fd.lear_columns(Xw)
    Xl = Xw[cols]
    pf, pe, pr, pv = (utils.in_dates(X, *r) for r in (FIT, ES, REFIT, EVAL))
    wf, we, wr, wv = (utils.in_dates(Xw, *r) for r in (FIT, ES, REFIT, EVAL))
    assert pd.to_datetime(X.index.get_level_values("date")).max() < pd.Timestamp("2025-01-01"), "2025 rows in the features"
    parts, trees = [], []

    def collect(model, target, setting, pred):
        parts.append(pd.DataFrame({"date": X.index.get_level_values("date")[pv], "hour": X.index.get_level_values("hour")[pv],
                                   "zone": zone, "model": model, "target": target, "setting": setting,
                                   "y_true": y.values[pv], "y_pred": pred, "scored": mask.values[pv]}))

    def wide_to_rows(P):
        s = pd.DataFrame(P, index=Xw.index[wv], columns=range(24)).stack(future_stack=True)
        s.index.names = ["date", "hour"]
        return s.reindex(X.index[pv]).values

    # LEAR, penalty by AIC on 2018-2023
    for norm, target in [("global", "standard"), ("level_scale", "level_scale")]:
        lear = baseline.fit_linear_aic(Xl[wr], Yw[wr], norm=norm)
        collect("lear", target, "aic", wide_to_rows(baseline.predict_linear(lear, Xl[wv])))
    print(f"  LEAR done ({time.time() - t0:.0f} s)", flush=True)

    def es_then_refit(Xf, yf, Xe, ye, Xr, yr, target, params):
        m = advanced.fit_lgbm(Xf, yf, Xe, ye, n_jobs=utils.N_JOBS, norm=target, params=params)
        ran = m.model.booster_.current_iteration()           # iterations actually trained before early stopping
        n = advanced.best_iteration(m)
        refit = advanced.fit_lgbm(Xr, yr, n_estimators=n, n_jobs=utils.N_JOBS, norm=target, params=params)
        return refit, n, ran >= CAP

    for params, target in itertools.product(grid, TARGETS):
        name = setting_name(params)
        pooled, n, cap = es_then_refit(X[pf], y[pf], X[pe], y[pe], X[pr], y[pr], target, params)
        collect("lgbm_pooled", target, name, pooled.predict(X[pv]))
        trees.append({"zone": zone, "model": "lgbm_pooled", "target": target, "setting": name, "hour": -1, "best_iteration": n, "cap_hit": cap})
        hourly_models = []
        for h in range(24):
            m, n, cap = es_then_refit(Xl[wf], Yw[wf][h], Xl[we], Yw[we][h], Xl[wr], Yw[wr][h], target, params)
            hourly_models.append(m)
            trees.append({"zone": zone, "model": "lgbm_hourly", "target": target, "setting": name, "hour": h, "best_iteration": n, "cap_hit": cap})
        collect("lgbm_hourly", target, name, wide_to_rows(advanced.predict_lgbm_hourly(hourly_models, Xl[wv])))
        print(f"  {name:22s} {target:12s} done ({time.time() - t0:.0f} s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(f"{OUT}/{zone}_classic.parquet", index=False)
    pd.DataFrame(trees).to_csv(f"{OUT}/{zone}_trees.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", nargs="+", default=utils.ALL_ZONES)
    parser.add_argument("--quick", action="store_true", help="one grid setting only (check run)")
    args = parser.parse_args()
    for z in args.zones:
        run_zone(z, GRID[:1] if args.quick else GRID)
