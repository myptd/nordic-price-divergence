"""
LightGBM models with a normalised price target.

Two layouts:
  pooled    one model for the 24 hours of D, one row per (date, hour), the hour as an input
  per-hour  24 models, model h predicts hour h of D from the LEAR inputs (one row per day)

Normalised target: trees cannot predict outside the range seen in training, so each
model learns the price relative to a level known at the forecast time and the level is added back.
  norm="none"         raw price
  norm="level"        p - L                L = mean price of D-1
  norm="level_scale"  (p - L) / S          S = standard deviation of the prices of D-1, floored at SPREAD_FLOOR
All price inputs are transformed the same way; spreads (std columns) are left as they are, and L and S stay inputs.
With a validation set the number of trees is set by early stopping, otherwise it is given.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42

LGBM_PARAMS = dict(
    objective="regression_l1",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)
EARLY_STOPPING = 50
LEVEL_COLUMNS = ["price_d1_mean"]     # pooled layout
SPREAD_COLUMNS = ["price_d1_std"]
SPREAD_FLOOR = 1.0                    # EUR/MWh; avoids dividing by ~0 on flat days


def price_level(X: pd.DataFrame) -> np.ndarray:
    """Mean price of D-1 for each row (pooled column, or the mean of the 24 LEAR D-1 prices)."""
    for c in LEVEL_COLUMNS:
        if c in X.columns:
            return X[c].values
    return X[[f"lear_price_d1_h{h}" for h in range(24)]].mean(axis=1).values


def price_spread(X: pd.DataFrame) -> np.ndarray:
    """Standard deviation of the prices of D-1 for each row, floored at SPREAD_FLOOR."""
    for c in SPREAD_COLUMNS:
        if c in X.columns:
            return np.maximum(X[c].values, SPREAD_FLOOR)
    return np.maximum(X[[f"lear_price_d1_h{h}" for h in range(24)]].std(axis=1).values, SPREAD_FLOOR)


def to_relative(X: pd.DataFrame, level: np.ndarray, spread=None) -> pd.DataFrame:
    """Price inputs as (price - level) / spread (spread = 1 for the level-only version); level and spread stay inputs."""
    cols = [c for c in X.columns if c.startswith(("price_", "lear_price_"))
            and c not in LEVEL_COLUMNS and not c.endswith("_std")]
    sc = 1.0 if spread is None else spread[:, None]
    Xr = X.copy()
    Xr[cols] = (X[cols].values - level[:, None]) / sc
    if not any(c in X.columns for c in LEVEL_COLUMNS):
        Xr["price_level"] = level
    if spread is not None and not any(c in X.columns for c in SPREAD_COLUMNS):
        Xr["price_spread"] = spread
    return Xr


class LevelLGBM:
    """
    LightGBM on a normalised target; predict() returns prices in EUR/MWh.
    The level-and-scale version follows reversible instance normalisation (Kim et al., 2022) and
    variance-stabilising transformations for price forecasting (Uniejewski et al., 2018).
    """

    def __init__(self, n_jobs=4, n_estimators=None, norm="level", seed=SEED, params=None):
        params = {**LGBM_PARAMS, **(params or {}), "n_jobs": n_jobs, "random_state": seed}
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        self.model = lgb.LGBMRegressor(**params)
        self.norm = norm

    def _prep(self, X):
        if self.norm == "none":
            return X, 0.0, 1.0
        level = price_level(X)
        spread = price_spread(X) if self.norm == "level_scale" else None
        return to_relative(X, level, spread), level, (1.0 if spread is None else spread)

    def fit(self, X, y, X_val=None, y_val=None):
        Xr, level, spread = self._prep(X)
        target = (np.asarray(y) - level) / spread
        if X_val is None:
            self.model.fit(Xr, target)
        else:
            Vr, lv, sv = self._prep(X_val)
            self.model.fit(Xr, target, eval_X=(Vr,), eval_y=((np.asarray(y_val) - lv) / sv,),
                           callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(-1)])
        return self

    def predict(self, X):
        Xr, level, spread = self._prep(X)
        return self.model.predict(Xr) * spread + level

    @property
    def best_iteration_(self):
        return self.model.best_iteration_

    @property
    def n_estimators(self):
        return self.model.n_estimators


def fit_lgbm(X_train: pd.DataFrame, y_train, X_val=None, y_val=None, n_estimators=None, n_jobs=4, norm="level", seed=SEED,
             params=None):
    """With a validation set: early stopping. Without: n_estimators trees. params overrides LGBM_PARAMS."""
    return LevelLGBM(n_jobs=n_jobs, n_estimators=n_estimators, norm=norm, seed=seed, params=params).fit(X_train, y_train, X_val, y_val)


def best_iteration(model) -> int:
    return model.best_iteration_ or model.n_estimators


def predict_lgbm_hourly(models, X: pd.DataFrame) -> np.ndarray:
    """Forecasts of the 24 per-hour models, (days x 24)."""
    return np.column_stack([m.predict(X) for m in models])
