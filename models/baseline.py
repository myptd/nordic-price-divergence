"""
LEAR: LASSO with the preprocessing of Lago et al. (2021), one model per hour of D on the 247 LEAR inputs.

Inputs and target are normalised by median and MAD and transformed with asinh; predictions are transformed back.
The penalty is chosen per hour by the AIC (LassoLarsIC), as in the epftoolbox LEAR, with no validation set.
  norm="global"       standard LEAR (median/MAD + asinh over the training set)
  norm="level_scale"  first (price - L) / S with L, S = mean and std of D-1 (advanced.price_level / price_spread),
                      then the standard LEAR steps
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

import advanced


class AsinhScaler:
    """Median/MAD normalisation followed by asinh."""

    def fit(self, a):
        a = np.asarray(a, float)
        self.med = np.median(a, axis=0)
        self.mad = np.median(np.abs(a - self.med), axis=0) * 1.4826
        self.mad = np.where(self.mad < 1e-6, 1.0, self.mad)
        return self

    def transform(self, a):
        return np.arcsinh((np.asarray(a, float) - self.med) / self.mad)

    def inverse(self, a):
        return np.sinh(a) * self.mad + self.med


def _predict_lasso(fit, X):
    return fit["y"].inverse(fit["model"].predict(fit["x"].transform(X)))


def fit_linear_aic(X_train: pd.DataFrame, y_train, norm="global"):
    """24 LASSO models, penalty per hour by the AIC. y_train is (days x 24)."""
    X, Y = X_train, np.asarray(y_train, float)
    if norm == "level_scale":
        L, S = advanced.price_level(X_train), advanced.price_spread(X_train)
        X, Y = advanced.to_relative(X_train, L, S), (Y - L[:, None]) / S[:, None]
    models = []
    for h in range(24):
        xs, ys = AsinhScaler().fit(X), AsinhScaler().fit(Y[:, h])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = LassoLarsIC(criterion="aic", max_iter=1000).fit(xs.transform(X), ys.transform(Y[:, h]))
        models.append({"x": xs, "y": ys, "model": m, "alpha": m.alpha_})
    return {"norm": norm, "models": models}


def predict_linear(model, X: pd.DataFrame) -> np.ndarray:
    """Forecasts in EUR/MWh, (days x 24)."""
    if model["norm"] == "level_scale":
        L, S = advanced.price_level(X), advanced.price_spread(X)
        Z = np.column_stack([_predict_lasso(m, advanced.to_relative(X, L, S)) for m in model["models"]])
        return Z * S[:, None] + L[:, None]
    return np.column_stack([_predict_lasso(m, X) for m in model["models"]])
