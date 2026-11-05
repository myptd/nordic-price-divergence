"""
Shared helpers: zones and study period, data loading, date masks, accuracy measures and the multivariate
Diebold-Mariano test. Feature matrices are indexed by (date, hour) of the delivery hour.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import features_day_ahead

SEED = 42
TZ = "Europe/Oslo"
TARGET = "price_eur_mwh"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

NO_ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ALL_ZONES = NO_ZONES + ["SE1", "SE2", "SE3", "SE4", "DE_LU"]

STUDY_START = "2018-01-08"
STUDY_END = "2025-12-30"


def load_zone(zone: str) -> pd.DataFrame:
    """data/cleaned/{zone}_hourly.parquet with a sorted Europe/Oslo index."""
    return pd.read_parquet(f"data/cleaned/{zone}_hourly.parquet").sort_index()


def require_commodities(columns, what: str) -> None:
    """Stop when the commodity inputs (com_*) are missing from data/cleaned/, instead of fitting on fewer inputs."""
    if not any(str(c).startswith("com_") for c in columns):
        raise RuntimeError(f"{what} uses the commodity inputs (com_* columns), which are not in data/cleaned/. Add them "
                           "as described in README.md (Commodity prices), then rerun sh_runs/02_preprocess.sh.")


def build_features(hourly: pd.DataFrame):
    """Pooled (X, y, mask) indexed by (date, hour); see features_day_ahead.build_features."""
    return features_day_ahead.build_features(hourly)


def year_range(year: int) -> tuple[str, str]:
    """Delivery dates of a calendar year, clipped to the study period."""
    return max(f"{year}-01-01", STUDY_START), min(f"{year}-12-31", STUDY_END)


def in_dates(obj, start: str, end: str) -> np.ndarray:
    """Boolean mask of rows of obj (indexed by (date, hour) or date) with date in [start, end]."""
    idx = obj.index
    dates = idx.get_level_values("date") if isinstance(idx, pd.MultiIndex) else idx
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def evaluate(y_true, y_pred, y_naive=None, mask=None) -> dict:
    """MAE, RMSE, sMAPE (%), R2 and rMAE (MAE relative to the naive forecast y_naive) on hourly values."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if y_naive is not None:
        y_naive = np.asarray(y_naive, float)
        ok &= np.isfinite(y_naive)
    t, p = y_true[ok], y_pred[ok]
    err = t - p
    mae = np.mean(np.abs(err))
    res = {
        "n": int(ok.sum()),
        "MAE": mae,
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "sMAPE": 100 * np.mean(2 * np.abs(err) / np.maximum(np.abs(t) + np.abs(p), 1e-8)),
        "R2": 1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2),
    }
    if y_naive is not None:
        res["rMAE"] = mae / np.mean(np.abs(t - y_naive[ok]))
    return res


def dm_test_lago(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """
    Multivariate Diebold-Mariano test as in Lago et al. (2021), eqs. 11-12, with the absolute loss (p = 1).
    errors_a, errors_b: (days x 24) forecast errors of models A and B on the same days.
    Loss differential per day: Delta_d = ||e_A,d||_1 - ||e_B,d||_1; DM = sqrt(N) * mean / std (asymptotically N(0, 1)).
    Returns (DM, p), p the one-sided p-value of H0: E(Delta) <= 0 against H1: E(Delta) > 0,
    i.e. a small p means that model B is significantly more accurate than model A.
    """
    # nansum: on 23-hour days the filled 02:00 slot is not scored (NaN in both models)
    delta = np.nansum(np.abs(np.asarray(errors_a, float)), axis=1) - np.nansum(np.abs(np.asarray(errors_b, float)), axis=1)
    delta = delta[np.isfinite(delta)]
    dm = np.sqrt(len(delta)) * delta.mean() / delta.std()
    return dm, 1 - stats.norm.cdf(dm)
