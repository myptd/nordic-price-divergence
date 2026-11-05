"""
Day-bootstrap intervals of the gain from the neighbouring prices, 2025 (used by Fig. 4 and analysis/11).

delta rMAE = (sum of absolute errors without - with the neighbouring prices) / (sum of absolute errors of the D-7
naive), per zone. The days of 2025 are resampled with replacement (each day's 24 hours stay together), 2,000 resamples,
fixed seed, percentile interval. Resampling single days does not keep dependence across days.

Output: results/crossborder_2025_bootstrap.csv.
Run from the repository root:  python analysis/09_crossborder_bootstrap.py
"""
import importlib.util

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("xb", "analysis/08_crossborder.py")
xb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xb)

B, SEED = 2000, 20250928


def bootstrap(zone, rng):
    W, y = xb.zone_frame(zone)
    daily = pd.DataFrame({k: (y - W[k]).abs() for k in ["base", "crossborder", "naive_d7"]}).groupby(level="date").sum()
    v = daily.values                                                  # days x (base, crossborder, naive)
    delta = (v[:, 0].sum() - v[:, 1].sum()) / v[:, 2].sum()
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = v[idx].sum(axis=1)                                            # B x 3
    reps = (s[:, 0] - s[:, 1]) / s[:, 2]
    return {"zone": zone, "n_days": len(v), "delta_rMAE": delta, "ci_low": np.percentile(reps, 2.5), "ci_high": np.percentile(reps, 97.5),
            "share_reps_above_0": float(np.mean(reps > 0))}


def main():
    rng = np.random.default_rng(SEED)
    t = pd.DataFrame([bootstrap(z, rng) for z in xb.ZONES])
    t["coupling_2024"] = t.zone.map(xb.COUPLING)
    t["split"] = t.zone.map(xb.SPLIT)
    t.to_csv(f"{xb.OUT}/crossborder_2025_bootstrap.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
