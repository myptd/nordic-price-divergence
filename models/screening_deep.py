"""
Screening of LSTM, TCN and Transformer on 2024 (no 2025 data), NO1-NO5 and SE1-SE4.

Same periods as models/screening.py: fit 2018-01-08..2022-12-31 with early stopping on 2023 (patience 25, at most
300 epochs), then refit on 2018-01-08..2023-12-31 for the epoch with the lowest 2023 loss, and forecast 2024.
The refit uses the same learning-rate schedule (warm-up and cosine decay) over that number of epochs. Scalers are
fitted on the training days of each fit. Architectures, hyperparameters and the level-only target as in
models/deep_learning.py. Three seeds; the reported forecast is their average.

Saves model_results/screening/{zone}_{model}.parquet (2024 forecasts per seed and averaged, long format) and
model_results/screening/{zone}_{model}_epochs.csv.
Run from the repository root:
  python models/screening_deep.py --model tcn --zones NO1
  python models/screening_deep.py --model tcn --zones NO1 --quick      # 1 seed, 3 epochs, for a check
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import deep_learning as dl
import screening
import utils

OUT = screening.OUT


def run(kind, zone, seeds, max_epochs):
    print(f"\n=== screening {kind} {zone} ({dl.DEVICE}) ===", flush=True)
    t0 = time.time()
    A = dl.build_arrays(screening.load_until_2024(zone))
    dates = pd.DatetimeIndex(A["dates"])
    assert dates.max() < pd.Timestamp("2025-01-01"), "2025 days in the arrays"
    in_range = lambda r: (dates >= pd.Timestamp(r[0])) & (dates <= pd.Timestamp(r[1]))
    fit_rows, es_rows, refit_rows, eval_rows = (in_range(r) for r in (screening.FIT, screening.ES, screening.REFIT, screening.EVAL))

    def scalers(rows):
        sc = {"past": dl.Scaler().fit(A["past"][rows], (0, 1)), "future": dl.Scaler().fit(A["future"][rows], (0, 1)),
              "static": dl.Scaler().fit(A["static"][rows], (0,))}
        yrel = (A["y"][rows] - A["level"][rows][:, None]) / A["spread"][rows][:, None]
        sc["y_m"], sc["y_s"] = yrel.mean(), yrel.std() + 1e-6
        return sc

    sc_fit, sc_refit = scalers(fit_rows), scalers(refit_rows)
    train, es = dl.tensors(A, fit_rows, sc_fit), dl.tensors(A, es_rows, sc_fit)
    full, ev = dl.tensors(A, refit_rows, sc_refit), dl.tensors(A, eval_rows, sc_refit)
    out, preds, epochs = [], [], []
    for seed in seeds:
        net, ran, best = dl.fit(kind, train, es, seed, max_epochs)
        n = net.best_epoch
        net, _, _ = dl.fit(kind, full, None, seed, n)                     # refit on 2018-2023 for n epochs
        p = dl.predict(net, ev, sc_refit, A["level"][eval_rows], A["spread"][eval_rows])
        preds.append(p)
        out.append(dl.long_format(A, eval_rows, p, "eval_2024", f"{kind}_s{seed}"))
        epochs.append({"zone": zone, "model": kind, "seed": seed, "epochs_run": ran, "best_epoch": n, "best_es_loss": best})
        print(f"  seed {seed}: early stopping at {ran} epochs, best epoch {n}, refit done ({time.time() - t0:.0f} s)", flush=True)
    out.append(dl.long_format(A, eval_rows, np.mean(preds, axis=0), "eval_2024", kind))
    os.makedirs(OUT, exist_ok=True)
    pd.concat(out, ignore_index=True).assign(zone=zone).to_parquet(f"{OUT}/{zone}_{kind}.parquet", index=False)
    pd.DataFrame(epochs).to_csv(f"{OUT}/{zone}_{kind}_epochs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lstm", "tcn", "transformer"])
    parser.add_argument("--zones", nargs="+", default=utils.NO_ZONES)
    parser.add_argument("--quick", action="store_true", help="1 seed and 3 epochs (check run)")
    args = parser.parse_args()
    for z in args.zones:
        run(args.model, z, dl.SEEDS[:1] if args.quick else dl.SEEDS, 3 if args.quick else dl.MAX_EPOCHS)
