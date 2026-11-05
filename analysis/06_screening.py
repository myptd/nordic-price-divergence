"""
Screening table and selection on 2024 (no 2025 data), from models/screening.py and models/screening_deep.py.

For every zone, model, target version and LightGBM setting: MAE and rMAE (relative to the D-7 naive forecast on the same
hours) in 2024, on the hours that every model of the zone covers (the filled 02:00 slot of 23-hour days excluded).
Selection, on the mean 2024 MAE over NO1-NO5:
  LightGBM setting   separately for each layout (pooled, per hour) x target (raw, level, level and scale)
  best target        per family: LEAR (standard vs normalised); pooled and per-hour LightGBM (at their chosen settings)
  ensemble           equal-weight average of the best LEAR and the best per-hour LightGBM, per zone
Also reports how often early stopping ran to the 1,000-tree cap, and the networks' epochs.

Two screening runs (RUNS):
  model_results/screening                 the run that fixed the frozen choices, made with the earlier carbon input
                                          (Table S1; not reproducible from the current data/cleaned)
  model_results/screening_rerun    the same screening with the corrected commodity inputs (Supplementary S2)
Outputs: results/screening_2024.csv, results/screening_summary.txt, and the same two files in
results/screening_rerun/.
Run from the repository root:  python analysis/06_screening.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "models")
import features_day_ahead as fd  # noqa: E402
import screening  # noqa: E402
import utils  # noqa: E402

RUNS = [("model_results/screening", "results"),
        ("model_results/screening_rerun", "results/screening_rerun")]   # (forecasts, outputs)
ZONES, NO = utils.ALL_ZONES, utils.NO_ZONES
DEEP = ["lstm", "tcn", "transformer"]


def zone_frame(R, zone):
    """Columns (model, target, setting) of 2024 forecasts, plus naive D-1 and D-7, on the hours all models cover."""
    c = pd.read_parquet(f"{R}/{zone}_classic.parquet")
    c["date"] = pd.to_datetime(c["date"])
    W = c.pivot_table(index=["date", "hour"], columns=["model", "target", "setting"], values="y_pred")
    y = c.groupby(["date", "hour"])["y_true"].mean()
    scored = c.groupby(["date", "hour"])["scored"].min().astype(bool)
    for kind in DEEP:
        path = f"{R}/{zone}_{kind}.parquet"
        if os.path.exists(path):
            d = pd.read_parquet(path)
            d = d[d.model == kind]
            d["date"] = pd.to_datetime(d["date"])
            W[(kind, "level", "-")] = d.set_index(["date", "hour"])["y_pred"].reindex(W.index)
    P, _ = fd.daily_matrix(fd.prepare_hourly(screening.load_until_2024(zone))[fd.TARGET])
    for name, k in [("naive_d1", 1), ("naive_d7", 7)]:
        s = P.shift(k).stack(future_stack=True)
        s.index.names = ["date", "hour"]
        W[(name, "-", "-")] = s.reindex(W.index).values
    keep = W.notna().all(axis=1) & scored
    return W[keep], y[keep]


def metrics(y, pred, naive):
    mae = float(np.mean(np.abs(y - pred)))
    return {"n_hours": int(len(y)), "MAE": mae, "rMAE": mae / float(np.mean(np.abs(y - naive)))}


def summarise(R, OUT):
    os.makedirs(OUT, exist_ok=True)
    frames = {z: zone_frame(R, z) for z in ZONES}
    rows = []
    for z, (W, y) in frames.items():
        for col in W.columns:
            rows.append({"zone": z, "model": col[0], "target": col[1], "setting": col[2], **metrics(y, W[col], W[("naive_d7", "-", "-")])})
    t = pd.DataFrame(rows)

    no = t[t.zone.isin(NO)]
    chosen = {}                                                # (model, target) -> chosen setting
    for (model, target), g in no[no.model.str.startswith("lgbm")].groupby(["model", "target"]):
        chosen[(model, target)] = g.groupby("setting").MAE.mean().idxmin()
    best_target = {}
    for model in ["lgbm_pooled", "lgbm_hourly"]:
        cand = {tg: no[(no.model == model) & (no.target == tg) & (no.setting == s)].MAE.mean() for (m, tg), s in chosen.items() if m == model}
        best_target[model] = min(cand, key=cand.get)
    lear = no[no.model == "lear"].groupby("target").MAE.mean()
    best_target["lear"] = lear.idxmin()

    ens_rows = []
    lear_col = ("lear", best_target["lear"], "aic")
    hour_col = ("lgbm_hourly", best_target["lgbm_hourly"], chosen[("lgbm_hourly", best_target["lgbm_hourly"])])
    for z, (W, y) in frames.items():
        ens_rows.append({"zone": z, "model": "ensemble", "target": f"{best_target['lear']} + {best_target['lgbm_hourly']}",
                         "setting": hour_col[2], **metrics(y, (W[lear_col] + W[hour_col]) / 2, W[("naive_d7", "-", "-")])})
    t = pd.concat([t, pd.DataFrame(ens_rows)], ignore_index=True)
    t["chosen_setting"] = [(m, tg) in chosen and chosen[(m, tg)] == s for m, tg, s in zip(t.model, t.target, t.setting)]
    t["best_target"] = [(m in best_target and best_target[m] == tg and (m == "lear" or chosen.get((m, tg)) == s)) or m == "ensemble"
                        for m, tg, s in zip(t.model, t.target, t.setting)]
    t.sort_values(["zone", "model", "target", "setting"]).to_csv(f"{OUT}/screening_2024.csv", index=False)

    trees = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/*_trees.csv"))], ignore_index=True)
    trees["chosen"] = [chosen.get((m, tg)) == s for m, tg, s in zip(trees.model, trees.target, trees.setting)]
    epochs = [pd.read_csv(f) for f in sorted(glob.glob(f"{R}/*_epochs.csv"))]

    pd.set_option("display.width", 220)
    sel = t[(t.chosen_setting | t.model.isin(["lear", "naive_d1", "naive_d7", "ensemble"] + DEEP))]
    wide = lambda col: sel.assign(version=sel.model + " | " + sel.target).pivot_table(index="version", columns="zone", values=col)[ZONES]
    lines = ["Screening, evaluated on 2024 (no 2025 data used). Selection on the mean 2024 MAE over NO1-NO5.", "",
             "Chosen LightGBM setting per layout and target"]
    for (m, tg), s in sorted(chosen.items()):
        g = no[(no.model == m) & (no.target == tg)].groupby("setting").MAE.mean().sort_values()
        lines.append(f"  {m:12s} {tg:12s} {s:22s} mean NO MAE {g.iloc[0]:.3f} (next best {g.index[1]} {g.iloc[1]:.3f})")
    lines += ["", "Best target per family (mean NO MAE)",
              f"  LEAR: {best_target['lear']}  ({', '.join(f'{k} {v:.3f}' for k, v in lear.items())})"]
    for m in ["lgbm_pooled", "lgbm_hourly"]:
        lines.append(f"  {m}: {best_target[m]}  (" + ", ".join(f"{tg} {no[(no.model == m) & (no.target == tg) & (no.setting == chosen[(m, tg)])].MAE.mean():.3f}"
                                                            for tg in screening.TARGETS) + ")")
    lines += [f"  ensemble = LEAR {best_target['lear']} + per-hour LightGBM {best_target['lgbm_hourly']} ({hour_col[2]})", "",
              "rMAE in 2024 (vs naive D-7), chosen settings", wide("rMAE").round(3).to_string(), "",
              "MAE in 2024 (EUR/MWh), chosen settings", wide("MAE").round(2).to_string(), "",
              "Early stopping runs that reached the 1,000-tree cap (share of fits)",
              trees.groupby(["model", "target"]).cap_hit.mean().round(3).to_string(),
              f"  all fits: {trees.cap_hit.mean():.3f} of {len(trees):,};  fits of the chosen settings: {trees[trees.chosen].cap_hit.mean():.3f} "
              f"of {int(trees.chosen.sum()):,}", "",
              "Cap hits by setting (share of fits)", trees.pivot_table(index="setting", columns="model", values="cap_hit").round(3).to_string()]
    if epochs:
        e = pd.concat(epochs)
        lines += ["", "Deep models: best epoch on 2023 (used for the refit), by zone and seed",
                  e.pivot_table(index=["model", "zone"], columns="seed", values="best_epoch").to_string()]
    open(f"{OUT}/screening_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    for R, OUT in RUNS:
        print(f"\n##### {R}")
        summarise(R, OUT)


if __name__ == "__main__":
    main()
