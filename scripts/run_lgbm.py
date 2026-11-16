"""LightGBM baseline: one model for all hours, refit every month on all prior days (or a rolling
window), target = price minus yesterday's mean price.

    python scripts/run_lgbm.py [--window 0] [--objective l1] [--tag v1]
    python scripts/run_lgbm.py --base <ledger id of Chronos forecasts>  # stack: target = price minus Chronos median
    python scripts/run_lgbm.py --model tabpfn2 --window 730 --n_estimators 4  # same protocol with TabPFN-2 (CPU)
"""
import argparse

import numpy as np
import pandas as pd

from epf.data import eval_days, load_panel
from epf.features import feature_table
from epf.ledger import FC, record

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=0, help="training days (0 = expanding)")
ap.add_argument("--objective", default="l1")
ap.add_argument("--n_estimators", type=int, default=1500)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="")
ap.add_argument("--gate_strict", action="store_true", help="drop features built from day-D wind/solar forecasts")
ap.add_argument("--halflife", type=float, default=0, help="recency weighting of training days, half-life in days; 0 = none")
ap.add_argument("--base", default="", help="stack on these saved Chronos forecasts (ledger id)")
ap.add_argument("--burn_in", default="2023-01-01", help="with --base: first scored day; earlier base forecasts only train")
ap.add_argument("--model", choices=["lgbm", "tabpfn2"], default="lgbm")
ap.add_argument("--max_rows", type=int, default=3000, help="tabpfn2: random training rows per refit")
a = ap.parse_args()
assert a.model == "lgbm" or not a.halflife, "TabPFN takes no sample weights"
if a.model == "lgbm":  # torch and lightgbm segfault together on macOS
    import lightgbm as lgb
else:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(a.threads)

panel = load_panel()
days = eval_days(panel)
df = feature_table(panel)
anchor = "p_d1_mean"
if a.base:
    z = np.load(FC / f"{a.base}.npz")
    q, lv = z["quantiles"], z["levels"].round(2).tolist()
    idx = [panel.idx(d) for d in z["days"]]
    n0 = len(idx) * 24
    base = pd.DataFrame({"day_idx": np.repeat(idx, 24), "hour": np.tile(np.arange(24), len(idx)),
                         "cf_p50": z["point"].ravel(), "cf_q10": q[..., lv.index(0.1)].ravel(),
                         "cf_q90": q[..., lv.index(0.9)].ravel()})
    base["cf_spread"] = base.cf_q90 - base.cf_q10
    base["cf_day_mean"] = base.groupby("day_idx").cf_p50.transform("mean")
    df = df.merge(base, on=["day_idx", "hour"])
    assert len(df) == n0, "base forecasts did not align with the feature table"
    anchor = "cf_p50"
    days = days[days >= pd.Timestamp(a.burn_in)]
feats = [c for c in df.columns if c not in ("y", "day_idx")]
if a.gate_strict:  # lagged *_d1 forecasts stay: they were published before this gate
    post_gate = {"wind_fc", "wind_fc_mean", "solar_fc", "solar_fc_mean", "ren_fc", "ren_fc_mean", "resid_fc",
                 "resid_fc_mean", "resid_fc_max", "resid_fc_min", "resid_fc_rank", "resid_fc_diff_d1", "ren_share_fc"}
    assert post_gate <= set(feats)
    feats = [c for c in feats if c not in post_gate]
df["target"] = df["y"] - df[anchor]
params = dict(objective=a.objective, n_estimators=a.n_estimators, learning_rate=a.lr, num_leaves=a.leaves,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=1.0, verbose=-1, n_jobs=a.threads)

preds = np.zeros((len(days), 24))
months = pd.Series(days).dt.to_period("M")
for m in months.unique():
    sel = np.where(months == m)[0]
    t0 = panel.idx(days[sel[0]])
    lo = 7 if a.window == 0 else max(7, t0 - a.window)
    tr = df[(df.day_idx >= lo) & (df.day_idx < t0)].dropna(subset=["target", "p_d7"])
    weight = 0.5 ** ((t0 - tr["day_idx"]) / a.halflife) if a.halflife else None
    if a.model == "lgbm":
        model = lgb.LGBMRegressor(**params).fit(tr[feats], tr["target"], sample_weight=weight)
    else:
        tr = tr.sample(min(a.max_rows, len(tr)), random_state=0)
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=a.n_estimators,
                                                           ignore_pretraining_limits=True, random_state=0)
        model.fit(tr[feats], tr["target"])
    te_idx = [panel.idx(days[s]) for s in sel]
    te = df[df.day_idx.isin(te_idx)].sort_values(["day_idx", "hour"])
    preds[sel] = (model.predict(te[feats]) + te[anchor].to_numpy()).reshape(-1, 24)
    print(m, end=" ", flush=True)
print()
cfg = {k: v for k, v in vars(a).items() if k not in ("tag", "threads")} | {"target": f"y - {anchor}", "refit": "monthly",
                                                           "n_features": len(feats)}
if not a.base:  # keep baseline config (and its ledger id) unchanged
    del cfg["base"], cfg["burn_in"]
if not a.gate_strict:
    del cfg["gate_strict"]
if not a.halflife:
    del cfg["halflife"]
if a.model == "lgbm":
    del cfg["model"], cfg["max_rows"]
else:
    del cfg["objective"], cfg["lr"], cfg["leaves"]
record((a.model + "_stack" if a.base else a.model) + (f"_{a.tag}" if a.tag else ""), cfg, days, preds, panel=panel)
