"""Fine-tune Chronos-2 walk-forward: for each test year, train on all days before it (minus a
validation block used for early stopping), then forecast that year's days.

    python scripts/run_chronos_finetune.py --mode lora --lr 1e-4 --steps 1000 [--calendar]
"""
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from epf.chronos_io import predict_days, training_series
from epf.data import ROOT, eval_days, load_panel, with_pregate
from epf.ledger import record  # noqa  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["lora", "full"], default="lora")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--context_days", type=int, default=85)
ap.add_argument("--val_days", type=int, default=60)
ap.add_argument("--lora_r", type=int, default=8)
ap.add_argument("--calendar", action="store_true")
ap.add_argument("--neighbours", action="store_true")
ap.add_argument("--gate_strict", action="store_true", help="no day-D wind/solar forecasts (published after the gate)")
ap.add_argument("--pre_gate", action="store_true", help="forecast with wind/solar from pre-gate weather runs (training keeps TSO forecasts)")
ap.add_argument("--zones", default="de", help="comma-separated markets to train on; forecasts and validation stay on DE")
ap.add_argument("--folds", default="2022,2023,2024,2025")
ap.add_argument("--tag", default="")
a = ap.parse_args()
assert a.zones == "de" or a.neighbours, "non-German targets need the neighbour channels"

dev = "mps" if torch.backends.mps.is_available() else "cpu"
panel = load_panel()
days = eval_days(panel)
if a.pre_gate:
    panel = with_pregate(panel)
    days = days[np.isfinite(panel["wind_fc_pre"]).all(1)[panel.days >= days[0]]]
kw = dict(covs=True, neighbours=a.neighbours, calendar=a.calendar, gate_strict=a.gate_strict)
ctx = a.context_days * 24
years = [int(y) for y in a.folds.split(",")]
points, quants, logs = [], [], []
t_all = time.time()
for y in years:
    fold_days = days[days.year == y]
    t0 = panel.idx(fold_days[0])
    v0 = t0 - a.val_days
    train = [training_series(panel, end=v0, zone=z, **kw) for z in a.zones.split(",")]
    val = [training_series(panel, end=d + 1, start=d + 1 - a.context_days, **kw) for d in range(v0, t0)]
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
    out_dir = ROOT / "runs" / "ckpt" / f"{a.mode}_{y}"
    lora = {"r": a.lora_r, "lora_alpha": 2 * a.lora_r,
            "target_modules": ["q", "k", "v", "o", "output_patch_embedding.output_layer"]} \
        if a.mode == "lora" else None
    ft = pipe.fit(train, prediction_length=24, validation_inputs=val, finetune_mode=a.mode,
                  lora_config=lora, context_length=ctx, learning_rate=a.lr, num_steps=a.steps,
                  batch_size=a.batch, output_dir=out_dir, min_past=7 * 24)
    point, q, levels = predict_days(ft, panel, fold_days, a.context_days, pre_gate=a.pre_gate, **kw)
    points.append(point)
    quants.append(q)
    yt = np.array([panel["price_de"][panel.idx(d)] for d in fold_days])
    logs.append(f"{y}: MAE {np.abs(point - yt).mean():.2f}")
    print(f"fold {y}: MAE {np.abs(point - yt).mean():.2f}  ({time.time() - t_all:.0f}s)", flush=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    del pipe, ft
    torch.mps.empty_cache() if dev == "mps" else None

cfg = {k: v for k, v in vars(a).items() if k != "tag"}
for flag in ("gate_strict", "pre_gate"):  # absent when off, so earlier ledger ids stay valid
    if not cfg[flag]:
        del cfg[flag]
if a.zones == "de":
    del cfg["zones"]
name = f"chronos2_ft_{a.mode}" + (f"_{a.tag}" if a.tag else "")
record(name, cfg, days[np.isin(days.year, years)], np.concatenate(points), np.concatenate(quants), levels,
       panel=panel, notes="; ".join(logs) + f"; {time.time() - t_all:.0f}s")
