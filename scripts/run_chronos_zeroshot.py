"""Zero-shot Chronos-2 on the dev window.

    python scripts/run_chronos_zeroshot.py --context_days 85 [--no_covs] [--neighbours] [--calendar]
"""
import argparse
import time

import numpy as np
import torch
from chronos import Chronos2Pipeline

from epf.chronos_io import predict_days
from epf.data import eval_days, load_panel, with_pregate
from epf.ledger import record

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="amazon/chronos-2")
ap.add_argument("--context_days", type=int, default=85)
ap.add_argument("--no_covs", action="store_true")
ap.add_argument("--neighbours", action="store_true")
ap.add_argument("--calendar", action="store_true")
ap.add_argument("--gate_strict", action="store_true", help="no day-D wind/solar forecasts (published after the gate)")
ap.add_argument("--pre_gate", action="store_true", help="day-D wind/solar forecasts from pre-gate weather runs")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
a = ap.parse_args()

panel = load_panel()
days = eval_days(panel)
if a.pre_gate:
    panel = with_pregate(panel)
    days = days[np.isfinite(panel["wind_fc_pre"]).all(1)[panel.days >= days[0]]]
pipe = Chronos2Pipeline.from_pretrained(a.model, device_map=a.device)
t0 = time.time()
point, q, levels = predict_days(pipe, panel, days, a.context_days, covs=not a.no_covs,
                                neighbours=a.neighbours, calendar=a.calendar, gate_strict=a.gate_strict, pre_gate=a.pre_gate)
cfg = {"model": a.model, "context_days": a.context_days, "covs": not a.no_covs,
       "neighbours": a.neighbours, "calendar": a.calendar, "mode": "zero-shot"}
if a.gate_strict:  # absent otherwise, so earlier ledger ids stay valid
    cfg["gate_strict"] = True
if a.pre_gate:
    cfg["pre_gate"] = True
record("chronos2_zs", cfg, days, point, q, levels, panel=panel, notes=f"{time.time() - t0:.0f}s on {a.device}")
