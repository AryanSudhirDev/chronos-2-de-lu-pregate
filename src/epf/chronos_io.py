"""Build Chronos-2 inputs from the daily panel (flattened to an hourly series of 24 values/day)."""
from __future__ import annotations

import holidays
import numpy as np

from .data import PRICE_ZONES, Panel

FUTURE_COVS = ["load_fc", "wind_fc", "solar_fc"]
NEIGHBOURS = [f"price_{z}" for z in PRICE_ZONES[1:] if z != "pl"]  # PL starts late 2019


_OFF: dict[int, np.ndarray] = {}


def _offday(panel: Panel) -> np.ndarray:
    """1 on weekends and German public holidays, as a (n_days, 24) array."""
    if id(panel) not in _OFF:
        hol = holidays.Germany(years=range(2018, 2028))
        off = np.array([d.weekday() >= 5 or d in hol for d in panel.days.date], dtype="float32")
        _OFF[id(panel)] = np.repeat(off[:, None], 24, 1)
    return _OFF[id(panel)]


def series_inputs(panel: Panel, t: int, context_days: int, covs=True, neighbours=False,
                  calendar=False, gate_strict=False, pre_gate=False) -> dict:
    """Inputs to forecast delivery day index t from days [t-context_days, t).
    gate_strict: day-D wind/solar forecasts are published after the gate, so only load_fc is future-known.
    pre_gate: use wind/solar forecasts built from pre-gate weather runs instead (see data.with_pregate)."""
    lo = max(0, t - context_days)
    flat = lambda a, a0, a1: a[a0:a1].reshape(-1).astype("float32")
    item = {"target": flat(panel["price_de"], lo, t)}
    past, fut = {}, {}
    if covs:
        for c in FUTURE_COVS:
            past[c] = flat(panel[c], lo, t)
            if pre_gate and c != "load_fc":
                fut[c] = flat(panel[f"{c}_pre"], t, t + 1)
            elif not gate_strict or c == "load_fc":
                fut[c] = flat(panel[c], t, t + 1)
    if calendar:
        off = _offday(panel)
        past["offday"], fut["offday"] = flat(off, lo, t), flat(off, t, t + 1)
    if neighbours:
        for c in NEIGHBOURS:
            past[c] = flat(panel[c], lo, t)
    if past:
        item["past_covariates"] = past
    if fut:
        item["future_covariates"] = fut
    return item


def training_series(panel: Panel, end: int, start: int = 0, covs=True, neighbours=False,
                    calendar=False, gate_strict=False, zone="de") -> dict:
    """One long series over days [start, end) for fine-tuning; known-future covariates are
    declared with None in future_covariates. zone: market to use as the target; its own channel
    among the neighbour covariates then carries the German price, so all series share one schema."""
    flat = lambda a: a[start:end].reshape(-1).astype("float32")
    item = {"target": flat(panel[f"price_{zone}"])}
    past, fut = {}, {}
    if covs:
        for c in FUTURE_COVS:
            past[c] = flat(panel[c])
            if not gate_strict or c == "load_fc":
                fut[c] = None
    if calendar:
        past["offday"], fut["offday"] = flat(_offday(panel)), None
    if neighbours:
        for c in NEIGHBOURS:
            past[c] = flat(panel["price_de" if c == f"price_{zone}" else c])
    if past:
        item["past_covariates"] = past
    if fut:
        item["future_covariates"] = fut
    return item


def predict_days(pipe, panel: Panel, days, context_days=85, batch_size=64, **kw):
    """Returns point (median) forecasts (n, 24), quantiles (n, 24, k) and the quantile levels."""
    levels = np.asarray(pipe.quantiles, dtype=float)
    mid = int(np.argmin(np.abs(levels - 0.5)))
    items = [series_inputs(panel, panel.idx(d), context_days, **kw) for d in days]
    out = []
    for b in range(0, len(items), batch_size):
        preds = pipe.predict(items[b:b + batch_size], prediction_length=24, batch_size=batch_size)
        out += [p[0].float().cpu().numpy().T for p in preds]  # (24, k)
    q = np.stack(out)
    return q[..., mid], q, levels
