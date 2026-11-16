"""Long (day, hour) feature table for gradient-boosted models. Row for delivery day i uses only
the information set of data.py: prices through D-1, TSO forecasts for D, actuals through D-2."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .data import PRICE_ZONES, Panel


def feature_table(panel: Panel, pre_gate=False) -> pd.DataFrame:
    n = len(panel.days)
    p = panel["price_de"]
    de_hol = holidays.Germany(years=range(2018, 2028))

    def lag(a, k):
        out = np.full_like(a, np.nan)
        out[k:] = a[:-k]
        return out

    cols: dict[str, np.ndarray] = {}
    H = np.tile(np.arange(24), (n, 1))
    cols["hour"] = H
    dow = panel.days.dayofweek.to_numpy()
    cols["dow"] = np.repeat(dow[:, None], 24, 1)
    cols["month"] = np.repeat(panel.days.month.to_numpy()[:, None], 24, 1)
    doy = panel.days.dayofyear.to_numpy()
    cols["doy_sin"] = np.repeat(np.sin(2 * np.pi * doy / 365.25)[:, None], 24, 1)
    cols["doy_cos"] = np.repeat(np.cos(2 * np.pi * doy / 365.25)[:, None], 24, 1)
    hol = np.array([d in de_hol for d in panel.days.date])
    cols["holiday"] = np.repeat(hol[:, None], 24, 1).astype(float)
    cols["holiday_d1"] = np.repeat(lag(hol.astype(float)[:, None], 1), 24, 1)

    p1, p2, p7 = lag(p, 1), lag(p, 2), lag(p, 7)
    cols["p_d1"], cols["p_d2"], cols["p_d7"] = p1, p2, p7
    for name, f in [("mean", np.mean), ("min", np.min), ("max", np.max), ("std", np.std)]:
        cols[f"p_d1_{name}"] = np.repeat(f(p1, axis=1)[:, None], 24, 1)
    cols["p_d1_last"] = np.repeat(p1[:, 23:24], 24, 1)
    week = np.stack([lag(p, k) for k in range(1, 8)])  # (7, n, 24)
    cols["p_w_mean"] = np.repeat(np.nanmean(week, axis=(0, 2))[:, None], 24, 1)
    cols["p_w_std"] = np.repeat(np.nanstd(week, axis=(0, 2))[:, None], 24, 1)
    cols["p_w_hour_mean"] = np.nanmean(week, axis=0)

    for z in PRICE_ZONES[1:]:
        pz1 = lag(panel[f"price_{z}"], 1)
        cols[f"p_{z}_d1"] = pz1
        cols[f"p_{z}_d1_mean"] = np.repeat(np.nanmean(pz1, axis=1)[:, None], 24, 1)

    current = {c: panel[c] for c in ["load_fc", "wind_fc", "solar_fc", "ren_fc", "resid_fc"]}
    if pre_gate:
        current["wind_fc"], current["solar_fc"] = panel["wind_fc_pre"], panel["solar_fc_pre"]
        current["ren_fc"] = current["wind_fc"] + current["solar_fc"]
        current["resid_fc"] = panel["load_fc"] - current["ren_fc"]
    for c in current:
        a = panel[c]
        cols[c] = current[c]
        cols[f"{c}_mean"] = np.repeat(current[c].mean(axis=1)[:, None], 24, 1)
        cols[f"{c}_d1"] = lag(a, 1)
    r = current["resid_fc"]
    cols["resid_fc_max"] = np.repeat(r.max(axis=1)[:, None], 24, 1)
    cols["resid_fc_min"] = np.repeat(r.min(axis=1)[:, None], 24, 1)
    cols["resid_fc_rank"] = np.argsort(np.argsort(r, axis=1), axis=1).astype(float)
    cols["resid_fc_diff_d1"] = r - lag(panel["resid_fc"], 1)
    cols["ren_share_fc"] = current["ren_fc"] / panel["load_fc"]
    cols["resid_act_d2"] = lag(panel["resid_act"], 2)

    df = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df["day_idx"] = np.repeat(np.arange(n), 24)
    df["y"] = p.reshape(-1)
    return df
