"""Long (day, hour) feature table for gradient boosting. Row (i, h) uses only the
information set of a forecast for day i issued on day i-1 (see data.py)."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .data import PRICE_ZONES, Panel

NEIGHBOURS = [z for z in PRICE_ZONES if z != "de"]


def tabular(panel: Panel, start: int = 7) -> pd.DataFrame:
    n = len(panel.days)
    I = np.repeat(np.arange(start, n), 24)
    H = np.tile(np.arange(24), n - start)
    p = panel["price_de"]
    de_hol = holidays.Germany(years=range(2018, 2028))
    days = panel.days[I]
    f = {
        "day_idx": I, "hour": H, "dow": days.dayofweek.values, "month": days.month.values,
        "doy_sin": np.sin(2 * np.pi * days.dayofyear.values / 365.25),
        "doy_cos": np.cos(2 * np.pi * days.dayofyear.values / 365.25),
        "holiday": np.array([d in de_hol for d in days.date]).astype(int),
        "t": I.astype(float),
    }
    for c in ["load_fc", "wind_fc", "solar_fc", "ren_fc", "resid_fc"]:
        x = panel[c]
        f[c] = x[I, H]
        f[f"{c}_dmean"] = x[I].mean(1)
        f[f"{c}_d1"] = x[I - 1, H]
    r = panel["resid_fc"]
    f["resid_fc_dmax"], f["resid_fc_dmin"] = r[I].max(1), r[I].min(1)
    f["resid_fc_rank"] = (r[I] < r[I, H][:, None]).sum(1)
    f["resid_fc_prev_h"] = r[I, np.maximum(H - 1, 0)]
    f["resid_fc_next_h"] = r[I, np.minimum(H + 1, 23)]
    f["ren_share"] = panel["ren_fc"][I, H] / panel["load_fc"][I, H]
    f["resid_fc_delta"] = r[I, H] - r[I - 1, H]
    for lag in (1, 2, 3, 7):
        f[f"p_d{lag}"] = p[I - lag, H]
    f["p_d1_mean"], f["p_d1_min"], f["p_d1_max"] = p[I - 1].mean(1), p[I - 1].min(1), p[I - 1].max(1)
    f["p_d1_std"], f["p_d1_last"] = p[I - 1].std(1), p[I - 1, 23]
    f["p_w1_mean"] = np.stack([p[I - k].mean(1) for k in range(1, 8)]).mean(0)
    for z in NEIGHBOURS:
        q = panel[f"price_{z}"]
        f[f"p_{z}_d1"] = q[I - 1, H]
        f[f"p_{z}_d1_mean"] = q[I - 1].mean(1)
    f["resid_act_d2"] = panel["resid_act"][I - 2, H]
    f["y"] = p[I, H]
    return pd.DataFrame(f)


FEATURES_EXCLUDE = {"day_idx", "y", "t"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]
