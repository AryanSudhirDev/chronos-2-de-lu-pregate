"""Daily 24-hour panel in Berlin local time, and the sealed holdout.

Every delivery day has 24 values: on the 23-hour spring DST day the missing 02:00 is
interpolated, on the 25-hour autumn day the two 02:00 hours are averaged (epftoolbox convention).

Forecast for delivery day D is issued on D-1 before the 12:00 gate. Information set:
  * prices of every zone through the end of D-1 (cleared in the D-2 auction)
  * TSO day-ahead forecasts for D: load (published before the gate) and wind/solar
    (published 18:00 on D-1, i.e. after the gate; the standard EPF-literature setup, see README)
  * actual load/generation through the end of D-2
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_START = pd.Timestamp("2025-11-01")  # sealed: delivery days from here on
DEV_START = pd.Timestamp("2022-01-01")       # walk-forward evaluation window on dev data
TZ = "Europe/Berlin"

PRICE_ZONES = ["de", "be", "fr", "nl", "pl", "dk1", "at", "ch", "cz"]


class Sealed(RuntimeError):
    pass


@dataclass
class Panel:
    days: pd.DatetimeIndex          # delivery days (naive dates)
    x: dict[str, np.ndarray]        # name -> (n_days, 24)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.x[k]

    def idx(self, day) -> int:
        return int(self.days.get_loc(pd.Timestamp(day)))

    def to_frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.x[name], index=self.days, columns=range(24))


def _to_daily(s: pd.Series) -> pd.DataFrame:
    loc = s.tz_convert(TZ)
    df = pd.DataFrame({"v": loc.values, "day": loc.index.tz_localize(None).normalize(),
                       "hour": loc.index.hour})
    # autumn DST day has 02:00 twice -> mean; spring day lacks 02:00 -> interpolate
    wide = df.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=range(24))
    wide = wide.interpolate(axis=1, limit_area="inside")
    return wide


def load_panel(allow_holdout: bool = False, path: Path | None = None) -> Panel:
    """Load the daily panel. Holdout days are cut off unless allow_holdout=True,
    which only scripts/open_holdout.py may pass (it records the opening in the ledger)."""
    if allow_holdout and os.environ.get("EPF_OPEN_HOLDOUT") != "1":
        raise Sealed("holdout is sealed; open it only via scripts/open_holdout.py")
    raw = pd.read_parquet(path or ROOT / "data" / "hourly_raw.parquet")
    raw["ren_fc"] = raw[["wind_on_fc", "wind_off_fc", "solar_fc"]].sum(axis=1, min_count=3)
    raw["wind_fc"] = raw[["wind_on_fc", "wind_off_fc"]].sum(axis=1, min_count=2)
    raw["resid_fc"] = raw["load_fc"] - raw["ren_fc"]
    raw["ren_act"] = raw[["wind_on_act", "wind_off_act", "solar_act"]].sum(axis=1, min_count=3)
    raw["resid_act"] = raw["load_act"] - raw["ren_act"]
    cols = [f"price_{z}" for z in PRICE_ZONES] + [
        "load_fc", "wind_fc", "solar_fc", "ren_fc", "resid_fc",
        "load_act", "ren_act", "resid_act"]
    daily = {c: _to_daily(raw[c]) for c in cols}
    days = daily["price_de"].dropna().index
    days = days[days >= pd.Timestamp("2018-10-01")]
    if not allow_holdout:
        days = days[days < HOLDOUT_START]
    x = {c: daily[c].reindex(days).to_numpy(dtype="float64") for c in cols}
    return Panel(days=pd.DatetimeIndex(days), x=x)


def with_pregate(panel: Panel, path: Path | None = None) -> Panel:
    """Adds wind_fc_pre and solar_fc_pre (scripts/make_pregate_fc.py); NaN on days without them."""
    df = pd.read_parquet(path or ROOT / "data" / "pregate_fc.parquet")
    for c in ("wind_fc_pre", "solar_fc_pre"):
        panel.x[c] = df.pivot(index="day", columns="hour", values=c).reindex(panel.days).to_numpy(dtype="float64")
    return panel


def eval_days(panel: Panel, start=DEV_START, end=None) -> pd.DatetimeIndex:
    d = panel.days[panel.days >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    return d
