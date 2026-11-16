"""Every evaluated variant is a trial: forecasts saved to runs/forecasts/, metrics appended
to runs/ledger.jsonl. Nothing is ever removed from the ledger."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import ROOT, Panel, load_panel
from .metrics import summarize

RUNS = ROOT / "runs"
LEDGER = RUNS / "ledger.jsonl"
FC = RUNS / "forecasts"


def naive_forecast(panel: Panel, days) -> np.ndarray:
    """epftoolbox naive: Mon, Sat, Sun use the same day last week; Tue-Fri use yesterday."""
    p = panel["price_de"]
    out = []
    for d in days:
        i = panel.idx(d)
        out.append(p[i - 7] if pd.Timestamp(d).dayofweek in (0, 5, 6) else p[i - 1])
    return np.array(out)


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def record(name: str, config: dict, days, point: np.ndarray, quantiles: np.ndarray | None = None,
           levels=None, panel: Panel | None = None, split: str = "dev", notes: str = ""):
    panel = panel or load_panel(allow_holdout=(split == "holdout"))
    days = pd.DatetimeIndex(days)
    y = np.array([panel["price_de"][panel.idx(d)] for d in days])
    naive = naive_forecast(panel, days)
    m = summarize(y, point, naive, quantiles, levels)
    key = json.dumps({"name": name, "config": config, "split": split,
                      "start": str(days[0].date()), "end": str(days[-1].date())}, sort_keys=True)
    rid = hashlib.sha1(key.encode()).hexdigest()[:10]
    FC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FC / f"{rid}.npz", days=days.values.astype("datetime64[D]").astype(str),
                        point=point, quantiles=quantiles if quantiles is not None else np.array([]),
                        levels=np.asarray(levels if levels is not None else []))
    row = {"id": rid, "name": name, "split": split, "config": config,
           "start": str(days[0].date()), "end": str(days[-1].date()), **m,
           "git": _git_rev(), "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "notes": notes}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{rid}] {name} {split} {row['start']}..{row['end']}: MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  "
          f"rMAE {m['rmae']:.3f}" + (f"  CRPS {m['crps']:.2f}" if 'crps' in m else "") +
          f"  battery {m['battery_pct_oracle']:.1f}% of oracle", flush=True)
    return row


def read_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame()
    return pd.read_json(LEDGER, lines=True)
