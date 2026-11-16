"""Online blend of saved forecasts. Weights for day D use only errors through D-1, whose prices are
known before the D gate. Weight of a model = (exponentially weighted recent loss) ** -k.

    python scripts/run_online_blend.py --ids a,b,c --tag pool7 [--halflife 30] [--k 20] [--per_hour] [--loss mae|regret]
"""
import argparse

import numpy as np

from epf.blend import online
from epf.data import load_panel
from epf.ledger import FC, record

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--halflife", type=float, default=30)
ap.add_argument("--k", type=float, default=20)
ap.add_argument("--per_hour", action="store_true")
ap.add_argument("--loss", choices=["mae", "regret"], default="mae")
a = ap.parse_args()
assert not (a.per_hour and a.loss == "regret"), "battery regret is a whole-day loss"
kw = dict(halflife=a.halflife, k=a.k, per_hour=a.per_hour, loss=a.loss)

ids = a.ids.split(",")
runs = [np.load(FC / f"{i}.npz") for i in ids]
days = sorted(set.intersection(*(set(r["days"]) for r in runs)))
F = np.stack([r["point"][np.isin(r["days"], days)] for r in runs])
panel = load_panel()
y = np.array([panel["price_de"][panel.idx(d)] for d in days])
assert F.shape[1:] == y.shape
point = online(F, y, **kw)
leak = y.copy()
leak[-1] += 1000  # the last day's own price must not change any forecast
assert np.array_equal(online(F, leak, **kw), point), "blend weights use the day being forecast"
record(f"blend_{a.tag}", {"ids": ids, "halflife": a.halflife, "k": a.k, "per_hour": a.per_hour, "loss": a.loss},
       days, point, panel=panel)
