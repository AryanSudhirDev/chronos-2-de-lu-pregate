"""Simple average of saved forecasts, on the days they all cover.

    python scripts/run_ensemble.py --ids 7a7adf81e5,69b2219339 --tag zs_lgbm
"""
import argparse

import numpy as np

from epf.data import load_panel
from epf.ledger import FC, record

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

ids = a.ids.split(",")
assert len(ids) > 1, "need at least two forecasts"
runs = [np.load(FC / f"{i}.npz") for i in ids]
days = sorted(set.intersection(*(set(r["days"]) for r in runs)))
point = np.mean([r["point"][np.isin(r["days"], days)] for r in runs], axis=0)
assert len(point) == len(days)
record(f"ensemble_{a.tag}", {"ids": ids, "weights": "equal"}, days, point, panel=load_panel())
