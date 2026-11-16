"""LEAR benchmark, daily recalibration over the dev window (parallel over days).

    python scripts/run_lear.py [--windows 56 84 728 1092] [--workers 12]
"""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # one BLAS thread per worker process; otherwise threads oversubscribe

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epf.data import DEV_START, eval_days, load_panel
from epf.lear import WINDOWS, lear_features, lear_forecast
from epf.ledger import record

_P = _X = None


def _init(allow_holdout):
    global _P, _X
    _P = load_panel(allow_holdout=allow_holdout)
    _X = np.stack([lear_features(_P, i) if i >= 7 else np.full(247, np.nan) for i in range(len(_P.days))])


def _one(args):
    t, windows = args
    return lear_forecast(_P, t, _X, windows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="*", default=list(WINDOWS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", default=str(DEV_START.date()))
    ap.add_argument("--holdout", action="store_true", help="only from open_holdout.py; needs EPF_OPEN_HOLDOUT=1 and --save")
    ap.add_argument("--save", default="", help="write forecasts to this npz instead of the ledger")
    a = ap.parse_args()
    if a.holdout:
        raise SystemExit("holdout LEAR now runs only inside scripts/open_holdout.py --offline")
    assert a.save or not a.holdout, "holdout forecasts are recorded by open_holdout.py"
    panel = load_panel(allow_holdout=a.holdout)
    days = eval_days(panel, start=a.start)
    days = days[[bool(np.isfinite(lear_features(panel, panel.idx(d))).all()) for d in days]]  # a gap in the inputs: not forecastable
    ts = [panel.idx(d) for d in days]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.holdout,)) as ex:
        preds = np.array(list(ex.map(_one, [(t, tuple(a.windows)) for t in ts], chunksize=8)))
    if a.save:
        np.savez(a.save, days=days.values.astype("datetime64[D]").astype(str), point=preds)
    else:
        record("lear", {"windows": a.windows, "exog": ["load_fc", "ren_fc"]}, days, preds, panel=panel)
