"""Online blend of forecasts: weights for day t use only errors through t-1, whose prices are known
before the gate of day t. Weight of a model = (exponentially weighted recent loss) ** -k."""
from __future__ import annotations

import numpy as np

from .metrics import battery_schedule


def _settle(y, f):
    s = battery_schedule(f)
    return float(np.sum(np.where(s > 0, -y / 0.95, np.where(s < 0, y * 0.95, 0.0))))


def _losses(F, y, per_hour, loss):
    """(models, days, 24 or 1): the loss of each model on each day."""
    if loss == "mae":
        e = np.abs(F - y)
        return e if per_hour else e.mean(-1, keepdims=True)
    return np.array([[[_settle(yt, yt) - _settle(yt, ft)] for ft, yt in zip(f, y)] for f in F])


def online(F, y, halflife=30.0, k=20.0, per_hour=False, loss="mae"):
    """F: (models, days, 24), y: (days, 24). Returns the blended forecast (days, 24)."""
    L, alpha = _losses(F, y, per_hour, loss), 1 - 0.5 ** (1 / halflife)
    ew, out = None, np.zeros(y.shape)
    for t in range(len(y)):
        w = np.ones_like(L[:, 0]) if ew is None else (ew + 1e-6) ** -k
        out[t] = (w / w.sum(0) * F[:, t]).sum(0)
        ew = L[:, t] if ew is None else (1 - alpha) * ew + alpha * L[:, t]
    return out
