"""Point, probabilistic and economic metrics for day-ahead (n_days, 24) forecasts."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def smape(y, f):
    return float(np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f) + 1e-9)) * 100)


def rmae(y, f, naive):
    return mae(y, f) / mae(y, naive)


def crps_from_quantiles(y, q, levels):
    """Mean pinball loss over quantile levels, times 2 (approximates CRPS).
    y: (n, 24); q: (n, 24, k); levels: (k,)"""
    levels = np.asarray(levels)
    d = y[..., None] - q
    pin = np.maximum(levels * d, (levels - 1) * d)
    return float(2 * pin.mean())


def coverage(y, q, levels, lo=0.1, hi=0.9):
    levels = list(np.round(levels, 4))
    a, b = q[..., levels.index(lo)], q[..., levels.index(hi)]
    return float(np.mean((y >= a) & (y <= b)))


def dm_test(y, f1, f2, norm=1):
    """Multivariate Diebold-Mariano (Ziel & Weron 2018): daily loss differential of the
    L-norm of the 24-hour error vectors. One-sided p-value for H1: f2 more accurate than f1."""
    e1 = np.linalg.norm(y - f1, ord=norm, axis=1)
    e2 = np.linalg.norm(y - f2, ord=norm, axis=1)
    d = e1 - e2
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(1 - stats.norm.cdf(t))


def battery_schedule(prices, eta=0.95, cap=1.0):
    """Optimal hourly schedule for a 1 MW / 1 MWh battery that starts and ends the day empty.
    Dynamic programme over state of charge {0, cap}. Returns actions in MWh at the grid:
    +1 = buy (charge), -1 = sell (discharge), 0 = idle."""
    n = len(prices)
    NEG = -1e18
    v = np.full((n + 1, 2), NEG)
    v[n, 0] = 0.0
    act = np.zeros((n, 2), dtype=int)
    for t in range(n - 1, -1, -1):
        p = prices[t]
        idle0, charge = v[t + 1, 0], v[t + 1, 1] - p * cap / eta
        v[t, 0], act[t, 0] = (charge, 1) if charge > idle0 else (idle0, 0)
        idle1, dis = v[t + 1, 1], v[t + 1, 0] + p * cap * eta
        v[t, 1], act[t, 1] = (dis, -1) if dis > idle1 else (idle1, 0)
    s, out = 0, np.zeros(n)
    for t in range(n):
        a = act[t, s]
        out[t] = a
        s = s + (a == 1) - (a == -1)
    return out


def battery_profit(y, f, eta=0.95, cap=1.0):
    """Realised profit (EUR) of scheduling on forecast f and settling at actual prices y,
    with the profit of perfect foresight for reference."""
    real, oracle = 0.0, 0.0
    for yt, ft in zip(y, f):
        a = battery_schedule(ft, eta, cap)
        real += float(np.sum(np.where(a > 0, -yt * cap / eta, np.where(a < 0, yt * cap * eta, 0.0))))
        b = battery_schedule(yt, eta, cap)
        oracle += float(np.sum(np.where(b > 0, -yt * cap / eta, np.where(b < 0, yt * cap * eta, 0.0))))
    return real, oracle


def summarize(y, f, naive, q=None, levels=None):
    out = {"mae": mae(y, f), "rmse": rmse(y, f), "smape": smape(y, f), "rmae": rmae(y, f, naive)}
    if q is not None:
        out["crps"] = crps_from_quantiles(y, q, levels)
        out["cov80"] = coverage(y, q, levels)
    profit, oracle = battery_profit(y, f)
    out["battery_eur"] = profit
    out["battery_pct_oracle"] = 100 * profit / oracle
    out["n_days"] = int(len(y))
    return out
