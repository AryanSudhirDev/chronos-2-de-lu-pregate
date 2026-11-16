"""LEAR (Lago et al. 2021, epftoolbox): per-hour Lasso on asinh-transformed inputs,
recalibrated daily, averaged over calibration windows."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import Lasso, LassoLarsIC

from .data import Panel

WINDOWS = (56, 84, 728, 1092)


def lear_features(panel: Panel, i: int, pre_gate=False) -> np.ndarray:
    p, load, ren = panel["price_de"], panel["load_fc"], panel["ren_fc"]
    dow = np.zeros(7)
    dow[panel.days[i].dayofweek] = 1
    return np.concatenate([p[i - 1], p[i - 2], p[i - 3], p[i - 7],
                           load[i], load[i - 1], load[i - 7],
                           panel["wind_fc_pre"][i] + panel["solar_fc_pre"][i] if pre_gate else ren[i],
                           ren[i - 1], ren[i - 7], dow])


class Invariant:
    """Median/MAD standardisation followed by asinh (Uniejewski et al. 2018)."""

    def fit(self, a):
        self.med = np.median(a, axis=0)
        mad = np.median(np.abs(a - self.med), axis=0) / 0.6745
        self.mad = np.where(mad < 1e-9, 1.0, mad)
        return self

    def transform(self, a):
        return np.arcsinh((a - self.med) / self.mad)

    def inverse(self, a):
        return np.sinh(a) * self.mad + self.med


def lear_forecast(panel: Panel, t: int, X_all: np.ndarray, windows=WINDOWS) -> np.ndarray:
    """Forecast day index t, training on the `w` days before t. X_all[i] = features of day i."""
    y_all = panel["price_de"]
    preds = []
    noise = np.full(24, np.nan)  # per-hour residual variance from the longest window
    for w in sorted(windows, reverse=True):
        lo = max(7, t - w)
        X, Y = X_all[lo:t], y_all[lo:t]
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)  # no effect on dev, which has no gaps
        X, Y = X[ok], Y[ok]
        sx, sy = Invariant().fit(X[:, :-7]), Invariant().fit(Y)
        Xs = np.hstack([sx.transform(X[:, :-7]), X[:, -7:]])
        Ys = sy.transform(Y)
        xt = np.hstack([sx.transform(X_all[t:t + 1, :-7]), X_all[t:t + 1, -7:]])
        out = np.zeros(24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for h in range(24):
                # sklearn cannot estimate the noise variance when days < features
                nv = noise[h] if len(Ys) <= Xs.shape[1] + 1 else None
                alpha = LassoLarsIC(criterion="aic", max_iter=2500, noise_variance=nv).fit(Xs, Ys[:, h]).alpha_
                m = Lasso(alpha=alpha, max_iter=2500).fit(Xs, Ys[:, h])
                if np.isnan(noise[h]):
                    noise[h] = max(np.var(Ys[:, h] - m.predict(Xs)), 1e-6)
                out[h] = m.predict(xt)[0]
        preds.append(sy.inverse(out[None, :])[0])
    return np.mean(preds, axis=0)
