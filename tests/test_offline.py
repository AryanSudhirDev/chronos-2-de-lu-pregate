"""Isolation and time-causality checks; synthetic data only, no holdout access."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from epf.chronos_io import series_inputs
from epf.data import Panel, PRICE_ZONES, ROOT
from epf.features import feature_table
from epf.offline import ModelWorker


class OfflineTests(unittest.TestCase):
    def test_inputs_ignore_unreleased_outcomes(self):
        rng = np.random.default_rng(42)
        names = [f"price_{z}" for z in PRICE_ZONES] + ["load_fc", "wind_fc", "solar_fc",
                 "ren_fc", "resid_fc", "resid_act", "wind_fc_pre", "solar_fc_pre"]
        panel = Panel(pd.date_range("2024-01-01", periods=40),
                      {k: rng.normal(100, 10, (40, 24)) for k in names})
        t = 25
        poisoned = Panel(panel.days, {k: v.copy() for k, v in panel.x.items()})
        for name, values in poisoned.x.items():
            start = t if name.startswith("price_") else t - 1 if name.endswith("_act") else t + 1
            values[start:] = 1e9
        for mode in [{}, {"gate_strict": True}, {"pre_gate": True}]:
            a = series_inputs(panel, t, 14, neighbours=True, calendar=True, **mode)
            b = series_inputs(poisoned, t, 14, neighbours=True, calendar=True, **mode)
            np.testing.assert_array_equal(a["target"], b["target"])
            for group in ("past_covariates", "future_covariates"):
                for key in a[group]:
                    np.testing.assert_array_equal(a[group][key], b[group][key])
        a, b = feature_table(panel), feature_table(poisoned)
        columns = [c for c in a if c != "y"]
        np.testing.assert_array_equal(a.loc[a.day_idx == t, columns], b.loc[b.day_idx == t, columns])

    def test_worker_denies_files_and_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "unreleased.txt"
            private.write_text("synthetic future outcome")
            with ModelWorker("test", threads=1) as worker:
                paths = [private, ROOT / "data/hourly_raw.parquet", ROOT / "runs/ledger.jsonl",
                         ROOT / "runs/forecasts/8851fdfc81.npz"]
                alternate = Path("/System/Volumes/Data") / str(ROOT / "data/hourly_raw.parquet").lstrip("/")
                if alternate.exists():
                    paths.append(alternate)
                for path in paths:
                    if path.exists():
                        self.assertIn("denied", worker.call("probe", forbidden=str(path)))

    def test_numeric_worker_matches_local_fit(self):
        import lightgbm as lgb
        rng = np.random.default_rng(123)
        X = rng.normal(size=(200, 3))
        y = X[:, 0] * 3 + rng.normal(size=200)
        params = dict(n_estimators=10, num_leaves=7, min_child_samples=5, n_jobs=1, verbose=-1)
        expected = lgb.LGBMRegressor(**params).fit(X, y).predict(X[:7])
        with ModelWorker("lgbm", threads=1) as worker:
            worker.call("lgbm_fit", params=params, X=X, y=y)
            np.testing.assert_array_equal(expected, worker.call("lgbm_predict", X=X[:7]))

    def test_blend_uses_only_released_targets(self):
        from epf.blend import online
        rng = np.random.default_rng(17)
        F, y = rng.normal(size=(3, 8, 24)), rng.normal(size=(8, 24))
        cfg = dict(halflife=30, k=20, per_hour=False, loss="mae")
        expected = online(F, y, **cfg)
        with ModelWorker("blend", threads=1) as worker:
            for t in range(len(y)):
                released = np.concatenate([y[:t], np.zeros((1, 24))])
                got = worker.call("blend", F=F[:, :t + 1], y=released, config=cfg)
                np.testing.assert_array_equal(got, expected[t])


if __name__ == "__main__":
    unittest.main()
