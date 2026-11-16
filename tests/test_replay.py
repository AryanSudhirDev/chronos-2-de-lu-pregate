"""Decision replay tests; synthetic prices only."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from epf.data import ROOT, Panel, PRICE_ZONES
from epf.chronos_io import series_inputs
from epf.offline import ModelWorker
from epf.readiness import Battery, dispatch

spec = importlib.util.spec_from_file_location('replay_holdout', ROOT/'scripts/replay_holdout.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_normal_day_rejects_both_dst_changes(self):
        self.assertFalse(replay.normal_day('2026-03-29'))
        self.assertFalse(replay.normal_day('2026-10-25'))
        self.assertTrue(replay.normal_day('2026-03-28'))

    def test_actual_prices_align_utc_to_berlin_and_reject_missing_duplicate_hours(self):
        index = pd.date_range('2025-10-31 23:00', periods=24, freq='h', tz='UTC')
        raw = pd.Series(np.arange(24.), index=index)
        np.testing.assert_array_equal(replay.actual_day(raw, '2025-11-01'), raw)
        for invalid in [raw.iloc[1:], pd.concat([raw,raw.iloc[:1]])]:
            with self.assertRaises(ValueError): replay.actual_day(invalid, '2025-11-01')

    def test_worker_dispatch_has_no_actual_price_interface(self):
        forecast = np.full(24, 50.)
        forecast[0], forecast[18] = -50, 200
        b = Battery()
        with ModelWorker('replay_test', threads=1) as worker:
            result = worker.call('dispatch', forecast=forecast, assumptions=replay.asdict(b))
            for actual in [np.zeros(24), np.full(24, 1e9)]:
                with self.assertRaisesRegex(RuntimeError, 'only one day'):
                    worker.call('dispatch', forecast=forecast, assumptions=replay.asdict(b), actual=actual)
            for path in [ROOT/'data/hourly_raw.parquet', ROOT/'runs/ledger.jsonl', ROOT/'runs/forecasts/15830042bf.npz']:
                if path.exists():
                    self.assertIn('denied', worker.call('probe', forbidden=str(path)))
            repeat = worker.call('dispatch', forecast=forecast, assumptions=replay.asdict(b))
        np.testing.assert_array_equal(result, repeat)
        np.testing.assert_array_equal(result, dispatch(forecast, b))
        replay.validate_dispatch(*result, b)
        self.assertGreater(result[0][0], 0)
        self.assertLess(result[0][18], 0)

    def test_inputs_ignore_target_future_and_current_postgate_wind(self):
        rng = np.random.default_rng(9)
        names = [f'price_{z}' for z in PRICE_ZONES] + ['load_fc','wind_fc','solar_fc', 'wind_fc_pre','solar_fc_pre']
        panel = Panel(pd.date_range('2024-01-01', periods=40),
                      {k:rng.normal(size=(40,24)) for k in names})
        t = 25
        before = series_inputs(panel,t,14,pre_gate=True,neighbours=True)
        for name, values in panel.x.items():
            values[t if name.startswith('price_') or name in ('wind_fc','solar_fc') else t+1:] = 1e9
        after = series_inputs(panel,t,14,pre_gate=True,neighbours=True)
        np.testing.assert_array_equal(before['target'],after['target'])
        for group in ['past_covariates','future_covariates']:
            for name in before[group]:
                np.testing.assert_array_equal(before[group][name],after[group][name])

    def test_artifacts_cannot_be_replaced_and_bad_dispatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'decisions.json'
            replay.durable(path, {'grid':[1,0,-1]})
            replay.durable(path, {'grid':[1,0,-1]})
            with self.assertRaises(ValueError): replay.durable(path, {'grid':[0,0,0]})
        with self.assertRaises(ValueError): replay.validate_dispatch(np.ones(24), np.zeros(24), Battery())

    def test_archive_hash_and_prediction_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'runs/forecasts').mkdir(parents=True)
            np.savez(root/'runs/forecasts/test.npz', days=['2025-11-01'], point=np.zeros((1,24)))
            buf=io.BytesIO(); np.save(buf,np.zeros(24)); data=buf.getvalue()
            archive=root/'archive.zip'; audit=root/'audit.json'
            name='pregate_2025-11-01.npy'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(name,data)
            audit.write_text(json.dumps({'daily_issued_sha256':{name:hashlib.sha256(data).hexdigest()}}))
            with patch.multiple(replay,ROOT=root,ARCHIVE=archive,AUDIT=audit):
                replay.audited_forecasts({'forecast_id':'test','archive_label':'pregate'})
                np.savez(root/'runs/forecasts/test.npz', days=['2025-11-01'], point=np.ones((1,24)))
                with self.assertRaises(AssertionError):
                    replay.audited_forecasts({'forecast_id':'test','archive_label':'pregate'})
                audit.write_text(json.dumps({'daily_issued_sha256':{name:'bad hash'}}))
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    replay.audited_forecasts({'forecast_id':'test','archive_label':'pregate'})


if __name__ == '__main__':
    unittest.main()
