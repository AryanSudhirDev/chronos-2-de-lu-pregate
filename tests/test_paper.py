import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from epf.paper import validate_issue, publish, durable_json, panel_from_snapshot


class PaperTests(unittest.TestCase):
    def test_live_weather_gap_is_rejected_before_model_work(self):
        import sys
        from epf.data import ROOT, Panel
        sys.path.insert(0,str(ROOT/'scripts'))
        import paper_day
        weather=pd.DataFrame(1.,index=pd.date_range('2026-11-11',periods=24,freq='h',tz='Europe/Berlin').tz_convert('UTC'),columns=['wind'])
        weather.iloc[5,0]=np.nan
        with patch('paper_day.pd.read_parquet',return_value=weather), patch('paper_day.ModelWorker',side_effect=AssertionError('worker must not start')):
            with self.assertRaisesRegex(ValueError,'24 complete original weather hours'):
                paper_day.map_weather(Panel(pd.DatetimeIndex(['2026-11-11']),{}),Path('/unused'),'2026-11-11')

    def test_controller_masks_unreleased_targets_and_actuals(self):
        cols=['price_'+z for z in ['de','be','fr','nl','pl','dk1','at','ch','cz']]+['load_fc','wind_on_fc','wind_off_fc','solar_fc','load_act','wind_on_act','wind_off_act','solar_act']
        ix=pd.date_range('2026-11-08',periods=72,freq='h',tz='Europe/Berlin').tz_convert('UTC')
        raw=pd.DataFrame(123.,index=ix,columns=cols)
        p=panel_from_snapshot(raw,'2026-11-10')
        self.assertTrue(np.isnan(p['price_de'][-1]).all())
        self.assertTrue(np.isfinite(p['price_de'][-2]).all())
        self.assertTrue(np.isnan(p['load_act'][-2:]).all())
        self.assertTrue(np.isfinite(p['load_act'][-3]).all())
        self.assertTrue(np.isnan(p['wind_fc'][-1]).all())
        self.assertTrue(np.isfinite(p['wind_fc'][-2]).all())
        self.assertTrue(np.isfinite(p['load_fc'][-1]).all())

    def test_deadlines_freshness_and_dst(self):
        now=datetime(2026,11,10,10,tzinfo=timezone.utc)  # 11:00 Berlin
        validate_issue('2026-11-11',now.isoformat(),now)
        for captured,clock in [(now.isoformat(),now+timedelta(hours=1)),
                               ((now-timedelta(hours=3)).isoformat(),now),
                               ((now+timedelta(seconds=1)).isoformat(),now)]:
            with self.assertRaises(ValueError):validate_issue('2026-11-11',captured,clock)
        with self.assertRaises(ValueError):
            validate_issue('2027-03-28','2027-03-27T10:00:00+00:00',datetime(2027,3,27,10,tzinfo=timezone.utc))

    def test_issue_is_immutable_and_atomic(self):
        now=datetime(2026,11,10,10,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'issue.json'
            publish(p,'2026-11-11',now.isoformat(),{'model':[42.]*24},{},now)
            original=p.read_bytes()
            with self.assertRaises(FileExistsError):
                publish(p,'2026-11-11',now.isoformat(),{'model':[99.]*24},{},now)
            self.assertEqual(p.read_bytes(),original)
            self.assertEqual(json.loads(original)['kind'],'prospective-paper')
            self.assertEqual(list(Path(t).glob('.pending_*')),[])
            with self.assertRaises(ValueError):durable_json(Path(t)/'bad.json',{'x':float('nan')})


if __name__=='__main__':unittest.main()
