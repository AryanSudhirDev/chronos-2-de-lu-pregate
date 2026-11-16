import unittest
import numpy as np
import pandas as pd

from epf.data import Panel, PRICE_ZONES
from epf.features import feature_table
from epf.lear import lear_features
from epf.readiness import Battery, dispatch, economics, interval_adjustment


class ReadinessTests(unittest.TestCase):
    def test_pregate_features_ignore_current_postgate_forecasts(self):
        rng = np.random.default_rng(2)
        names = [f"price_{z}" for z in PRICE_ZONES]+["load_fc","wind_fc","solar_fc","ren_fc","resid_fc","resid_act","wind_fc_pre","solar_fc_pre"]
        p = Panel(pd.date_range("2025-01-01", periods=40), {k:rng.normal(100,10,(40,24)) for k in names})
        before = feature_table(p, pre_gate=True)
        before_lear = lear_features(p,25,pre_gate=True)
        for k in ["wind_fc","solar_fc","ren_fc","resid_fc"]:
            p.x[k][25:] = 1e9
        after = feature_table(p, pre_gate=True)
        np.testing.assert_array_equal(before[before.day_idx==25],after[after.day_idx==25])
        np.testing.assert_array_equal(before_lear,lear_features(p,25,pre_gate=True))

    def test_dispatch_respects_power_soc_and_cycles(self):
        b = Battery(max_cycles=1)
        prices = np.tile([-100,200],12)
        grid, discharged = dispatch(prices,b)
        soc = np.cumsum(np.where(grid>0,grid*b.eta,grid/b.eta))
        self.assertTrue((abs(grid)<=b.power_mw+1e-9).all())
        self.assertTrue((soc>=-1e-9).all() and (soc<=b.capacity_mwh+1e-9).all())
        self.assertAlmostEqual(soc[-1],0)
        self.assertLessEqual(discharged.sum(),b.capacity_mwh+1e-9)
        e=economics(prices[None],prices[None],b,annual_fixed_eur=365.25)
        self.assertAlmostEqual(e['margin_eur'],e['oracle_margin_eur'])
        self.assertAlmostEqual(e['scenario_net_eur'],e['margin_eur']-1)
        self.assertTrue(np.all(dispatch(np.full(24,100),b)[0]==0))

    def test_dispatch_known_arbitrage(self):
        prices=np.full(24,50.);prices[0]=0;prices[1]=100
        b=Battery(eta=1,steps=10,max_cycles=1,fee_slippage_eur_mwh=0,wear_eur_mwh=0)
        grid,_=dispatch(prices,b)
        self.assertAlmostEqual(-prices@grid,100)

    def test_calibration_finite_sample_and_past_only(self):
        y=np.arange(50)[:,None]*np.ones((1,24))
        a=interval_adjustment(y,np.zeros_like(y),np.zeros_like(y))
        np.testing.assert_array_equal(a,np.full(24,40.))
        with self.assertRaises(ValueError):interval_adjustment(y[:4],y[:4],y[:4])


if __name__=='__main__':unittest.main()
