"""Synthetic checks for cost attribution and cashflow decomposition."""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from epf.data import ROOT

spec=importlib.util.spec_from_file_location('diagnose_replay',ROOT/'scripts/diagnose_replay.py')
diagnosis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


class DiagnosisTests(unittest.TestCase):
    def test_skips_still_pay_fixed_costs_and_drawdown_starts_at_zero(self):
        d=pd.DataFrame(dict(day=['2025-11-01','2025-11-03'],actual_mean=[50.,60.],
            high=[100.,110.],low=[0.,10.],margin_eur=[10.,20.],oracle_margin_eur=[15.,25.]))
        h=pd.DataFrame(dict(day=['2025-11-01']*2,actual_eur_mwh=[0.,100.],grid_mwh=[1.,-.9],variable_cost_eur=[3.,21.]))
        x,m,s=diagnosis.diagnose(d,h,annual_fixed=365.25)
        self.assertAlmostEqual(x.loc['2025-11-02','net'],-1.)
        self.assertAlmostEqual(m.fixed.sum(),374.)
        self.assertAlmostEqual(s['ending_net'],30.-374.)
        self.assertAlmostEqual(s['worst_drawdown_eur'],-371.)
        self.assertEqual(s['scored_days'],2)
        no_margin=d.copy();no_margin.margin_eur=0.
        _,_,losses=diagnosis.diagnose(no_margin,h,annual_fixed=365.25)
        self.assertAlmostEqual(losses['worst_drawdown_eur'],-374.)

    def test_daily_mean_shift_hurts_fixed_lossy_schedule(self):
        d=pd.DataFrame(dict(day=['2025-11-01','2025-11-02'],actual_mean=[50.,60.],
            high=[100.,110.],low=[0.,10.],margin_eur=[66.,65.],oracle_margin_eur=[70.,69.]))
        h=pd.DataFrame(dict(day=['2025-11-01']*2,actual_eur_mwh=[0.,100.],grid_mwh=[1.,-.9],variable_cost_eur=[3.,21.]))
        _,_,a=diagnosis.diagnose(d,h)
        shifted=h.copy();shifted.actual_eur_mwh+=10
        _,_,b=diagnosis.diagnose(d,shifted)
        self.assertAlmostEqual(a['gross_decomposition']['gross'],90.)
        self.assertAlmostEqual(a['gross_decomposition']['mean_level_component'],-5.)
        self.assertAlmostEqual(a['gross_decomposition']['within_day_pattern_component'],95.)
        self.assertAlmostEqual(b['gross_decomposition']['within_day_pattern_component'],95.)
        self.assertAlmostEqual(b['gross_decomposition']['gross']-a['gross_decomposition']['gross'],-1.)


if __name__=='__main__':unittest.main()
