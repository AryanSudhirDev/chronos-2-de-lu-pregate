"""Synthetic chronology and update checks for adaptive forecast correction."""
import unittest
import numpy as np
from epf.adaptive import AdaptiveBias
from epf.offline import ModelWorker


class AdaptiveTests(unittest.TestCase):
    def test_predict_then_observe_and_calendar_gap(self):
        model=AdaptiveBias(halflife=1)
        with self.assertRaises(ValueError):model.observe('2025-01-01',np.ones(24))
        forecast=np.full(24,100.)
        first=model.predict('2025-01-01',forecast)
        np.testing.assert_array_equal(first['point'],forecast)
        with self.assertRaises(ValueError):model.predict('2025-01-02',forecast)
        with self.assertRaises(ValueError):model.observe('2025-01-02',np.full(24,110.))
        model.observe('2025-01-01',np.full(24,110.))
        np.testing.assert_array_equal(model.bias,np.full(24,5.))
        result=model.predict('2025-01-04',forecast)
        np.testing.assert_array_equal(result['weights'],np.full(24,1.25))
        model.observe('2025-01-04',forecast)
        np.testing.assert_array_equal(model.bias,np.full(24,.625))
        with self.assertRaises(ValueError):model.predict('2025-01-04',forecast)

    def test_current_target_cannot_change_saved_prediction(self):
        forecast=np.arange(24.)
        outputs=[]
        for target in [np.zeros(24),np.full(24,1e9)]:
            model=AdaptiveBias();issued=model.predict('2025-01-01',forecast)
            saved=issued['point'].copy();model.observe('2025-01-01',target)
            np.testing.assert_array_equal(issued['point'],saved)
            outputs.append(saved)
        np.testing.assert_array_equal(*outputs)

    def test_input_validation_and_state_copy(self):
        for kwargs in [dict(halflife=0),dict(bias=np.zeros(3)),dict(bias=np.full(24,np.nan))]:
            with self.assertRaises(ValueError):AdaptiveBias(**kwargs)
        bias=np.ones(24);model=AdaptiveBias(bias=bias);bias[:]=1e9
        issued=model.predict('2025-01-01',np.zeros(24))
        np.testing.assert_array_equal(issued['point'],np.ones(24))
        with self.assertRaises(ValueError):model.observe('2025-01-01',np.full(24,np.nan))

    def test_isolated_worker_rejects_outcomes_during_prediction(self):
        with ModelWorker('adaptive_test',threads=1) as worker:
            worker.call('adaptive_init',halflife=30)
            with self.assertRaisesRegex(RuntimeError,'no outcomes'):
                worker.call('adaptive_predict',day='2025-01-01',forecast=np.zeros(24),actual=np.ones(24))
            predicted=worker.call('adaptive_predict',day='2025-01-01',forecast=np.zeros(24))
            np.testing.assert_array_equal(predicted['point'],np.zeros(24))
            worker.call('adaptive_observe',day='2025-01-01',actual=np.ones(24))
            second=worker.call('adaptive_predict',day='2025-01-02',forecast=np.zeros(24))
            np.testing.assert_allclose(second['point'],np.full(24,1-.5**(1/30)))


if __name__=='__main__':unittest.main()
