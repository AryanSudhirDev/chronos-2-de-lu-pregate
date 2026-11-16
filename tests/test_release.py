"""Validate the public input contract without loading weights or accessing outcomes."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = load(ROOT / "publish/inference.py", "public_inference")
example = load(ROOT / "scripts/make_example_input.py", "example_input")


class ReleaseTests(unittest.TestCase):
    def test_example_matches_full_training_schema(self):
        item = example.example()
        self.assertEqual(set(item["past_covariates"]), inference.PAST_KEYS)
        self.assertEqual(set(item["future_covariates"]), inference.FUTURE_KEYS)
        self.assertEqual(len(item["target"]), 2040)
        self.assertTrue(all(len(v) == 24 for v in item["future_covariates"].values()))

    def test_invalid_inputs_fail_before_loading_model(self):
        for change in ("missing_covariate", "nan", "future_length", "short_context"):
            item = example.example()
            if change == "missing_covariate":
                del item["past_covariates"]["price_fr"]
            elif change == "nan":
                item["future_covariates"]["wind_fc"][0] = np.nan
            elif change == "future_length":
                item["future_covariates"]["solar_fc"].pop()
            else:
                item["target"] = [1.] * 24
            with self.subTest(change=change), patch.object(inference.Chronos2Pipeline, "from_pretrained") as loader:
                with self.assertRaises(ValueError):
                    inference.forecast("unused", item)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
