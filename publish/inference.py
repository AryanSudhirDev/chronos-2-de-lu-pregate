"""Forecast from prepared numeric inputs; no data downloads or model training.

python inference.py --model . --input inputs.json --output forecast.json
Input keys: target (hourly history), past_covariates (matching history arrays),
future_covariates (24-hour arrays). See README.md for the required information set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from chronos import Chronos2Pipeline

PAST_KEYS = {"load_fc", "wind_fc", "solar_fc", "price_be", "price_fr", "price_nl",
             "price_dk1", "price_at", "price_ch", "price_cz"}
FUTURE_KEYS = {"load_fc", "wind_fc", "solar_fc"}


def forecast(model_path, item, device="cpu", calibration_path=None):
    item = dict(item)
    item["target"] = np.asarray(item["target"], dtype=np.float32)
    if item["target"].ndim != 1 or not np.isfinite(item["target"]).all():
        raise ValueError("target must be a finite hourly vector")
    if not 168 <= len(item["target"]) <= 2040:
        raise ValueError("supply 168..2040 hourly values; the research setup used 2040")
    for key, length in [("past_covariates", len(item["target"])), ("future_covariates", 24)]:
        item[key] = {k: np.asarray(v, dtype=np.float32) for k, v in item.get(key, {}).items()}
        if any(v.shape != (length,) or not np.isfinite(v).all() for v in item[key].values()):
            raise ValueError(f"{key} must contain finite arrays of length {length}")
        expected = PAST_KEYS if key == "past_covariates" else FUTURE_KEYS
        if set(item[key]) != expected:
            raise ValueError(f"{key} requires exactly these keys: {sorted(expected)}")
    pipe = Chronos2Pipeline.from_pretrained(model_path, device_map=device)
    q = pipe.predict([item], prediction_length=24)[0][0].float().cpu().numpy().T
    if not np.isfinite(q).all():
        raise ValueError("model produced non-finite quantiles")
    levels = np.asarray(pipe.quantiles)
    idx = lambda level: int(np.argmin(np.abs(levels - level)))
    result = dict(median=q[:, idx(.5)].tolist(), quantile_levels=levels.tolist(), quantiles=q.tolist())
    if calibration_path:
        adjustment = np.asarray(json.loads(Path(calibration_path).read_text())["adjustment"])
        if adjustment.shape != (24,) or not np.isfinite(adjustment).all():
            raise ValueError("calibration must contain 24 finite hourly adjustments")
        result.update(lower80=(q[:, idx(.1)] - adjustment).tolist(),
                      upper80=(q[:, idx(.9)] + adjustment).tolist())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=".")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration", help="Optional path to calibration.json")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    result = forecast(args.model, json.loads(Path(args.input).read_text()), args.device, args.calibration)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
