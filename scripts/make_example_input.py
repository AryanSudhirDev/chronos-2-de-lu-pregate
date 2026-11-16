"""Write a complete synthetic input schema; these values are not market forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np


def example():
    x = np.arange(2040, dtype=np.float32)
    future = np.arange(2040, 2064, dtype=np.float32)
    price = 70 + 20 * np.sin(2 * np.pi * x / 24) + 5 * np.cos(2 * np.pi * x / 168)
    past = {
        "load_fc": 50000 + 5000 * np.sin(2 * np.pi * x / 24),
        "wind_fc": 15000 + 3000 * np.cos(2 * np.pi * x / 72),
        "solar_fc": 10000 * np.maximum(0, np.sin(2 * np.pi * (x % 24 - 6) / 24)),
    }
    for i, zone in enumerate(["be", "fr", "nl", "dk1", "at", "ch", "cz"]):
        past[f"price_{zone}"] = price + i - 3
    future_covariates = {
        "load_fc": 50000 + 5000 * np.sin(2 * np.pi * future / 24),
        "wind_fc": 15000 + 3000 * np.cos(2 * np.pi * future / 72),
        "solar_fc": 10000 * np.maximum(0, np.sin(2 * np.pi * (future % 24 - 6) / 24)),
    }
    return dict(target=price.tolist(), past_covariates={k:v.tolist() for k,v in past.items()},
                future_covariates={k:v.tolist() for k,v in future_covariates.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example_input.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(example(), indent=2, allow_nan=False) + "\n")
    print(f"Wrote synthetic example: {args.output}")
