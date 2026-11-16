---
license: apache-2.0
library_name: chronos-forecasting
pipeline_tag: time-series-forecasting
base_model: amazon/chronos-2
base_model_relation: finetune
tags:
  - time-series
  - electricity-price-forecasting
  - chronos-2
  - lora
  - safetensors
---

# Chronos-2 DE-LU pre-gate electricity price model

**Author:** Aryan Sudhir ([AryanSudhirDev](https://huggingface.co/AryanSudhirDev)).
**Source code:** [GitHub repository](https://github.com/AryanSudhirDev/chronos-2-de-lu-pregate).
**Detailed documentation:** [results](docs/RESULTS.md), [research process](docs/METHODOLOGY.md),
[reproduction](docs/REPRODUCIBILITY.md), [data contract](docs/DATA.md), [artifacts](docs/ARTIFACTS.md).

Research checkpoint for forecasting 24 hourly German/Luxembourg electricity prices in EUR/MWh.
This is the frozen pre-gate paper candidate, exported as standalone merged weights.
It is an electricity model; the source project's folder name reflects an earlier crypto study.

## Contents and provenance

- `model.safetensors` and `config.json`: complete Chronos-2 base plus learned LoRA updates.
- `adapter/`: original learned adapter tensors and portable, revision-pinned configuration.
- `calibration.json`: separate development-only hourly interval expansion, not neural weights.
- `inference.py`: inference from numeric inputs; `requirements.txt`: export environment versions.
- `manifest.json`: file hashes, pinned base revision, and synthetic reload verification.
- `LICENSE`, any upstream `NOTICE`, and `MODIFICATIONS.md`: licensing and change attribution.
- `example_input.json` and `make_example_input.py`: full-schema synthetic example, without real observations.
- `source-release.zip`: code, tests, recorded forecasts, protocols and results from the linked GitHub snapshot.
- `docs/`, `results.png`, `CITATION.cff`: full methodology/results, a labelled results figure and citation.

Base: [amazon/chronos-2](https://huggingface.co/amazon/chronos-2), revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. LoRA rank 8, alpha 16, dropout 0,
with attention q/k/v/o and output patch embedding updates. The original base weights
were frozen during adapter training. Export merges the trained updates without additional training.

The frozen checkpoint is identified by adapter SHA-256
`01fe7c491f30f3bbcb1265d3bbd53db226b5f0cfe3769d0519b94679462c48b5`.
Calibration uses 303 earlier development days ending 2025-06-30.
The holdout record declares a fitting cutoff of 2025-09-01.

## Run

Use Python 3.12 and install `pip install -r requirements.txt` in an isolated environment.
After downloading this repository:

```python
from chronos import Chronos2Pipeline

pipe = Chronos2Pipeline.from_pretrained("/path/to/downloaded/model", device_map="cpu")
# item contains the arrays described below, represented as numpy float32 arrays.
quantiles = pipe.predict([item], prediction_length=24)
```

For JSON input and optional calibrated intervals:

```sh
python make_example_input.py --output example_input.json
python inference.py --model . --input inputs.json --output forecast.json --calibration calibration.json
```

Use `--input example_input.json` for the generated synthetic example; `inputs.json` represents
your own prepared market inputs. The synthetic output is a plumbing check, not an accuracy claim.

`--device mps` can be used on a compatible Mac. CPU is the portable default.
Merged weights load without PEFT; loading the optional standalone adapter requires PEFT.

## Required inputs

The research inference setup uses 85 days of hourly history (2040 values), flattened
from 24-hour Berlin delivery-day blocks:

- `target`: historical DE-LU prices, in EUR/MWh.
- `past_covariates`: matching hourly `load_fc`, `wind_fc`, `solar_fc`, plus neighbouring prices
  `price_be`, `price_fr`, `price_nl`, `price_dk1`, `price_at`, `price_ch`, `price_cz`.
- `future_covariates`: 24 values each of `load_fc`, `wind_fc`, `solar_fc`.

All listed channels are required by the helper. It validates finite aligned arrays and accepts
168–2040 history values; use 2040 to match the research context. Keep original units: prices
in EUR/MWh and load/generation in MW. No external standardization is required.

Only use information available before the D-1 noon Europe/Berlin auction deadline.
For the forecast day, `wind_fc` and `solar_fc` must be pre-gate weather-derived estimates;
the actual day-ahead TSO forecasts arrive after the deadline. Historical wind/solar covariates
use historical TSO forecasts. The original pipeline maps DST delivery days onto 24 slots.
Model outputs have axes target series, quantile, forecast hour; the helper returns hourly rows.

Raw feeds, weather-to-generation LightGBM regressors, feed collection, DST preprocessing,
reference models and battery dispatch are external to these neural weights. This package
accepts prepared inputs; it does not reconstruct the full operational research pipeline.
No raw market data or private credentials are included. Original study sources were SMARD,
Energy-Charts and Open-Meteo; consult their terms for any separate data redistribution.

## Evaluation and limitations

The pre-gate candidate's recorded one-time holdout MAE was **15.386 EUR/MWh** over
372 days, 2025-11-01 through 2026-11-09 (two delivery days missing). LEAR scored approximately
15.34 on the common window; this candidate did not establish an accuracy improvement
(one-sided daily multivariate Diebold-Mariano p = 0.543). LEAR used post-gate TSO inputs,
so the information sets differ. Other, post-gate Chronos variants performed better,
but those results must not be attributed to this checkpoint.

Walk-forward development and one-time holdout evaluation used isolated offline forecasting
workers. This does not establish the base model's original pretraining cutoff or rule out
overlap with its pretraining data. Evaluation involved multiple recorded research trials.
Nominal quantiles do not guarantee coverage; interval expansion is a separate historical calibration.
Paper observation stopped at five scored days at the user's request; no claim of a completed
30-day prospective validation is made.

This is a research artifact, with no demonstrated live trading edge. Battery simulations
depend on assumptions and do not establish deployable profitability. No trades are placed.
Performance may change with market conditions, data availability and input preparation.

## What the broader study achieved

On the same 372-day holdout, the post-auction research LoRA setup scored **12.3510 MAE**,
versus zero-shot **13.4993** (8.51% lower); a compact blend reached **11.8143** against LEAR
**15.3446** (23.01% lower). Those rows use wind/solar forecasts released after the gate,
and the blend includes components outside these neural weights. The pre-gate setup remains
**15.3860**. The original fit was reused with different future-covariate inputs.

A later development-only pre-gate comparison (123 days, July–October 2025) scored LoRA
**12.15**, LightGBM **12.69**, and LEAR **14.40**. Interval coverage rose from **71.1% to 83.6%**
after past-only calibration. A single adaptive correction improved MAE but lowered constrained
margin and was rejected. Prospective observation stopped at **five scored days**, with LoRA
MAE **22.36** versus LEAR **20.12**. These follow-ups do not supply another untouched test.

The research process included naive/LEAR/LightGBM baselines, covariate ablations, zero-shot
Chronos, LoRA/full adaptation, residual stacks, blending, a timing audit, isolated one-time holdout
evaluation and separate readiness/paper work. [Read the full account](docs/METHODOLOGY.md).

![Separated evaluation windows; lower MAE is better](results.png)

## Citation and licensing

See `CITATION.cff` for this research artifact. Credit Amazon's Chronos-2 base model separately.
The derivative weights and accompanying release code are Apache-2.0. `LICENSE`, upstream
`NOTICE` and `MODIFICATIONS.md` preserve attribution and identify changes. Source data terms
remain separate. No optimizer-resume state or every trained reference-model object is claimed.
