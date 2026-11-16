# Research process and model construction

## Question and scope

The initial plan asked whether fine-tuning a time-series foundation model could improve crypto
return forecasts. The implemented research moved to German/Luxembourg electricity prices.
This release contains the completed electricity study; it makes no claim of a completed crypto
benchmark or established crypto edge. The target is a vector of 24 hourly day-ahead prices
in EUR/MWh, rather than returns or price direction.

The original machine was an Apple M4 Pro, 14 cores, 24 GB unified memory. Python 3.12 ran
in a project environment. Chronos used MPS where available, otherwise CPU. Models were
loaded serially to control memory. Tree models and Chronos were separated in workers where
their OpenMP runtimes conflicted on macOS.

## Data and information sets

SMARD provides DE-LU/neighbouring prices, load/generation actuals and TSO wind/solar forecasts.
Energy-Charts supplies load forecasts. Open-Meteo Previous Runs supplies historical weather
features used to estimate pre-auction wind/solar generation. Data starts in October 2018,
when the DE-LU bidding zone begins. The original weather archive is shorter.

Delivery days use Europe/Berlin time. Historical processing averages duplicate autumn hours
and interpolates the missing spring hour to obtain 24 bins. The later paper prototype rejects
23/25-hour delivery days. Quarter-hour prices are averaged into hours; actual execution in
quarter-hour market products is outside this study.

For delivery day D, prices through D-1 are available from the earlier cleared auction.
Actual load/generation features stop at D-2. Load forecasts for D are treated as pre-gate.
TSO wind/solar forecasts arriving at 18:00 on D-1 are **after** the noon auction deadline:
they define the post-auction research track, not deployable pre-auction information.

The strict timing experiment removed those current forecasts. The weather-derived pre-gate
track replaced them with estimates from older weather runs, using 14 geographic points,
ECMWF/ICON features, and wind speed at 100m, radiation and temperature. Separate LightGBM
weather mappers refit monthly using earlier data. Their weights are not inside Chronos.
The archive's lead-offset naming alone does not establish original publication timestamps.

## References and zero-shot comparisons

The naive reference uses last week's same day for Monday/Saturday/Sunday, yesterday for
Tuesday–Friday. LEAR uses lagged hourly prices and exogenous load/renewable regressors,
recalibrated across several windows. LightGBM learns a residual relative to the previous day's
mean using lagged price, volatility and exogenous features, with monthly refits.

Chronos-2 was tested without covariates, with load/renewable forecasts, with neighbouring
prices, with calendar features and with a longer context. Poland was excluded from the
neighbouring Chronos channels because its available history starts later. The chosen schema
uses DE prices as the target; load, wind, solar and seven neighbouring price channels as past
covariates; and load/wind/solar as future covariates.

## Actual neural weight updates

The base is `amazon/chronos-2`, pinned to revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. Its published architecture has approximately
120 million parameters. In LoRA experiments the base weights were frozen while small learned
low-rank updates were trained. The retained adapter uses rank 8, alpha 16, dropout 0, and targets
attention q/k/v/o and `output_patch_embedding.output_layer`.

The main LoRA setting used LR 1e-4, up to 1000 training steps, batch size 32, 85 context days,
24 prediction hours, a minimum past of seven days, and a later 60-day validation block passed
to the library for checkpoint selection. Full fine-tuning settings used LR 1e-6 and 1e-5.
Joint neighbouring-market training was also evaluated, but did not improve the main LoRA row.
See ledger configurations rather than assuming every trial used identical folds.

Development folds were ordered by year, with all fitting history before each fold and a
separate later validation window. The final retained fit precedes the holdout, with the ledger
recording a fitting end of 2025-09-01. Forecasting with pre-gate inputs reused that fit;
historical training covariates remained the TSO forecast series. This introduces an input
distribution difference that readers should consider.

The Hugging Face export merges the original adapter into base weights without optimization
or new fitting. It also preserves the adapter separately. CPU save/reload parity was verified
on synthetic inputs; this packaging check is not another accuracy evaluation.

## Stacks and blending

LightGBM residual stacks consume Chronos forecasts/quantile information and tabular features
to predict a correction. Compact blends combine Chronos, its residual stack and LightGBM.
Online blend weights update from past errors only. These add trainable components around
Chronos; the merged neural checkpoint does not incorporate a tree model or blend weights.

## Holdout and isolation

Development dates end on 2025-10-31; holdout dates begin 2025-11-01. Candidates were selected
before the one-time opening on 2026-11-09. Three candidates were evaluated together with four
declared references. The append-only ledger makes the search visible, including later dev trials.

The final runner requires macOS sandbox-exec. Model workers cannot access networking, raw
data files, the experiment ledger or unreleased future files. A trusted controller passes causal
numeric inputs over a pipe. Each forecast is durably saved before a later outcome can be released.
Synthetic poisoning tests check that unreleased values do not alter features. Probe tests verify
file and IPv4/IPv6 network denial. A development rehearsal preceded the opening.

The controller/scorer remains trusted and outside the worker sandbox. An environment variable
guards data access in ordinary code but is not itself OS isolation. Local pickle IPC is trusted
internal serialization, not a public service interface for untrusted requests. No assertion is made
about the base model's original training data or a generic Linux isolation implementation.

## Metrics and follow-up

MAE is primary; RMSE, naive-relative MAE, CRPS, quantile coverage and daily multivariate DM
comparisons are also recorded. The original battery metric is gross profit relative to perfect
foresight. Later readiness/replay studies add constraints and illustrative variable/fixed costs.
The DM implementation is simple and not selection/serial-dependence adjusted.

Later readiness, calibration and adaptive studies are explicitly development-only. Calibration
widens q10/q90 by hourly corrections learned before the readiness scoring block. A single
EWMA hourly bias correction failed its economic promotion rule. The prospective paper stream
kept neural weights and calibration frozen; separate references refit under fixed past-only rules.
The paper stream stopped early. No live orders or market bids were made.
