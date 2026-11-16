> Historical study notes. Scheduling/future-tense statements below are superseded by README.md and docs/RESULTS.md. The paper stream stopped.

# Day-ahead electricity price forecasting (DE-LU) with a fine-tuned Chronos-2

**Public release:** [Hugging Face model](https://huggingface.co/AryanSudhirDev/chronos-2-de-lu-pregate)
and [GitHub source](https://github.com/AryanSudhirDev/chronos-2-de-lu-pregate).
See [detailed results](docs/RESULTS.md), [methodology](docs/METHODOLOGY.md),
[reproduction guide](docs/REPRODUCIBILITY.md) and [data contract](docs/DATA.md).
The research/paper processes remain stopped. Historical plans below are preserved as study notes.

Research question: can a fine-tuned time-series foundation model (Chronos-2) beat the strong,
established benchmarks for forecasting tomorrow's 24 hourly German/Luxembourg day-ahead prices,
judged honestly on unseen data, in error and in the money a battery would make with it?

The project started as crypto return forecasting; it was switched because the evidence
(papers and practitioner reports) shows no reliable edge there, while day-ahead power prices are
predictable from known-in-advance drivers (load, wind and solar forecasts). The crypto downloader is
kept in `scripts/crypto/` but is not part of the study.

Published reference points (Germany, 2021-2025, [arXiv 2609.00089](https://arxiv.org/html/2609.00089)),
MAE in EUR/MWh: TabPFN-3 12.59, DDNN 14.58, LEAR 15.93, zero-shot Chronos-2 16.13. Different test
window from ours, so they are context, not a direct comparison; our own benchmarks are rerun below.

## Setup

- Data (free, no keys, CC BY 4.0): [SMARD](https://www.smard.de) (Bundesnetzagentur) for prices of
  DE-LU and 8 neighbours, actual load/generation and TSO wind/solar forecasts;
  [Energy-Charts](https://api.energy-charts.info) for the TSO day-ahead load forecast. Hourly, Oct 2018 on.
- Task: on day D-1 before the 12:00 auction gate, forecast the 24 hourly prices of day D (Berlin time;
  DST days mapped to 24 hours). 15-minute prices since Oct 2025 are averaged to hours.
- Information set: prices through D-1; TSO forecasts for D; actuals through D-2.
  **Caveat:** the TSO wind/solar forecasts are published at 18:00 on D-1, after the gate (the load
  forecast is published before it). Using them is the standard EPF-literature setup and is what the
  published numbers use; gate-strict and pre-gate weather variants have also been evaluated (see STATUS.md).
- Splits: dev walk-forward window 2022-01-01..2025-10-31 (all model selection happens here).
  **Sealed holdout: delivery days from 2025-11-01**, unreadable by the code unless opened through
  `scripts/open_holdout.py`, once, for the final candidates.
- Holdout model workers now use a macOS OS sandbox: network access is denied, raw data and
  the ledger are unreadable, and only causal numeric inputs arrive over a pipe. The trusted
  controller releases one delivery day's prediction inputs at a time and saves each prediction
  before releasing subsequent outcomes. LightGBM refits monthly on earlier days; blend weights
  update from earlier errors; LEAR recalibrates daily. The controller/scorer stays outside the
  worker sandbox, disables Python networking and uses cached weather and pinned cached weights.
  This does not establish the foundation model's original pretraining cutoff. Post-gate TSO
  inputs remain research-only even with process isolation; the pre-gate candidate uses older
  weather runs. Holdout opening requires `--offline`, after a successful dev rehearsal.
- Every evaluated variant is appended to `runs/ledger.jsonl` (the trial count matters when judging the
  best of many tries); forecasts are saved in `runs/forecasts/`.

## Metrics

MAE (primary), RMSE, rMAE (MAE relative to the epftoolbox naive forecast), CRPS for probabilistic
models, multivariate Diebold-Mariano tests, and the realised profit of a 1 MW / 1 MWh battery
(95% one-way efficiency, empty at midnight) scheduled on the forecast and settled at actual prices,
as % of perfect-foresight profit. This simplified simulation includes conversion losses, but excludes
fees, battery degradation, capital costs and execution constraints. It does not establish a live trading edge;
post-gate TSO results are research comparisons, while the pre-gate track tests usable auction-time inputs.

## Final holdout result

The one-time offline opening completed on 2026-11-09 after the three candidates were frozen.
The original study contains 36 dev trials and seven holdout evaluations (three candidates, four references).
Two later dev-readiness reference trials are recorded separately.
On the **372 days shared by all seven models**, 2025-11-01..2026-11-09, the compact blend scored
**11.81 EUR/MWh MAE**, LoRA 12.35, zero-shot Chronos 13.50, LEAR 15.34 and LightGBM 15.49.
The blend reduced MAE by 23.0% against LEAR; LoRA reduced it by 8.5% against zero-shot
(one-sided daily multivariate DM p = 5.3e-12). These are **post-gate research comparisons**,
using TSO wind/solar inputs published after the auction deadline.

The **pre-gate usable-input candidate** scored **15.39 MAE**, versus LEAR's 15.34 on the same days;
it did not establish an accuracy gain (one-sided DM p = 0.543). LEAR here uses post-gate TSO inputs,
so this is not a comparison against an equally informed pre-gate benchmark. The pre-gate forecast
is missing 2026-04-10 and 2026-04-11; the other models cover all 374 days. The ledger table below
uses each model's available days, which explains its slightly different MAEs.

Simulated battery profit was 96.2% of oracle for the blend and 94.6% for pre-gate LoRA on the
shared days. These are shares of a simplified perfect-foresight profit, not net trading returns.
Fees, degradation, capital costs and execution constraints remain excluded. LoRA's nominal 80%
interval covered only 72.0% of outcomes (pre-gate: 61.9%), so uncertainty is under-calibrated.
Original Chronos pretraining provenance and historical source publication vintages are not proven
by process isolation. The trusted controller/scorer remains outside the OS worker sandbox.

No models or weather settings were retuned after opening. All 2,616 daily issued forecasts were
checked against the final forecast artifacts and match exactly. See
[runs/compare_holdout.md](runs/compare_holdout.md), [the numeric summary](runs/holdout_summary.json)
and [the issued-forecast audit archive](runs/holdout_issued_audit.zip).

## Forward paper validation

An additional [historical decision replay](runs/replay/report.md) reuses the saved pre-gate
holdout forecasts without retraining. Its interactive report (local historical HTML, not redistributed) and
hourly decisions (local historical CSV, not redistributed) show forecast-only battery actions against actual
prices, with explicit assumed costs. This is a retrospective diagnostic; historical source
publication vintages and original model pretraining remain unverified.
The [drawdown diagnosis](runs/replay/diagnosis.md) corrects the isolated falling-day
interpretation: the first four months lost EUR 6,561 after assumed costs, and even
perfect forecasts could not cover those months' assumed fixed costs. The paper Chronos
checkpoint is fixed; recent inputs refresh daily, while a separate research blend adapts weights.

The next phase is separate from the completed holdout. A new dev-only readiness protocol
was frozen in commit `27c9d9e`; it evaluates pre-gate LEAR/LightGBM references on July–October
2025 and fits interval expansion only on the 303 dev forecast days through June 2025.
On the 123 dev days, MAE was **12.15 EUR/MWh for LoRA**, 12.69 for pre-gate LightGBM
and 14.40 for pre-gate LEAR. LoRA's gain over LightGBM was 4.3%; this is dev evidence,
not a new unseen test. Nominal 80% interval coverage rose from 71.1% to 83.6% after the
past-only expansion, with wider intervals. Under the base illustrative battery costs,
LoRA's margin exceeded LightGBM's by only about 265 EUR over those 123 days.
Higher cost scenarios produced losses. See [the complete readiness report](runs/readiness/report.md).

These references use the same current pre-gate wind/solar inputs as LoRA. Their weather
training history starts in September 2024; Chronos has a longer history and pretraining.
The original 43 ledger rows and holdout results are unchanged. Two new references are appended
under the separate `readiness_dev` split. New outputs are under `runs/readiness/`.

The paper policy is frozen in [runs/paper/policy.json](runs/paper/policy.json): reuse the
retained LoRA adapter, pin adapter/calibration/protocol hashes, forecast before the next
auction, and score in a separate later invocation. Reference models refit under their fixed
protocols using past released data only. No new Chronos fit is required. All model workers
remain OS-isolated with networking denied. Collectors retain timestamped public source
snapshots under `data/paper/`; issued forecasts and scores use exclusive, durable file creation
and cannot be replaced by rerunning the command. Late, stale, incomplete and already-cleared
inputs are rejected. This relies on the trusted controller and local clock.

A later [single adaptive correction experiment](runs/adaptive/report.md) learned 24 hourly
offsets from prior dev errors. It improved dev MAE by 3.25% but changed constrained battery
margin by -4.62 EUR over 123 days, failing its predeclared promotion rule. It was not added
to the prospective paper stream. This tested daily correction weights, not neural-network
retraining; the paper control remains fixed. One separate `adaptive_dev` ledger row records it.

The feeds are [SMARD](https://www.smard.de), [Energy-Charts](https://api.energy-charts.info/)
and [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api). Open-Meteo
defines `previous_day2` as a fixed 48-hour lead offset. Live collection checks availability
before issuance, rather than assuming that a historical archive proves publication timing.
The paper prototype skips 23/25-hour DST days and models hourly price averages; execution
against actual 15-minute products still requires a dedicated integration.

Battery readiness calculations enforce 1 MW grid power, 1 MWh capacity, empty start/end,
95% one-way efficiency, a 0.1 MWh SOC grid, and at most two full equivalent cycles/day.
Illustrative assumptions are 3 EUR per grid MWh for fees/slippage and 20 EUR per discharged
battery MWh for wear. Sensitivities include 0/15,000/30,000 EUR annual fixed costs, prorated
to the scored window. These are assumptions, not asset quotes; taxes, outages and imbalance
risk remain excluded. A positive scenario margin does not establish a profitable investment.

```bash
.venv/bin/python -W ignore scripts/validate_readiness.py  # dev-only, resume saved reference forecasts
.venv/bin/python -W ignore scripts/paper_day.py auto     # before D-1 noon Europe/Berlin only
.venv/bin/python scripts/paper_day.py score              # score already-issued days, never reissue
.venv/bin/python -W ignore -m unittest discover -s tests -v
```

The first issue attempt is scheduled for **2026-11-10 11:00 Berlin / 02:00 Pacific**
(delivery 2026-11-11); each run schedules the next daily continuation. The machine and
network must remain available. The first review is after **30 scored prospective days**. No tuning is allowed on that stream;
30 days are an initial diagnostic and do not establish performance across seasons. Actual
new observations must accumulate over time. No broker, trade or bid interface is present.

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -e .
.venv/bin/python scripts/fetch_data.py          # ~4 min
.venv/bin/python scripts/run_naive.py
.venv/bin/python scripts/run_lear.py            # ~20-30 min, 10 processes
.venv/bin/python scripts/run_lgbm.py
scripts/sweep_zeroshot.sh                       # Chronos-2 zero-shot variants (MPS)
.venv/bin/python scripts/run_chronos_finetune.py --mode lora
.venv/bin/python scripts/report.py              # refresh the table below
```

## Results

<!-- results:start -->
Unique trials in ledger: **46** (47 append-only records; 1 duplicate recording).

### Dev window (walk-forward)

| Model | Config | Window | MAE | RMSE | rMAE | CRPS | Battery % of oracle | id |
|---|---|---|---|---|---|---|---|---|
| blend_pool7 | ids=['51d3abf5b2', '3f75dbb0b1', '69b2219339', '580124a4ca', 'f2f5ca7406', 'dbd170cd58', ' | 2023-01-01..2025-10-31 | 9.86 | 19.03 | 0.352 |  | 95.1 | fd73754ae2 |
| blend_pool7 | ids=['51d3abf5b2', '3f75dbb0b1', '69b2219339', '580124a4ca', 'f2f5ca7406', 'dbd170cd58', ' | 2023-01-01..2025-10-31 | 9.88 | 19.06 | 0.353 |  | 94.3 | d99e45c506 |
| blend_pool3 | ids=['51d3abf5b2', '3f75dbb0b1', '69b2219339'], halflife=30.0, k=20.0, per_hour=False, los | 2023-01-01..2025-10-31 | 9.88 | 18.97 | 0.353 |  | 94.9 | 2dae4f56b9 |
| blend_pool7 | ids=['51d3abf5b2', '3f75dbb0b1', '69b2219339', '580124a4ca', 'f2f5ca7406', 'dbd170cd58', ' | 2023-01-01..2025-10-31 | 10.04 | 19.43 | 0.359 |  | 95.1 | 9371338e11 |
| lgbm_stack_lora | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, base=51d3abf5b2, burn_in=20 | 2023-01-01..2025-10-31 | 10.19 | 19.10 | 0.364 |  | 94.6 | 3f75dbb0b1 |
| blend_pool7 | ids=['51d3abf5b2', '3f75dbb0b1', '69b2219339', '580124a4ca', 'f2f5ca7406', 'dbd170cd58', ' | 2023-01-01..2025-10-31 | 10.44 | 20.15 | 0.373 |  | 95.0 | 2dbad3c104 |
| tabpfn2_stack_lora | window=730, n_estimators=4, base=51d3abf5b2, burn_in=2023-01-01, max_rows=3000, target=y - | 2023-01-01..2025-10-31 | 10.58 | 19.35 | 0.378 |  | 95.2 | 8cf7cb666a |
| lgbm_stack_zs | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, base=7a7adf81e5, burn_in=20 | 2023-01-01..2025-10-31 | 10.61 | 20.34 | 0.379 |  | 94.7 | 85313a733a |
| chronos2_ft_lora | lr=0.0001, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, n | 2022-01-01..2025-10-31 | 13.52 | 23.44 | 0.356 | 9.40 | 94.0 | 51d3abf5b2 |
| chronos2_ft_lora_pregate | lr=0.0001, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, n | 2024-09-01..2025-10-31 | 13.58 | 26.93 | 0.446 | 9.60 | 93.0 | 864e91aa35 |
| chronos2_ft_lora_multi | lr=0.0001, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, n | 2022-01-01..2025-10-31 | 13.59 | 23.49 | 0.358 | 9.44 | 93.9 | de49f09033 |
| chronos2_ft_full | lr=1e-06, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, ne | 2022-01-01..2025-10-31 | 13.66 | 23.81 | 0.360 | 9.55 | 93.9 | f2f5ca7406 |
| ensemble_lora_lgbm | ids=['51d3abf5b2', '69b2219339'], weights=equal | 2022-01-01..2025-10-31 | 14.04 | 24.35 | 0.370 |  | 94.3 | 13f7720eea |
| chronos2_ft_full | lr=1e-05, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, ne | 2022-01-01..2025-10-31 | 14.12 | 24.33 | 0.372 | 9.81 | 93.4 | dbd170cd58 |
| chronos2_zs | context_days=85, covs=True, neighbours=True, calendar=False, pre_gate=True | 2024-09-01..2025-10-31 | 14.25 | 29.03 | 0.468 | 10.07 | 92.7 | 9998c635ba |
| ensemble_lora_lgbm_lear | ids=['51d3abf5b2', '69b2219339', '580124a4ca'], weights=equal | 2022-01-01..2025-10-31 | 14.32 | 24.61 | 0.378 |  | 93.7 | f846a34128 |
| ensemble_zs_lgbm | ids=['7a7adf81e5', '69b2219339'], weights=equal | 2022-01-01..2025-10-31 | 14.53 | 25.45 | 0.383 |  | 94.2 | ea25dc5487 |
| ensemble_zs_lgbm_lear | ids=['7a7adf81e5', '69b2219339', '580124a4ca'], weights=equal | 2022-01-01..2025-10-31 | 14.66 | 25.33 | 0.386 |  | 93.7 | d9d89d4949 |
| chronos2_zs | context_days=85, covs=True, neighbours=True, calendar=False | 2022-01-01..2025-10-31 | 14.66 | 25.69 | 0.386 | 10.23 | 93.5 | 7a7adf81e5 |
| chronos2_zs | context_days=85, covs=True, neighbours=False, calendar=False | 2022-01-01..2025-10-31 | 14.81 | 25.64 | 0.390 | 10.34 | 93.3 | 297007c63e |
| chronos2_zs | context_days=341, covs=True, neighbours=False, calendar=True | 2022-01-01..2025-10-31 | 14.99 | 25.99 | 0.395 | 10.49 | 93.2 | 8324d03f4e |
| chronos2_zs | context_days=85, covs=True, neighbours=False, calendar=True | 2022-01-01..2025-10-31 | 15.07 | 26.17 | 0.397 | 10.54 | 93.2 | 6fa0d05a99 |
| ensemble_zs_lear | ids=['7a7adf81e5', '580124a4ca'], weights=equal | 2022-01-01..2025-10-31 | 15.09 | 25.89 | 0.398 |  | 92.4 | 3b4d9631f6 |
| lgbm_stack_gate | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, gate_strict=True, base=4e71 | 2023-01-01..2025-10-31 | 16.98 | 28.63 | 0.607 |  | 91.4 | 2ee96d5f3e |
| lgbm_stack_gate_lora | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, gate_strict=True, base=d8d9 | 2023-01-01..2025-10-31 | 17.02 | 28.49 | 0.608 |  | 91.9 | 221a0547f8 |
| tabpfn2 | window=730, n_estimators=4, max_rows=3000, target=y - p_d1_mean, refit=monthly, n_features | 2022-01-01..2025-10-31 | 17.21 | 29.58 | 0.453 |  | 90.9 | 8851fdfc81 |
| lgbm | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, target=y - p_d1_mean, refit | 2022-01-01..2025-10-31 | 17.31 | 29.23 | 0.456 |  | 92.2 | 69b2219339 |
| lgbm_hl | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, halflife=180.0, target=y -  | 2022-01-01..2025-10-31 | 17.35 | 29.37 | 0.457 |  | 92.1 | 8133309c04 |
| lear | windows=[56, 84, 728, 1092], exog=['load_fc', 'ren_fc'] | 2022-01-01..2025-10-31 | 18.20 | 30.33 | 0.480 |  | 88.9 | 580124a4ca |
| lgbm_hl | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, halflife=30.0, target=y - p | 2022-01-01..2025-10-31 | 19.41 | 32.09 | 0.512 |  | 91.2 | f10dc65157 |
| chronos2_ft_lora_gate | lr=0.0001, steps=1000, batch=32, context_days=85, val_days=60, lora_r=8, calendar=False, n | 2022-01-01..2025-10-31 | 20.83 | 34.43 | 0.549 | 14.54 | 91.8 | d8d9083325 |
| chronos2_zs | context_days=85, covs=True, neighbours=True, calendar=False, gate_strict=True | 2022-01-01..2025-10-31 | 21.21 | 34.97 | 0.559 | 14.84 | 91.5 | 4e71fd0d1a |
| chronos2_zs | context_days=85, covs=False, neighbours=False, calendar=False | 2022-01-01..2025-10-31 | 23.04 | 37.56 | 0.607 | 16.10 | 90.9 | 2c0716d255 |
| lgbm_gate | window=0, objective=l1, n_estimators=1500, lr=0.03, leaves=63, gate_strict=True, target=y  | 2022-01-01..2025-10-31 | 25.89 | 41.76 | 0.682 |  | 88.4 | 1f92d7dfe1 |
| naive | rule=D-7 Mon/Sat/Sun, else D-1 | 2022-01-01..2025-10-31 | 37.94 | 60.17 | 1.000 |  | 84.5 | 73450863f2 |
| naive_weekly | rule=D-7 | 2022-01-01..2025-10-31 | 47.30 | 73.48 | 1.247 |  | 82.5 | 17ca79b952 |

### Sealed holdout

| Model | Config | Window | MAE | RMSE | rMAE | CRPS | Battery % of oracle | id |
|---|---|---|---|---|---|---|---|---|
| blend_pool3 | role=candidate, dev_id=2dae4f56b9, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 11.84 | 22.51 | 0.340 |  | 96.2 | 9256383124 |
| lgbm_stack_lora | role=reference, dev_id=3f75dbb0b1, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 11.99 | 22.45 | 0.344 |  | 96.0 | b0e0553ae7 |
| chronos2_ft_lora | role=candidate, dev_id=51d3abf5b2, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 12.38 | 23.78 | 0.355 | 8.73 | 96.0 | 4b84e50737 |
| chronos2_zs | role=reference, dev_id=7a7adf81e5, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 13.53 | 25.46 | 0.388 | 9.52 | 95.3 | a7536ac266 |
| chronos2_ft_lora_pregate | role=candidate, dev_id=864e91aa35, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 15.39 | 27.31 | 0.443 | 11.10 | 94.6 | 15830042bf |
| lear | role=reference, dev_id=580124a4ca, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 15.40 | 25.93 | 0.442 |  | 92.4 | b9d6688033 |
| lgbm | role=reference, dev_id=69b2219339, chronos=51d3abf5b2, fit_end=2025-09-01, isolation=macos | 2025-11-01..2026-11-09 | 15.51 | 26.00 | 0.445 |  | 94.3 | 51cfdbe45f |

### Post-study dev readiness references

These were added after the holdout study; they were not part of candidate selection. Battery figures here retain the original gross metric; constrained cost results are in runs/readiness/report.md.

| Model | Config | Window | MAE | RMSE | rMAE | CRPS | Battery % of oracle | id |
|---|---|---|---|---|---|---|---|---|
| pregate_lgbm | protocol=27c9d9e, current_wind_solar=pregate_weather, parameters={'objective': 'l1', 'n_es | 2025-07-01..2025-10-31 | 12.69 | 22.02 | 0.473 |  | 92.9 | d89d09c3c9 |
| pregate_lear | protocol=27c9d9e, current_wind_solar=pregate_weather, parameters=[56, 84, 728, 1092] | 2025-07-01..2025-10-31 | 14.40 | 23.69 | 0.536 |  | 88.9 | c6d2f0377c |

### Post-study dev adaptive experiment

A separate dev-only trial after holdout inspection; not a new unseen test. Decision and constrained costs: runs/adaptive/report.md.

| Model | Config | Window | MAE | RMSE | rMAE | CRPS | Battery % of oracle | id |
|---|---|---|---|---|---|---|---|---|
| pregate_lora_adaptive_bias | protocol_sha256=eac61fd83fadf54bd0531034e836045f144277782e0d8745220ac7dcb9b0d0c5, base=864 | 2025-07-01..2025-10-31 | 11.75 | 20.13 | 0.438 |  | 94.9 | f2f52c9e2d |
<!-- results:end -->

Not trading advice. No trades or bids are placed by anything in this repository.
