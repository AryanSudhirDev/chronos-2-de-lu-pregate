# Historical pre-gate decision replay

This is a retrospective diagnostic of forecasts already produced during the one-time holdout opening.
It is not a fresh unseen experiment or evidence of live execution. No model was retrained or selected.

## Results

Window: 2025-11-01 through 2026-11-09; 370 scored days out of 374 calendar days.
Forecast MAE: 15.38 EUR/MWh. Physical battery: 1 MW grid power / 1 MWh capacity,
95% efficiency each way, empty at start/end of each day, maximum two full discharge cycles/day,
SOC grid 0.1 MWh. Positive grid energy is a purchase; negative is a sale.

| Item | EUR |
|---|---:|
| Energy cashflow at actual historical hourly average prices | 50,797.29 |
| Assumed variable fees/slippage and wear | -11,974.03 |
| Margin after variable costs | 38,823.25 |
| Assumed fixed costs for all 374 calendar days | -30,718.69 |
| Illustrative remaining margin | 8,104.57 |

Fees/slippage: EUR 3/grid MWh bought or sold; wear: EUR 20/discharged battery MWh.
Fixed costs: EUR 30,000/year, prorated at 365.25 days/year, including skipped days.
These are assumptions, not asset quotes or actual net trading returns.
328 days had positive margin after variable costs, 20 negative,
and 22 zero. These day counts exclude fixed costs.
No-action energy margin is zero before the same fixed-cost assumption.
Scorer-only perfect-foresight margin after variable costs: EUR 41,449.84; never supplied to the decision worker.

## What prevents looking ahead

1. Reused forecast 15830042bf. Every one of its 372 daily forecasts exactly matches the saved
   median of the hashed original issued-forecast archive. No model rerun, tuning or ledger append.
2. Separate `issue` invocation has an audit hook rejecting all data/ and ledger reads.
   An OS-sandboxed decision worker cannot read raw outcomes, ledger, forecast files or network;
   its pipe accepts only that day's 24 forecast prices and the fixed battery assumptions.
3. Persisted all 370 decisions with exclusive creation and fsync before invoking `score`.
   Protocol, source forecast, original archive, original manifest and dispatch-code hashes are checked.
4. The scorer alone reads the actual price column. It requires exactly the 24 hourly timestamps
   for each Berlin delivery day, checks battery power/SOC/cycles, and settles the unchanged schedule.
   A UTC/Berlin representation mismatch was caught by this validation and fixed; missing/duplicate
   timestamp tests now cover it. No outcome rows were filled or shifted to bypass the check.
5. Synthetic poisoning checks alter current/future targets and current post-gate wind/solar;
   pre-gate prediction inputs stay exactly unchanged. Actual prices are rejected by the dispatch API.

## Exclusions and evidence limits

Missing forecasts: 2026-04-10, 2026-04-11. DST days excluded: 2026-03-29, 2026-10-25.
Skipped days are assumed idle, while fixed costs still accrue.
The original holdout normalized DST days; this execution replay deliberately excludes them.
Only pre-gate LoRA is replayed here. The better-scoring post-gate blend uses late TSO inputs
and cannot support an auction-time claim. Equally informed pre-gate baseline forecasts were
not retained for this full holdout; their dev-only comparison is in runs/readiness/report.md.

- Forecasts and decisions are historical simulations created now, not timestamped live orders
- The original holdout has already been inspected; this is a retrospective diagnostic with no parameter selection
- Source publication/revision vintages and foundation-model pretraining provenance are not independently proven
- Hourly averaged prices are a proxy for actual quarter-hour products; full fills at the average are an assumption
- Costs are illustrative; taxes, outages, imbalance risk and actual execution are excluded
- Pre-gate mapping fits historically used past post-gate TSO forecasts as labels, not current-day post-gate inputs

The archive hashes prove consistency with retained artifacts, not their historical live publication.
The trusted controller remains outside the OS sandbox. The original model may have been pretrained
on data whose exact provenance is unknown. Historical input archives may include later revisions.
These limits prevent calling this a fully certified leakage-free historical trading test.
The scheduled forward paper stream collects timestamped inputs before future auctions to address
the historical publication uncertainty. It does not place trades or bids.

## Inspect or reproduce

- `hourly.csv`: every buy/sell/hold, forecast, actual price, grid MWh, SOC, cashflow and variable cost.
- `daily.csv`: daily accuracy and economics. Fixed costs for the four skipped days are additionally
  accounted for in summary.json; summing the scored-day CSV alone excludes that EUR 328.54.
- `decisions.json`: frozen forecast-only schedules with source/code hashes.
- `summary.json`: totals, actual-source hash and exclusions.
- `report.html`: interactive day selector, forecast/actual chart and actions.

Run `.venv/bin/python scripts/replay_holdout.py issue`, then a separate
`.venv/bin/python scripts/replay_holdout.py score`. Existing differing artifacts cause refusal.
Run `.venv/bin/python scripts/render_replay.py` to rebuild this presentation.
