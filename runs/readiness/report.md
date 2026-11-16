# Dev-only pre-gate readiness study

New protocol after the original holdout was opened; no new untouched test result.
Window: 2025-07-01..2025-10-31, 123 days. Models/settings fixed before this run.

| Model | MAE EUR/MWh | DM p vs pre-gate LEAR | Margin after variable costs EUR |
|---|---|---|---|
| pregate_lora | 12.15 | 4.1461315985813485e-05 | 12407 |
| pregate_lgbm | 12.69 | 0.0001099399783890398 | 12142 |
| pregate_lear | 14.40 | None | 11528 |
| naive | 26.84 | 0.99999999999881 | 11659 |

80% interval coverage: raw 71.1%, calibrated 83.6%.
Per-hour conformal expansion fitted only through June 2025; dependence/shift preclude a guarantee.

Battery: 1 MWh / 1 MW, 95% one-way efficiency, 0.1 MWh SOC grid, at most two equivalent cycles/day,
empty midnight; 3 EUR/grid MWh fees+slippage and 20 EUR/discharged battery MWh wear assumed.
These are illustrative sensitivities, not asset quotes or established net returns.
Fixed-cost sensitivity: 0/15,000/30,000 EUR annually, prorated to scored days.
Hourly auction prices omit 15-minute execution, imbalance risk, outages and taxes.
Dev uses the original 24-bin DST convention; prospective paper issues skip 23/25-hour delivery days.
Weather features for these baselines start in September 2024; Chronos has a longer training/pretraining history.
DM p-values are the existing daily-loss test and do not adjust for serial correlation or trial selection.

| Fees+slippage EUR/grid MWh | Wear EUR/discharged MWh | Annual fixed EUR | Scenario net EUR |
|---|---|---|---|
| 0 | 0 | 0 | 17578 |
| 3 | 20 | 0 | 12407 |
| 3 | 20 | 15000 | 7355 |
| 3 | 20 | 30000 | 2304 |
| 6 | 40 | 30000 | -1842 |
