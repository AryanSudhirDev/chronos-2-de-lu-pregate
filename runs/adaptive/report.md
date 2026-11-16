# One adaptive correction experiment

Dev only, 2025-07-01..2025-10-31 (123 days); no new unseen result.
Protocol fixed before this run. A 24-weight hourly correction learns only from already-released
actual errors, halflife 30 calendar days. Chronos neural weights remain fixed. Every corrected
forecast and its pre-update weights were saved before an isolated worker received that day's target.

| Forecast | MAE EUR/MWh | Margin after assumed variable costs EUR |
|---|---:|---:|
| Original pre-gate Chronos | 12.1483 | 12406.51 |
| Daily adaptive correction | 11.7535 | 12401.88 |

MAE change: 3.25% improvement. Variable-cost margin change EUR -4.62.
Fixed costs are identical and cancel. Battery settings/cost assumptions are the existing readiness
defaults. Original dev DST normalization applies; prospective paper skips DST days.

Predeclared gate: at least 1% MAE improvement AND no lower constrained variable-cost margin.
Gate passed: False. Decision: **retain frozen control; do not add this challenger**.

One variant, no follow-up hyperparameter search from these outcomes. Original study/holdout forecasts,
ledger rows and paper control policy remain unchanged. Actual error observations drive learning,
not predictions used as self-labels. Core-model gradients were not tested by this lightweight trial.
Old publication vintages, Chronos pretraining, real execution and asset costs remain unverified.
An adaptive layer cannot guarantee improvement or manufacture profitable price spreads.
