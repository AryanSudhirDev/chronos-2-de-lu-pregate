# Correction: the strategy did suffer a sustained drawdown

My earlier statement about profitable falling-price days was too reassuring. It pooled isolated
falling days across the full year. It does not establish resilience over a sustained weak period.
These are descriptive diagnostics of the existing replay after inspection, not new untouched tests.

From November 2025 through February 2026 (all 120 calendar days), margin after variable
costs was EUR 3,295.42, below EUR 9,856.26 assumed fixed costs.
The simulation lost EUR 6,560.84. Even the same battery with perfect future-price
knowledge lost EUR 5,669.45 after the same costs. Forecast mistakes explain the
difference from that upper bound; even perfect forecasts cannot remove the rest of the winter loss
within these dispatch assumptions.

Maximum cumulative drawdown from starting zero/running peak was EUR 6,563.51
on 2026-03-01. Cumulative margin first recovered to zero on 2026-06-26.
The EUR 8,104.57 final scenario margin masks that early loss. This is cash margin in EUR,
not an equity curve including a battery purchase or an investment return percentage.

## All calendar months, with no months selected or omitted

The March/October DST days and missing April 10/11 forecasts remain idle, with fixed costs accrued.
November 2026 is partial (nine days). Price and spread averages use scored days; money and fixed
costs include the complete calendar month within the study range.

| Month | Mean price EUR/MWh | Mean daily high-low EUR/MWh | Margin after variable EUR | Fixed EUR | Model net EUR | Perfect-foresight net EUR |
|---|---:|---:|---:|---:|---:|---:|
| 2025-11 | 101.88 | 84.16 | 1,223.95 | 2,464.07 | -1,240.11 | -995.44 |
| 2025-12 | 93.47 | 61.79 | 565.83 | 2,546.20 | -1,980.37 | -1,763.79 |
| 2026-01 | 110.09 | 72.84 | 948.83 | 2,546.20 | -1,597.37 | -1,373.75 |
| 2026-02 | 96.58 | 63.76 | 556.81 | 2,299.79 | -1,742.99 | -1,536.48 |
| 2026-03 | 100.26 | 157.53 | 3,958.59 | 2,546.20 | 1,412.39 | 1,576.24 |
| 2026-04 | 78.19 | 175.99 | 3,793.16 | 2,464.07 | 1,329.09 | 1,636.73 |
| 2026-05 | 97.54 | 181.10 | 4,387.24 | 2,546.20 | 1,841.04 | 1,960.71 |
| 2026-06 | 109.52 | 208.91 | 4,970.00 | 2,464.07 | 2,505.94 | 2,612.67 |
| 2026-07 | 105.45 | 162.67 | 3,802.55 | 2,546.20 | 1,256.35 | 1,378.00 |
| 2026-08 | 126.89 | 173.98 | 4,043.05 | 2,546.20 | 1,496.85 | 1,679.63 |
| 2026-09 | 144.82 | 225.96 | 6,047.04 | 2,464.07 | 3,582.97 | 3,763.95 |
| 2026-10 | 174.69 | 159.85 | 3,692.52 | 2,546.20 | 1,146.32 | 1,559.13 |
| 2026-11 | 266.41 | 143.32 | 833.68 | 739.22 | 94.46 | 233.55 |

January prices rose versus December, yet the strategy still lost money. April prices fell versus
March, yet it made money. Average price direction alone does not explain the results. Daily high-low
spreads were mostly about EUR 62–84/MWh during November–February, then about EUR 158–226 in many
later months. Higher trading opportunities coincided with the recovery. A high-low range is a rough
opportunity indicator, not a feasible battery profit measure; the constrained oracle is the upper bound.

## What the cashflow actually depends on

For each unchanged day-ahead schedule g (positive = buy), actual prices p, and daily mean m:

`energy cashflow = -sum(p*g) = -m*sum(g) - sum((p-m)*g)`

With an empty battery at start/end and conversion losses, total grid purchases are at least total
sales. The mean-level term for the fixed historical schedules is EUR -5,830.77;
the within-day-pattern term is EUR 56,628.05. Their sum is the original
EUR 50,797.29 gross cashflow. Costs are deducted separately. These are algebraic accounting
terms, not independent causal effects: dispatch was chosen from forecasts and would change in another market.
Uniformly increasing each day's prices while holding these schedules fixed makes the mean-level
loss term more negative. Thus it was not mechanically earning by holding a rising asset, but it
was strongly dependent on favorable within-day trading opportunities.

Daily margin correlation with the daily mean price was 0.197; with daily high-low
spread it was 0.980. Correlation is descriptive, not proof of causality or stable future performance.

## What we can conclude

The model added about EUR 3,407 versus the simple lagged-price forecast on the prior 367-day matched
comparison, but the physical battery economics still failed during the first four months. Both facts
can be true. Positive total scenario margin does not establish reliable performance across market
conditions. No claim of a deployable, robust profitable strategy is justified yet.

Historical input versions/publication timing and original Chronos pretraining remain unverified.
Hourly average execution and EUR 30k/year fixed costs are illustrative; real asset costs, outages,
taxes and imbalance exposure are not established. Daily averages can hide hourly forecasting errors.
No model fits, strategy changes, holdout reopening, original ledger edits or paper-policy changes
were made. Forward paper collection remains the next independent evidence; its first 30 scored days
are only an initial check, not coverage of all seasons or permission to trade.

## Is the model self-improving?

For this pre-gate graph and the prospective paper candidate, the Chronos LoRA weights are fixed.
The checkpoint was fine-tuned through September 1, 2025 with a later pre-holdout validation block.
Each day's inference gets an updated 85-day context of already-known prices/covariates and that
day's pre-gate inputs. Fresh context changes predictions without performing gradient updates or
guaranteeing improvement. The graph joins those separate day-ahead forecasts; it is not a single
forecast made a year in advance. No target day's actual prices are prediction inputs.

The separate research blend updates component weights from already-revealed earlier errors.
Its Chronos component is not thereby retrained. Reference LightGBM uses monthly refits and LEAR
daily recalibration under fixed protocols; the weather mapper also refits using past data. The
blend's better historical scores used late TSO inputs, so it is not the pre-gate battery replay here.
Dev Chronos experiments used yearly walk-forward fine-tuning folds; that is different from daily
self-training of the fixed holdout/paper checkpoint.

An online learner could be tested honestly by predicting day D, persisting that forecast,
revealing D's outcome only when available, then updating for later forecasts under a predeclared
rule. It could improve, degrade or fail under changing conditions. Changing training now would
create a new experiment; it would not revise the original historical predictions or provide a
fresh untouched test on the same outcomes. No such training change was made in this diagnosis.
