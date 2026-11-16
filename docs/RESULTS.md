# Results and what they establish

The study shows that LoRA adaptation improved Chronos-2 under the post-auction research
information set. It does not establish a better pre-auction forecaster on the original holdout,
or a live trading advantage. This distinction is central to interpreting the release.

## Original development study

Walk-forward development covered 2022-01-01 through 2025-10-31. The 2022 crisis makes
this full window harder than comparisons beginning in 2023; scores from those windows must
not be ranked together. Selected full-window results from `runs/ledger.jsonl`:

| Model | MAE EUR/MWh | Trial ID |
|---|---:|---|
| Chronos-2 without covariates | 23.04 | See ledger |
| LEAR | 18.20 | `580124a4ca` |
| LightGBM | 17.31 | `69b2219339` |
| Chronos-2 zero-shot, load/renewables/neighbours | 14.66 | `7a7adf81e5` |
| Chronos-2 LoRA | 13.52 | `51d3abf5b2` |
| Full fine-tune, LR 1e-6 | 13.66 | `f2f5ca7406` |
| Full fine-tune, LR 1e-5 | 14.12 | `dbd170cd58` |

These rows use post-auction TSO wind/solar forecasts. Covariates explain a large part of the
gain over univariate inference. LoRA had lower development MAE than the tested full-tuning
settings. Longer context and calendar features did not improve the best zero-shot setup.
These are selected comparisons among recorded experiments, not selection-adjusted evidence.

On the shorter shared 2023-01..2025-10 window, the zero-shot residual stack scored 10.61
and the LoRA residual stack 10.19. Excluding 2022 changes the problem substantially; neither
number should be presented as a direct improvement over the full-window 13.52 result.

## One-time holdout

Three candidates and four references were evaluated after candidate selection. The common
window is **372 days, 2025-11-01..2026-11-09**. Other rows have 374 available days, but
the common comparison removes 2026-04-10 and 2026-04-11, which lack pre-gate forecasts.

| Model | MAE EUR/MWh | Battery gross profit as % of oracle | Role |
|---|---:|---:|---|
| Compact blend | 11.8143 | 96.22% | Candidate, post-auction |
| Residual LightGBM stack on LoRA | 11.9670 | 96.01% | Reference, post-auction |
| LoRA Chronos-2 | 12.3510 | 96.00% | Candidate, post-auction |
| Zero-shot Chronos-2 | 13.4993 | 95.32% | Reference, post-auction |
| LEAR | 15.3446 | 92.38% | Reference, post-auction |
| **LoRA with pre-auction inputs** | **15.3860** | **94.61%** | **Published candidate input setup** |
| LightGBM | 15.4930 | 94.28% | Reference, post-auction |

Sources: `runs/holdout_summary.json` and `runs/compare_holdout.md`. The best blend reduced MAE
by **23.01%** against LEAR. LoRA reduced MAE by **8.51%** against zero-shot; the recorded
one-sided daily multivariate DM p-value is **5.31e-12**. The pre-gate candidate's p-value
against LEAR is **0.543**, providing no improvement evidence. That LEAR reference itself
uses post-auction inputs, so it is not an equal-information pre-auction benchmark.

Some saved p-values are floating-point `0.0`; this is numerical underflow, not a literal
probability of zero. Existing tests do not correct for serial correlation or repeated trial
selection. No deflated-Sharpe or selection-adjusted significance claim is made.

All 2,616 issued daily forecasts were checked against the saved outputs. The audit archive
and hash manifest are included. The OS isolation protects the worker's information flow,
not the provenance of the foundation model's original pretraining.

## Which result belongs to the published weights?

The exported checkpoint is the frozen adapter retained for the paper stream, not a merged
ensemble. The original holdout runner trained the LoRA fit once and evaluated it with two
future-covariate setups: post-auction TSO wind/solar and weather-derived pre-auction estimates.
The **15.3860** row is the relevant result for its intended pre-auction input setup. The
**12.3510** row describes use of the research information set. The **11.8143** blend requires
other fitted models and historical blending rules; publishing these neural weights does not
publish those other fitted components as part of the model tensor file.

## Later development-only readiness comparison

After the original holdout, a new readiness protocol used **123 development days,
2025-07-01..2025-10-31**, with pre-auction inputs for all three models.

| Model | MAE EUR/MWh | Margin after assumed variable costs, EUR |
|---|---:|---:|
| LoRA | 12.1483 | 12,406.51 |
| LightGBM | 12.69 | About 12,142 |
| LEAR | 14.40 | About 11,528 |
| Naive | 26.84 | About 11,659 |

LoRA's MAE improvement against LightGBM was about **4.3%**, with only about **EUR 265**
additional variable-cost margin. Baseline weather training begins in September 2024;
Chronos benefits from a longer history and base-model pretraining. This is useful development
evidence, not a second unseen test. Exact values are in `runs/readiness/summary.json`.

Calibration used 303 earlier forecast days ending 2025-06-30. Nominal 80% coverage on the
later readiness block rose from **71.1% to 83.6%**, with wider intervals. Distribution shift
and temporal dependence prevent treating that observed coverage as a future guarantee.
Unadjusted holdout coverage was only about **72.0%** for post-auction LoRA and **61.9%**
for the pre-gate input setup.

## Costs and the failed adaptive experiment

Readiness dispatch assumes a 1 MW / 1 MWh battery, 95% one-way efficiency, empty start/end,
a 0.1 MWh SOC grid and at most two equivalent full cycles a day. Fees/slippage are assumed
at EUR 3 per grid MWh and wear at EUR 20 per discharged battery MWh.

| Fees/slippage | Wear | Annual fixed cost | LoRA scenario net over 123 days |
|---|---|---|---:|
| 0 | 0 | 0 | EUR 17,578 |
| 3 | 20 | 0 | EUR 12,407 |
| 3 | 20 | EUR 15,000 | EUR 7,355 |
| 3 | 20 | EUR 30,000 | EUR 2,304 |
| 6 | 40 | EUR 30,000 | **EUR -1,842** |

These are sensitivity assumptions, not asset quotes. Taxes, outages, imbalance, financing,
capital expenditure and real 15-minute execution are excluded.

One later dev-only adaptive hourly bias correction improved MAE from **12.1483 to 11.7535**
(3.25%), but reduced variable-cost margin by **EUR 4.62**. Its predeclared promotion rule
required at least 1% better MAE and no lower margin. It failed and was not promoted.
This changed 24 correction weights, not Chronos neural weights.

The retrospective annual replay also had a sustained winter loss: **EUR -6,560.84** over
November 2025–February 2026 after assumed fixed costs. Even perfect forecasts lost
**EUR -5,669.45** under those assumptions. Maximum cumulative drawdown was **EUR 6,563.51**;
the final **EUR 8,104.57** scenario margin concealed that early loss. These are cash margins,
not investment returns. See `runs/replay/diagnosis.md` for all months and assumptions.

## Prospective paper observation: five days only

The paper stream froze the adapter and calibration, collected timestamped inputs before the
gate and scored issued forecasts later. It stopped at the user's request on 2026-11-15,
after **six issued days and five scored days**.

| Model | MAE EUR/MWh | Assumed variable-cost margin, EUR |
|---|---:|---:|
| Pre-gate LEAR | 20.1169 | 306.03 |
| Published LoRA | 22.3631 | 320.89 |
| Pre-gate LightGBM | 41.5882 | 343.37 |
| Naive | 75.2869 | 225.51 |

Observed calibrated interval coverage was 80.0%. Five days are far too few to assess seasonal
accuracy or reliability. The planned 30-day review was not completed. This small observation
does not confirm the development ranking: LEAR had lower MAE, while LightGBM had higher
simulated margin. Source: `runs/paper/summary.json`.

## What the work contributed

The useful outputs are a working adapted checkpoint, evidence that covariates and LoRA help
under the specified research setup, a tested causal evaluation boundary, a reproducible record
of successful and failed variants, and explicit separation of research accuracy from auction-time
usability and economics. A profitable live strategy has not been demonstrated.
