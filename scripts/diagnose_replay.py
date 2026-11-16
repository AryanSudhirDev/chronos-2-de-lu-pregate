"""Read-only attribution of the frozen replay: drawdowns, costs and daily spreads.

Diagnostic after inspection; no fitting, strategy selection, new forecasts or trades.
"""
import json
import numpy as np
import pandas as pd
from epf.data import ROOT

OUT=ROOT/'runs/replay'


def diagnose(daily, hourly, annual_fixed=30000):
    calendar=pd.date_range('2025-11-01','2026-11-09')
    x=daily.set_index(pd.to_datetime(daily.day)).reindex(calendar)
    for col in ['margin_eur','oracle_margin_eur']:
        x[col]=x[col].fillna(0)
    x['fixed']=annual_fixed/365.25
    x['net']=x.margin_eur-x.fixed
    x['oracle_net']=x.oracle_margin_eur-x.fixed
    x['cumulative_net']=x.net.cumsum()
    x['cumulative_oracle_net']=x.oracle_net.cumsum()
    x['cumulative_variable_margin']=x.margin_eur.cumsum()
    x['drawdown']=x.cumulative_net-x.cumulative_net.cummax().clip(lower=0)
    x['spread']=x.high-x.low
    monthly=x.resample('MS').agg(mean_price=('actual_mean','mean'),mean_spread=('spread','mean'),
        variable_margin=('margin_eur','sum'),fixed=('fixed','sum'),net=('net','sum'),
        oracle_net=('oracle_net','sum'),scored_days=('actual_mean','count'))
    monthly['price_change']=monthly.mean_price.diff()
    # Calendar months all reported; the first four months are an ex-post drawdown description.
    winter=x.loc['2025-11-01':'2026-02-28']
    trough=x.cumulative_net.idxmin()
    recovery=x.loc[trough:]; recovery=recovery[recovery.cumulative_net>=0]
    daily_mean=hourly.groupby('day').actual_eur_mwh.transform('mean')
    level_component=float((-daily_mean*hourly.grid_mwh).sum())
    spread_component=float((-(hourly.actual_eur_mwh-daily_mean)*hourly.grid_mwh).sum())
    gross=float((-hourly.actual_eur_mwh*hourly.grid_mwh).sum())
    np.testing.assert_allclose(level_component+spread_component,gross,atol=1e-8,rtol=0)
    # Same fixed schedule in a market with flat within-day prices, preserving each day's mean.
    flat_variable_margin=level_component-float(hourly.variable_cost_eur.sum())
    correlation=x[['actual_mean','spread','margin_eur']].corr()
    summary=dict(calendar_days=len(x),scored_days=int(x.actual_mean.notna().sum()),
        winter=dict(start='2025-11-01',end='2026-02-28',n_days=len(winter),
            variable_margin=float(winter.margin_eur.sum()),fixed=float(winter.fixed.sum()),
            net=float(winter.net.sum()),oracle_net=float(winter.oracle_net.sum())),
        worst_drawdown_eur=float(x.drawdown.min()),trough_date=str(trough.date()),
        first_recovery_date=None if recovery.empty else str(recovery.index[0].date()),
        ending_net=float(x.net.sum()),daily_corr_margin_price=float(correlation.loc['margin_eur','actual_mean']),
        daily_corr_margin_spread=float(correlation.loc['margin_eur','spread']),
        correlations_note='Descriptive associations within this historical sample, not causal estimates or out-of-sample guarantees',
        gross_decomposition=dict(mean_level_component=level_component,within_day_pattern_component=spread_component,gross=gross),
        frozen_schedule_flat_day_variable_margin=flat_variable_margin,
        flat_day_note='Accounting counterfactual for unchanged schedules, not a policy rerun or a realistic alternate market',
        explanation='Empty start/end eliminates overnight holding. Higher daily means increase conversion-loss expense for fixed schedules; within-day spreads and fixed costs determine opportunity. No regime-independent profit is established.')
    return x,monthly,summary


def main():
    daily=pd.read_csv(OUT/'market_daily.csv')
    hourly=pd.read_csv(OUT/'hourly.csv')
    x,monthly,s=diagnose(daily,hourly)
    original=json.loads((OUT/'summary.json').read_text())
    np.testing.assert_allclose(s['ending_net'],original['totals']['scenario_net_eur'],atol=1e-8,rtol=0)
    assert (x.oracle_net>=x.net-1e-8).all()
    np.testing.assert_allclose(monthly.net.sum(),x.net.sum(),atol=1e-8,rtol=0)
    np.testing.assert_allclose(monthly.fixed.sum(),original['totals']['fixed_cost_eur'],atol=1e-8,rtol=0)
    (OUT/'diagnosis.json').write_text(json.dumps(s,indent=2)+'\n')
    x.to_csv(OUT/'diagnosis_daily.csv',index_label='date')
    monthly.to_csv(OUT/'diagnosis_monthly.csv',index_label='month')
    table='\n'.join(f"| {i:%Y-%m} | {r.mean_price:.2f} | {r.mean_spread:.2f} | {r.variable_margin:,.2f} | {r.fixed:,.2f} | {r.net:,.2f} | {r.oracle_net:,.2f} |" for i,r in monthly.iterrows())
    report=f"""# Correction: the strategy did suffer a sustained drawdown

My earlier statement about profitable falling-price days was too reassuring. It pooled isolated
falling days across the full year. It does not establish resilience over a sustained weak period.
These are descriptive diagnostics of the existing replay after inspection, not new untouched tests.

From November 2025 through February 2026 (all {s['winter']['n_days']} calendar days), margin after variable
costs was EUR {s['winter']['variable_margin']:,.2f}, below EUR {s['winter']['fixed']:,.2f} assumed fixed costs.
The simulation lost EUR {-s['winter']['net']:,.2f}. Even the same battery with perfect future-price
knowledge lost EUR {-s['winter']['oracle_net']:,.2f} after the same costs. Forecast mistakes explain the
difference from that upper bound; even perfect forecasts cannot remove the rest of the winter loss
within these dispatch assumptions.

Maximum cumulative drawdown from starting zero/running peak was EUR {-s['worst_drawdown_eur']:,.2f}
on {s['trough_date']}. Cumulative margin first recovered to zero on {s['first_recovery_date']}.
The EUR {s['ending_net']:,.2f} final scenario margin masks that early loss. This is cash margin in EUR,
not an equity curve including a battery purchase or an investment return percentage.

## All calendar months, with no months selected or omitted

The March/October DST days and missing April 10/11 forecasts remain idle, with fixed costs accrued.
November 2026 is partial (nine days). Price and spread averages use scored days; money and fixed
costs include the complete calendar month within the study range.

| Month | Mean price EUR/MWh | Mean daily high-low EUR/MWh | Margin after variable EUR | Fixed EUR | Model net EUR | Perfect-foresight net EUR |
|---|---:|---:|---:|---:|---:|---:|
{table}

January prices rose versus December, yet the strategy still lost money. April prices fell versus
March, yet it made money. Average price direction alone does not explain the results. Daily high-low
spreads were mostly about EUR 62–84/MWh during November–February, then about EUR 158–226 in many
later months. Higher trading opportunities coincided with the recovery. A high-low range is a rough
opportunity indicator, not a feasible battery profit measure; the constrained oracle is the upper bound.

## What the cashflow actually depends on

For each unchanged day-ahead schedule g (positive = buy), actual prices p, and daily mean m:

`energy cashflow = -sum(p*g) = -m*sum(g) - sum((p-m)*g)`

With an empty battery at start/end and conversion losses, total grid purchases are at least total
sales. The mean-level term for the fixed historical schedules is EUR {s['gross_decomposition']['mean_level_component']:,.2f};
the within-day-pattern term is EUR {s['gross_decomposition']['within_day_pattern_component']:,.2f}. Their sum is the original
EUR {s['gross_decomposition']['gross']:,.2f} gross cashflow. Costs are deducted separately. These are algebraic accounting
terms, not independent causal effects: dispatch was chosen from forecasts and would change in another market.
Uniformly increasing each day's prices while holding these schedules fixed makes the mean-level
loss term more negative. Thus it was not mechanically earning by holding a rising asset, but it
was strongly dependent on favorable within-day trading opportunities.

Daily margin correlation with the daily mean price was {s['daily_corr_margin_price']:.3f}; with daily high-low
spread it was {s['daily_corr_margin_spread']:.3f}. Correlation is descriptive, not proof of causality or stable future performance.

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
"""
    (OUT/'diagnosis.md').write_text(report)
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
