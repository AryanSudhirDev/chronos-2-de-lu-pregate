"""Describe saved replay versus market prices; causal naive benchmark, no model tuning."""
from dataclasses import asdict
import json
import numpy as np
import pandas as pd

from epf.data import ROOT
from epf.offline import ModelWorker, disable_controller_network
from epf.readiness import Battery
from replay_holdout import actual_day, durable, digest, validate_dispatch

OUT=ROOT/'runs/replay'


def main():
    disable_controller_network()
    protocol=json.loads((OUT/'protocol.json').read_text())
    source=json.loads((OUT/'summary.json').read_text())
    b=Battery(**protocol['battery'])
    hourly=pd.read_csv(OUT/'hourly.csv')
    daily=pd.read_csv(OUT/'daily.csv').set_index('day')
    means=hourly.groupby('day').agg(actual_mean=('actual_eur_mwh','mean'),
        forecast_mean=('forecast_eur_mwh','mean'),low=('actual_eur_mwh','min'),high=('actual_eur_mwh','max'))
    daily=daily.join(means)
    successive=pd.Series(pd.to_datetime(daily.index).to_series().diff().values,index=daily.index)==pd.Timedelta(days=1)
    daily['market_change']=daily.actual_mean.diff().where(successive)
    daily['regime']=np.where(daily.market_change<0,'falling',np.where(daily.market_change>0,'rising','undefined'))
    groups=[]
    for regime,frame in daily.groupby('regime'):
        groups.append(dict(regime=regime,n_days=len(frame),
            positive_variable_margin_days=int((frame.margin_eur>0).sum()),
            margin_eur=float(frame.margin_eur.sum()),net_eur=float(frame.scenario_net_eur.sum())))
    # Trusted controller releases only the benchmark's already-cleared lagged day.
    raw_path=ROOT/'data/hourly_raw.parquet'
    raw=pd.read_parquet(raw_path,columns=['price_de'])['price_de']
    records=[]; skipped=[]
    with ModelWorker('market_naive_dispatch',threads=1) as worker:
        for day in daily.index:
            target=pd.Timestamp(day)
            lag=7 if target.weekday() in (0,5,6) else 1
            lag_day=(target-pd.Timedelta(days=lag)).strftime('%Y-%m-%d')
            assert lag_day<day
            try:
                known=actual_day(raw,lag_day)
            except ValueError:
                skipped.append(dict(day=day,source_day=lag_day,reason='lagged day missing or not 24 hours'))
                continue
            prediction=known.to_numpy()
            grid,wear=worker.call('dispatch',forecast=prediction,assumptions=asdict(b))
            validate_dispatch(grid,wear,b)
            records.append(dict(day=day,source_day=lag_day,source_last_timestamp=known.index[-1].isoformat(),
                forecast_eur_mwh=prediction.tolist(),grid_mwh=grid.tolist(),discharged_mwh=wear.tolist()))
    manifest=dict(market_protocol_sha256=digest(OUT/'market_protocol.json'),
        battery_protocol_sha256=digest(OUT/'protocol.json'),actual_source_sha256=digest(raw_path),
        primary_decisions_sha256=digest(OUT/'decisions.json'),
        dispatch_code_sha256=digest(ROOT/'src/epf/readiness.py'),records=records,skipped=skipped)
    durable(OUT/'market_benchmark_decisions.json',manifest)
    # All benchmark schedules are persisted before target-day settlement starts.
    scored=[]
    for row in records:
        day=row['day']; y=actual_day(raw,day).to_numpy()
        grid=np.asarray(row['grid_mwh']);wear=np.asarray(row['discharged_mwh'])
        margin=float(-y@grid - b.fee_slippage_eur_mwh*abs(grid).sum()-b.wear_eur_mwh*wear.sum())
        scored.append(dict(day=day,benchmark_margin_eur=margin,
            benchmark_mae=float(abs(y-np.asarray(row['forecast_eur_mwh'])).mean()),
            model_margin_eur=float(daily.loc[day,'margin_eur']),model_mae=float(daily.loc[day,'mae'])))
    comp=pd.DataFrame(scored).set_index('day')
    fixed=protocol['annual_fixed_eur']*source['calendar_days']/365.25
    comparison=dict(n_common_days=len(comp),benchmark_skipped=skipped,
        model_margin_eur=float(comp.model_margin_eur.sum()),benchmark_margin_eur=float(comp.benchmark_margin_eur.sum()),
        model_mae=float(comp.model_mae.mean()),benchmark_mae=float(comp.benchmark_mae.mean()),
        fixed_cost_eur=fixed,model_net_eur=float(comp.model_margin_eur.sum()-fixed),
        benchmark_net_eur=float(comp.benchmark_margin_eur.sum()-fixed))
    comparison['incremental_margin_eur']=comparison['model_margin_eur']-comparison['benchmark_margin_eur']
    summary=dict(market_first_day=float(daily.actual_mean.iloc[0]),market_last_day=float(daily.actual_mean.iloc[-1]),
        first30_mean=float(daily.actual_mean.iloc[:30].mean()),last30_mean=float(daily.actual_mean.iloc[-30:].mean()),
        first30_dates=[daily.index[0],daily.index[29]],last30_dates=[daily.index[-30],daily.index[-1]],
        market_level_pct=float(100*(daily.actual_mean.iloc[-30:].mean()/daily.actual_mean.iloc[:30].mean()-1)),
        groups=groups,comparison=comparison,market_protocol_sha256=digest(OUT/'market_protocol.json'),
        benchmark_decisions_sha256=digest(OUT/'market_benchmark_decisions.json'))
    durable(OUT/'market_summary.json',summary)
    daily.to_csv(OUT/'market_daily.csv');comp.to_csv(OUT/'market_comparison.csv')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
