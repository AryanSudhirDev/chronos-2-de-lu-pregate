"""One declared dev-only adaptive correction trial. Never loads holdout data."""
import hashlib
import json
import numpy as np
import pandas as pd

from epf.data import ROOT, load_panel
from epf.offline import ModelWorker, disable_controller_network
from epf.paper import durable_json
from epf.readiness import economics
from epf.ledger import record, LEDGER

OUT=ROOT/'runs/adaptive'


def main():
    disable_controller_network()
    protocol=json.loads((OUT/'protocol.json').read_text())
    panel=load_panel()  # default cuts at 2025-10-31, never allow_holdout=True
    path=ROOT/f"runs/forecasts/{protocol['source_dev_forecasts']}.npz"
    z=np.load(path,allow_pickle=False)
    days=pd.DatetimeIndex(z['days']); forecasts=z['point']
    assert days.max()<pd.Timestamp('2025-11-01') and days.is_monotonic_increasing and days.is_unique
    target=np.array([panel['price_de'][panel.idx(d)] for d in days])
    if not np.isfinite([forecasts,target]).all():raise ValueError('incomplete dev forecast/outcome')
    config_hash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    corrected=[];weights=[]
    with ModelWorker('adaptive',threads=1) as worker:
        worker.call('adaptive_init',halflife=protocol['halflife_calendar_days'])
        for day,forecast,actual in zip(days,forecasts,target):
            day=str(day.date())
            issued=worker.call('adaptive_predict',day=day,forecast=forecast)
            # The file is durable before the target crosses the boundary.
            row=dict(day=day,protocol_sha256=config_hash,point=issued['point'].tolist(),
                original=forecast.tolist(),weights=issued['weights'].tolist(),last_observed=issued['last_observed'])
            dest=OUT/'issued'/f'{day}.json'
            if dest.exists():
                if json.loads(dest.read_text())!=row:raise ValueError('saved adaptive dev prediction changed')
            else:durable_json(dest,row)
            corrected.append(issued['point']);weights.append(issued['weights'])
            state=worker.call('adaptive_observe',day=day,actual=actual)
    corrected=np.asarray(corrected)
    sel=(days>=protocol['evaluate_start'])&(days<=protocol['evaluate_end'])
    evaluated=days[sel];y=target[sel];base=forecasts[sel];adaptive=corrected[sel]
    base_mae=float(abs(y-base).mean());adaptive_mae=float(abs(y-adaptive).mean())
    base_e=economics(y,base);adaptive_e=economics(y,adaptive)
    improvement=1-adaptive_mae/base_mae
    passed=bool(improvement>=.01 and adaptive_e['margin_eur']>=base_e['margin_eur'])
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    summary=dict(protocol_sha256=config_hash,source_forecast_sha256=source_hash,
        window=[str(evaluated[0].date()),str(evaluated[-1].date())],n_days=len(evaluated),
        base=dict(mae=base_mae,economics=base_e),adaptive=dict(mae=adaptive_mae,economics=adaptive_e),
        mae_improvement_fraction=improvement,variable_margin_change_eur=adaptive_e['margin_eur']-base_e['margin_eur'],
        dev_gate_passed=passed,decision='add separate prospective challenger' if passed else 'retain frozen control; do not add this challenger',
        scope='One new dev trial after holdout inspection. No new unseen result. Small correction weights, not continual core Chronos training.')
    durable_json(OUT/'summary.json',summary) if not (OUT/'summary.json').exists() else None
    if json.loads((OUT/'summary.json').read_text())!=summary:raise ValueError('adaptive summary changed on rerun')
    seed=dict(protocol_sha256=config_hash,source_forecast_sha256=source_hash,**state)
    if (OUT/'seed.json').exists():
        if json.loads((OUT/'seed.json').read_text())!=seed:raise ValueError('seed changed')
    else:durable_json(OUT/'seed.json',seed)
    cfg=dict(protocol_sha256=config_hash,base=protocol['source_dev_forecasts'],halflife=protocol['halflife_calendar_days'],correction='EWMA hourly original forecast residual')
    existing=[json.loads(line) for line in LEDGER.read_text().splitlines()]
    matching=[r for r in existing if r['name']=='pregate_lora_adaptive_bias' and r['split']=='adaptive_dev' and r['config']==cfg]
    if matching:
        np.testing.assert_array_equal(np.load(ROOT/f"runs/forecasts/{matching[-1]['id']}.npz")['point'],adaptive)
    else:
        record('pregate_lora_adaptive_bias',cfg,evaluated,adaptive,panel=panel,split='adaptive_dev',
            notes='One predeclared daily adaptive correction. Dev only, not unseen; constrained-cost selection in runs/adaptive/summary.json. No holdout or core-weight updates.')
    pd.DataFrame(dict(day=evaluated.strftime('%Y-%m-%d'),base_mae=np.abs(y-base).mean(1),adaptive_mae=np.abs(y-adaptive).mean(1),
        mean_weight=np.asarray(weights)[sel].mean(1))).to_csv(OUT/'daily.csv',index=False)
    text=f"""# One adaptive correction experiment

Dev only, {summary['window'][0]}..{summary['window'][1]} ({summary['n_days']} days); no new unseen result.
Protocol fixed before this run. A 24-weight hourly correction learns only from already-released
actual errors, halflife 30 calendar days. Chronos neural weights remain fixed. Every corrected
forecast and its pre-update weights were saved before an isolated worker received that day's target.

| Forecast | MAE EUR/MWh | Margin after assumed variable costs EUR |
|---|---:|---:|
| Original pre-gate Chronos | {base_mae:.4f} | {base_e['margin_eur']:.2f} |
| Daily adaptive correction | {adaptive_mae:.4f} | {adaptive_e['margin_eur']:.2f} |

MAE change: {100*improvement:.2f}% improvement. Variable-cost margin change EUR {summary['variable_margin_change_eur']:.2f}.
Fixed costs are identical and cancel. Battery settings/cost assumptions are the existing readiness
defaults. Original dev DST normalization applies; prospective paper skips DST days.

Predeclared gate: at least 1% MAE improvement AND no lower constrained variable-cost margin.
Gate passed: {passed}. Decision: **{summary['decision']}**.

One variant, no follow-up hyperparameter search from these outcomes. Original study/holdout forecasts,
ledger rows and paper control policy remain unchanged. Actual error observations drive learning,
not predictions used as self-labels. Core-model gradients were not tested by this lightweight trial.
Old publication vintages, Chronos pretraining, real execution and asset costs remain unverified.
An adaptive layer cannot guarantee improvement or manufacture profitable price spreads.
"""
    (OUT/'report.md').write_text(text)
    print(text,flush=True)


if __name__=='__main__':main()
