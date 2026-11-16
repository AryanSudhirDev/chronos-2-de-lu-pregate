"""Collect public feeds, issue a frozen pre-gate paper forecast, or separately score issues.

    python scripts/paper_day.py collect --day YYYY-MM-DD
    python scripts/paper_day.py issue --bundle data/paper/<bundle>
    python scripts/paper_day.py score
    python scripts/paper_day.py auto  # score earlier issues, then collect/issue tomorrow before gate
    python scripts/paper_day.py rehearse  # dev-only end-to-end model plumbing, never a live issue
"""
import argparse
import hashlib
import json
import shutil
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from epf.data import ROOT, TZ, PRICE_ZONES, _to_daily, load_panel, with_pregate
from epf.paper import utcnow, deadline, validate_issue, durable_json, publish, panel_from_snapshot
from epf.offline import ModelWorker, disable_controller_network
from epf.chronos_io import series_inputs
from epf.features import feature_table
from epf.lear import lear_features, WINDOWS
from epf.readiness import economics
from fetch_data import SMARD, SMARD_SERIES, EC

OUT=ROOT/'runs/paper'
ADAPTER=ROOT/'runs/ckpt/offline_chronos_vftxarhg/fit/finetuned-ckpt'
POINTS=[(54.6,8.8),(53.5,8.0),(52.5,7.5),(53.5,13.0),(52.3,12.5),(51.8,11.5),(51.0,9.5),(50.0,7.5),
        (49.0,10.5),(48.3,11.5),(48.5,9.0),(51.0,13.5),(54.3,6.5),(54.8,14.0)]
VARIABLES=['wind_speed_100m','shortwave_radiation','temperature_2m']
MODELS=['ecmwf_ifs025','icon_seamless']


def check_policy():
    policy=json.loads((OUT/'policy.json').read_text())
    checks=[(ADAPTER/'adapter_model.safetensors','adapter_sha256'),
            (ROOT/'runs/readiness/calibration.json','calibration_sha256'),
            (ROOT/'runs/readiness/protocol.json','dev_protocol_sha256')]
    for path,key in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=policy[key]:
            raise ValueError(f'frozen paper policy changed: {path}; do not silently retune')
    return policy


def get(url, **params):
    r=requests.get(url,params=params,timeout=90);r.raise_for_status();return r


def recent_smard(name):
    fid=SMARD_SERIES[name]
    stamps=get(f'{SMARD}/{fid}/DE/index_hour.json').json()['timestamps'][-2:]
    values={}
    for t in stamps:
        for ts,v in get(f'{SMARD}/{fid}/DE/{fid}_DE_hour_{t}.json').json()['series']:
            if v is not None:values[pd.Timestamp(ts,unit='ms',tz='UTC')]=v
    return pd.Series(values,dtype=float).sort_index()


def collect(day):
    captured=utcnow().isoformat()
    validate_issue(day,captured)
    bundle=ROOT/'data/paper'/f'{day}_{utcnow():%Y%m%dT%H%M%SZ}'
    bundle.mkdir(parents=True)
    raw=pd.read_parquet(ROOT/'data/hourly_raw.parquet').copy()
    for name in SMARD_SERIES:
        recent=recent_smard(name)
        raw=raw.reindex(raw.index.union(recent.index))
        raw.loc[recent.index,name]=recent
    start=(pd.Timestamp(day)-pd.Timedelta(days=8)).strftime('%Y-%m-%d')
    d=get(f'{EC}/public_power_forecast',country='de',production_type='load',forecast_type='day-ahead',start=start,
          end=(pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')).json()
    load=pd.Series(d['forecast_values'],index=pd.to_datetime(d['unix_seconds'],unit='s',utc=True),dtype=float).resample('1h').mean().dropna()
    raw=raw.reindex(raw.index.union(load.index));raw.loc[load.index,'load_fc']=load
    raw.to_parquet(bundle/'raw.parquet')
    # Timestamped raw public weather responses are retained with the collection.
    frames=[]
    for i,(lat,lon) in enumerate(POINTS):
        response=get('https://previous-runs-api.open-meteo.com/v1/forecast',latitude=lat,longitude=lon,
            hourly=','.join(v+'_previous_day2' for v in VARIABLES),models=','.join(MODELS),
            start_date=(pd.Timestamp(day)-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),end_date=day,timezone='GMT')
        (bundle/f'weather_{i}.json').write_text(response.text)
        df=pd.DataFrame(response.json()['hourly']);df.index=pd.to_datetime(df.pop('time'),utc=True)
        frames.append(df.add_prefix(f'p{i}_').rename(columns=lambda c:c.replace('_previous_day2','')))
    pd.concat(frames,axis=1).to_parquet(bundle/'weather.parquet')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
    durable_json(bundle/'manifest.json',dict(day=day,captured_at=captured,finished_at=utcnow().isoformat(),sha256=hashes,
        historical_raw_sha256=hashlib.sha256((ROOT/'data/hourly_raw.parquet').read_bytes()).hexdigest(),
        sources=[SMARD,EC,'https://open-meteo.com/en/docs/previous-runs-api']))
    print(bundle,flush=True)
    return bundle


def map_weather(panel,bundle,day):
    weather=pd.concat([pd.read_parquet(ROOT/'data/weather_prev2.parquet'),
        pd.read_parquet(ROOT/'data/weather_prev2_holdout.parquet'),pd.read_parquet(bundle/'weather.parquet')])
    weather=weather[~weather.index.duplicated(keep='last')].sort_index()
    local=weather.index.tz_convert(TZ)
    original=weather.loc[local.normalize()==pd.Timestamp(day,tz=TZ)]
    if len(original)!=24 or not np.isfinite(original.to_numpy()).all():
        raise ValueError('24 complete original weather hours required; no interpolation for live inputs')
    X=pd.DataFrame({c:_to_daily(weather[c]).reindex(panel.days).to_numpy().ravel() for c in weather})
    X['hour']=np.tile(np.arange(24),len(panel.days));X['doy']=np.repeat(panel.days.dayofyear,24)
    dates=np.repeat(panel.days.values,24);cut=pd.Timestamp(day).to_period('M').to_timestamp().to_datetime64()
    today=dates==pd.Timestamp(day).to_datetime64();have=X.notna().all(axis=1).to_numpy()
    if not have[today].all():raise ValueError('current weather inputs incomplete; skip paper issue')
    for c in ['wind_fc','solar_fc']:
        target=panel[c].ravel();train=have & (dates<cut) & np.isfinite(target)
        params=dict(n_estimators=400,learning_rate=.05,num_leaves=63,min_child_samples=30,subsample=.8,
                    subsample_freq=1,colsample_bytree=.7,verbose=-1,n_jobs=4)
        with ModelWorker('paper_weather',threads=4) as worker:
            worker.call('lgbm_fit',params=params,X=X[train],y=target[train])
            panel.x[c+'_pre'][-1]=np.maximum(worker.call('lgbm_predict',X=X[today]),0)


def predict(panel):
    """All model fitting/prediction runs in the existing fail-closed offline sandbox."""
    check_policy()
    t=len(panel.days)-1
    item=series_inputs(panel,t,85,neighbours=True,pre_gate=True)
    arrays=[item['target'],*item['past_covariates'].values(),*item['future_covariates'].values()]
    if not all(np.isfinite(v).all() for v in arrays):raise ValueError('missing Chronos context or current pre-gate covariates')
    with ModelWorker('chronos',threads=4) as worker:
        adapter=worker.work/'adapter';shutil.copytree(ADAPTER,adapter)
        levels=worker.call('chronos_zero',snapshot=str(adapter),device='mps')
        q=worker.call('chronos_predict',items=[item])[0]
    levels=list(np.round(levels,4));point=q[:,levels.index(.5)]
    df=feature_table(panel,pre_gate=True);cols=[c for c in df if c not in ['day_idx','y']]
    # Match monthly reference training; original post-gate outcome stream is never a selection set.
    month_start=panel.days[t].to_period('M').to_timestamp()
    cut=panel.idx(month_start);train=df[(df.day_idx>=7)&(df.day_idx<cut)].dropna(subset=cols+['y'])
    params=json.loads((ROOT/'runs/readiness/protocol.json').read_text())['lgbm']
    with ModelWorker('paper_lgbm',threads=4) as worker:
        worker.call('lgbm_fit',params=params,X=train[cols],y=train.y-train.p_d1_mean)
        te=df[df.day_idx==t];lgb=worker.call('lgbm_predict',X=te[cols])+te.p_d1_mean.to_numpy()
    X=np.stack([lear_features(panel,i,pre_gate=True) if i>=7 else np.full(247,np.nan) for i in range(t+1)])
    if not np.isfinite(X[t]).all():raise ValueError('missing LEAR current inputs')
    from epf.data import Panel
    causal=Panel(panel.days,{'price_de':panel['price_de'].copy()});causal.x['price_de'][t]=np.nan
    with ModelWorker('paper_lear',threads=1) as worker:
        lear=worker.call('lear',panel=causal,t=t,X=X,windows=WINDOWS)
    cal=json.loads((ROOT/'runs/readiness/calibration.json').read_text())
    adjustment=np.asarray(cal['adjustment'])
    forecasts=dict(pregate_lora=point.tolist(),pregate_lgbm=lgb.tolist(),pregate_lear=lear.tolist(),
        naive=panel['price_de'][t-(7 if panel.days[t].weekday() in (0,5,6) else 1)].tolist())
    return forecasts,dict(lower80=(q[:,levels.index(.1)]-adjustment).tolist(),
        upper80=(q[:,levels.index(.9)]+adjustment).tolist(),calibration=cal,
        adapter_sha256=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest())


def issue(bundle):
    policy=check_policy()
    meta=json.loads((bundle/'manifest.json').read_text());day=meta['day']
    if day<policy['start_delivery_day']:raise ValueError('day predates prospective paper stream')
    validate_issue(day,meta['captured_at'])
    destination=OUT/'issued'/f'{day}.json'
    if destination.exists():raise FileExistsError('paper forecast already issued; never replace it')
    for name,digest in meta['sha256'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:raise ValueError('source snapshot changed')
    raw=pd.read_parquet(bundle/'raw.parquet')
    if np.isfinite(_to_daily(raw.price_de).reindex([pd.Timestamp(day)]).to_numpy()).any():
        raise ValueError('target auction prices already in snapshot; refusing prospective claim')
    panel=panel_from_snapshot(raw,day)
    local=raw.index.tz_convert(TZ)
    current_load=raw.loc[local.normalize()==pd.Timestamp(day,tz=TZ),'load_fc']
    if len(current_load)!=24 or not np.isfinite(current_load).all():
        raise ValueError('all 24 original hourly load forecasts required; no interpolation for live inputs')
    dev=pd.read_parquet(ROOT/'data/pregate_fc.parquet');old=pd.read_parquet(ROOT/'data/pregate_fc_holdout.parquet')
    pg=pd.concat([dev,old]).drop_duplicates(['day','hour'],keep='last')
    for c in ['wind_fc_pre','solar_fc_pre']:panel.x[c]=pg.pivot(index='day',columns='hour',values=c).reindex(panel.days).to_numpy(float).copy()
    disable_controller_network();map_weather(panel,bundle,day)
    forecasts,details=predict(panel)
    details.update(source_bundle=str(bundle.relative_to(ROOT)),source_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256((ROOT/'runs/readiness/protocol.json').read_bytes()).hexdigest(),
                   policy_sha256=hashlib.sha256((OUT/'policy.json').read_bytes()).hexdigest(),
                   historical_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [ROOT/'data/weather_prev2.parquet',ROOT/'data/weather_prev2_holdout.parquet',ROOT/'data/pregate_fc.parquet',ROOT/'data/pregate_fc_holdout.parquet']})
    publish(destination,day,meta['captured_at'],forecasts,details)
    print('Issued',destination,flush=True)


def score():
    for path in sorted((OUT/'issued').glob('*.json')):
        record=json.loads(path.read_text());day=record['day'];dest=OUT/'scored'/path.name
        if dest.exists() or utcnow()<deadline(day)+timedelta(hours=2):continue
        prices=recent_smard('price_de')
        local=prices.index.tz_convert(TZ)
        original=prices.loc[local.normalize()==pd.Timestamp(day,tz=TZ)]
        if len(original)!=24 or not np.isfinite(original).all():continue
        actual=_to_daily(prices).reindex([pd.Timestamp(day)]).to_numpy(float)
        if not np.isfinite(actual).all():continue
        models={}
        for name,f in record['predictions'].items():
            models[name]=dict(mae=float(np.abs(actual-np.array(f)).mean()),economics=economics(actual,np.array([f])))
        lo=np.array(record['metadata']['lower80']);hi=np.array(record['metadata']['upper80'])
        durable_json(dest,dict(day=day,scored_at=utcnow().isoformat(),issue_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actual=actual[0].tolist(),models=models,coverage80=float(np.mean((actual[0]>=lo)&(actual[0]<=hi)))))
        print('Scored',day,flush=True)
    scored=[json.loads(p.read_text()) for p in sorted((OUT/'scored').glob('*.json'))]
    totals=dict(n_scored_days=len(scored),n_issued_days=len(list((OUT/'issued').glob('*.json'))),models={})
    if scored:
        for name in scored[0]['models']:
            totals['models'][name]=dict(mae=float(np.mean([d['models'][name]['mae'] for d in scored])),
                margin_eur=float(sum(d['models'][name]['economics']['margin_eur'] for d in scored)))
        totals['coverage80']=float(np.mean([d['coverage80'] for d in scored]))
    (OUT/'summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    print(json.dumps(totals),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['collect','issue','score','auto','rehearse'])
    ap.add_argument('--day');ap.add_argument('--bundle',type=Path);a=ap.parse_args()
    day=a.day or (utcnow().astimezone(ZoneInfo(TZ)).date()+timedelta(days=1)).isoformat()
    if a.command=='collect':collect(day)
    elif a.command=='issue':
        if a.bundle is None:ap.error('issue requires --bundle')
        issue(a.bundle)
    elif a.command=='score':score()
    elif a.command=='auto':
        score()
        policy=check_policy()
        if len(list((OUT/'scored').glob('*.json')))>=policy['target_scored_days']:
            print('Paper review threshold reached; stop scheduling and review without retuning.')
            return
        if not (OUT/'issued'/f'{day}.json').exists():
            try:
                issue(collect(day))
            except Exception as exc:
                durable_json(OUT/'skipped'/f'{day}_{utcnow():%Y%m%dT%H%M%SZ}.json',
                    dict(day=day,checked_at=utcnow().isoformat(),reason=str(exc),issued=False))
                raise
        score()
    else:
        disable_controller_network()
        p=with_pregate(load_panel());p.x['price_de']=p.x['price_de'].copy();p.x['price_de'][-1]=np.nan
        forecasts,meta=predict(p)
        durable_json(ROOT/'runs/readiness/paper_rehearsal.json',dict(kind='dev-rehearsal',day=str(p.days[-1].date()),predictions=forecasts,metadata=meta))
        print('Dev-only paper rehearsal passed; no prospective issue created.',flush=True)


if __name__=='__main__':
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
