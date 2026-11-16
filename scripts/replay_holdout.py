"""Audit existing forecasts, freeze forecast-only dispatch, then score in a separate invocation.

    python scripts/replay_holdout.py issue
    python scripts/replay_holdout.py score

Never trains a model, calls the holdout opener, or writes the study ledger.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from epf.data import ROOT, TZ
from epf.offline import ModelWorker, disable_controller_network
from epf.readiness import Battery, dispatch

OUT = ROOT / 'runs/replay'
PROTOCOL = OUT / 'protocol.json'
ARCHIVE = ROOT / 'runs/holdout_issued_audit.zip'
AUDIT = ROOT / 'runs/holdout_audit_manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing artifact differs: {path}; preserve and inspect')
        return
    import os
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def normal_day(day):
    begin = pd.Timestamp(day).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    return (end - begin).total_seconds() == 86400


def audited_forecasts(protocol):
    path = ROOT / f"runs/forecasts/{protocol['forecast_id']}.npz"
    with np.load(path, allow_pickle=False) as z:
        days, point = z['days'].astype(str), z['point'].copy()
    if len(set(days)) != len(days) or list(days) != sorted(days):
        raise ValueError('duplicate or unordered forecast dates')
    if point.shape != (len(days), 24) or not np.isfinite(point).all():
        raise ValueError('invalid saved forecasts')
    manifest = json.loads(AUDIT.read_text())['daily_issued_sha256']
    with zipfile.ZipFile(ARCHIVE) as archive:
        for day, forecast in zip(days, point):
            name = f"{protocol['archive_label']}_{day}.npy"
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != manifest[name]:
                raise ValueError(f'archived issuance hash mismatch: {name}')
            saved = np.load(io.BytesIO(data), allow_pickle=False)
            # Pre-gate issuance retained all quantiles; final point is their median.
            if saved.ndim == 2:
                with np.load(path, allow_pickle=False) as z:
                    saved = saved[:, int(np.argmin(abs(z['levels'] - .5)))]
            np.testing.assert_array_equal(saved, forecast)
    return days, point


def validate_dispatch(grid, wear, battery):
    grid, wear = np.asarray(grid), np.asarray(wear)
    if grid.shape != (24,) or wear.shape != (24,) or not np.isfinite([grid, wear]).all():
        raise ValueError('invalid dispatch')
    energy = np.where(grid > 0, grid * battery.eta, grid / battery.eta)
    soc = np.cumsum(energy)
    if (abs(grid) > battery.power_mw + 1e-9).any() or (soc < -1e-9).any() or (soc > battery.capacity_mwh + 1e-9).any():
        raise ValueError('dispatch violates power or capacity')
    if abs(soc[-1]) > 1e-9 or wear.sum() > battery.max_cycles * battery.capacity_mwh + 1e-9:
        raise ValueError('dispatch violates final SOC or cycles')
    np.testing.assert_allclose(wear, np.maximum(-energy, 0), atol=1e-9)
    return soc


def actual_day(raw, day):
    begin = pd.Timestamp(day).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    actual = raw.loc[(raw.index >= begin) & (raw.index < end)].sort_index()
    expected = pd.date_range(begin, end, freq='h', inclusive='left').tz_convert(raw.index.tz)
    if not actual.index.equals(expected) or not np.isfinite(actual).all() or len(actual) != 24:
        raise ValueError(f'incomplete/duplicate/nonhourly actual prices: {day}')
    return actual


def issue():
    def deny_outcomes(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(args[0]).resolve()
            if path.is_relative_to(ROOT / 'data') or path == ROOT / 'runs/ledger.jsonl':
                raise PermissionError('decision issuer cannot open outcomes or ledger')
    sys.addaudithook(deny_outcomes)
    protocol = json.loads(PROTOCOL.read_text())
    b = Battery(**protocol['battery'])
    days, forecasts = audited_forecasts(protocol)
    records, skipped = [], []
    with ModelWorker('replay_dispatch', threads=1) as worker:
        for day, forecast in zip(days, forecasts):
            if not protocol['start'] <= day <= protocol['end']:
                continue
            if not normal_day(day):
                skipped.append(dict(day=day, reason='DST: not 24 actual delivery hours'))
                continue
            grid, wear = worker.call('dispatch', forecast=forecast, assumptions=asdict(b))
            validate_dispatch(grid, wear, b)
            records.append(dict(day=day, forecast_eur_mwh=forecast.tolist(),
                                grid_mwh=grid.tolist(), discharged_mwh=wear.tolist()))
    expected = pd.date_range(protocol['start'], protocol['end']).strftime('%Y-%m-%d')
    missing = [d for d in expected if d not in set(days)]
    durable(OUT / 'decisions.json', dict(protocol_sha256=digest(PROTOCOL),
        forecast_sha256=digest(ROOT / f"runs/forecasts/{protocol['forecast_id']}.npz"),
        archive_sha256=digest(ARCHIVE), audit_sha256=digest(AUDIT),
        dispatch_code_sha256=digest(ROOT / 'src/epf/readiness.py'),
        worker_code_sha256=digest(ROOT / 'scripts/offline_worker.py'),
        missing_forecast_days=missing, skipped=skipped, records=records))
    print(f'Persisted {len(records)} forecast-only daily decisions; {len(missing)} missing, {len(skipped)} DST excluded.')


def score():
    protocol = json.loads(PROTOCOL.read_text())
    decisions = json.loads((OUT / 'decisions.json').read_text())
    for key, path in [('protocol', PROTOCOL), ('forecast', ROOT / f"runs/forecasts/{protocol['forecast_id']}.npz"),
                      ('archive', ARCHIVE), ('audit', AUDIT), ('dispatch_code', ROOT / 'src/epf/readiness.py'),
                      ('worker_code', ROOT / 'scripts/offline_worker.py')]:
        if decisions[key + '_sha256'] != digest(path):
            raise ValueError(f'{key} changed after decisions were frozen')
    b = Battery(**protocol['battery'])
    # Outcome data enters only here, after the separate issue invocation finished.
    raw_path = ROOT / 'data/hourly_raw.parquet'
    raw = pd.read_parquet(raw_path, columns=['price_de'])['price_de']
    scored, hours = [], []
    for row in decisions['records']:
        day = row['day']
        actual = actual_day(raw, day)
        y = actual.to_numpy()
        grid, wear, forecast = map(np.asarray, (row['grid_mwh'], row['discharged_mwh'], row['forecast_eur_mwh']))
        soc = validate_dispatch(grid, wear, b)
        gross = -y * grid
        costs = b.fee_slippage_eur_mwh * abs(grid) + b.wear_eur_mwh * wear
        fixed = protocol['annual_fixed_eur'] / 365.25
        oracle_grid, oracle_wear = dispatch(y, b)  # scorer-only hindsight upper bound
        oracle = float(-y @ oracle_grid - b.fee_slippage_eur_mwh * abs(oracle_grid).sum()
                       - b.wear_eur_mwh * oracle_wear.sum())
        margin = float((gross - costs).sum())
        if margin > oracle + 1e-8:
            raise ValueError('scored margin exceeds feasible oracle')
        scored.append(dict(day=day, mae=float(abs(y-forecast).mean()), gross_eur=float(gross.sum()),
            variable_cost_eur=float(costs.sum()), margin_eur=margin,
            fixed_cost_eur=fixed, scenario_net_eur=margin-fixed, oracle_margin_eur=oracle))
        for h in range(24):
            hours.append(dict(day=day, berlin_hour=h, timestamp=actual.index[h].isoformat(),
                action='buy' if grid[h] > 1e-9 else 'sell' if grid[h] < -1e-9 else 'hold',
                forecast_eur_mwh=float(forecast[h]), actual_eur_mwh=float(y[h]),
                grid_mwh=float(grid[h]), soc_mwh=float(soc[h]),
                gross_eur=float(gross[h]), variable_cost_eur=float(costs[h]),
                margin_eur=float(gross[h]-costs[h])))
    totals = {k: float(sum(r[k] for r in scored)) for k in scored[0] if k not in ('day', 'mae')}
    calendar_days = len(pd.date_range(protocol['start'], protocol['end']))
    unscored_fixed = protocol['annual_fixed_eur'] * (calendar_days-len(scored)) / 365.25
    totals['fixed_cost_eur'] += unscored_fixed
    totals['scenario_net_eur'] -= unscored_fixed
    summary = dict(n_days=len(scored), window=[scored[0]['day'], scored[-1]['day']],
        calendar_days=calendar_days, unscored_fixed_cost_eur=unscored_fixed,
        fixed_cost_accounting='all calendar days, including four skipped days assumed idle',
        mae=float(np.mean([r['mae'] for r in scored])), totals=totals,
        profitable_days=sum(r['margin_eur'] > 0 for r in scored),
        losing_days=sum(r['margin_eur'] < 0 for r in scored),
        decisions_sha256=digest(OUT / 'decisions.json'), actual_source_sha256=digest(raw_path),
        missing_forecast_days=decisions['missing_forecast_days'], skipped=decisions['skipped'],
        limitations=protocol['limits'], days=scored)
    durable(OUT / 'summary.json', summary)
    durable(OUT / 'hourly.json', hours)
    for filename, records in [('daily.csv', scored), ('hourly.csv', hours)]:
        text = pd.DataFrame(records).to_csv(index=False)
        path = OUT / filename
        if path.exists() and path.read_text() != text:
            raise ValueError(f'existing report differs: {path}')
        path.write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('days','limitations')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['issue', 'score'])
    args = parser.parse_args()
    disable_controller_network()
    issue() if args.mode == 'issue' else score()
