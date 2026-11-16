"""Immutable paper issues and independent scoring. No broker or order interfaces."""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import Panel, PRICE_ZONES, TZ, _to_daily


def utcnow():
    return datetime.now(timezone.utc)


def deadline(day):
    d = datetime.fromisoformat(str(day)).replace(tzinfo=ZoneInfo(TZ))
    return d - timedelta(days=1) + timedelta(hours=12)


def validate_issue(day, captured, now=None):
    now = now or utcnow()
    captured = datetime.fromisoformat(captured)
    if captured.tzinfo is None or now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if not (timedelta(0) <= now-captured <= timedelta(hours=2)) or now >= deadline(day):
        raise ValueError("stale input snapshot or auction deadline passed")
    if now.astimezone(ZoneInfo(TZ)).date() != deadline(day).date():
        raise ValueError("paper issues must be on the day before delivery")
    start = pd.Timestamp(day, tz=TZ)
    if ((start+pd.DateOffset(days=1)).tz_convert('UTC')-start.tz_convert('UTC')).total_seconds()!=86400:
        raise ValueError("paper dispatcher currently skips 23/25-hour DST days")


def durable_json(path, record):
    """Exclusive create: existing predictions/scores are never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    # Write a complete temporary file, then exclusively link it into the published name.
    import tempfile
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.pending_')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(name,path)
    finally:
        os.unlink(name)
    return hashlib.sha256(payload).hexdigest()


def publish(path, day, captured, predictions, metadata, now=None):
    validate_issue(day,captured,now)
    if not predictions:
        raise ValueError("no predictions to publish")
    for model, values in predictions.items():
        if np.asarray(values).shape != (24,) or not np.isfinite(values).all():
            raise ValueError(f"invalid forecast: {model}")
    if 'lower80' in metadata or 'upper80' in metadata:
        lower,upper=np.asarray(metadata['lower80']),np.asarray(metadata['upper80'])
        if lower.shape!=(24,) or upper.shape!=(24,) or not np.isfinite([lower,upper]).all() or (lower>upper).any():
            raise ValueError('invalid uncertainty interval')
    record = dict(day=day, issued_at=(now or utcnow()).isoformat(),captured_at=captured,
                  deadline=deadline(day).isoformat(),predictions=predictions,metadata=metadata,
                  kind="prospective-paper", trades=False)
    return durable_json(path,record)


def panel_from_snapshot(raw, day):
    """Controller view; mask unreleased prices/actuals before any worker request."""
    raw=raw.copy()
    raw['wind_fc']=raw[['wind_on_fc','wind_off_fc']].sum(axis=1,min_count=2)
    raw['ren_fc']=raw['wind_fc']+raw['solar_fc']
    raw['resid_fc']=raw['load_fc']-raw['ren_fc']
    raw['ren_act']=raw[['wind_on_act','wind_off_act','solar_act']].sum(axis=1,min_count=3)
    raw['resid_act']=raw['load_act']-raw['ren_act']
    names=[f'price_{z}' for z in PRICE_ZONES]+['load_fc','wind_fc','solar_fc','ren_fc','resid_fc','load_act','ren_act','resid_act']
    days=pd.date_range('2018-10-01',day)
    p=Panel(days,{c:_to_daily(raw[c]).reindex(days).to_numpy(float).copy() for c in names})
    for c in p.x:
        if c.startswith('price_'):p.x[c][-1]=np.nan
        elif c.endswith('_act'):p.x[c][-2:]=np.nan
        elif c!='load_fc':p.x[c][-1]=np.nan
    return p
