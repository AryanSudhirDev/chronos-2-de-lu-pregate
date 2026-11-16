"""Pre-gate German wind/solar forecasts from weather-model runs made 48 h before each valid hour
(Open-Meteo Previous Runs API), so every hour of day D comes from a run issued before the D-1 12:00 gate.
Weather is mapped to the TSO forecast level by LightGBM, refit monthly on earlier days only.
Writes data/pregate_fc.parquet (dev days only).

    python scripts/make_pregate_fc.py
    python scripts/make_pregate_fc.py --fetch_only                 # holdout-period weather only; reads no sealed data
    EPF_OPEN_HOLDOUT=1 python scripts/make_pregate_fc.py --holdout --offline  # opening step
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from epf.data import HOLDOUT_START, ROOT, _to_daily, load_panel
from epf.offline import ModelWorker, disable_controller_network

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START, END, FIRST_FC = "2024-03-12", "2025-10-31", pd.Timestamp("2024-09-01")  # ECMWF wind archive starts 2024-03-10
assert pd.Timestamp(END) < HOLDOUT_START
END_HOLDOUT = "2026-11-09"  # last delivery day in data/hourly_raw.parquet
MODELS = ["ecmwf_ifs025", "icon_seamless"]
VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]
POINTS = [(54.6, 8.8), (53.5, 8.0), (52.5, 7.5), (53.5, 13.0), (52.3, 12.5), (51.8, 11.5), (51.0, 9.5), (50.0, 7.5),
          (49.0, 10.5), (48.3, 11.5), (48.5, 9.0), (51.0, 13.5), (54.3, 6.5), (54.8, 14.0)]  # onshore belt, south, North Sea, Baltic


ap = argparse.ArgumentParser()
ap.add_argument("--holdout", action="store_true")
ap.add_argument("--fetch_only", action="store_true")
ap.add_argument("--offline", action="store_true", help="OS-isolated monthly weather models; required for holdout")
ap.add_argument("--rehearse", action="store_true", help="offline dev October 2025 check; preserve cached dev forecasts")
a = ap.parse_args()
assert not a.rehearse or (a.offline and not a.holdout and not a.fetch_only)
if a.holdout and not a.offline:
    raise SystemExit("holdout weather models require --offline")
if a.offline:
    disable_controller_network()


def fetch(name=None, start=START, end=END):
    out = ROOT / "data" / (name or "weather_prev2.parquet")
    if out.exists():
        return pd.read_parquet(out)
    if a.offline:
        raise RuntimeError(f"offline weather cache missing: {out}; never fetch during evaluation")
    frames = []
    for i, (lat, lon) in enumerate(POINTS):
        r = requests.get(URL, timeout=180, params={
            "latitude": lat, "longitude": lon, "hourly": ",".join(f"{v}_previous_day2" for v in VARS),
            "models": ",".join(MODELS), "start_date": start, "end_date": end, "timezone": "GMT"})
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        frames.append(df.add_prefix(f"p{i}_").rename(columns=lambda c: c.replace("_previous_day2", "")))
        print(f"point {i}: {df.notna().mean().min():.3f} filled", flush=True)
        time.sleep(8)  # free tier is rate-limited
    w = pd.concat(frames, axis=1)
    w.to_parquet(out)
    return w


w = fetch()
if a.holdout or a.fetch_only:
    w = pd.concat([w, fetch("weather_prev2_holdout.parquet", str(HOLDOUT_START.date()), END_HOLDOUT)])
if a.fetch_only:
    raise SystemExit
panel = load_panel(allow_holdout=a.holdout)
if a.holdout:
    FIRST_FC = HOLDOUT_START  # earlier months are the dev forecasts in pregate_fc.parquet
elif a.rehearse:
    FIRST_FC = pd.Timestamp("2025-10-01")
n, days = len(panel.days), panel.days
X = pd.DataFrame({c: _to_daily(w[c]).reindex(days).to_numpy().ravel() for c in w})
X["hour"] = np.tile(np.arange(24), n)
X["doy"] = np.repeat(days.dayofyear, 24)
day = np.repeat(days.values, 24)
have = X.drop(columns=["hour", "doy"]).notna().all(axis=1).to_numpy()
pre = {}
for tgt in ["wind_fc", "solar_fc"]:
    y = panel[tgt].ravel()
    out = np.full(n * 24, np.nan)
    for m in pd.period_range(FIRST_FC, days[-1], freq="M"):
        first = m.to_timestamp()
        tr = have & (day < first.to_datetime64()) & np.isfinite(y)
        te = have & (pd.DatetimeIndex(day).to_period("M") == m)
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=6)
        if a.offline:
            with ModelWorker("weather", threads=6) as worker:
                worker.call("lgbm_fit", params=params, X=X[tr], y=y[tr])
                for d in days[days.to_period("M") == m]:
                    today = te & (day == d.to_datetime64())
                    if today.any():
                        out[today] = np.maximum(worker.call("lgbm_predict", X=X[today]), 0)
                        issued = ROOT / "runs/ckpt/weather_issued" / ("holdout" if a.holdout else "dev")
                        issued.mkdir(parents=True, exist_ok=True)
                        path = issued / f"{tgt}_{d.date()}.npy"
                        if path.exists():
                            np.testing.assert_array_equal(np.load(path), out[today])
                        else:
                            with path.open("xb") as f:
                                np.save(f, out[today])
                                f.flush()
                                os.fsync(f.fileno())
                    (ROOT / "runs/HEARTBEAT").touch()
        else:
            model = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
            out[te] = np.maximum(model.predict(X[te]), 0)
    pre[tgt] = out.reshape(n, 24)

ok = np.isfinite(pre["wind_fc"]).all(1) & np.isfinite(pre["solar_fc"]).all(1) & np.isfinite(panel["ren_act"]).all(1)
print(f"{ok.sum()} days, {days[ok][0].date()}..{days[ok][-1].date()}, days with gaps after first forecast: "
      f"{int((days >= FIRST_FC).sum() - ok.sum())}")
for tgt, p in pre.items():
    print(f"{tgt}: pre-gate MAE vs TSO forecast {np.abs(p[ok] - panel[tgt][ok]).mean():.0f} MW, mean level {panel[tgt][ok].mean():.0f} MW")
act = panel["ren_act"][ok]
print(f"wind+solar MAE vs actual: pre-gate {np.abs(pre['wind_fc'][ok] + pre['solar_fc'][ok] - act).mean():.0f} MW, "
      f"TSO {np.abs(panel['ren_fc'][ok] - act).mean():.0f} MW")
result = pd.DataFrame({"day": day, "hour": X["hour"], "wind_fc_pre": pre["wind_fc"].ravel(), "solar_fc_pre": pre["solar_fc"].ravel()})
if a.rehearse:
    reference = pd.read_parquet(ROOT / "data/pregate_fc.parquet")
    check = result.merge(reference, on=["day", "hour"], suffixes=("", "_reference"))
    check = check[check.day >= FIRST_FC]
    for col in ["wind_fc_pre", "solar_fc_pre"]:
        np.testing.assert_allclose(check[col], check[col + "_reference"], rtol=0, atol=1e-6)
    result.to_parquet(ROOT / "runs/ckpt/weather_rehearsal.parquet")
    print("offline weather rehearsal matches existing dev forecasts", flush=True)
else:
    result.to_parquet(ROOT / "data" / ("pregate_fc_holdout.parquet" if a.holdout else "pregate_fc.parquet"))
