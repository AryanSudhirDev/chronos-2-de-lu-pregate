"""Fetch German (DE-LU) day-ahead market data from free, keyless sources into data/raw/.

SMARD (Bundesnetzagentur, CC BY 4.0): prices for DE-LU and neighbours, actual load and
generation, and the TSOs' day-ahead wind/solar forecasts, hourly.
Energy-Charts (Fraunhofer ISE, CC BY 4.0): the TSOs' day-ahead load forecast, 15-minute.

    python scripts/fetch_data.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SMARD = "https://www.smard.de/app/chart_data"
EC = "https://api.energy-charts.info"
START = pd.Timestamp("2018-10-01", tz="Europe/Berlin")  # DE-LU bidding zone starts here

SMARD_SERIES = {
    "price_de": 4169,
    "price_be": 4996,
    "price_fr": 254,
    "price_nl": 256,
    "price_pl": 257,
    "price_dk1": 252,
    "price_at": 4170,
    "price_ch": 259,
    "price_cz": 261,
    "load_act": 410,
    "wind_on_act": 4067,
    "wind_off_act": 1225,
    "solar_act": 4068,
    "wind_on_fc": 123,
    "wind_off_fc": 3791,
    "solar_fc": 125,
}

session = requests.Session()


def smard_series(name: str, fid: int, region: str = "DE") -> pd.Series:
    idx = session.get(f"{SMARD}/{fid}/{region}/index_hour.json", timeout=30).json()["timestamps"]
    idx = [t for t in idx if t >= START.timestamp() * 1000 - 7 * 86400_000]
    cache = RAW / "smard" / name
    cache.mkdir(parents=True, exist_ok=True)
    newest = idx[-2:]  # the last weeks are still filling in; always refetch them

    def week(t):
        p = cache / f"{t}.json"
        if p.exists() and t not in newest:
            return json.loads(p.read_text())["series"]
        r = session.get(f"{SMARD}/{fid}/{region}/{fid}_{region}_hour_{t}.json", timeout=30)
        r.raise_for_status()
        p.write_text(r.text)
        return r.json()["series"]

    with ThreadPoolExecutor(8) as ex:
        rows = [row for chunk in ex.map(week, idx) for row in chunk]
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): v for t, v in rows if v is not None},
                  name=name, dtype="float64")
    return s.sort_index()


def energy_charts_load_forecast() -> pd.Series:
    parts = []
    for y in range(START.year, date.today().year + 1):
        p = RAW / "energy_charts" / f"load_fc_{y}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or y >= date.today().year:
            r = session.get(f"{EC}/public_power_forecast", timeout=120, params={
                "country": "de", "production_type": "load", "forecast_type": "day-ahead",
                "start": f"{y}-01-01", "end": f"{y + 1}-01-01"})
            r.raise_for_status()
            p.write_text(r.text)
        d = json.loads(p.read_text())
        parts.append(pd.Series(d["forecast_values"],
                               index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)))
    s = pd.concat(parts).dropna()
    s = s[~s.index.duplicated()].sort_index()
    return s.resample("1h").mean().rename("load_fc")  # MW averaged over the hour = MWh per hour


def main():
    out = {}
    for name, fid in SMARD_SERIES.items():
        out[name] = smard_series(name, fid)
        s = out[name]
        print(f"{name:13s} {len(s):6d} rows  {s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d %H:%M}", flush=True)
    out["load_fc"] = energy_charts_load_forecast()
    s = out["load_fc"]
    print(f"{'load_fc':13s} {len(s):6d} rows  {s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d %H:%M}")
    df = pd.DataFrame(out)
    df = df[df.index >= START.tz_convert("UTC")]
    df.index.name = "ts"
    df.to_parquet(ROOT / "data" / "hourly_raw.parquet")
    print("saved data/hourly_raw.parquet", df.shape)


if __name__ == "__main__":
    main()
