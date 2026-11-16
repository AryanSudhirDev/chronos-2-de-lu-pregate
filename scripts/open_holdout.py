"""One-time evaluation of the pre-declared candidates on the sealed holdout (see STATUS.md).

Refuses once the ledger has any holdout row. Every forecast is computed and checked before the first
row is recorded, so a crash leaves nothing recorded or printed and the script can be rerun. Stage 1 output
stays in runs/ckpt/holdout.npz after a stage 2 crash; rerun the same command with --lgbm_stage to skip the fit.

Candidates (at most 3): --chronos, --blend, --pre_gate. References, recorded with role=reference and never
picked from: --stack and --lgbm (blend members), --lear, --zero_shot.

    EPF_OPEN_HOLDOUT=1 python scripts/make_pregate_fc.py --holdout --offline
    EPF_OPEN_HOLDOUT=1 python scripts/open_holdout.py --chronos <id> --stack <id> --lgbm <id> --blend <id> \
        --pre_gate <id> --zero_shot <id> --lear <id> --offline
    python scripts/open_holdout.py ... --offline --rehearse --device cpu --steps 5
        # rehearsal: pretends the holdout starts 2025-09-01, reads only dev data, records nothing
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from epf.blend import online
from epf.chronos_io import series_inputs, training_series
from epf.data import HOLDOUT_START, ROOT, Panel, eval_days, load_panel, with_pregate
from epf.features import feature_table
from epf.ledger import FC, read_ledger, record
from epf.metrics import mae
from epf.offline import ModelWorker, disable_controller_network

ap = argparse.ArgumentParser()
ap.add_argument("--chronos", required=True, help="ledger id of the fine-tuned Chronos-2 run")
ap.add_argument("--stack", default="", help="ledger id of a LightGBM stack on --chronos")
ap.add_argument("--lgbm", default="", help="ledger id of the LightGBM benchmark")
ap.add_argument("--blend", default="", help="ledger id of an online blend of --chronos, --stack, --lgbm")
ap.add_argument("--pre_gate", default="", help="ledger id of the pre-gate run: same fit as --chronos, forecast with pre-gate wind/solar")
ap.add_argument("--zero_shot", default="", help="ledger id of a zero-shot Chronos-2 run")
ap.add_argument("--lear", default="", help="ledger id of the LEAR benchmark")
ap.add_argument("--rehearse", action="store_true")
ap.add_argument("--steps", type=int, default=0, help="rehearsal only: override training steps")
ap.add_argument("--device", default="")
ap.add_argument("--lgbm_stage", action="store_true", help="internal: second process, see below")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--offline", action="store_true", help="required: use OS-isolated numeric model workers")
a = ap.parse_args()
assert a.rehearse or not a.steps, "--steps is for rehearsals only"
if not a.offline:
    raise SystemExit("holdout opening requires --offline; rehearse the isolated runner on dev first")
disable_controller_network()

ledger = read_ledger()
if not a.rehearse and (ledger.split == "holdout").any():
    raise SystemExit("holdout already opened: the ledger has holdout rows")


def dev_row(i):
    r = ledger[(ledger.id == i) & (ledger.split == "dev")]
    assert len(r), f"{i} is not a dev run in the ledger"
    return r.iloc[-1]


cut = pd.Timestamp("2025-09-01") if a.rehearse else HOLDOUT_START
panel = load_panel(allow_holdout=not a.rehearse)
days = eval_days(panel, start=cut)
complete = np.all([np.isfinite(panel[k]).all(1) for k in ("price_de", "load_fc", "wind_fc", "solar_fc")], 0)
kept = complete[panel.days >= days[0]]
print(f"{len(days)} days from {days[0].date()}, {int((~kept).sum())} dropped for missing price or TSO forecasts", flush=True)
days = days[kept]
chronos = dev_row(a.chronos)
c = chronos.config
assert not c.get("gate_strict"), "gate-strict candidates are not supported here"

# Same protocol as one walk-forward fold of run_chronos_finetune.py, with the cut as fold start.
kw = dict(covs=True, neighbours=c["neighbours"], calendar=c["calendar"])
t0 = panel.idx(days[0])
v0 = t0 - c["val_days"]
train = [training_series(panel, end=v0, **kw)]
val = [training_series(panel, end=d + 1, start=d + 1 - c["context_days"], **kw) for d in range(v0, t0)]
lora = {"r": c["lora_r"], "lora_alpha": 2 * c["lora_r"],
        "target_modules": ["q", "k", "v", "o", "output_patch_embedding.output_layer"]} if c["mode"] == "lora" else None
out_dir = ROOT / "runs" / "ckpt" / ("rehearsal" if a.rehearse else "holdout")
tmp = out_dir.with_suffix(".npz")
if tmp.exists() and not a.rehearse and not a.lgbm_stage:
    raise SystemExit("holdout checkpoint exists: inspect it and resume with --lgbm_stage; do not launch another fit")
manifest = json.dumps({"args": {k: v for k, v in vars(a).items() if k != "lgbm_stage"},
                       "cut": str(cut), "config": c}, sort_keys=True)
audit = out_dir.parent / (out_dir.name + "_issued_" + hashlib.sha256(manifest.encode()).hexdigest()[:12])
audit.mkdir(parents=True, exist_ok=True)


def persist(label, day, forecast):
    """Durably issue each forecast before the next request can reveal its realised target."""
    path = audit / f"{label}_{pd.Timestamp(day).date()}.npy"
    forecast = np.asarray(forecast)
    if path.exists():
        if not np.array_equal(np.load(path), forecast):
            raise RuntimeError(f"issued forecast changed on resume: {path}")
        return
    with path.open("xb") as f:
        np.save(f, forecast)
        f.flush()
        os.fsync(f.fileno())


def isolated_predict(worker, ds, label, **options):
    # Only one delivery day's inputs cross the boundary. No target for that day is sent.
    out = []
    for day in ds:
        item = series_inputs(panel, panel.idx(day), c["context_days"], **options)
        forecast = worker.call("chronos_predict", items=[item])[0]
        persist(label, day, forecast)
        out.append(forecast)
        (ROOT / "runs/HEARTBEAT").touch()
    quantiles = np.stack(out)
    return quantiles[..., int(np.argmin(np.abs(levels - 0.5)))], quantiles, levels


if not a.lgbm_stage:
    # torch and lightgbm load different libomp copies and segfault in one process on macOS
    device = a.device or "mps"
    worker = ModelWorker("chronos", threads=a.threads)
    levels = worker.call("chronos_fit", train=train, snapshot=str(worker.snapshot), device=device,
        threads=a.threads, prediction_length=24, validation_inputs=val, finetune_mode=c["mode"], lora_config=lora,
        context_length=c["context_days"] * 24, learning_rate=c["lr"], num_steps=a.steps or c["steps"],
        batch_size=c["batch"], min_past=7 * 24)
    point, q, levels = isolated_predict(worker, days, "lora", **kw)
    saved = dict(point=point, quantiles=q, levels=levels, manifest=manifest)
    if a.pre_gate:  # the dev pre-gate run trains on TSO forecasts too, so the same fit serves both
        p = dev_row(a.pre_gate).config
        assert p["pre_gate"] and all(p[k] == c[k] for k in ("mode", "lr", "steps", "batch", "context_days", "val_days", "lora_r", "calendar", "neighbours"))
        with_pregate(panel, None if a.rehearse else ROOT / "data" / "pregate_fc_holdout.parquet")
        ok = np.isfinite(panel["wind_fc_pre"]).all(1) & np.isfinite(panel["solar_fc_pre"]).all(1)
        pg_days = days[ok[[panel.idx(d) for d in days]]]
        print(f"pre-gate forecasts on {len(pg_days)} of {len(days)} days", flush=True)
        saved["pg_point"], saved["pg_quantiles"], _ = isolated_predict(worker, pg_days, "pregate", pre_gate=True, **kw)
        saved["pg_days"] = pg_days.values.astype("datetime64[D]").astype(str)
    worker.close()  # retain fit checkpoints, including after a later-stage failure
    if a.zero_shot:
        z = dev_row(a.zero_shot).config
        assert z["covs"] and not z.get("gate_strict") and not z.get("pre_gate")
        assert (z["context_days"], z["neighbours"], z["calendar"]) == (c["context_days"], c["neighbours"], c["calendar"])
        with ModelWorker("chronos", threads=a.threads) as base:
            base_levels = base.call("chronos_zero", snapshot=str(base.snapshot), device=device)
            assert np.array_equal(base_levels, levels)
            saved["zs_point"], saved["zs_quantiles"], _ = isolated_predict(base, days, "zs", **kw)
    np.savez(tmp, **saved)
    subprocess.run([sys.executable, "-W", "ignore", *sys.argv, "--lgbm_stage"], check=True)
    tmp.unlink()
    raise SystemExit

saved = dict(np.load(tmp))
assert saved["manifest"].item() == manifest, "checkpoint belongs to different candidates/settings"
point, q, levels = saved["point"], saved["quantiles"], saved["levels"]


def monthly_lgbm(cfg, base=False):
    """Refit monthly on all earlier days, as run_lgbm.py does; base=True stacks on the Chronos forecasts."""
    assert not cfg.get("gate_strict") and not cfg.get("halflife") and cfg["window"] == 0
    df = feature_table(panel)
    anchor = "p_d1_mean"
    if base:
        z = np.load(FC / f"{a.chronos}.npz")
        keep = pd.DatetimeIndex(z["days"]) < cut  # only forecasts made before the cut train the stack
        lv = np.round(levels, 2).tolist()
        assert z["levels"].round(2).tolist() == lv
        idx = [panel.idx(d) for d in z["days"][keep]] + [panel.idx(d) for d in days]
        p50, qq = np.concatenate([z["point"][keep], point]), np.concatenate([z["quantiles"][keep], q])
        b = pd.DataFrame({"day_idx": np.repeat(idx, 24), "hour": np.tile(np.arange(24), len(idx)), "cf_p50": p50.ravel(),
                          "cf_q10": qq[..., lv.index(0.1)].ravel(), "cf_q90": qq[..., lv.index(0.9)].ravel()})
        b["cf_spread"] = b.cf_q90 - b.cf_q10
        b["cf_day_mean"] = b.groupby("day_idx").cf_p50.transform("mean")
        df = df.merge(b, on=["day_idx", "hour"])
        assert len(df) == len(idx) * 24, "base forecasts did not align with the feature table"
        anchor = "cf_p50"
    feats = [f for f in df.columns if f not in ("y", "day_idx")]
    assert len(feats) == cfg["n_features"], "feature set differs from the dev run"
    assert cfg["target"] == f"y - {anchor}"
    df["target"] = df["y"] - df[anchor]
    params = dict(objective=cfg["objective"], n_estimators=cfg["n_estimators"], learning_rate=cfg["lr"],
                  num_leaves=cfg["leaves"], min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=a.threads)
    preds = np.zeros((len(days), 24))
    months = pd.Series(days).dt.to_period("M")
    with ModelWorker("lgbm", threads=a.threads) as worker:
        for m in months.unique():
            sel = np.where(months == m)[0]
            first = panel.idx(days[sel[0]])
            tr = df[(df.day_idx >= 7) & (df.day_idx < first)].dropna(subset=["target", "p_d7"])
            worker.call("lgbm_fit", params=params, X=tr[feats], y=tr["target"])
            for s in sel:
                te = df[df.day_idx == panel.idx(days[s])].sort_values("hour")
                preds[s] = worker.call("lgbm_predict", X=te[feats]) + te[anchor].to_numpy()
                persist("stack" if base else "lgbm", days[s], preds[s])
                (ROOT / "runs/HEARTBEAT").touch()
    return preds


def prices(ds):
    return np.array([panel["price_de"][panel.idx(d)] for d in ds]).reshape(-1, 24)


results = [("candidate", chronos["name"], a.chronos, days, point, q, levels)]
members = {a.chronos: point}
if a.stack:
    r = dev_row(a.stack)
    assert r.config["base"] == a.chronos, "stack was built on a different Chronos run"
    members[a.stack] = monthly_lgbm(r.config, base=True)
    results.append(("reference", r["name"], a.stack, days, members[a.stack], None, None))
if a.lgbm:
    r = dev_row(a.lgbm)
    members[a.lgbm] = monthly_lgbm(r.config)
    results.append(("reference", r["name"], a.lgbm, days, members[a.lgbm], None, None))
if a.blend:
    r = dev_row(a.blend)
    bk = {k: r.config[k] for k in ("halflife", "k", "per_hour", "loss")}
    assert set(r.config["ids"]) <= set(members), "a blend member was not computed: pass --stack and --lgbm"
    hist = [np.load(FC / f"{i}.npz") for i in r.config["ids"]]
    past = np.array(sorted(set.intersection(*(set(z["days"]) for z in hist))))
    past = past[pd.DatetimeIndex(past) < cut]
    F0, y0 = np.stack([z["point"][np.isin(z["days"], past)] for z in hist]), prices(past)
    dev = np.load(FC / f"{a.blend}.npz")
    assert np.allclose(online(F0, y0, **bk), dev["point"][np.isin(dev["days"], past)]), "blend replay differs from the dev run"
    F = np.concatenate([F0, np.stack([members[i] for i in r.config["ids"]])], 1)
    blended = []
    with ModelWorker("blend", threads=1) as worker:
        for j, day in enumerate(days):
            # The placeholder target is arbitrary; today's outcome is not sent.
            history = np.concatenate([y0, prices(days[:j]), np.zeros((1, 24))])
            f = worker.call("blend", F=F[:, :len(past) + j + 1], y=history, config=bk)
            persist("blend", day, f)
            blended.append(f)
            (ROOT / "runs/HEARTBEAT").touch()
    results.append(("candidate", r["name"], a.blend, days, np.array(blended), None, None))
if a.pre_gate:
    results.append(("candidate", dev_row(a.pre_gate)["name"], a.pre_gate, pd.DatetimeIndex(saved["pg_days"]),
                    saved["pg_point"], saved["pg_quantiles"], levels))
if a.zero_shot:
    results.append(("reference", dev_row(a.zero_shot)["name"], a.zero_shot, days, saved["zs_point"], saved["zs_quantiles"], levels))
if a.lear:
    from epf.lear import lear_features
    cfg = dev_row(a.lear).config
    X = np.stack([lear_features(panel, i) if i >= 7 else np.full(247, np.nan) for i in range(len(panel.days))])
    ld, lf = [], []
    with ModelWorker("lear", threads=1) as worker:
        for d in days:
            t = panel.idx(d)
            if not np.isfinite(X[t]).all():
                continue
            lo = max(0, t - max(cfg["windows"]) - 7)
            history = panel["price_de"][lo:t + 1].copy()
            history[-1] = np.nan  # current target is never released to the worker
            causal = Panel(panel.days[lo:t + 1], {"price_de": history})
            lf.append(worker.call("lear", panel=causal, t=t - lo, X=X[lo:t + 1], windows=cfg["windows"]))
            persist("lear", d, lf[-1])
            ld.append(d)
            (ROOT / "runs/HEARTBEAT").touch()
    results.append(("reference", dev_row(a.lear)["name"], a.lear, pd.DatetimeIndex(ld), np.array(lf), None, None))
assert sum(r[0] == "candidate" for r in results) <= 3

for role, name, dev_id, ds, f, qf, lf in results:  # all checks before any row is recorded
    assert f.shape == prices(ds).shape and np.isfinite(f).all(), f"bad forecast from {name}"
for role, name, dev_id, ds, f, qf, lf in results:
    if a.rehearse:
        print(f"[rehearsal] {role} {name}: MAE {mae(prices(ds), f):.2f} on {ds[0].date()}..{ds[-1].date()} ({len(ds)} days)", flush=True)
    else:
        record(name, {"role": role, "dev_id": dev_id, "chronos": a.chronos, "fit_end": str(panel.days[v0 - 1].date()),
                      "isolation": "macos-sandbox-offline-v1", "audit": str(audit.relative_to(ROOT))}, ds, f, qf, lf,
               panel=panel, split="holdout")
