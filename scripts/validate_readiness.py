"""New dev-only readiness study. Never opens or appends to the original holdout ledger."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epf.data import ROOT, Panel, load_panel, with_pregate
from epf.features import feature_table
from epf.lear import WINDOWS, lear_features
from epf.metrics import mae, dm_test
from epf.offline import ModelWorker, disable_controller_network
from epf.readiness import interval_adjustment, economics, Battery
from epf.ledger import record, LEDGER

OUT = ROOT / "runs/readiness"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    disable_controller_network()
    panel = with_pregate(load_panel())
    z = np.load(ROOT / "runs/forecasts/864e91aa35.npz")
    ds = pd.DatetimeIndex(z["days"])
    q = z["quantiles"]; levels = list(np.round(z["levels"], 4))
    lo, hi = q[..., levels.index(.1)], q[..., levels.index(.9)]
    y = np.array([panel["price_de"][panel.idx(d)] for d in ds])
    cal = ds <= protocol["calibration_end"]
    adjustment = interval_adjustment(y[cal], lo[cal], hi[cal])
    (OUT / "calibration.json").write_text(json.dumps(dict(alpha=.2, end=protocol["calibration_end"],
        n_days=int(cal.sum()), adjustment=adjustment.tolist(), source="864e91aa35"), indent=2)+"\n")
    days = panel.days[(panel.days >= protocol["evaluate_start"]) & (panel.days <= protocol["evaluate_end"])]
    features = feature_table(panel, pre_gate=True)
    cols = [c for c in features if c not in ("day_idx", "y")]
    features["target"] = features.y - features.p_d1_mean
    ids = np.array([panel.idx(d) for d in days])
    # Both baselines get the same pre-gate current wind/solar, plus legitimately available lags.
    lgb_path = OUT / "lgbm_pregate.npz"
    if not lgb_path.exists():
        predictions = []
        with ModelWorker("readiness_lgbm", threads=4) as worker:
            month = None
            for d, t in zip(days, ids):
                if d.to_period("M") != month:
                    month = d.to_period("M")
                    tr = features[(features.day_idx >= 7) & (features.day_idx < t)].dropna(subset=cols+["target"])
                    worker.call("lgbm_fit", params=protocol["lgbm"], X=tr[cols], y=tr.target)
                te = features[features.day_idx == t]
                predictions.append(worker.call("lgbm_predict", X=te[cols]) + te.p_d1_mean.to_numpy())
                (ROOT / "runs/HEARTBEAT").touch()
        np.savez(lgb_path, days=days.strftime("%Y-%m-%d"), point=predictions)
    learner = OUT / "lear_pregate.npz"
    if not learner.exists():
        X = np.stack([lear_features(panel, i, pre_gate=True) if i >= 7 else np.full(247, np.nan) for i in range(len(panel.days))])
        predictions = []
        with ModelWorker("readiness_lear", threads=1) as worker:
            for d, t in zip(days, ids):
                issued = OUT / f"lear_{d.date()}.npy"
                if issued.exists():
                    predictions.append(np.load(issued))
                    continue
                if not np.isfinite(X[t]).all():
                    raise ValueError(f"missing pre-gate inputs for {d}")
                causal = Panel(panel.days[:t+1], {"price_de": panel["price_de"][:t+1].copy()})
                causal.x["price_de"][t] = np.nan
                prediction = worker.call("lear", panel=causal, t=t, X=X[:t+1], windows=WINDOWS)
                with issued.open("xb") as f:
                    np.save(f, prediction)
                predictions.append(prediction)
                print("LEAR", d.date(), flush=True)
                (ROOT / "runs/HEARTBEAT").touch()
        np.savez(learner, days=days.strftime("%Y-%m-%d"), point=predictions)
    sel = np.isin(z["days"], days.strftime("%Y-%m-%d"))
    yy = y[sel]
    forecasts = {"pregate_lora": z["point"][sel], "pregate_lgbm": np.load(lgb_path)["point"],
                 "pregate_lear": np.load(learner)["point"],
                 "naive": np.array([panel["price_de"][t-(7 if d.weekday() in (0,5,6) else 1)] for d,t in zip(days,ids)])}
    raw_cov = np.mean((yy >= lo[sel]) & (yy <= hi[sel]))
    adjusted_cov = np.mean((yy >= lo[sel]-adjustment) & (yy <= hi[sel]+adjustment))
    summary = dict(protocol=protocol, coverage80_raw=float(raw_cov), coverage80_calibrated=float(adjusted_cov),
                   mean_width_raw=float(np.mean(hi[sel]-lo[sel])),
                   mean_width_calibrated=float(np.mean(hi[sel]-lo[sel]+2*adjustment)), models=[])
    # Append separate readiness trials; preserve every original dev/holdout row.
    ledger = [json.loads(line) for line in LEDGER.read_text().splitlines()]
    summary['reference_trial_ids'] = {}
    for model in ['pregate_lgbm','pregate_lear']:
        cfg=dict(protocol='27c9d9e',current_wind_solar='pregate_weather',
                 parameters=protocol['lgbm'] if model=='pregate_lgbm' else list(WINDOWS))
        existing=[r for r in ledger if r['name']==model and r['split']=='readiness_dev' and r['config']==cfg]
        if existing:
            rid=existing[-1]['id']
            np.testing.assert_array_equal(np.load(ROOT/f'runs/forecasts/{rid}.npz')['point'],forecasts[model])
        else:
            rid=record(model,cfg,days,forecasts[model],panel=panel,split='readiness_dev',
                 notes='Post-study dev-only reference. Legacy gross battery metric; constrained costs in runs/readiness/report.md. Not part of frozen holdout selection.')['id']
        summary['reference_trial_ids'][model]=rid
    for name, f in forecasts.items():
        e = economics(yy, f)
        summary["models"].append(dict(name=name, mae=mae(yy,f),
             dm_p_better_than_lear=dm_test(yy,forecasts["pregate_lear"],f) if name!="pregate_lear" else None,
             economics=e))
    summary["lora_cost_sensitivity"] = [economics(yy,forecasts["pregate_lora"],Battery(fee_slippage_eur_mwh=fee,wear_eur_mwh=wear),fixed)
            for fee,wear,fixed in [(0,0,0),(3,20,0),(3,20,15000),(3,20,30000),(6,40,30000)]]
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    text = ["# Dev-only pre-gate readiness study", "", "New protocol after the original holdout was opened; no new untouched test result.",
            f"Window: {days[0].date()}..{days[-1].date()}, {len(days)} days. Models/settings fixed before this run.", "",
            "| Model | MAE EUR/MWh | DM p vs pre-gate LEAR | Margin after variable costs EUR |", "|---|---|---|---|"]
    for m in summary["models"]:
        text.append(f'| {m["name"]} | {m["mae"]:.2f} | {m["dm_p_better_than_lear"]} | {m["economics"]["margin_eur"]:.0f} |')
    text += ["", f"80% interval coverage: raw {100*raw_cov:.1f}%, calibrated {100*adjusted_cov:.1f}%.",
             "Per-hour conformal expansion fitted only through June 2025; dependence/shift preclude a guarantee.", "",
             "Battery: 1 MWh / 1 MW, 95% one-way efficiency, 0.1 MWh SOC grid, at most two equivalent cycles/day,",
             "empty midnight; 3 EUR/grid MWh fees+slippage and 20 EUR/discharged battery MWh wear assumed.",
             "These are illustrative sensitivities, not asset quotes or established net returns.",
             "Fixed-cost sensitivity: 0/15,000/30,000 EUR annually, prorated to scored days.",
             "Hourly auction prices omit 15-minute execution, imbalance risk, outages and taxes.",
             "Dev uses the original 24-bin DST convention; prospective paper issues skip 23/25-hour delivery days.",
             "Weather features for these baselines start in September 2024; Chronos has a longer training/pretraining history.",
             "DM p-values are the existing daily-loss test and do not adjust for serial correlation or trial selection.", "",
             "| Fees+slippage EUR/grid MWh | Wear EUR/discharged MWh | Annual fixed EUR | Scenario net EUR |",
             "|---|---|---|---|"]
    for e in summary['lora_cost_sensitivity']:
        b=e['assumptions']
        text.append(f"| {b['fee_slippage_eur_mwh']} | {b['wear_eur_mwh']} | {e['annual_fixed_eur']} | {e['scenario_net_eur']:.0f} |")
    (OUT / "report.md").write_text("\n".join(text)+"\n")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
