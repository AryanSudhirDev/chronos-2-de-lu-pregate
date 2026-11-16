"""Private numeric model worker. Inputs arrive over a pipe, never from data files."""
import sys

protocol_stdout = sys.stdout
wire = protocol_stdout.buffer
sys.stdout = sys.stderr  # model libraries may print; stdout is reserved for replies

import socket
import traceback
from pathlib import Path

import numpy as np

from epf.offline import receive, send

model = None


def execute(p):
    global model
    cmd = p.pop("command")
    if cmd == "probe":
        # Fail closed if either OS restriction is ineffective. No actual data is read.
        try:
            with open(p["forbidden"], "rb"):
                raise RuntimeError("sandbox allowed opening a forbidden data file")
        except PermissionError:
            pass
        for family, address in [(socket.AF_INET, ("127.0.0.1", 9)),
                                (socket.AF_INET6, ("::1", 9))]:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as s:
                    s.connect(address)
            except PermissionError:
                continue
            raise RuntimeError("sandbox failed to deny network access")
        return "filesystem and IPv4/IPv6 network denied"
    if cmd == "chronos_fit":
        from chronos import Chronos2Pipeline
        import torch
        torch.set_num_threads(p.pop("threads"))
        pipe = Chronos2Pipeline.from_pretrained(p.pop("snapshot"), device_map=p.pop("device"), local_files_only=True)
        model = pipe.fit(p.pop("train"), output_dir=Path.cwd() / "fit", **p)
        return np.asarray(model.quantiles)
    if cmd == "chronos_zero":
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(p["snapshot"], device_map=p["device"], local_files_only=True)
        return np.asarray(model.quantiles)
    if cmd == "chronos_predict":
        return np.stack([v[0].float().cpu().numpy().T for v in
                         model.predict(p["items"], prediction_length=24, batch_size=len(p["items"]))])
    if cmd == "lgbm_fit":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**p["params"]).fit(p["X"], p["y"])
        return None
    if cmd == "lgbm_predict":
        return model.predict(p["X"])
    if cmd == "lear":
        from epf.lear import lear_forecast
        return lear_forecast(p["panel"], p["t"], p["X"], p["windows"])
    if cmd == "blend":
        from epf.blend import online
        return online(p["F"], p["y"], **p["config"])[-1]
    if cmd == "dispatch":
        from epf.readiness import Battery, dispatch
        if set(p) != {"forecast", "assumptions"}:
            raise ValueError("dispatch accepts only one day's forecast and frozen assumptions")
        return dispatch(p["forecast"], Battery(**p["assumptions"]))
    if cmd == "adaptive_init":
        from epf.adaptive import AdaptiveBias
        model=AdaptiveBias(**p)
        return None
    if cmd == "adaptive_predict":
        if set(p)!={"day","forecast"}:raise ValueError("adaptive prediction accepts no outcomes")
        return model.predict(**p)
    if cmd == "adaptive_observe":
        return model.observe(**p)
    raise ValueError(f"unknown worker command {cmd}")


while True:
    try:
        payload = receive(sys.stdin.buffer)
    except RuntimeError:
        break
    try:
        send(wire, {"value": execute(payload)})
    except Exception:
        send(wire, {"error": traceback.format_exc()})
