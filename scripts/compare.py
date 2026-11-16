"""Compare every dev forecast in the ledger on the window they all share, with multivariate
Diebold-Mariano p-values (H1: this model is more accurate than the reference). Writes runs/compare.md.

    python scripts/compare.py [--ref 580124a4ca] [--since 2023-01-01]
    EPF_OPEN_HOLDOUT=1 python scripts/compare.py --holdout  # after the opening; reference = LEAR; writes runs/compare_holdout.md
"""
import argparse
from functools import reduce

import numpy as np

from epf.data import HOLDOUT_START, ROOT, load_panel
from epf.ledger import FC, read_ledger
from epf.metrics import battery_profit, dm_test, mae, rmse

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="580124a4ca", help="DM reference forecast (default LEAR)")
ap.add_argument("--since", default="2023-01-01")
ap.add_argument("--holdout", action="store_true")
a = ap.parse_args()

split = "holdout" if a.holdout else "dev"
rows = read_ledger().query("split == @split").drop_duplicates("id", keep="last")
if a.holdout:
    a.since = str(HOLDOUT_START.date())
    a.ref = rows[rows.name == "lear"].id.iloc[-1]
runs = {r.id: (r.name, np.load(FC / f"{r.id}.npz")) for r in rows.itertuples() if a.holdout or r.start <= a.since}  # shorter dev runs would shrink the window
days = reduce(np.intersect1d, [z["days"] for _, z in runs.values()])
days = days[days >= a.since]
panel = load_panel(allow_holdout=a.holdout)
y = np.array([panel["price_de"][panel.idx(d)] for d in days])
fc = {i: z["point"][np.isin(z["days"], days)] for i, (_, z) in runs.items()}
assert all(f.shape == y.shape for f in fc.values())
_, oracle = battery_profit(y, y)
table = sorted(fc, key=lambda i: mae(y, fc[i]))
lines = [f"Window {days[0]}..{days[-1]} ({len(days)} days), DM reference {a.ref} ({runs[a.ref][0]}).", "",
         "| Model | id | MAE | RMSE | Battery % of oracle | DM p vs ref |", "|---|---|---|---|---|---|"]
for i in table:
    p = "" if i == a.ref else f"{dm_test(y, fc[a.ref], fc[i]):.4f}"
    lines.append(f"| {runs[i][0]} | {i} | {mae(y, fc[i]):.2f} | {rmse(y, fc[i]):.2f} | "
                 f"{100 * battery_profit(y, fc[i])[0] / oracle:.1f} | {p} |")
text = "\n".join(lines) + "\n"
(ROOT / "runs" / ("compare_holdout.md" if a.holdout else "compare.md")).write_text(text)
print(text)
