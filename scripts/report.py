"""Rebuild the results table in README.md from runs/ledger.jsonl."""
import re

import pandas as pd

from epf.data import ROOT
from epf.ledger import read_ledger

df = read_ledger()
record_count = len(df)
df = df.drop_duplicates('id', keep='last')
README = ROOT / "README.md"


def fmt_cfg(c: dict) -> str:
    skip = {"model", "mode", "folds", "tag", "workers"}
    return ", ".join(f"{k}={v}" for k, v in c.items() if k not in skip)[:90]


def table(split: str) -> str:
    d = df[df.split == split].copy()
    if d.empty:
        return "_none yet_"
    d = d.sort_values("mae")
    rows = ["| Model | Config | Window | MAE | RMSE | rMAE | CRPS | Battery % of oracle | id |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        crps = f"{r['crps']:.2f}" if "crps" in r and pd.notna(r.get("crps")) else ""
        rows.append(f"| {r['name']} | {fmt_cfg(r['config'])} | {r['start']}..{r['end']} | {r['mae']:.2f} | "
                    f"{r['rmse']:.2f} | {r['rmae']:.3f} | {crps} | {r['battery_pct_oracle']:.1f} | {r['id']} |")
    return "\n".join(rows)


body = (f"Unique trials in ledger: **{len(df)}** ({record_count} append-only records; "
        f"{record_count-len(df)} duplicate recording).\n\n### Dev window (walk-forward)\n\n{table('dev')}\n\n"
        f"### Sealed holdout\n\n{table('holdout')}\n\n"
        f"### Post-study dev readiness references\n\nThese were added after the holdout study; they were not part of candidate selection. "
        f"Battery figures here retain the original gross metric; constrained cost results are in runs/readiness/report.md.\n\n{table('readiness_dev')}\n\n"
        f"### Post-study dev adaptive experiment\n\nA separate dev-only trial after holdout inspection; not a new unseen test. "
        f"Decision and constrained costs: runs/adaptive/report.md.\n\n{table('adaptive_dev')}\n")
text = README.read_text()
text = re.sub(r"<!-- results:start -->.*<!-- results:end -->",
              f"<!-- results:start -->\n{body}<!-- results:end -->", text, flags=re.S)
README.write_text(text)
print(body)
