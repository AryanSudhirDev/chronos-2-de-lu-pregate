#!/bin/zsh
# Only after the candidate declaration is committed and both offline dev rehearsals pass.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - "${1:-}" <<'PY'
import os
from pathlib import Path
import sys
import time
from epf.ledger import read_ledger
if sys.argv[1]:
    pid = int(sys.argv[1])
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        Path('runs/HEARTBEAT').touch()
        time.sleep(30)
log = Path('runs/offline_rehearsal_stage2.log').read_text()
assert 'Traceback' not in log and sum(line.startswith('[rehearsal]') for line in log.splitlines()) == 7, 'full offline dev rehearsal did not pass'
assert 'offline weather rehearsal matches existing dev forecasts' in Path('runs/weather_rehearsal.log').read_text()
assert '\nOK\n' in Path('runs/offline_tests.log').read_text(), 'isolation tests did not pass'
assert not (read_ledger().split == 'holdout').any(), 'holdout already recorded; do not retrain'
PY
# A failed opening deliberately leaves its lock and checkpoints for the steering agent.
mkdir runs/ckpt/holdout_opening.lock
touch runs/HEARTBEAT
EPF_OPEN_HOLDOUT=1 .venv/bin/python -W ignore scripts/make_pregate_fc.py --holdout --offline
EPF_OPEN_HOLDOUT=1 .venv/bin/python -W ignore scripts/open_holdout.py \
  --chronos 51d3abf5b2 --stack 3f75dbb0b1 --lgbm 69b2219339 \
  --blend 2dae4f56b9 --pre_gate 864e91aa35 --zero_shot 7a7adf81e5 --lear 580124a4ca --offline
EPF_OPEN_HOLDOUT=1 .venv/bin/python scripts/compare.py --holdout
.venv/bin/python scripts/report.py
rmdir runs/ckpt/holdout_opening.lock
