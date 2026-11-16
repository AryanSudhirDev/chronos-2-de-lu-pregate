#!/bin/zsh
# Zero-shot Chronos-2 variants, one model at a time.
cd "$(dirname "$0")/.."
run() { .venv/bin/python -W ignore scripts/run_chronos_zeroshot.py "$@" 2>&1 | grep -E "^\[|Error|error" }
run --no_covs
run
run --calendar
run --neighbours
run --calendar --neighbours
run --calendar --context_days 341
