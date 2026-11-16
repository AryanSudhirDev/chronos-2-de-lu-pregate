# Reproduce, inspect and extend

## 1. Load the exact published model

The easiest reproducible unit is the frozen checkpoint. Use Python 3.12 in a fresh environment:

```sh
uv venv --python 3.12
uv pip install -r requirements-inference.txt
.venv/bin/hf download AryanSudhirDev/chronos-2-de-lu-pregate --local-dir model
.venv/bin/python scripts/make_example_input.py --output example_input.json
.venv/bin/python publish/inference.py --model model --input example_input.json --output forecast.json --calibration model/calibration.json
```

The example checks loading, input validation, quantile output and calibration. It is synthetic
and has no expected accuracy score. `forecast.json` returns `median` (24 values), `quantile_levels`,
`quantiles` (24 rows), and, with calibration enabled, `lower80` / `upper80`.
Use `--device mps` on a compatible Mac; CPU is the portable default. `--threads` defaults to 4.

For immutable weight provenance, the initial model publication is revision
`70aec15186fd38d649cb6145a2f44dba68c9fa75`. Later documentation updates retain the same
model and adapter tensors. The base is revision `29ec3766d36d6f73f0696f85560a422f50e8498c`.
File hashes are listed in the release's `manifest.json`.

## 2. Install research dependencies and run checks

```sh
uv pip install -r requirements-research.txt
uv pip install --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
```

Use an editable install: code resolves `data/`, `runs/` and worker scripts relative to the checkout.
Direct dependencies are pinned to the observed environment; `environment-observed.txt` records
its full package inventory. This is an environment record, not a universal cross-platform lock.
Optional historical TabPFN trials require their separately licensed/access-controlled dependency.
Default inference and the release checks do not require it.

The offline model runner and its integration tests require macOS sandbox-exec, available
dependencies under `.venv/`, and appropriate local runtime permissions. There is no silent
fallback that weakens the historical isolation claim. Non-macOS users can run portable unit tests:

```sh
.venv/bin/python -m unittest discover -s tests -p test_readiness.py -v
.venv/bin/python -m unittest discover -s tests -p test_paper.py -v
.venv/bin/python -m unittest discover -s tests -p test_diagnosis.py -v
.venv/bin/python -m unittest discover -s tests -p test_release.py -v
```

## 3. Rebuild inputs for a new development run

The following commands download public data and perform development fitting when explicitly run
by a researcher. Publication did not run them. Keep your work in a separate checkout or copy of
`runs/` so the published historical record remains intact.

```sh
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/make_pregate_fc.py
```

The collector fetches through current availability; the historical study cutoff is fixed in
the code. New responses may be revised. Weather collection is rate-limited and can fail if the
upstream archive or terms change. See DATA.md before using raw inputs for actual forecasts.

## 4. Development baselines, zero-shot and adaptation

Run heavy models sequentially. The default Chronos script chooses MPS when available, else CPU.

```sh
.venv/bin/python scripts/run_naive.py
.venv/bin/python scripts/run_lear.py
.venv/bin/python scripts/run_lgbm.py
.venv/bin/python scripts/run_chronos_zeroshot.py --neighbours
.venv/bin/python scripts/run_chronos_finetune.py --mode lora --lr 1e-4 --neighbours
.venv/bin/python scripts/run_chronos_finetune.py --mode full --lr 1e-6 --neighbours
.venv/bin/python scripts/run_chronos_zeroshot.py --neighbours --pre_gate
.venv/bin/python scripts/run_chronos_finetune.py --mode lora --lr 1e-4 --neighbours --pre_gate --folds 2024,2025
```

These are new experiments, not guaranteed bit-identical reconstructions: floating-point kernels,
MPS behavior, source revisions, library versions and training randomness can differ. Some original
development scripts use the public base model name; for exact provenance pin its revision in a
new study before fitting. Each evaluation appends to `runs/ledger.jsonl` and saves predictions.
Use a new ledger for a new study rather than mixing fresh outcomes into the published record.

Stacking and blending scripts accept ledger IDs; consult their `--help` and the recorded
configurations. A stack must be trained on earlier forecasts/errors, never the target dates being
scored. `scripts/report.py` renders the trial table; reports alone are not an unseen evaluation.

## 5. Inspect the already-opened holdout

The original holdout was opened once. This release preserves its ledger rows, seven-model
common-window summary, model forecast arrays, daily-issuance audit archive and hash manifest.
`scripts/open_holdout.py` refuses another opening when holdout rows already exist. Do not remove
that guard or erase rows to obtain a more favourable result. Original historical commit IDs remain
provenance labels; they refer to the private development history, not this clean public snapshot.

Inspect `runs/holdout_summary.json`, `runs/compare_holdout.md` and the saved `.npz` forecasts
without raw data. Recomputing target-based metrics requires source observations matching the
original study. The raw data is not redistributed and may be revised upstream. This release
does not promise exact re-scoring from today's downloads.

Later historical replays intentionally reuse these forecasts and are diagnostic, not independent
test data. Readiness and adaptive reports use development dates and are labelled accordingly.

## 6. Connect the released adapter to the research pipeline

```sh
.venv/bin/python scripts/download_checkpoint.py
```

This downloads the original adapter and pinned base, checks adapter/calibration hashes against
the frozen policy, and places the adapter at the path expected by `scripts/paper_day.py`.
It does not start feed collection, training or paper issuance. It refuses a differing existing
adapter. The full model can be loaded directly from the Hugging Face repository instead.

## 7. Paper prototype prerequisites

The paper prototype is stopped, not a deployed service. There is no active schedule, broker or
bid connection in the release. The original observation ended after five scored days. A new
paper study would require rebuilt historical raw/weather caches, timestamped live source
snapshots, working pre-gate weather estimates, pinned adapter/calibration/protocol files,
macOS isolation and separate issuance/scoring invocations. `paper_day.py` documents its commands.

Do not treat the synthetic inference example as a functioning live collector. Do not claim
that the original 30-day paper review completed. New observations must accumulate in real time.

## Known reproduction limits

- Historical publication vintages and foundation-model pretraining overlap are unverified.
- Raw market/weather snapshots and ephemeral fitted weather mapper objects are absent.
- The actual original training run is not represented by an optimizer-resume checkpoint.
- Hardware and packages can change optimization and inference numerics.
- The source release preserves code and results; the model release preserves the retained fit,
  not every fitted reference, stack or full-fine-tuning variant.
- A GitHub source archive carries no private development history or local execution logs.
