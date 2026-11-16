# Release artifact map

## Hugging Face model repository

`model.safetensors` and `config.json` load a complete merged Chronos-2 model. The LoRA updates
are already applied; no separate base-model download is needed for this path. The adapter is
also available separately under `adapter/`, with a public base ID and immutable revision.
No tokenizer is required for this numeric time-series architecture.

`inference.py`, `requirements.txt`, `example_input.json` and `make_example_input.py` provide
a runnable numeric example. `calibration.json` is an optional per-hour interval correction,
stored separately from neural weights. `manifest.json` hashes all packaged files except itself.
`source-release.zip` contains the corresponding GitHub source snapshot. Documentation is under
`docs/`, and `results.png` illustrates the clearly separated evaluation windows.

The adapter hash is `01fe7c491f30f3bbcb1265d3bbd53db226b5f0cfe3769d0519b94679462c48b5`.
The calibration hash is `b31f52fcad40dc9c0460b433f707b34663f9778c608b65a5da596d32d1d30808`.
The published checkpoint was merged from the retained original adapter, not retrained for release.

## GitHub source repository

`src/epf` and `scripts` contain the implementation. `tests` uses synthetic values and temporary
files; no holdout re-evaluation is required for the test suite. `publish` contains the portable
inference helper, model card and packaging sources. `docs` explains results and reconstruction.

`runs/ledger.jsonl` retains all original records. `runs/forecasts/*.npz` contains date arrays,
point predictions, optional quantiles and quantile levels. It does not contain the raw target
feed. Seven holdout rows share a common-day summary; ledger rows can have different available
days. `runs/holdout_issued_audit.zip` stores daily prediction arrays and its companion manifest
records their hashes. These preserve prediction artifacts; they do not supply original raw inputs.

Selected readiness, replay, adaptive and paper protocols/reports are included for provenance.
Run logs, active queue files, source-response snapshots, secrets, caches, machine-specific paths
and unrelated project inventories are excluded from the clean source snapshot. Original ledger
commit IDs refer to historical development commits outside the public snapshot.

The full raw data, every intermediate fit, all issued live-source snapshots and all trained
reference models are not part of this release. Collection and fitting code document how to
reconstruct new instances, with the limits described in REPRODUCIBILITY.md.
