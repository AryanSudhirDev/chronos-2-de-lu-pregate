# Release verification

Release preparation on 2026-11-15 used the original Python 3.12 macOS environment.

- The frozen adapter and calibration SHA-256 hashes matched the paper policy.
- The full merged model was exported to safetensors. Reloaded synthetic predictions had
  maximum absolute difference **0.0** from the in-memory merged model in the export check.
- **26 synthetic unit/integration tests passed**, covering input causality, unreleased outcomes,
  worker file/network denial, immutable issuance, dispatch constraints, DST rules, calibration,
  replay integrity, adaptive state updates and the public release input schema.
- The portable inference helper was exercised with the complete synthetic covariate schema
  and the saved calibration. This is an inference plumbing check, not a new accuracy evaluation.
- A fresh Python 3.12 environment installed the pinned research dependencies from the local
  package cache and installed this source snapshot in editable mode. The full **26-test suite
  passed again from that clean source snapshot**.
- Git whitespace checks passed. The public snapshot is assembled from explicit source/result
  allowlists, without source logs, credentials, raw feed archives or private Git history.

The first fresh-checkout test exposed a startup probe that assumed the raw-data file existed.
Release code now probes an existing synthetic canary outside the worker's readable paths;
it still requires actual access denial and does not treat a missing file as evidence of isolation.
Tests probe optional historical artifacts only when present. This changes packaging/startup
verification, not the historical forecasts or saved model weights.

No development model, holdout evaluation, live collection or paper schedule was restarted for
publication. Synthetic tests instantiate short-lived test workers only. Historical scientific
results remain the original saved records; packaging checks do not strengthen their evidence.

The GitHub workflow runs portable synthetic tests on Linux. OS-isolation integration tests
require macOS and are intentionally absent from that Linux workflow. Cross-platform training
equivalence and a deployed operational forecast service have not been verified.
