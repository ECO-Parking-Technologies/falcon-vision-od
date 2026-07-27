# 01 — Housekeeping & Repo Restructure

## Done (2026-07-27)

- [x] Removed ExecuTorch everywhere (deps, artifacts, code paths); preannotation runs plain PyTorch/TorchScript.
- [x] Deleted MobileNetV4 training runs + exports, aborted runs, 17 stale regenerable `split_*` dirs (~8 GB freed).
- [x] Deleted stale ONNX→TF→TFLite artifacts, `patches/` (dead onnx2tflite workaround), `build/`, `dist/`.
- [x] Moved downloaded COCO checkpoints to gitignored `weights/`.
- [x] Removed redundant `.venv` (5.6 GB; canonical venv is `falcon-vision-od-venv`) and upstream `.github/` (FUNDING.yml, ISSUE_TEMPLATE).
- [x] Rewrote README; fixed `.vscode/launch.json`; trimmed `generate_model_files.py` to a checkpoint→TorchScript exporter.
- [x] Converted the portal API HTML dump to parseable docs: [../portal-api/](../portal-api/).

## Remaining

- [ ] **Commit the pending working-tree changes** (cleanup touched tracked files; review diff first).
- [ ] Decide fate of the uncommitted `effdet/config/model_config.py` mnv4 additions — harmless to keep; commit or drop.
- [ ] `effdet.egg-info/` left in place to avoid disturbing a possible editable install; remove on next venv rebuild.
- [ ] **Repo restructure** (after the TFLite export path proves out, before large-scale data lands):
  - Separate our code from vendored upstream: e.g. `falcon/` (or `tools/`) containing `training/`, `preannotation/`, `export/`, `eval/`, `portal/`, with `configs/` at top level; leave `effdet/` as the library.
  - One canonical config entry point; kill duplicate model-size tables (preannotation config duplicates `model_config.py` input sizes).
  - Consider re-syncing with upstream effdet or pinning it as a dependency instead of vendoring (needs the mnv4 configs upstreamed or kept as a patch layer).
- [ ] Data layout redesign for all-garages scale — see [04-portal-integration.md](04-portal-integration.md).
