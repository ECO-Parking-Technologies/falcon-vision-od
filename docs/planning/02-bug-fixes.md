# 02 — Known Bug Fixes (DONE 2026-07-27)

All fixed and verified against real garage images. Two additional, more severe bugs were found during verification (items 7–8).

1. [x] **Inference preprocessing mismatch** — `preprocess()` now applies the training dataloader's normalization (ImageNet mean/std over 0-255 pixels) instead of bare `/255`.
2. [x] **Box coordinate-order confusion** — standardized on effdet's actual output (`[x_min, y_min, x_max, y_max, score, class]`, verified in `effdet/anchors.py generate_detections`); `infer()` debug and `annotate()` fixed, convention documented at the top of `efficient_det_model.py`.
3. [x] **Label lookup in debug output** — now goes through the contiguous remap instead of indexing a sparse dict by `len()`.
4. [x] **Dead config key** — code now reads `train_val_split` (matching the YAML).
5. [x] **Threshold default drift** — `convert_detections` threshold is a required parameter.
6. [x] **num_classes derivation** — from the full label map, not `allowed_labels + 1`.
7. [x] **State-dict loading silently loaded nothing** — `create_model(bench_task="predict")` returns a `DetBenchPredict`; the old code stripped every `model.` prefix and loaded into the bench with `strict=False` (0 keys matched → random weights), then double-wrapped in another `DetBenchPredict`. All prior TorchScript/ONNX/TFLite/ExecuTorch exports — and every preannotation produced from them — came from an **effectively untrained model**. Fixed in `efficient_det_model.py` and `generate_model_files.py`: keys normalized (leading `module.`/`model.` stripped once) and loaded into `bench.model` with missing-keys treated as fatal. Also: `.pt` files are tried as TorchScript first with fallback to checkpoint loading (the downloaded pretrained `.pt` files are re-saved checkpoints, not traces — the `use_pretrained_model` path was broken too).
8. [x] **Training labels were 0-based → `person` was trained as background** — effdet's COCO parser (`cat_ids_as_labels = True`) uses category ids directly as labels with **0 reserved for background**, but `remap_label_map` produced ids 0..5. Consequence: person (id 0) could never be learned, and one class-head logit was dead. `remap_label_map` now emits **1-based ids (1..6)**, which also makes detection class ids (1-based, bg=0) map back through the plain inverse map.

## ⚠️ Consequence: retrain required

The existing `model_best.pth.tar` (mAP 0.231) was trained with the 0-based labels — person is background, class logits shifted. **The next training run (with `pretrained: true`, per track 05) supersedes it**; don't generate preannotations from the old checkpoint expecting correct class names. Detection *localization* from the old checkpoint is fine.

Verified: correct-weight model on a busy arlington frame produces tight car boxes at 0.92/0.63/0.61/0.47 vs. the degenerate ~0.5-score slivers before the fix; full preannotation pipeline dry-run passes.
