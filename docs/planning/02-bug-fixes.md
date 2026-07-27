# 02 — Known Bug Fixes (queued, not started)

Do these before generating new preannotations, benchmarks, or eval numbers — they silently skew results.

1. **Inference preprocessing mismatch** — `preannotation/efficient_det_model.py` `preprocess()` does BGR→RGB + `/255` only; the training dataloader normalizes with ImageNet mean/std. Inference inputs are therefore distribution-shifted vs. training. Fix: apply the same mean/std as `train.py` (and mirror whatever the final firmware preprocessing contract is — the baseline Lite2 uses uint8 input with scale 1/128, zero-point 127, i.e. ≈ `(x-127)/128`).
2. **Box coordinate-order confusion** — same file: `infer()` debug unpacks `y1,x1,y2,x2`, `annotate()` unpacks `ymin,xmin,ymax,xmax`, but `utils.convert_detections` unpacks `xmin,ymin,xmax,ymax`. effdet `DetBenchPredict` outputs **xyxy**; make all three consistent and add a comment stating the convention.
3. **Label lookup indexing** — `infer()` debug does `labels[int(cls)] if int(cls) < len(labels)` where `labels` is a dict keyed by sparse COCO ids ({1,2,3,4,6,8}); classes 6/8 break. Detections at that point carry *contiguous* model ids, so the lookup should go through the remap (`remap_label_map`) like `run_preannotation.py` does.
4. **Dead config key** — `config/train_wrapper_config.yaml` has `train_val_split`, code reads `train_split` (hardcoded 0.8 fallback always wins). Rename one side.
5. **Threshold default drift** — `utils.convert_detections` defaults `threshold=0.25` while config uses 0.3; harmless today (config value is passed) but remove the duplicate default.
6. Nice-to-have: `run_preannotation.py` `num_classes = len(allowed_labels) + 1` (=6) only matches the trained model because 6 classes were trained; derive from `label_map` instead of `allowed_labels` so shrinking `allowed_labels` doesn't rebuild a wrong-sized model.
