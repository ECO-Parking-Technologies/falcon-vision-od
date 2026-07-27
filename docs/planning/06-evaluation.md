# 06 — Evaluation & Benchmarking (not started)

Two-layer evaluation (roadmap): detection quality ("where are cars?") and business accuracy ("which spots are occupied?"). Plus on-device speed.

## Spot-occupancy evaluator (the decision-gate tool)

- [ ] Script: detections + spot definitions → per-spot occupancy score 0–1 (IoU vs spot region; scale/clamp around T=0.5 per roadmap). Use OpenCV primitives for later C++ port.
- [ ] Spot definitions: portal `parkingSpaces` data may provide geometry ([04-portal-integration.md](04-portal-integration.md)); otherwise reuse existing sensor spot configs.
- [ ] Ground truth: CVAT annotations with the roadmap's `InEcoParkingSpot` attribute (start tagging it in the next annotation round), and/or historical `parkingSpaceDataPoints` from the current classifier.
- [ ] Baseline comparison: run the firmware's `efficientdet_lite2.tflite` and our candidates through the same evaluator → per-spot accuracy/precision/recall vs the current classifier (metrics functions exist in falcon-vision-ml `dnn_validation.py`).

## Detection metrics

- [ ] Per-class + aggregate PR curves, mAP; confusion matrix. Consider FiftyOne for inspection (roadmap recommendation).
- [ ] Garage-only val split alongside the COCO-merged one.

## On-device benchmarking

- [ ] Benchmark harness for the three tiers (CM3+, DART-MX8M-PLUS, DART-MX95): TFLite `benchmark_model` or a small runner, single-thread CPU + delegated NPU, 1000 runs, report p50/p95; random input is fine for latency.
- [ ] Verify NPU op coverage on i.MX (fallback ops show up as CPU time).
- [ ] Record results per model×tier in a tracked doc/CSV so speed/accuracy trade-offs are decided from data.
