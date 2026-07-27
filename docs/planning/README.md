# Planning

Working plan for what remains to be done, split by track. Status snapshot as of 2026-07-27.

| # | Track | Doc | State |
|---|-------|-----|-------|
| 1 | Housekeeping / repo restructure | [01-housekeeping.md](01-housekeeping.md) | in progress |
| 2 | Known bug fixes | [02-bug-fixes.md](02-bug-fixes.md) | not started |
| 3 | int8 TFLite export pipeline | [03-tflite-export.md](03-tflite-export.md) | not started |
| 4 | Portal integration (training images) | [04-portal-integration.md](04-portal-integration.md) | blocked on API gap |
| 5 | Training improvements | [05-training-improvements.md](05-training-improvements.md) | not started |
| 6 | Evaluation & benchmarking | [06-evaluation.md](06-evaluation.md) | not started |

## Context

- **Goal:** replace the sensor firmware's off-the-shelf **EfficientDet-Lite2** (448×448, full-int8 TFLite, `TFLite_Detection_PostProcess` output, max 25 detections) with a fine-tuned model that is faster and more accurate, especially for vehicles in monitored spots.
- **Hardware tiers:** Raspberry Pi CM3+ (CPU/XNNPACK) · [DART-MX8M-PLUS](https://variscite.com/system-on-module-som/i-mx-8/i-mx-8m-plus/dart-mx8m-plus/) (i.MX8M Plus NPU) · [DART-MX95](https://variscite.com/system-on-module-som/i-mx-9/i-mx-95/dart-mx95/) (i.MX95 eIQ Neutron NPU). All three consume **full-int8 TFLite** — one export format for the fleet.
- **Decisions made:** stay in the EfficientDet family (firmware compatibility); PyTorch training (this repo); TFLite via ai-edge-torch for deployment; ExecuTorch abandoned; CVAT is the annotation source of truth.
- **Best training run so far:** `tf_efficientdet_lite0` @320, eval mAP 0.231 (trained without COCO detector pretraining — known headroom).
- Long-term background reading: [../road-map/](../road-map/).
