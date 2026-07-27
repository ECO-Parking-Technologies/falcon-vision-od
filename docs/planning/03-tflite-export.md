# 03 — int8 TFLite Export Pipeline (not started)

Goal: `model_best.pth.tar` → **full-int8 TFLite** that is a drop-in for the firmware's current model on all three hardware tiers.

## Baseline model contract (measured from `efficientdet_lite2.tflite`)

- Input: `[1,448,448,3]` **uint8**, quant `(scale=1/128, zero_point=127)` → effectively `(x-127)/128`.
- Body: full-int8 (CONV_2D / DEPTHWISE_CONV_2D etc.), post-processing via **`TFLite_Detection_PostProcess` custom op**.
- Outputs (4 tensors, float32): boxes `[1,25,4]`, classes `[1,25]`, scores `[1,25]`, num_detections `[1]`.

## Tasks

- [ ] Build exporter with **ai-edge-torch**: effdet model (without `DetBenchPredict` NMS) → LiteRT, full-int8 PTQ with a representative dataset of real garage images (a `rep_data_gen` existed in the deleted `export_to_tflite.py` — reuse the idea).
- [ ] **Resolve the post-process contract** (open question, talk to FW owner):
  - Option A: graft `TFLite_Detection_PostProcess` onto the exported graph (this is what google/automl's exporter does) so outputs match byte-for-byte → zero firmware changes.
  - Option B: firmware adapts to raw head outputs + on-device decode (more FW work, more flexibility).
- [ ] Parity test: PyTorch vs TFLite outputs on N sample images (IoU/score deltas within tolerance).
- [ ] Verify **NPU delegation**: on i.MX8M Plus check every op maps to the NPU (VX delegate; int64 ops and odd shapes fall back to CPU — a known NXP pitfall); run the eIQ Neutron converter for i.MX95.
- [ ] Wire into repo as the successor of `generate_model_files.py` (which currently only emits the float TorchScript for preannotation).
- [ ] Note: quantization must be **static PTQ (or QAT later)** — PyTorch `quantize_dynamic` does not quantize convs (the pre-cleanup ".quant" artifacts were effectively fp32).

## Benchmark targets

Beat the Lite2 baseline's on-device latency and mAP-on-our-data. Per-tier model scaling once the pipeline works: CM3+ likely lite0@320; NPU tiers can afford lite2+/higher input res.
