# 03 — int8 TFLite Export Pipeline (toolchain DONE 2026-07-27; device verification pending)

Goal: `model_best.pth.tar` → **full-int8 TFLite** that is a drop-in for the firmware's current model on all three hardware tiers.

## Baseline model contract (measured from `efficientdet_lite2.tflite`)

- Input: `[1,448,448,3]` **uint8**, quant `(scale=1/128, zero_point=127)` → effectively `(x-127)/128`.
- Body: full-int8, post-processing via **`TFLite_Detection_PostProcess` custom op**.
- Outputs (4 tensors, float32): boxes `[1,25,4]`, classes `[1,25]`, scores `[1,25]`, num_detections `[1]`.

## Done

- [x] **`export_tflite.py`** (run in the export venv from `setup_export_venv.sh` — litert-torch pulls TensorFlow + its own torch pin, kept separate from the training venv):
  - Exports the raw network (backbone + BiFPN + heads, no NMS) via **litert-torch** (the renamed ai-edge-torch), input **NHWC `[1,H,W,3]`** like the baseline.
  - Outputs pre-NMS tensors mirroring `DetBenchPredict`: boxes `[1,N,4]` (anchor-encoded), sigmoid scores `[1,N,C]`.
  - **Full-int8 static PTQ via ai-edge-quantizer** (`static_wi8_ai8` recipe) calibrated on 64 real garage images with training-time preprocessing. (The litert-torch PT2E path hit a converter layout-pass bug; quantizing the float flatbuffer with ai-edge-quantizer is the robust route.)
  - Built-in parity checks. Current lite0 numbers: float p99 score Δ=0.0004; int8 p99 score Δ=0.0026, top-20 anchor overlap 15/20, top-100 box Δ≤0.08.
- [x] Desktop x86 single-thread sanity benchmark: our int8 lite0@320 **32 ms** vs baseline lite2@448 **97 ms** (~3×); int8 model 4.6 MB vs 7.6 MB. (ARM/NPU numbers are what actually matter — below.)

## Remaining

- [ ] **Post-process contract** (talk to FW owner): baseline ends in `TFLite_Detection_PostProcess`; our export stops at pre-NMS tensors + anchors. Either graft the custom op (automl-style, needs anchor constants baked in) for a byte-compatible drop-in, or firmware decodes raw outputs. Anchor generation lives in `effdet/anchors.py` (`Anchors.boxes`).
- [ ] **Input quantization contract**: ours is int8 `(scale≈0.02, asymmetric per-tensor from ImageNet-normalized floats)`; baseline is uint8 `(1/128, 127)`. If the FW feeds raw camera bytes, retrain/export with TF-style `x/128-1` normalization or bake the transform in.
- [ ] **On-device verification**: run on CM3+ (XNNPACK, 1 thread) and both NPUs; check full delegation on i.MX8M Plus (VX delegate) and run the eIQ Neutron converter for i.MX95; record per-tier latency table.
- [ ] Re-export after the pending retrain (track 05) — current artifacts are from the label-broken checkpoint (see 02) and are for pipeline validation only.
