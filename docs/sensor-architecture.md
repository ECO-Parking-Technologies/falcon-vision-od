# Sensor Architecture Reference — How falcon-vision-sensor Uses Model Files

Findings from exploring `falcon-vision-sensor` (2026-07-28, HEAD ~`2813f3f2`). This is the
firmware our models deploy into; every claim below has file:line grounding in that repo.

## Platform

- C++17, CMake, cross-compiled via Buildroot for **Raspberry Pi CM3+** (armhf, Cortex-A53, `arm_freq=600` MHz).
- Processes under `Eco-AppMonitor`: `Eco-CameraCtrl` (capture + ML), `Eco-GatewayProxy`, `Eco-LightingManager`, `Eco-WebServer` (civetweb).
- Camera: MMAL capture, analysis frames 640×480 @ ~2 Hz effective (`normalModeSpotDetectorScheduleTickMs=500`).

## The two models

**Classifier (primary)** — `spot_detector_dnn.tflite` (built-in) or installed `custom.tflite`:
- One engine per parking space; spot polygon crop (homography `WarpToBoundingBox` for non-rectangles), **plain stretch resize**, BGR→RGB; uint8 raw or float `x/127.5−1`; single scalar occupancy output.
- Multiple preprocess runners (raw/CLAHE/HSV-eq/bilateral) blended only when the raw score is ambiguous; result feeds CUSUM/adaptive scoring.
- Pool of 2 single-threaded interpreters (`inferencePoolSize=2`, RT_FIFO workers).

**Object detection** — installed `od_model.tflite` (no built-in fallback; today = off-the-shelf efficientdet-lite2):
- `Model/AnalyzerParkingTensorFlowLiteOdModel.cpp`. `SetNumThreads(2)`; XNNPACK code exists but is compiled out (`USE_XNNPACK` never defined).
- Runs NOT on the full frame: a per-camera **ROI crop** = bounding box of all spot polygons + padding (`padTop/Bottom=0.1` defaults, optional expand-to-tensor-width/aspect). Optional CLAHE.
- Duty-cycled, self-pacing: after each pass, idle `elapsed×(1−0.40)/0.40`, floor 1 s (`dutyCycle=0.40`). Round-robin over cameras.

## The model-file contract (what our exports must satisfy)

1. **Input size: read from the model tensor** (config dims default 0 → "Trusting model input tensor dimensions"). Nothing hardcodes 448 → shipping lite0 at native 320 needs **no firmware change**.
2. **Input dtype**: uint8 (raw byte memcpy, no normalization — our drop-in's in-graph normalization is compatible) or float32 (`pixel/127.5 − 1`).
3. **Preprocessing: LETTERBOX** — aspect-preserving resize (INTER_LINEAR), centered on a black canvas, BGR→RGB. ⚠️ Our training/eval/calibration stretch-resize → distribution mismatch to fix on our side (see planning 05).
4. **Outputs parsed by tensor index**, two accepted layouts:
   - 4 tensors (`TFLite_Detection_PostProcess`): boxes `[1,N,4]` normalized ymin,xmin,ymax,xmax / classes (float→int) / scores / count — what our `.dropin.tflite` emits;
   - OR a single `[1,N,6]` tensor `[ymin,xmin,ymax,xmax,class,score]` (alternative contract, no custom op needed).
   Boxes are treated as normalized to the letterboxed canvas and un-letterboxed in `Extract()`.
5. **Class mapping hardcoded** (0-based COCO): `{0:PERSON, 1:BICYCLE, 2:CAR, 3:MOTORCYCLE, 4:AIRPLANE, 5:BUS, 7:TRUCK}` → PERSON/VEHICLE types; `forceUnknownToVehicle=true`. Our 6-class indices (0..5 = person,bicycle,car,moto,bus,truck) all resolve to the correct PERSON/VEHICLE type — bus/truck subtype labels are cosmetically wrong (airplane/bus) but occupancy logic only uses the type.
6. **Thresholds**: model `minConfidence=0.25`, analyzer `0.35`, fusion "strong OD" `0.40`; `maxDetections=10` (config) caps the model's 25.
7. **Runtime: TFLite 2.6.0** (C++, Bazel-built, BuiltinOpResolver — includes the `TFLite_Detection_PostProcess` custom op). Registered op versions comfortably cover our exports (e.g. RESIZE_NEAREST_NEIGHBOR max v4 vs our v3; CONV_2D max v4). **int8 quantized models are within 2.6's capabilities on paper** — verify by on-device load, but "the runtime can't do int8" is not supported by its source.

## OD → occupancy decisions

- Per spot: polygon containment math (inscribed-ellipse), not IoU — `objectOverlap`/`spaceOverlap` ≥ 0.50/0.50 → `VEHICLE_IN_SPOT`; straddle 0.20/0.20; hysteresis on every threshold; 2-frame track maturity; space-aware NMS; known-background bbox filter (IoU 0.65).
- Modes `INFERENCE` (classifier only, default) / `OBJECT_DETECT` / `FUSION`. `ConfirmFusion` v2: OD ≥0.40 can override classifier NO_DETECT; ambiguous classifier band resolved by consecutive OD presence/absence; shadow-gating for gradual CUSUM rises.

## Model deployment (no firmware changes needed to swap models)

- Paths: built-ins `/etc/opt/eco-sensor/computer-vision/`; installed `/var/opt/eco-sensor/computer-vision/{custom.tflite, od_model.tflite}`.
- **HTTP endpoints** (Eco-WebServer): `/plugin/od-model/staging/upload` → `/plugin/od-model/install` (and `/uninstall`, `/info` with sha256+size). Staged at `/tmp/od_model`, size limit 1 KB–10 MB, atomic rename install + `.sha256` sidecar, engines auto-rebuild on install. Web UI: `od-model.shtml`.
- Second channel: gateway-driven download (`ObjectModelManager` "odmodel", `/tmp/od_model_downloads`).
- Firmware OTA (A/B `image.tar`) is separate; models are not part of it (except the built-in classifier).

## Training images & snapshots on the sensor (feeds our data pipeline)

- Training mode writes analysis frames as PNG every `imageCaptureIvalSec` to tmpfs, transferred to `/media/flash/images/training` (backup `/media/sdcard/...`), layout `camera-<N>/<YYYY>/<M>/<D>/<H>/` (non-zero-padded), pruned by storage manager.
- Served by `Eco-WebServer`: `GET /training-image/files/{y}/{m}/{d}/{h}` (JSON list) and `/training-image/image/<file>.png`; also `/training-image/times`, `/coverage`.
- Snapshots: `/plugin/snapshot/create|get/{uuid}|list|delete`, stored `/media/flash/snapshots` (max 16, JPEG q90 + per-space PNGs).

## Implications for falcon-vision-od (tracked in docs/planning/)

- Drop-in `.dropin.tflite` deploys with zero firmware changes (upload API). ✅ validated against contract items 1–7.
- Next accuracy wins on our side: letterbox + ROI-crop alignment in training/eval (05), native-320 drop-in build (03), on-device int8 load test (03).
- Real-device latency: 600 MHz, 2 threads, no XNNPACK — expect much slower than desktop benches; relative lite0-vs-lite2 advantage should hold or grow.
