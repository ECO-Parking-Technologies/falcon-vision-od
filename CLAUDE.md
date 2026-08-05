# CLAUDE.md — Falcon Vision OD operational brief

Fine-tuned EfficientDet object detectors for Eco Parking garage sensors,
replacing the off-the-shelf lite2 (car AP 26.7% on our yardstick). Core
approach: **SAM 3 distillation** — SAM 3 drafts boxes for every store frame,
students train on drafts, humans grade/correct rather than annotate volume.
Deep docs: [docs/training-and-experiments.md](docs/training-and-experiments.md)
(pipeline + results), [preannotation/README.md](preannotation/README.md),
[cvat/README.md](cvat/README.md), [docs/sensor-architecture.md](docs/sensor-architecture.md).

## Hard rules

- **`/media/lopezemi/Expansion/falcon-vision-ml` is READ-ONLY** — active prod
  classifier dataset. Copy from it, never modify/delete.
- **Credentials are RAM-only**: portal/CF/CVAT/HF secrets are prompted at
  runtime — never in files, args, env profiles, logs, or git.
- **CVAT stays behind Cloudflare Zero Trust** (customer CCTV + open
  self-registration). Never raw-public. Never publish store frames anywhere.
- **Never pip-install/upgrade in a venv while a long run uses it** (lazy
  imports crash mid-run — happened once).
- **Never change `split_salt`** in training configs — it would move the
  frozen val set and invalidate all cross-run comparisons.

## Environments

| venv | purpose |
|---|---|
| `falcon-vision-od-venv` | training, eval, preannotation (torch, transformers 5, tensorboard) |
| `falcon-vision-od-export-venv` | TFLite export/packaging (litert-torch, ai-edge-*, onnx) |
| `…-data/onnx2tf-venv` | clean converter (onnx2tf) — via `setup_convert_venvs.sh` |
| `…-data/tf28-venv` | TF 2.8.4: int8 PTQ + old-runtime (2.6-proxy) compat checks |

Data root: `data/` → `/media/lopezemi/Expansion/falcon-vision-od-data` (1.8T).
Store: `data/images/<garage>/<sensor>/<YYYY>/<MM>/*.jpg` + per-sensor
`preannotations.coco.json` (SAM3 drafts, 117k frames / 724k boxes, complete).
GPU: single RTX 3090 — check `nvidia-smi` + `pgrep -f "run_sweep|run_training|run_preannotation"`
before launching anything heavy.

## Pipeline & commands (repo root, main venv unless noted)

```bash
# SAM3 drafts (resumable, incremental; facebook/sam3 is GATED — prompts for
# HF token in-process on first download, cached after)
cd preannotation && PYTHONPATH=.. python3 run_preannotation.py --config config.yaml --all-frames --skip-existing

# training (adapter auto-discovers drafts; audited CVAT exports in
# data/cvat_exports/ OVERRIDE drafts per-frame automatically)
python3 run_training_from_config.py --config config/train_sam3_full.yaml

# size sweep / capacity ladder
python3 run_sweep.py --config config/train_sam3_full.yaml [--arm-b]
python3 run_sweep.py --config ... --models tf_efficientdet_lite1,tf_efficientdet_d2

# per-run metrics backfill · TFLite scoring · draft visualization
python3 run_metrics.py <train-run-dir>
falcon-vision-od-export-venv/bin/python eval_tflite_coco.py <eval_root> out.json <model.tflite> ours
python3 preannotation/render_drafts.py --garage <slug> --count 24

# packaging (EXPORT venv): clean path default, builds+validates f32/dyn/int8
PYTHONPATH=. falcon-vision-od-export-venv/bin/python package_dropin.py \
    --checkpoint <ckpt> --model <effdet-name> --input-size <native>
```

Auto after every training run: per-class metrics (`coco_metrics.json`,
`run.json`), dashboard refresh (`experiments/falcon-vision-effdet/report.html`),
TFLite exports + drop-in packages. Ladder visualization:
`experiments/falcon-vision-effdet/ladder.html`.

## Artifact naming (post-2026-08-05 — size/quant ALWAYS explicit)

`<run>.dropin-<size>-{f32,dyn,int8}.tflite` = sensor-ready (uint8 NHWC in,
TFLite_Detection_PostProcess out, validated vs `baseline/efficientdet_lite2.tflite`).
`<run>.raw-{f32,dyn,int8}.tflite` = dev/eval only. `<run>.ts.pt` = TorchScript
(full NMS, `torch.jit.load`). Native sizes: lite0=320 lite1=384 lite2=448
lite3=512 lite4/d1=640 d2=768. **Never ship a non-native size** (448 build of
a 320 model = exactly 2× latency; caused a real incident).

## Metrics discipline

- Garage data is ~94% car → **read per-class AP (car, person), never 6-class
  mean** (motorcycle/bus have no support; mean is meaningless).
- Val labels are SAM3's → sweep/ladder numbers are distillation fidelity.
  Honest accuracy = audited gold (gaprisco's CVAT exports) + eventually the
  spot-occupancy evaluator vs portal validations (not built yet).
- Frozen val = 5k frames via `split_by: sensor-hash`; identical across all
  runs since 2026-08-03. Person gate: prod baseline = 3.8% person AP.

## State (2026-08-05) — see auto-memory for full history

- **Deployment targets**: PRIMARY **lite1 dropin-384-int8** (3.1 s CM3, car
  53.9% = 2× prod); fallback lite0-320-int8 (1.6 s, 47.6%). lite2 (59.3%)
  needs the plain-sum-BiFPN retrain to fit ≤4 s (5.4 s today).
- **Ladder** (94,628 train imgs each): lite0 47.6 → lite1 53.9 → lite2 59.3 →
  lite3 60.0 → lite4 60.8 → d1 62.7 (car %). d1 = provisional FVS2/NPU pick.
  d2 pending. Person clears the 3.8% gate from lite2 up.
- **Latency lessons (CM3, 2.6 runtime)**: int8 ≈ 1.7× faster than f32;
  dynamic-range is SLOWER than f32 (hybrid-conv penalty); 2.6-era XNNPACK is
  float-only; clean export (onnx2tf, default since 86cdf5e) bought 6-19%;
  residual gap vs off-the-shelf is architectural (weighted-BiFPN SUM chains +
  unfused RELU6) — plain-sum retrain is the remaining lever.
- **Open threads**: lite4/d1 clean-int8 packaging fails (d-series SiLU vs TF2.8
  converter suspected; checkpoints fine per ts.pt) — d2 hits the same at
  auto-export; round 1 = photometric aug (night/glare/WB) + person fix;
  gaprisco mid-audit (~4k frames in CVAT, exports override drafts when they
  land); FVS2 (1-2 yr): up-to-4K + NPU — firmware spec must include native-res
  training-image capture + modern runtime.

## Sensor/firmware contract (details: docs/sensor-architecture.md)

CM3 @600MHz, TFLite **2.6**, 2 threads, no XNNPACK, **runs new-style full-int8**
(verified). FW reads input dims from the model file; feeds ROI crops,
letterboxed; parses 4-tensor postprocess by index; only PERSON-vs-VEHICLE
matters to fusion. Model swap: upload via sensor `/plugin/od-model/staging/upload`
→ `install`; verify sha256 + Tensor Size on the Fusion Analysis page.
