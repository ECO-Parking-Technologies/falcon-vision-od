# Falcon Vision Object Detection (EfficientDet)

This repository is a fork of [`rwightman/efficientdet-pytorch`](https://github.com/rwightman/efficientdet-pytorch), adapted for the **Falcon Vision** parking guidance system. It trains and fine-tunes custom EfficientDet models on Falcon Vision camera data to replace the off-the-shelf detection model currently running in the sensor firmware with one that is **faster and more accurate** — particularly at detecting vehicles in monitored parking spots.

---

## Goals & Deployment Targets

The sensor firmware already runs an off-the-shelf **EfficientDet-Lite2** (448×448, full-int8 TFLite). This repo exists to beat it. The deployment format is **full-int8 quantized TFLite (LiteRT)**, which serves all three sensor hardware tiers:

| Tier | Hardware | Inference path |
|------|----------|----------------|
| Low  | Raspberry Pi CM3+ (quad Cortex-A53) | TFLite int8 on CPU (2.6 runtime, no XNNPACK in current FW) |
| Mid  | [DART-MX8M-PLUS](https://variscite.com/system-on-module-som/i-mx-8/i-mx-8m-plus/dart-mx8m-plus/) (i.MX8M Plus) | TFLite int8 on 2.3 TOPS NPU (VX delegate / eIQ) |
| High | [DART-MX95](https://variscite.com/system-on-module-som/i-mx-9/i-mx-95/dart-mx95/) (i.MX95) | TFLite int8 on eIQ Neutron NPU |

Different model scales (input resolution / variant) can be targeted per tier from the same training pipeline.

See [docs/road-map/](docs/road-map/) for the full research roadmap (model selection, compute budget, data preparation, training, evaluation, deployment), [docs/planning/](docs/planning/) for the current working plan of remaining tasks, and [docs/portal-api/](docs/portal-api/) for the ECO Parking portal GraphQL API reference.

## Pipeline Overview (SAM 3 distillation)

```
Portal snapshot pull (all garages) → SAM 3 preannotation (drafts EVERY frame)
      ├→ CVAT audit subset (humans grade + correct — gold eval, not volume labeling)
      └→ Training on drafts (audited exports override them) → Checkpoint
            → Auto: per-class metrics + dashboard + TFLite variants
              + native-res drop-in packages (dynamic / f32 / int8) → sensor file swap
```

- **Classes** ([config/label_map.yaml](config/label_map.yaml)): person, bicycle, car, motorcycle, bus, truck (COCO ids, remapped to contiguous ids for training).
- **Data root**: `data/images` (symlink to the unified store) — `<garage>/<sensor>/<YYYY>/<MM>/*.jpg` + per-sensor `preannotations.coco.json` drafts.
- **Docs**: [docs/training-and-experiments.md](docs/training-and-experiments.md) (experiment machinery, results, packaging + latency), [preannotation/README.md](preannotation/README.md) (SAM 3 setup incl. gated-weights access), [cvat/README.md](cvat/README.md) (annotation infra).

---

## From scratch — the full path

### 1. Environments

```bash
bash setup_venv.sh              # main venv (training/eval/preannotation) — recreates each run
bash setup_export_venv.sh       # export venv (TFLite toolchain)
bash setup_convert_venvs.sh     # clean-converter venvs (onnx2tf + TF 2.8, on the data disk)
./setup_backup_service.sh       # OPTIONAL: nightly unattended backup (systemd user timer)
```

The backup service rsyncs all irreplaceable untracked data (runs/, images,
manifests) to a NAS — and optionally Azure Blob — nightly with retries, under a
per-machine subdir (`<hostname>-<machine-id8>`). One-off manual backups:
`./backup_valuables.sh <dest>`. Restoring onto a fresh machine:
[docs/disaster-recovery.md](docs/disaster-recovery.md).

### 2. Data (portal pulls — credentials prompted, RAM only)

```bash
python3 portal/pull_training_images.py --source b   # garage frames -> data/images/<garage>/<sensor>/…
python3 portal/pull_spot_polygons.py                # per-run spot calibrations -> data/spot_polygons.json
```

### 3. Preannotation (SAM 3 — the teacher)

One-time: accept the gated-weights license (see [preannotation/README.md](preannotation/README.md)).

```bash
cd preannotation && PYTHONPATH=.. python3 run_preannotation.py --config config.yaml --all-frames --skip-existing
cd .. && python3 preannotation/label_inspot.py      # stamp InEcoParkingSpot/spot attributes
python3 preannotation/render_drafts.py              # visual QA -> data/draft_previews/ (local only)
```

Resumable and incremental — rerun after any new portal pull; only new frames are drafted.

### 4. CVAT (optional — audits are on-demand, not required for training)

Self-hosted setup: [cvat/README.md](cvat/README.md). Then:

```bash
PYTHONPATH=preannotation python3 preannotation/export_cvat_tasks.py    # per-garage bundles
python3 cvat/create_tasks.py --host http://<host>:8085 --project "Falcon Vision v2"
# cvat/purge_tasks.py wipes tasks (keeps project/labels/users) before a re-import
```

Labels + attributes sync automatically from [config/cvat_labels.json](config/cvat_labels.json).
Audited exports saved to `data/cvat_exports/` override the drafts in every training build.

### 5. Training

```bash
python3 run_training_from_config.py --config config/train_sam3_full.yaml
```

The config's `levels:` list drives the session — every listed level (or
variant entry, e.g. `{name: lite1-coco, model: lite1, include_coco: true}`)
trains into one datetime session dir with a shared split.

Every run auto-produces: per-class metrics (`coco_metrics.json`, `run.json`),
TensorBoard logs, a refreshed dashboard (`runs/report.html`),
and packaged sensor artifacts. Read **per-class car/person AP, never the 6-class mean**.

### The `runs/` directory

One launch = one datetime session; everything a session produced lives under it:

```
runs/
├── 20260810-090000/            ← a session (e.g. levels: [lite0 … d2])
│   ├── split/                    the data split — built once, shared by every level
│   ├── lite0/
│   │   ├── train/               checkpoints, tb/, summary.csv, run.json, metrics
│   │   └── export/              dropin-<size>-{f32,dyn,int8}.tflite, raw-*, ts.pt, manifest
│   ├── lite1/ … d2/             one dir per level (variants: lite1-coco/, lite0-25k/,
│   │                            which carry their own local split/)
│   └── report.html              this session's report
├── latest-<level>/              symlink to that level's newest export/
├── report.html                  global dashboard (all sessions)
└── ladder.html                  architecture comparison
```

### 6. Evaluation

```bash
python3 run_metrics.py <train-run-dir>       # per-class + size-banded AP
python3 eval_inspot.py <train-run-dir>       # THE product metric: in-spot car AP
falcon-vision-od-export-venv/bin/python eval_tflite_coco.py <root> out.json <model.tflite> ours
```

Dashboards: `report.html` (all runs), `ladder.html` (architecture comparison).

### 7. Export & packaging

Automatic after training; manual for any checkpoint (export venv):

```bash
PYTHONPATH=. falcon-vision-od-export-venv/bin/python package_dropin.py \
    --checkpoint <ckpt> --model <effdet-name> --input-size <native>
```

Artifacts per run — size and quantization always explicit:

```
<run>.dropin-<size>-{f32,dyn,int8}.tflite   # sensor-ready, validated vs the baseline
<run>.raw-{f32,dyn,int8}.tflite             # dev/eval only
<run>.ts.pt · manifest.json                 # TorchScript + provenance
```

Conversion is the clean path (torch → ONNX → onnx2tf; [clean_convert.py](clean_convert.py)).
Native sizes: lite0=320 · lite1=384 · lite2=448 · lite3=512 · lite4/d1=640 · d2=768 —
**never ship a non-native size** (2× latency).

### 8. Deploy

Upload the dropin via the sensor's `/plugin/od-model/staging/upload` → `install`;
verify sha256 + Tensor Size on the Fusion Analysis page. Contract details:
[docs/sensor-architecture.md](docs/sensor-architecture.md).

---

## Based On

- [EfficientDet PyTorch (rwightman)](https://github.com/rwightman/efficientdet-pytorch)
- [Official EfficientDet (TensorFlow)](https://github.com/google/automl)
- [EfficientDet Paper: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

## Planned

- Spot-occupancy evaluator (detections + spot polygons + portal validations → per-spot accuracy vs the firmware model).
- Round-2 accuracy experiments on lite2 (spot-weighted loss, ROI-crop training, EMA/schedule) gated on `eval_inspot.py`.
- Photometric augmentation (night/glare/white-balance) — round 1.
- Anchor box tuning from garage box-size distributions.
