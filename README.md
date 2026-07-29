# Falcon Vision Object Detection (EfficientDet)

This repository is a fork of [`rwightman/efficientdet-pytorch`](https://github.com/rwightman/efficientdet-pytorch), adapted for the **Falcon Vision** parking guidance system. It trains and fine-tunes custom EfficientDet models on Falcon Vision camera data to replace the off-the-shelf detection model currently running in the sensor firmware with one that is **faster and more accurate** — particularly at detecting vehicles in monitored parking spots.

---

## Goals & Deployment Targets

The sensor firmware already runs an off-the-shelf **EfficientDet-Lite2** (448×448, full-int8 TFLite). This repo exists to beat it. The deployment format is **full-int8 quantized TFLite (LiteRT)**, which serves all three sensor hardware tiers:

| Tier | Hardware | Inference path |
|------|----------|----------------|
| Low  | Raspberry Pi CM3+ (quad Cortex-A53) | TFLite int8 on CPU (XNNPACK) |
| Mid  | [DART-MX8M-PLUS](https://variscite.com/system-on-module-som/i-mx-8/i-mx-8m-plus/dart-mx8m-plus/) (i.MX8M Plus) | TFLite int8 on 2.3 TOPS NPU (VX delegate / eIQ) |
| High | [DART-MX95](https://variscite.com/system-on-module-som/i-mx-9/i-mx-95/dart-mx95/) (i.MX95) | TFLite int8 on eIQ Neutron NPU |

Different model scales (input resolution / variant) can be targeted per tier from the same training pipeline.

See [docs/road-map/](docs/road-map/) for the full research roadmap (model selection, compute budget, data preparation, training, evaluation, deployment), [docs/planning/](docs/planning/) for the current working plan of remaining tasks, and [docs/portal-api/](docs/portal-api/) for the ECO Parking portal GraphQL API reference.

## Pipeline Overview

```
Portal snapshot pull (all garages) → Preannotation → CVAT (human confirm)
      → COCO export → Training → Checkpoint
      → Auto-export (TFLite variants + drop-in package) → Sensor firmware (file swap)
```

- **Classes** ([config/label_map.yaml](config/label_map.yaml)): person, bicycle, car, motorcycle, bus, truck (COCO ids, remapped to contiguous ids for training).
- **Data root**: `/media/lopezemi/Expansion/falcon-vision-ml/artifacts/data_pipeline`, laid out as `<garage>/training_images/<sensor>/<sensor>-<camera>-<date>-<time>-<nn>.png`.

---

## Installation

```bash
bash setup_venv.sh                        # creates falcon-vision-od-venv from requirements.txt
source falcon-vision-od-venv/bin/activate # activate before running anything
```

Note: `setup_venv.sh` deletes and recreates the venv each run.

---

## Training Images (preliminary step)

Pull diverse snapshot images from every garage via the ECO Parking portal
(garages auto-discovered; credentials prompted, kept in RAM only; already-pulled
images are never downloaded twice):

```bash
python3 portal/pull_training_images.py --source b
```

Images land in `data/images/<garage>/<sensor>/…` with a manifest. See
[portal/README.md](portal/README.md) for options.

## Preannotation

Sample a diverse annotation queue from the store, then generate first-draft
boxes with Grounding DINO (garages auto-discovered):

```bash
python3 preannotation/sample_queue.py --target 5000
```


Runs a model over garage images and writes COCO 1.0 annotation files per sensor for CVAT import:

```bash
python3 preannotation/run_preannotation.py --config preannotation/config.yaml --visualize 3
```

- Configured by [preannotation/config.yaml](preannotation/config.yaml): model, garages, confidence `threshold`, `allowed_labels`.
- Uses either downloaded COCO-pretrained weights (`use_pretrained_model: True`) or a trained TorchScript model from this repo (`model_file`).
- Outputs `preannotations.coco.json` per sensor plus `cvat_labels.json` for CVAT label setup.

## Annotation (CVAT)

CVAT is the source of truth for annotations, including train/val/test splits. See [cvat/README.md](cvat/README.md) for self-hosted setup (v2.23.1), task structure (one task per garage+sensor), importing preannotations, and exporting COCO for training.

## Training

```bash
python3 run_training_from_config.py --config config/train_wrapper_config.yaml
```

Configured by [config/train_wrapper_config.yaml](config/train_wrapper_config.yaml). This:

1. Merges CVAT-exported garage annotations, splits train/val, and symlinks images into a timestamped `split_*` dir under `output_dir`.
2. Downloads MS-COCO 2017 (~20 GB, once), filters it to our classes, and merges it into the split.
3. Invokes the upstream `train.py` in-process with the configured model, batch size, epochs, etc.

Outputs land in `experiments/falcon-vision-effdet/train/<timestamp>-<model>/` (checkpoints, `summary.csv` with per-epoch loss/mAP).

## Export

```bash
python3 generate_model_files.py            # newest best checkpoint → TorchScript .pt (for preannotation)

bash setup_export_venv.sh                  # once: separate venv for the TFLite toolchain
PYTHONPATH=. falcon-vision-od-export-venv/bin/python export_tflite.py
```

[export_tflite.py](export_tflite.py) exports a checkpoint to float32 and **full-int8 TFLite** (litert-torch conversion + ai-edge-quantizer static PTQ calibrated on real garage images), with built-in PyTorch↔TFLite parity checks. Input is NHWC like the firmware baseline; outputs are pre-NMS boxes/scores (post-process contract tracked in [docs/planning/03-tflite-export.md](docs/planning/03-tflite-export.md)).

Both exporters accept `--model` / `--checkpoint` and write versioned artifacts (see [artifact_paths.py](artifact_paths.py)):

```
<output_dir>/artifacts/<model>/<train-run>/
    model.ts.pt  model.f32.tflite  model.int8.tflite  manifest.json
<output_dir>/artifacts/<model>/latest -> <train-run>/   # stable path for configs
```

`manifest.json` records checkpoint provenance, input/output contract, quantization recipe, parity metrics, and git commit per artifact. Off-the-shelf COCO checkpoints downloaded for evaluation live in `weights/` (gitignored).

---

## Based On

- [EfficientDet PyTorch (rwightman)](https://github.com/rwightman/efficientdet-pytorch)
- [Official EfficientDet (TensorFlow)](https://github.com/google/automl)
- [EfficientDet Paper: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

## Planned

- int8 TFLite export path (ai-edge-torch) matching the firmware's model contract.
- Eco Parking portal integration to pull training images from all garages.
- Spot-occupancy evaluation (detections + spot definitions → per-spot accuracy vs. the current firmware model).
- Anchor box tuning from garage box-size distributions.
