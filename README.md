# Falcon Vision Object Detection (EfficientDet)

This repository is a fork of [`rwightman/efficientdet-pytorch`](https://github.com/rwightman/efficientdet-pytorch), adapted for use in the **Falcon Vision** parking guidance system.  
We are transitioning from image classification to full **object detection** using **EfficientDet** for better accuracy, scalability, and real-time performance on embedded devices.

---

## Purpose

- Detect vehicles and pedestrians in real-time on embedded hardware.
- Determine parking space occupancy by matching detections to parking zone regions.
- Train and fine-tune custom EfficientDet models on Falcon Vision camera data.
- Automate annotation with pre-annotation models to save time.

---

## Features

- Pre-annotation pipeline using EfficientDet variants (Lite0–Lite4, D0–D7).
- Fully configurable detection classes and thresholds.
- Automatic model downloading (from HuggingFace/Google Checkpoints).
- MS COCO 1.0 dataset export compatible with CVAT, FiftyOne, ClearML, etc.
- GPU acceleration supported for fast inference (requires NVIDIA drivers + PyTorch CUDA install).
- Modular config (`config.yaml`) to easily switch models and parameters.
- Optimized for future edge deployment (TFLite/ONNX export planned).

---

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/falcon-vision-od.git
cd falcon-vision-od
```

### 2. Set Up Python Virtual Environment

Use the provided setup script:

```bash
bash setup_venv.sh
```

This will:
- Create a new virtual environment `falcon-vision-od-venv`
- Install all required dependencies from `requirements.txt`

✅ After running the script, **activate the virtual environment** every time you start a new terminal session:

```bash
source falcon-vision-od-venv/bin/activate
```

> **Important:** Always activate the venv before running Python commands!

---

## Preannotation Pipeline

You can pre-annotate your dataset automatically using pretrained EfficientDet models.

**Command to run:**

```bash
python3 preannotation/run_preannotation.py --config preannotation/config.yaml --visualize 3
```

**What this does:**
- Downloads and loads the correct EfficientDet model automatically.
- Runs inference over all images under your configured data path.
- Filters detections based on allowed labels and thresholds.
- Outputs COCO 1.0 formatted annotation files per sensor.
- Generates a `labels.json` to assist with CVAT label setup.

---

## Configuration (`config.yaml`)

Key fields:

- `model_type`: EfficientDet model variant (e.g., `tf_efficientdet_d3_ap`).
- `efficientdet_models`: Lookup table for input size and download URL.
- `allowed_labels`: Only export these labels (e.g., `car`, `person`).
- `threshold`: Confidence threshold for filtering weak detections.
- `garages`: List of garages to pre-annotate.

---

## Annotation Setup (CVAT)

We use **CVAT** to manage image annotations for Falcon Vision.

- See [cvat/README.md](cvat/README.md) for complete instructions on:
  - Setting up CVAT
  - Project and task creation (one task per sensor)
  - Running pre-annotation using EfficientDet
  - Importing pre-annotations into tasks
  - Finalizing and exporting annotations for training


---

## Based On

- [Official EfficientDet (TensorFlow)](https://github.com/google/automl)
- [EfficientDet Paper: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- [EfficientDet PyTorch (rwightman)](https://github.com/rwightman/efficientdet-pytorch)

---

## Coming Soon (Planned)

- Fine-tuning scripts with PyTorch Lightning.
- Dataset management using FiftyOne.
- Automatic export to ONNX and TFLite for embedded deployment.

---
