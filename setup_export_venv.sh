#!/usr/bin/env bash
# Separate venv for the int8 TFLite export toolchain (ai-edge-torch pulls in
# TensorFlow and its own torch pin — keep it away from the training venv).
set -e

VENV_NAME="falcon-vision-od-export-venv"

rm -rf "$VENV_NAME"
python3 -m venv "$VENV_NAME"
"$VENV_NAME/bin/python" -m pip install --upgrade pip wheel

# CPU torch is enough for export and avoids multi-GB CUDA wheels
"$VENV_NAME/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$VENV_NAME/bin/python" -m pip install ai-edge-torch ai-edge-litert timm pyyaml opencv-python-headless numpy omegaconf pycocotools

echo "Export venv ready: source $VENV_NAME/bin/activate"
