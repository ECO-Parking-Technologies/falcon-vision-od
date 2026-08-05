#!/usr/bin/env bash
# Toolchain venvs for the clean TFLite export path (clean_convert.py):
#   onnx2tf-venv : ONNX -> NHWC-native TFLite/saved_model conversion
#   tf28-venv    : TF 2.8.4 — int8 static PTQ with sensor-era op versions,
#                  doubles as the old-runtime (2.6-proxy) compatibility check
# Both live on the data disk (large, not repo material).
set -euo pipefail
ROOT=/media/lopezemi/Expansion/falcon-vision-od-data

if [ ! -x "$ROOT/onnx2tf-venv/bin/python" ]; then
  python3 -m venv "$ROOT/onnx2tf-venv"
  "$ROOT/onnx2tf-venv/bin/pip" install --no-cache-dir \
    onnx2tf onnx onnx_graphsurgeon sng4onnx onnxsim "tensorflow>=2.15" ai-edge-litert
fi

if [ ! -x "$ROOT/tf28-venv/bin/python" ]; then
  python3 -m venv "$ROOT/tf28-venv"
  "$ROOT/tf28-venv/bin/pip" install --no-cache-dir \
    "tensorflow==2.8.4" "protobuf<3.20" numpy==1.23.5 "opencv-python-headless<4.8"
fi

# the export venv needs the ONNX exporter helpers
./falcon-vision-od-export-venv/bin/pip install --no-cache-dir -q onnx onnxscript
echo "convert venvs ready"
