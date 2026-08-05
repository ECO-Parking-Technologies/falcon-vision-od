#!/usr/bin/env python3
"""Clean TFLite conversion pipeline (the default since 2026-08-05).

torch (NCHW wrapper) -> ONNX (opset 17) -> onnx2tf (NHWC-native graph,
~1 transpose vs ~107 from the legacy litert-torch path) -> quantize:
  - f32:  onnx2tf's float32 output, as-is
  - dyn:  ai-edge-quantizer dynamic_wi8_afp32 (int8 weights, f32 activations)
  - int8: TF-2.8 TFLiteConverter static PTQ on the saved_model — fused int8
          chain + era-appropriate op versions for the sensor's 2.6 runtime
          (measured better than AEQ static: +1.3 car AP, 24->2 dequant ops)

Runs inside the EXPORT venv; onnx2tf and the TF-2.8 quantizer live in their
own venvs (see setup_convert_venvs.sh) and are invoked as subprocesses.

On-device context (CM3 bench, 2026-08-05): clean export buys ~6-19% vs the
legacy path and matches Google's own export op-for-op on the conv chain; the
residual lite2 gap vs the off-the-shelf model is architectural (weighted
BiFPN fusion SUM chains + unfused RELU6) — see docs/training-and-experiments.md.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_ROOT = Path("/media/lopezemi/Expansion/falcon-vision-od-data")
ONNX2TF_VENV = DATA_ROOT / "onnx2tf-venv"
TFQ_VENV = DATA_ROOT / "tf28-venv"

_TFQ_SCRIPT = r"""
import sys, glob, random
import numpy as np, tensorflow as tf, cv2
saved_model, size, calib_root, count, out = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
conv = tf.lite.TFLiteConverter.from_saved_model(saved_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
files = sorted(glob.glob(calib_root + "/*/*/*/*/*.jpg")) or \
        sorted(glob.glob(calib_root + "/*/training_images/*/*.png"))
random.Random(0).shuffle(files)
def rep():
    n = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
        yield [rgb[None].transpose(0, 3, 1, 2).astype(np.float32)]
        n += 1
        if n >= count:
            break
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
open(out, "wb").write(conv.convert())
print("tfq int8 ->", out)
"""


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"[clean_convert] FAILED: {' '.join(str(c) for c in cmd[:3])}…\n"
                 f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


def check_venvs():
    for v in (ONNX2TF_VENV, TFQ_VENV):
        if not (v / "bin" / "python").exists():
            sys.exit(f"[clean_convert] missing venv {v} — run setup_convert_venvs.sh "
                     "(or use --legacy-export)")


def to_onnx(wrapper, size, out_path):
    """NCHW wrapper -> ONNX. Caller (export venv) has torch."""
    import torch
    x = torch.zeros(1, 3, size, size)
    torch.onnx.export(wrapper, (x,), str(out_path),
                      input_names=["images"], output_names=["boxes", "scores"],
                      opset_version=17, do_constant_folding=True)


def onnx_to_tf(onnx_path, out_dir):
    """onnx2tf conversion. -kat keeps the NCHW input untouched (the NHWC input
    path both mangles the layout and trips an onnx2tf crash); the packaging
    surgery bridges uint8-NHWC -> NCHW with one input-sized transpose.
    Returns (saved_model_dir, float32_tflite_path)."""
    env = {"PATH": f"{ONNX2TF_VENV}/bin:/usr/bin:/bin"}
    _run([str(ONNX2TF_VENV / "bin" / "python"), "-m", "onnx2tf",
          "-i", str(onnx_path), "-o", str(out_dir), "-kat", "images"], env=env)
    f32 = next(Path(out_dir).glob("*_float32.tflite"))
    return Path(out_dir), f32


def quantize_int8(saved_model_dir, size, calib_root, calib_count, out_path):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_TFQ_SCRIPT)
        script = f.name
    _run([str(TFQ_VENV / "bin" / "python"), script, str(saved_model_dir),
          str(size), str(calib_root), str(calib_count), str(out_path)])
    Path(script).unlink()


def quantize_dynamic(f32_path, out_path):
    from ai_edge_quantizer import quantizer as aeq_quantizer
    from ai_edge_quantizer import recipe as aeq_recipe
    qt = aeq_quantizer.Quantizer(str(f32_path), aeq_recipe.dynamic_wi8_afp32())
    qt.quantize(serialize_to_path=str(out_path))
