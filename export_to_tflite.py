#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Step 0) Stub out tensorflow_probability before any TF-P imports
import sys, types

# Create a fake tensorflow_probability module
_tfp = types.ModuleType("tensorflow_probability")
_tfp.distributions = types.ModuleType("tensorflow_probability.distributions")
# Bernoulli is only ever imported in onnx_tf.handlers.backend.bernoulli
setattr(_tfp.distributions, "Bernoulli", lambda *args, **kwargs: None)
sys.modules["tensorflow_probability"] = _tfp
sys.modules["tensorflow_probability.distributions"] = _tfp.distributions

# ──────────────────────────────────────────────────────────────────────────────
# Step 1) Patch torchvision NMS to pure-tensor “coordinate trick”
import torchvision.ops.boxes as _bops
_bops.batched_nms = _bops._batched_nms_coordinate_trick

# ──────────────────────────────────────────────────────────────────────────────
import os
import yaml
import torch
import onnx
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

# Now safe to import onnx_tf
#from onnx_tf.backend import prepare

from effdet import create_model
from effdet.config.model_config import efficientdet_model_param_dict as MODEL_CONFIG
from config.label_loader import load_label_map


def load_cfg():
    cfg_p = Path(__file__).parent / "config" / "train_wrapper_config.yaml"
    if not cfg_p.is_file():
        sys.exit(f"Config not found: {cfg_p}")
    return yaml.safe_load(cfg_p.read_text())


def find_ckpt(d: Path) -> Path:
    ckpts = list(d.rglob("model_best.pth.tar"))
    if not ckpts:
        sys.exit(f"No checkpoint found under {d}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def rep_data_gen(dir: Path, size: tuple[int,int], maxs=100):
    imgs = sorted(p for p in dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    for img in imgs[:maxs]:
        im = Image.open(img).convert("RGB").resize(size, Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
        yield [arr[np.newaxis, ...]]


def main():
    cfg         = load_cfg()
    out_dir     = Path(cfg["output_dir"])
    model_name  = cfg["model"]
    num_classes = cfg.get("num_classes", len(load_label_map()))
    rep_dir     = cfg.get("rep_images_dir", None)

    if model_name not in MODEL_CONFIG:
        sys.exit(f"Unknown model: {model_name}")

    H, W = MODEL_CONFIG[model_name]["image_size"]
    print(f"🔍 Exporting {model_name} ({H}×{W}), classes={num_classes}")

    # 1) Locate & load the .tar checkpoint
    ckpt = find_ckpt(out_dir)
    print("✔️  Loading checkpoint:", ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)

    # 2) Extract & save just the state_dict as a .pt file ← NEW
    sd = raw["state_dict"]
    pt_path = out_dir / f"{model_name}.pt"
    print("💾 Saving raw state_dict to:", pt_path)
    torch.save(sd, pt_path)

    # 3) Build your model and load that .pt back in
    model = create_model(model_name, bench_task="predict",
                         num_classes=num_classes, pretrained=False)
    # if your keys are namespaced under "model.", strip it here
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 4) JIT-trace to eliminate Python control‐flow
    dummy  = torch.randn(1, 3, H, W)
    traced = torch.jit.trace(model, dummy, strict=False)

    # 5) Export to ONNX
    onnx_p = out_dir / f"{model_name}.onnx"
    print("🚀 Exporting ONNX →", onnx_p)
    torch.onnx.export(
        traced, dummy, str(onnx_p),
        opset_version=11,
        input_names=["input"],
        output_names=["detections"],
        dynamic_axes={"input": {0: "batch"}, "detections": {0: "batch"}},
    )

    # 6) Convert ONNX → TensorFlow SavedModel (in-process)
    tf_saved = out_dir / "tf_saved_model"
    print("🔄 Converting ONNX → SavedModel via onnx-tf API →", tf_saved)
    shutil.rmtree(tf_saved, ignore_errors=True)
    onnx_model = onnx.load(str(onnx_p))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_saved))

    # 7) Convert SavedModel → TFLite
    tflite_p = out_dir / f"{model_name}.tflite"
    print("🔄 Converting SavedModel → TFLite →", tflite_p)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if rep_dir:
        rd = Path(rep_dir)
        if not rd.is_dir():
            sys.exit(f"rep_images_dir not found: {rd}")
        converter.representative_dataset    = lambda: rep_data_gen(rd, (H, W))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type      = tf.uint8
        converter.inference_output_type     = tf.uint8
    else:
        print("⚠️  No rep_images_dir; using dynamic-range quant")

    tflite_model = converter.convert()
    tflite_p.write_bytes(tflite_model)
    print("✅ Done. TFLite model at:", tflite_p)


if __name__ == "__main__":
    main()
