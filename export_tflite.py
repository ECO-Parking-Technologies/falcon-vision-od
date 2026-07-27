#!/usr/bin/env python3
"""Export a trained effdet checkpoint to TFLite (float32 + full-int8 PTQ).

Run inside the export venv (setup_export_venv.sh), NOT the training venv:

    PYTHONPATH=. falcon-vision-od-export-venv/bin/python export_tflite.py

Exports the raw detection network (backbone + BiFPN + heads, no NMS).
Outputs, mirroring effdet's DetBenchPredict pre-NMS tensors:
  - boxes:  [1, N, 4] anchor-relative box regressions (per-level concat)
  - scores: [1, N, num_classes] sigmoid class scores
Box decode + NMS happen downstream (firmware / TFLite_Detection_PostProcess
grafting is tracked in docs/planning/03-tflite-export.md).

int8 quantization is static PT2E, calibrated on real garage images using the
training-time preprocessing (resize, RGB, ImageNet mean/std over 0-255).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import litert_torch
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer as aeq_quantizer
from ai_edge_quantizer import recipe as aeq_recipe

from effdet import create_model
from effdet.config.model_config import efficientdet_model_param_dict as MODEL_CONFIG
from config.label_loader import load_label_map


class ExportWrapper(torch.nn.Module):
    """Raw EfficientDet -> (boxes [B,N,4], scores [B,N,C]), pre-NMS.

    Level flattening matches effdet.bench._post_process:
    permute NCHW -> NHWC then reshape, concatenated over levels.
    """

    def __init__(self, net, num_classes):
        super().__init__()
        self.net = net
        self.num_classes = num_classes

    def forward(self, x):
        class_out, box_out = self.net(x)
        batch = x.shape[0]
        scores = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes) for c in class_out],
            dim=1,
        ).sigmoid()
        boxes = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(batch, -1, 4) for b in box_out], dim=1
        )
        return boxes, scores


def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, (size[1], size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = torch.tensor([m * 255 for m in IMAGENET_DEFAULT_MEAN]).view(3, 1, 1)
    std = torch.tensor([s * 255 for s in IMAGENET_DEFAULT_STD]).view(3, 1, 1)
    t = torch.from_numpy(img).permute(2, 0, 1).float().sub_(mean).div_(std)
    return t.unsqueeze(0)


def calibration_images(calib_root, count, seed=0):
    root = Path(calib_root)
    files = sorted(root.glob("*/training_images/*/*.png"))
    if not files:
        sys.exit(f"No calibration images found under {root}")
    random.Random(seed).shuffle(files)
    return files[:count]


def run_tflite(path, x):
    """Run a (possibly int8) tflite model on a float NCHW tensor; return float (boxes, scores)."""
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    xi = x.numpy() if hasattr(x, "numpy") else x
    if inp["dtype"] == np.int8:
        s, z = inp["quantization"]
        xi = np.clip(np.round(xi / s + z), -128, 127).astype(np.int8)
    it.set_tensor(inp["index"], xi)
    it.invoke()
    outs = []
    for od in it.get_output_details():
        o = it.get_tensor(od["index"])
        if od["dtype"] == np.int8:
            s, z = od["quantization"]
            o = (o.astype(np.float32) - z) * s
        outs.append(o)
    boxes = next(o for o in outs if o.shape[-1] == 4)
    scores = next(o for o in outs if o.shape[-1] != 4)
    return boxes, scores


def parity(tag, ref_boxes, ref_scores, boxes, scores, topk=100):
    """Report agreement, weighted toward the anchors that matter (top scores)."""
    top = np.argsort(ref_scores.max(-1).ravel())[::-1][:topk]
    d_scores = np.abs(scores - ref_scores)
    d_boxes = np.abs(boxes - ref_boxes).reshape(-1, 4)[top]
    print(f"{tag}: scores max|Δ|={d_scores.max():.4f} p99|Δ|={np.percentile(d_scores, 99):.4f}; "
          f"top{topk} boxes max|Δ|={d_boxes.max():.4f}")
    top_o = np.argsort(scores.max(-1).ravel())[::-1][:20]
    top_r = np.argsort(ref_scores.max(-1).ravel())[::-1][:20]
    print(f"{tag}: top-20 anchor overlap {len(set(top_o.tolist()) & set(top_r.tolist()))}/20")


def find_ckpt(output_dir: Path) -> Path:
    ckpts = sorted(
        output_dir.glob("train/*/model_best.pth.tar"), key=lambda p: p.stat().st_mtime
    )
    if not ckpts:
        sys.exit(f"No model_best.pth.tar found under {output_dir}/train")
    return ckpts[-1]


def load_network(model_name, num_classes, ckpt_path):
    net = create_model(
        model_name,
        bench_task="",  # raw network, no bench/NMS
        num_classes=num_classes,
        pretrained=False,
        pretrained_backbone=False,
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        sys.exit(f"checkpoint is missing {len(missing)} network keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[INFO] ignored {len(unexpected)} non-network checkpoint keys")
    return net.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="Root with <garage>/training_images/<sensor>/*.png "
                         "(default: base_data_path from preannotation/config.yaml)")
    ap.add_argument("--calib-count", type=int, default=64)
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config/train_wrapper_config.yaml").read_text())
    model_name = cfg["model"]
    num_classes = cfg.get("num_classes", len(load_label_map()))
    H, W = MODEL_CONFIG[model_name]["image_size"]
    out_dir = Path(cfg["output_dir"])

    ckpt = args.checkpoint or find_ckpt(out_dir)
    print(f"Exporting {model_name} ({H}x{W}, {num_classes} classes) from {ckpt}")

    net = load_network(model_name, num_classes, ckpt)
    model = ExportWrapper(net, num_classes).eval()

    # TFLite input is NHWC uint-style layout like the firmware baseline;
    # the wrapper transposes to NCHW internally.
    export_model = litert_torch.to_channel_last_io(model, args=[0]).eval()
    sample_nchw = torch.randn(1, 3, H, W)
    sample = (sample_nchw.permute(0, 2, 3, 1).contiguous(),)

    with torch.no_grad():
        ref_boxes, ref_scores = model(sample_nchw)
    n_anchors = ref_boxes.shape[1]
    print(f"Output contract: input [1,{H},{W},3], boxes [1,{n_anchors},4], "
          f"scores [1,{n_anchors},{num_classes}]")

    # ---- float32 export ----
    edge_model = litert_torch.convert(export_model, sample)
    f32_path = out_dir / f"{model_name}.f32.tflite"
    edge_model.export(str(f32_path))
    print(f"Wrote {f32_path}")

    # float parity vs PyTorch
    tfl_boxes, tfl_scores = run_tflite(f32_path, sample[0])
    parity("float parity", ref_boxes.numpy(), ref_scores.numpy(), tfl_boxes, tfl_scores)

    if args.skip_int8:
        return

    # ---- full-int8 static PTQ (ai-edge-quantizer on the float flatbuffer) ----
    calib_root = args.calib_dir
    if calib_root is None:
        pre_cfg = yaml.safe_load(Path("preannotation/config.yaml").read_text())
        calib_root = Path(pre_cfg["base_data_path"])
    calib_files = calibration_images(calib_root, args.calib_count)
    print(f"Calibrating on {len(calib_files)} images from {calib_root}")

    interp = Interpreter(model_path=str(f32_path))
    interp.allocate_tensors()
    sig_key = list(interp.get_signature_list().keys())[0]
    input_name = interp.get_signature_list()[sig_key]["inputs"][0]

    calib_inputs = []
    for f in calib_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        nhwc = preprocess(img, (H, W)).permute(0, 2, 3, 1).contiguous().numpy()
        calib_inputs.append({input_name: nhwc})

    qt = aeq_quantizer.Quantizer(f32_path, aeq_recipe.static_wi8_ai8())
    calib_result = qt.calibrate({sig_key: calib_inputs})
    q_path = out_dir / f"{model_name}.int8.tflite"
    qt.quantize(calib_result, serialize_to_path=q_path)
    print(f"Wrote {q_path}")

    # int8-vs-PyTorch parity on a real image
    test_img = cv2.imread(str(calib_files[0]))
    x = preprocess(test_img, (H, W))
    with torch.no_grad():
        fb, fs = model(x)
    qb, qs = run_tflite(q_path, x.permute(0, 2, 3, 1).contiguous())
    parity("int8 parity", fb.numpy(), fs.numpy(), qb, qs)

    print("\nArtifacts:")
    for p in (f32_path, q_path):
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
