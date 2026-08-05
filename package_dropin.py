#!/usr/bin/env python3
"""Package a trained effdet checkpoint as a byte-compatible drop-in for the
production model (baseline/efficientdet_lite2.tflite).

Run inside the export venv (setup_export_venv.sh), NOT the training venv:

    PYTHONPATH=. falcon-vision-od-export-venv/bin/python package_dropin.py

The firmware contract (verified against the baseline with ai_edge_litert):
  input : [1, 448, 448, 3] uint8 RGB (firmware memcpys raw bytes)
  output: TFLite_Detection_PostProcess -> 4 float32 tensors, in order:
          boxes [1, 25, 4] (normalized ymin, xmin, ymax, xmax)
          classes [1, 25] (0-based), scores [1, 25], num_detections [1]

Pipeline:
  1. litert_torch converts an export wrapper (raw 0-255 NHWC float input,
     in-graph ImageNet normalization, per-level head flattening + sigmoid).
  2. ai-edge-quantizer dynamic_wi8_afp32 for weight compression (the
     deliverable; a float32 variant is also written).
  3. Flatbuffer surgery (ai_edge_litert.schema_py_generated):
       a. uint8 input tensor (scale=1.0, zp=0) + DEQUANTIZE into the graph
       b. TFLite_Detection_PostProcess(box_regressions, scores, anchors)
          with anchors from effdet.anchors.Anchors, normalized to 0-1
          (ycenter, xcenter, h, w); scales all 1.0 because effdet
          regressions are unscaled ty, tx, th, tw.
  4. Validation vs the baseline on real garage images (interface match +
     car-detection overlap + empty-frame silence).

Outputs land in the checkpoint's artifact dir (artifact_paths.py) as
<run_name>.dropin.tflite (deliverable) and <run_name>.dropin.f32.tflite.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import flatbuffers
import flatbuffers.flexbuffers as flexbuffers
import litert_torch
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer as aeq_quantizer
from ai_edge_quantizer import recipe as aeq_recipe

from artifact_paths import artifact_dir, update_latest_symlink, update_manifest
from config.label_loader import load_label_map
from effdet import create_model, get_efficientdet_config
from effdet.anchors import Anchors

IMAGE_SIZE = 448  # default; firmware reads input dims from the model, so --input-size works too
MAX_DETECTIONS = 25
NMS_IOU_THRESHOLD = 0.5
CAR_CLASS = 2  # 0-based class index for "car" in both baseline (COCO) and ours

DEFAULT_BASELINE = Path("baseline/efficientdet_lite2.tflite")
_VAL_ROOT = Path("/media/lopezemi/Expansion/falcon-vision-ml/artifacts/data_pipeline"
                 "/arlington/training_images/fv1b999e")
DEFAULT_CAR_IMAGE = _VAL_ROOT / "fv1b999e-camera2-20210401-210231-01.png"
DEFAULT_EMPTY_IMAGE = _VAL_ROOT / "fv1b999e-camera1-20210401-000203-01.png"


# ---------------------------------------------------------------------------
# checkpoint loading (same pattern as export_tflite.py)
# ---------------------------------------------------------------------------

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
        # FPN resample sizes are frozen at build time, so the net must be
        # constructed at the firmware's 448 input (weights are fully conv).
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
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


class NCHWDropinWrapper(torch.nn.Module):
    """[1,3,H,W] float RGB raw 0-255 -> (box_regressions, sigmoid scores).
    The clean-export input convention (onnx2tf converts layout itself); the
    packaging surgery bridges the firmware's uint8-NHWC to this."""

    def __init__(self, net, num_classes):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        mean = torch.tensor([m * 255 for m in IMAGENET_DEFAULT_MEAN]).view(1, 3, 1, 1)
        std = torch.tensor([s * 255 for s in IMAGENET_DEFAULT_STD]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        class_out, box_out = self.net(x)
        batch = x.shape[0]
        scores = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes)
             for c in class_out], dim=1).sigmoid()
        boxes = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(batch, -1, 4) for b in box_out],
            dim=1)
        return boxes, scores


class DropinWrapper(torch.nn.Module):
    """[1,H,W,3] float RGB in raw 0-255 scale -> (box_regressions, sigmoid scores).

    Normalization and NHWC->NCHW live inside the graph so the (surgically
    inserted) uint8 input carries raw image bytes, exactly like the firmware
    feeds the baseline today. Level flattening matches effdet.bench._post_process.
    """

    def __init__(self, net, num_classes):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        mean = torch.tensor([m * 255 for m in IMAGENET_DEFAULT_MEAN]).view(1, 3, 1, 1)
        std = torch.tensor([s * 255 for s in IMAGENET_DEFAULT_STD]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
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


def build_anchors(model_name, size=IMAGE_SIZE):
    """effdet anchors -> [N,4] float32 (ycenter, xcenter, h, w), normalized 0-1.

    Anchors.boxes is (ymin, xmin, ymax, xmax) in input pixels (see
    effdet/anchors.py decode_box_outputs, which reads it exactly that way).
    """
    cfg = get_efficientdet_config(model_name)
    cfg.image_size = (size, size)
    boxes = Anchors.from_config(cfg).boxes.numpy()  # [N,4] yxyx pixels
    yc = (boxes[:, 0] + boxes[:, 2]) / 2.0 / size
    xc = (boxes[:, 1] + boxes[:, 3]) / 2.0 / size
    h = (boxes[:, 2] - boxes[:, 0]) / size
    w = (boxes[:, 3] - boxes[:, 1]) / size
    return np.stack([yc, xc, h, w], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# flatbuffer surgery
# ---------------------------------------------------------------------------

def _add_opcode(model, builtin=None, custom=None, version=1):
    oc = schema.OperatorCodeT()
    if custom is not None:
        oc.builtinCode = schema.BuiltinOperator.CUSTOM
        oc.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
        oc.customCode = custom
    else:
        oc.builtinCode = builtin
        oc.deprecatedBuiltinCode = builtin if builtin <= 127 else 127
    oc.version = version
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def _add_tensor(sg, name, shape, ttype, buffer=0, quant=None):
    t = schema.TensorT()
    t.name = name.encode()
    t.shape = list(shape)
    t.type = ttype
    t.buffer = buffer
    t.quantization = quant
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def postprocess_options(num_classes, score_threshold):
    # Mirrors the baseline's decoded flexbuffer options exactly (num_classes aside).
    return flexbuffers.Dumps({
        "max_detections": MAX_DETECTIONS,
        "max_classes_per_detection": 1,
        "nms_score_threshold": score_threshold,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "num_classes": num_classes,
        "use_regular_nms": 0,
        "y_scale": 1.0,   # effdet regressions are unscaled ty,tx,th,tw:
        "x_scale": 1.0,   # ycenter = ty/y_scale * ha + ya reduces to ty*ha + ya
        "h_scale": 1.0,
        "w_scale": 1.0,
    })


def _inline_external_buffers(model, raw: bytes):
    """LiteRT writes big constants outside the flatbuffer (buffer offset/size);
    repacking with the object API would drop that region. Inline them --
    plain in-flatbuffer buffers are also what old runtimes expect."""
    for buf in model.buffers:
        offset = getattr(buf, "offset", 0) or 0
        if offset > 1:
            buf.data = np.frombuffer(raw[offset:offset + buf.size], dtype=np.uint8)
            buf.offset = 0
            buf.size = 0


# Ops whose paddings input (index 1) old TFLite kernels (<= 2.8 pad.cc /
# mirror_pad.cc) read unconditionally as int32.
_PADDINGS_OPS = (schema.BuiltinOperator.PAD, schema.BuiltinOperator.PADV2,
                 schema.BuiltinOperator.MIRROR_PAD)


def _downcast_int64_paddings(model):
    """litert-torch emits PAD/PADV2 paddings as INT64 constants. Old TFLite
    runtimes (2.6-2.8) read the paddings tensor as int32 regardless of its
    declared type, so each int64 pair [0,0],[0,1],... is misread as
    [0,0],[0,0],[0,1],... -- shifting the pads onto the wrong dims (observed:
    'reshape 1040 != 1600' at prepare, channels padded 64->65). Rewriting the
    constants as INT32 (values are tiny) is valid on every runtime version."""
    for sg in model.subgraphs:
        for op in sg.operators:
            oc = model.operatorCodes[op.opcodeIndex]
            builtin = max(oc.builtinCode or 0, oc.deprecatedBuiltinCode or 0)
            if builtin not in _PADDINGS_OPS or len(op.inputs) < 2:
                continue
            t = sg.tensors[op.inputs[1]]
            buf = model.buffers[t.buffer]
            if t.type != schema.TensorType.INT64 or buf.data is None:
                continue
            vals = np.frombuffer(bytes(buf.data), dtype=np.int64)
            if vals.size and (np.abs(vals) > np.iinfo(np.int32).max).any():
                sys.exit(f"paddings tensor {t.name} has values too large for int32")
            # fresh buffer: the int64 one may be shared with another tensor
            nb = schema.BufferT()
            nb.data = np.frombuffer(vals.astype(np.int32).tobytes(), dtype=np.uint8)
            model.buffers.append(nb)
            t.buffer = len(model.buffers) - 1
            t.type = schema.TensorType.INT32


def repack_for_old_runtimes(src: Path, dst: Path = None):
    """Repack a litert-converted .tflite so old (2.6-2.8) TFLite runtimes parse
    it correctly: inline out-of-flatbuffer buffers, int32 pad paddings.
    Shared by package_dropin (via apply_surgery) and export_tflite."""
    raw = src.read_bytes()
    model = schema.ModelT.InitFromPackedBuf(bytearray(raw), 0)
    _inline_external_buffers(model, raw)
    _downcast_int64_paddings(model)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    (dst or src).write_bytes(builder.Output())


def apply_surgery(src: Path, dst: Path, anchors: np.ndarray, num_classes: int,
                  score_threshold: float):
    """uint8 input + DEQUANTIZE (float body) or QUANTIZE requant (int8 body),
    then append TFLite_Detection_PostProcess (dequantizing int8 heads first)."""
    raw = src.read_bytes()
    model = schema.ModelT.InitFromPackedBuf(bytearray(raw), 0)
    sg = model.subgraphs[0]

    _inline_external_buffers(model, raw)
    _downcast_int64_paddings(model)

    # --- (a) uint8 interface input feeding the graph's real input ---
    if len(sg.inputs) != 1:
        sys.exit(f"expected a single graph input, got {list(sg.inputs)}")
    graph_in = int(sg.inputs[0])
    in_t = sg.tensors[graph_in]
    in_shape = list(in_t.shape)
    # clean (onnx2tf) exports keep an NCHW graph input; the firmware contract
    # is NHWC uint8 bytes, so bridge via one input-sized TRANSPOSE (~free)
    nchw = len(in_shape) == 4 and in_shape[1] == 3 and in_shape[3] != 3
    if nchw:
        in_shape = [in_shape[0], in_shape[2], in_shape[3], in_shape[1]]
    quant = schema.QuantizationParametersT()
    if in_t.type == schema.TensorType.FLOAT32:
        # raw byte value == float value the graph expects
        quant.scale = [1.0]
        quant.zeroPoint = [0]
        bridge_code = _add_opcode(model, builtin=schema.BuiltinOperator.DEQUANTIZE,
                                  version=1)  # v1 = uint8 dequant, oldest-runtime safe
    elif in_t.type == schema.TensorType.INT8:
        # same scale, zero-point shifted +128: uint8 byte b == int8 (b-128),
        # so QUANTIZE is an exact requantization (pure offset)
        s = float(in_t.quantization.scale[0])
        z = int(in_t.quantization.zeroPoint[0])
        if not (0 <= z + 128 <= 255):
            sys.exit(f"int8 input zero-point {z} cannot shift into uint8")
        quant.scale = [s]
        quant.zeroPoint = [z + 128]
        bridge_code = _add_opcode(model, builtin=schema.BuiltinOperator.QUANTIZE,
                                  version=2)  # v2 = requantize support
    else:
        sys.exit(f"unsupported graph input type {in_t.type}")
    u8_idx = _add_tensor(sg, "serving_default_images:0", in_shape,
                         schema.TensorType.UINT8, buffer=0, quant=quant)
    if nchw:
        # uint8 NHWC -> (dequant|quant) -> NHWC tmp -> TRANSPOSE -> NCHW input
        tmp_quant = None
        if in_t.type == schema.TensorType.INT8:
            tmp_quant = schema.QuantizationParametersT()
            tmp_quant.scale = list(in_t.quantization.scale)
            tmp_quant.zeroPoint = list(in_t.quantization.zeroPoint)
        tmp_idx = _add_tensor(sg, "images_nhwc", in_shape, in_t.type,
                              quant=tmp_quant)
        perm_buf = schema.BufferT()
        perm_buf.data = np.frombuffer(
            np.array([0, 3, 1, 2], dtype=np.int32).tobytes(), dtype=np.uint8)
        model.buffers.append(perm_buf)
        perm_idx = _add_tensor(sg, "images_perm", [4], schema.TensorType.INT32,
                               buffer=len(model.buffers) - 1)
        tr_code = _add_opcode(model, builtin=schema.BuiltinOperator.TRANSPOSE,
                              version=2)
        tr = schema.OperatorT()
        tr.opcodeIndex = tr_code
        tr.inputs = [tmp_idx, perm_idx]
        tr.outputs = [graph_in]
        sg.operators.insert(0, tr)
        bridge_target = tmp_idx
    else:
        bridge_target = graph_in
    bridge = schema.OperatorT()
    bridge.opcodeIndex = bridge_code
    bridge.inputs = [u8_idx]
    bridge.outputs = [bridge_target]
    sg.operators.insert(0, bridge)  # operators must stay in execution order
    sg.inputs = [u8_idx]
    float_in = graph_in  # signature fix-up below

    # --- (b) TFLite_Detection_PostProcess over (boxes, scores, anchors) ---
    outs = list(sg.outputs)
    if len(outs) != 2:
        sys.exit(f"expected 2 graph outputs (boxes, scores), got {outs}")
    box_idx = next(i for i in outs if list(sg.tensors[i].shape)[-1] == 4)
    score_idx = next(i for i in outs if i != box_idx)

    # int8 heads must be dequantized: the postprocess op reads float tensors
    deq8_code = None
    def _ensure_float(idx):
        nonlocal deq8_code
        t = sg.tensors[idx]
        if t.type != schema.TensorType.INT8:
            return idx
        if deq8_code is None:
            deq8_code = _add_opcode(model, builtin=schema.BuiltinOperator.DEQUANTIZE,
                                    version=2)  # v2 = int8 dequant
        f_idx = _add_tensor(sg, t.name.decode() + "_f32", list(t.shape),
                            schema.TensorType.FLOAT32)
        d = schema.OperatorT()
        d.opcodeIndex = deq8_code
        d.inputs = [idx]
        d.outputs = [f_idx]
        sg.operators.append(d)
        return f_idx

    box_idx = _ensure_float(box_idx)
    score_idx = _ensure_float(score_idx)
    n_anchors = list(sg.tensors[box_idx].shape)[1]
    if anchors.shape != (n_anchors, 4):
        sys.exit(f"anchor mismatch: anchors {anchors.shape} vs boxes [1,{n_anchors},4]")

    anchor_buf = schema.BufferT()
    anchor_buf.data = np.frombuffer(anchors.tobytes(), dtype=np.uint8)
    model.buffers.append(anchor_buf)
    anchors_idx = _add_tensor(sg, "anchors", [n_anchors, 4],
                              schema.TensorType.FLOAT32,
                              buffer=len(model.buffers) - 1)

    # 4 dynamic-shaped float outputs, named/ordered exactly like the baseline
    # (shape [] like the baseline: the op's Prepare marks them dynamic).
    out_names = ["StatefulPartitionedCall:3", "StatefulPartitionedCall:2",
                 "StatefulPartitionedCall:1", "StatefulPartitionedCall:0"]
    pp_outs = [_add_tensor(sg, n, [], schema.TensorType.FLOAT32) for n in out_names]

    pp_code = _add_opcode(model, custom=b"TFLite_Detection_PostProcess", version=1)
    pp = schema.OperatorT()
    pp.opcodeIndex = pp_code
    pp.inputs = [box_idx, score_idx, anchors_idx]
    pp.outputs = pp_outs
    pp.customOptions = np.frombuffer(
        postprocess_options(num_classes, score_threshold), dtype=np.uint8)
    pp.customOptionsFormat = schema.CustomOptionsFormat.FLEXBUFFERS
    sg.operators.append(pp)
    sg.outputs = pp_outs

    # keep signatures coherent with the new I/O
    for sig in model.signatureDefs or []:
        for tm in sig.inputs or []:
            if tm.tensorIndex == float_in:
                tm.tensorIndex = u8_idx
        new_out = []
        for name, ti in zip(("boxes", "classes", "scores", "num_detections"), pp_outs):
            tm = schema.TensorMapT()
            tm.name = name.encode()
            tm.tensorIndex = ti
            new_out.append(tm)
        sig.outputs = new_out

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    dst.write_bytes(builder.Output())


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def load_uint8_rgb(path, size=IMAGE_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"cannot read validation image {path}")
    img = cv2.resize(img, (size, size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)


def run_detector(path, x_uint8):
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], x_uint8)
    it.invoke()
    boxes, classes, scores, count = (it.get_tensor(d["index"])
                                     for d in it.get_output_details())
    return boxes[0], classes[0], scores[0], float(count[0])


def detections(boxes, classes, scores, thresh, cls=None):
    keep = scores > thresh
    if cls is not None:
        keep &= classes.astype(int) == cls
    return boxes[keep], classes[keep].astype(int), scores[keep]


def iou(a, b):
    yi1, xi1 = max(a[0], b[0]), max(a[1], b[1])
    yi2, xi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, yi2 - yi1) * max(0.0, xi2 - xi1)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def fmt_dets(boxes, classes, scores, limit=6):
    lines = []
    for b, c, s in list(zip(boxes, classes, scores))[:limit]:
        lines.append(f"    class={c} score={s:.3f} "
                     f"box=[{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}]")
    return "\n".join(lines) if lines else "    (none)"


def validate(packaged: Path, baseline: Path, car_image: Path, empty_image: Path,
             thresh=0.3):
    """Interface + behavior checks vs the baseline. Returns (ok, stats)."""
    ok = True
    stats = {}

    it_b = Interpreter(model_path=str(baseline))
    it_b.allocate_tensors()
    it_p = Interpreter(model_path=str(packaged))
    it_p.allocate_tensors()

    bi, pi = it_b.get_input_details()[0], it_p.get_input_details()[0]
    s_b = int(bi["shape"][1])
    in_ok = (list(pi["shape"]) == [1, IMAGE_SIZE, IMAGE_SIZE, 3]
             and pi["dtype"] == np.uint8 and np.dtype(bi["dtype"]) == np.uint8)
    print(f"[{'ok' if in_ok else 'FAIL'}] input: packaged {list(pi['shape'])} "
          f"{np.dtype(pi['dtype']).name} q={pi['quantization']} "
          f"| baseline {list(bi['shape'])} {np.dtype(bi['dtype']).name} "
          f"q={bi['quantization']}")
    ok &= in_ok
    stats["input"] = {"shape": [int(v) for v in pi["shape"]],
                      "dtype": np.dtype(pi["dtype"]).name,
                      "quantization": list(pi["quantization"])}

    bo = [([int(v) for v in d["shape"]], np.dtype(d["dtype"]).name)
          for d in it_b.get_output_details()]
    po = [([int(v) for v in d["shape"]], np.dtype(d["dtype"]).name)
          for d in it_p.get_output_details()]
    out_ok = bo == po
    print(f"[{'ok' if out_ok else 'FAIL'}] outputs: packaged {po} | baseline {bo}")
    ok &= out_ok
    stats["outputs"] = [{"shape": s, "dtype": t} for s, t in po]

    # --- busy frame: car boxes must overlap the baseline's ---
    pb, pc, ps, pn = run_detector(packaged, load_uint8_rgb(car_image, IMAGE_SIZE))
    bb, bc, bs, bn = run_detector(baseline, load_uint8_rgb(car_image, s_b))
    p_cars = detections(pb, pc, ps, thresh, CAR_CLASS)
    b_cars = detections(bb, bc, bs, thresh, CAR_CLASS)
    print(f"\ncar image: {car_image.name}")
    print(f"  packaged (num_detections={pn:.0f}), cars > {thresh}:")
    print(fmt_dets(*p_cars))
    print(f"  baseline (num_detections={bn:.0f}), cars > {thresh}:")
    print(fmt_dets(*b_cars))
    ious = [max((iou(p, b) for b in b_cars[0]), default=0.0) for p in p_cars[0]]
    best_iou = max(ious, default=0.0)
    boxes_sane = bool(len(p_cars[0]) and
                      np.all(p_cars[0] >= -0.05) and np.all(p_cars[0] <= 1.05) and
                      np.all(p_cars[0][:, 2:] > p_cars[0][:, :2]))
    car_ok = len(p_cars[0]) > 0 and len(b_cars[0]) > 0 and best_iou > 0.3 and boxes_sane
    print(f"  [{'ok' if car_ok else 'FAIL'}] packaged cars={len(p_cars[0])} "
          f"baseline cars={len(b_cars[0])} best IoU={best_iou:.3f} "
          f"boxes sane={boxes_sane}")
    ok &= car_ok
    stats["car_image"] = {
        "file": car_image.name,
        "packaged_cars": int(len(p_cars[0])),
        "baseline_cars": int(len(b_cars[0])),
        "packaged_top_car_score": round(float(p_cars[2].max()), 4) if len(p_cars[2]) else 0.0,
        "baseline_top_car_score": round(float(b_cars[2].max()), 4) if len(b_cars[2]) else 0.0,
        "best_iou_vs_baseline": round(float(best_iou), 4),
    }

    # --- empty frame: silence above threshold ---
    xe = load_uint8_rgb(empty_image, IMAGE_SIZE)
    eb, ec, es, en = run_detector(packaged, xe)
    hits = detections(eb, ec, es, thresh)
    empty_ok = len(hits[0]) == 0
    print(f"\nempty image: {empty_image.name}")
    print(f"  packaged max score={es.max():.3f}; detections > {thresh}:")
    print(fmt_dets(*hits))
    print(f"  [{'ok' if empty_ok else 'FAIL'}] no detections above {thresh}")
    ok &= empty_ok
    stats["empty_image"] = {"file": empty_image.name,
                            "max_score": round(float(es.max()), 4),
                            "detections_above_threshold": int(len(hits[0]))}
    stats["score_threshold"] = thresh
    return ok, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="default: newest experiments/.../train/*/model_best.pth.tar")
    ap.add_argument("--model", default=None,
                    help="effdet model name (default: `model` from train_wrapper_config.yaml)")
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--car-image", type=Path, default=DEFAULT_CAR_IMAGE)
    ap.add_argument("--empty-image", type=Path, default=DEFAULT_EMPTY_IMAGE)
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="validation reporting threshold (the packaged model itself "
                         "keeps the baseline's -inf NMS score threshold)")
    ap.add_argument("--input-size", type=int, default=448,
                    help="square input size baked into the packaged model "
                         "(firmware resizes frames to match; 320 = lite0 native)")
    ap.add_argument("--variants", default="f32,dyn,int8",
                    help="comma subset of f32,dyn,int8 (default: all three)")
    ap.add_argument("--calib-dir", type=Path, default=Path("data/images"),
                    help="calibration image root for the int8 variant")
    ap.add_argument("--calib-count", type=int, default=256)
    ap.add_argument("--legacy-export", action="store_true",
                    help="old litert-torch conversion path (transpose-heavy; "
                         "~15-20%% slower on-device) — fallback only")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip building; validate the existing packaged files")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config/train_wrapper_config.yaml").read_text())
    model_name = args.model or cfg["model"]
    global IMAGE_SIZE
    IMAGE_SIZE = args.input_size
    num_classes = args.num_classes or cfg.get("num_classes", len(load_label_map()))
    out_dir = Path(cfg["output_dir"])
    ckpt = args.checkpoint or find_ckpt(out_dir)
    art_dir = artifact_dir(out_dir, model_name, ckpt)
    # naming: <run>.dropin-<size>-<quant>.tflite — size and quant ALWAYS
    # explicit so nobody ships the wrong variant again
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - {"f32", "dyn", "int8"}
    if bad:
        sys.exit(f"unknown variants {bad} (choose from f32,dyn,int8)")
    paths = {q: art_dir / f"{art_dir.name}.dropin-{IMAGE_SIZE}-{q}.tflite"
             for q in variants}

    if not args.validate_only:
        print(f"Packaging {model_name} ({IMAGE_SIZE}x{IMAGE_SIZE}, "
              f"{num_classes} classes) from {ckpt}")
        print(f"Artifacts -> {art_dir}")

        net = load_network(model_name, num_classes, ckpt)
        wrapper = DropinWrapper(net, num_classes).eval()
        sample = (torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, 3),)
        with torch.no_grad():
            ref_boxes, _ = wrapper(sample[0])
        n_anchors = ref_boxes.shape[1]

        anchors = build_anchors(model_name, IMAGE_SIZE)
        if anchors.shape[0] != n_anchors:
            sys.exit(f"anchor count {anchors.shape[0]} != model output {n_anchors}")
        print(f"Graph: input [1,{IMAGE_SIZE},{IMAGE_SIZE},3] raw RGB, "
              f"{n_anchors} anchors, {num_classes} classes")

        score_thr = float("-inf")  # same as the baseline's op options
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bodies = {}
            if args.legacy_export:
                f32_raw = td / "f32.tflite"
                edge_model = litert_torch.convert(wrapper, sample)
                edge_model.export(str(f32_raw))
                if "f32" in variants:
                    bodies["f32"] = f32_raw
                if "dyn" in variants:
                    qt = aeq_quantizer.Quantizer(f32_raw,
                                                 aeq_recipe.dynamic_wi8_afp32())
                    qt.quantize(serialize_to_path=td / "dyn.tflite")
                    bodies["dyn"] = td / "dyn.tflite"
                if "int8" in variants:
                    sys.exit("--legacy-export has no int8 path anymore; drop "
                             "int8 from --variants or use the clean default")
            else:
                import clean_convert as cc
                cc.check_venvs()
                nchw = NCHWDropinWrapper(net, num_classes).eval()
                onnx_path = td / "model.onnx"
                cc.to_onnx(nchw, IMAGE_SIZE, onnx_path)
                sm_dir, f32_body = cc.onnx_to_tf(onnx_path, td / "tf")
                if "f32" in variants:
                    bodies["f32"] = f32_body
                if "dyn" in variants:
                    cc.quantize_dynamic(f32_body, td / "dyn.tflite")
                    bodies["dyn"] = td / "dyn.tflite"
                if "int8" in variants:
                    print(f"int8 static PTQ (TF converter, {args.calib_count} "
                          "calibration frames)…")
                    cc.quantize_int8(sm_dir, IMAGE_SIZE, args.calib_dir,
                                     args.calib_count, td / "int8.tflite")
                    bodies["int8"] = td / "int8.tflite"
            for q, body in bodies.items():
                apply_surgery(body, paths[q], anchors, num_classes, score_thr)
                print(f"Wrote {paths[q].name} "
                      f"({paths[q].stat().st_size / 1e6:.1f} MB)")

    for p in list(paths.values()) + [args.baseline]:
        if not p.exists():
            sys.exit(f"missing {p}")

    QUANT_DESC = {
        "f32": "float32 (no quantization)",
        "dyn": "dynamic-range: int8 weights / f32 activations (ai-edge-quantizer)",
        "int8": "full-int8 static PTQ (TF-2.8 converter, garage-calibrated)",
    }
    all_ok = True
    for q in variants:
        print(f"\n=== validating {paths[q].name} vs {args.baseline} ===")
        ok, stats = validate(paths[q], args.baseline, args.car_image,
                             args.empty_image, args.score_threshold)
        all_ok &= ok
        update_manifest(art_dir, f"dropin_{IMAGE_SIZE}_{q}", {
            "model": model_name,
            "checkpoint": str(ckpt),
            "file": paths[q].name,
            "size_bytes": paths[q].stat().st_size,
            "num_classes": num_classes,
            "quantization": QUANT_DESC[q],
            "export": ("legacy litert-torch" if args.legacy_export
                       else "clean: onnx -> onnx2tf (clean_convert.py)"),
            "contract": {
                "input": f"[1,{IMAGE_SIZE},{IMAGE_SIZE},3] uint8 RGB raw bytes "
                         "(in-graph dequant/quant + ImageNet norm)",
                "outputs": "TFLite_Detection_PostProcess: boxes [1,25,4] norm "
                           "ymin,xmin,ymax,xmax; classes [1,25] 0-based; "
                           "scores [1,25]; num_detections [1]",
                "postprocess": f"max_detections={MAX_DETECTIONS}, "
                               f"iou={NMS_IOU_THRESHOLD}, score_threshold=-inf, "
                               "scales=1.0 (matches baseline)",
                "baseline": str(args.baseline),
            },
            "validation": stats,
        })
    if not all_ok:
        sys.exit("\nVALIDATION FAILED (files left in place for debugging)")
    update_latest_symlink(art_dir)
    print("\nVALIDATION PASSED\nDeliverables:")
    for q in variants:
        print(f"  {paths[q]}")


if __name__ == "__main__":
    main()
