#!/usr/bin/env python3
"""Export a trained checkpoint to a TorchScript trace (.pt) for the preannotation pipeline.

Note: the deployable int8 TFLite export path (via ai-edge-torch) is a separate,
upcoming script; this only produces the float TorchScript model used for
preannotation inference.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

from artifact_paths import artifact_dir, update_latest_symlink, update_manifest
from config.label_loader import load_label_map
from effdet import create_model
from effdet.config.model_config import efficientdet_model_param_dict as MODEL_CONFIG


def load_cfg():
    cfg_p = Path(__file__).parent / "config" / "train_wrapper_config.yaml"
    if not cfg_p.is_file():
        sys.exit(f"Config not found: {cfg_p}")
    return yaml.safe_load(cfg_p.read_text())


def find_ckpt(output_dir: Path) -> Path:
    ckpts = sorted(
        output_dir.glob("train/*/model_best.pth.tar"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        sys.exit(f"No model_best.pth.tar found under {output_dir}/train")
    return ckpts[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint to export (default: newest model_best.pth.tar in output_dir)")
    ap.add_argument("--model", default=None,
                    help="effdet model name (default: `model` from train_wrapper_config.yaml)")
    args = ap.parse_args()

    cfg = load_cfg()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or cfg["model"]
    num_classes = cfg.get("num_classes", len(load_label_map()))
    H, W = MODEL_CONFIG[model_name]["image_size"]
    print(f"Exporting {model_name} ({H}x{W}), classes={num_classes}")

    ckpt = args.checkpoint or find_ckpt(out_dir)
    print("Loading checkpoint:", ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)

    # create_model(bench_task="predict") returns a DetBenchPredict; training
    # checkpoints are bench-level ("model."-prefixed) — normalize keys to the
    # raw network and load strictly into bench.model so mistakes are loud.
    base = create_model(
        model_name,
        bench_task="predict",
        num_classes=num_classes,
        pretrained_backbone=False,
        pretrained=False,
    )
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = base.model.load_state_dict(sd, strict=False)
    if missing:
        sys.exit(f"checkpoint is missing {len(missing)} network keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[INFO] ignored {len(unexpected)} non-network checkpoint keys (e.g. {unexpected[:3]})")
    base.eval()

    dummy = torch.randn(1, 3, H, W)
    traced = torch.jit.trace(base, dummy, strict=False)
    art_dir = artifact_dir(out_dir, model_name, ckpt)
    ts_path = art_dir / "model.ts.pt"
    traced.save(ts_path)
    print("Saved TorchScript model to:", ts_path)

    update_manifest(art_dir, "torchscript", {
        "model": model_name,
        "checkpoint": str(ckpt),
        "image_size": [H, W],
        "num_classes": num_classes,
        "file": ts_path.name,
        "size_bytes": ts_path.stat().st_size,
        "torch_version": torch.__version__,
        "note": "DetBenchPredict trace (includes box decode + NMS); "
                "outputs [B,100,6] xyxy,score,1-based class",
    })
    update_latest_symlink(art_dir)


if __name__ == "__main__":
    main()
