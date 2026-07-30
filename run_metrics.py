#!/usr/bin/env python3
"""Per-run evaluation report: per-class + size-banded COCO metrics.

Runs the run's best checkpoint over its val split (via validate.py), then
writes into the run directory:
  - val_predictions.json   raw COCO-format detections
  - coco_metrics.json      overall + per-class + per-size AP/AR breakdown
  - run.json               provenance manifest (merged from the split's
                           run.json) + final headline metrics

Why per-class is mandatory here: garage frames are ~94% car, so 6-class mean
mAP is dominated by classes with almost no support (the 4.1k prove-out run
scored 0.09 mean while car AP was 0.44). Read car AP / person AP, not the mean.

Called automatically after training by run_training_from_config.py; run
manually to backfill an old run:

    python3 run_metrics.py experiments/falcon-vision-effdet/train/<run>/
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _breakdown(gt, preds_file):
    from pycocotools.cocoeval import COCOeval
    dt = gt.loadRes(str(preds_file))
    E = COCOeval(gt, dt, "bbox")
    E.evaluate()
    E.accumulate()
    prec = E.eval["precision"]  # [iou, recall, cat, area, maxdet]
    rec = E.eval["recall"]      # [iou, cat, area, maxdet]
    names = {c["id"]: c["name"] for c in gt.loadCats(E.params.catIds)}

    def ap(cat_idx=None, area=0, iou_idx=None):
        p = prec[:, :, :, area, 2] if cat_idx is None else prec[:, :, cat_idx, area, 2]
        if iou_idx is not None:
            p = p[iou_idx]
        p = p[p > -1]
        return round(float(p.mean()), 4) if p.size else None

    per_class = {}
    for k, cid in enumerate(E.params.catIds):
        n_gt = len(gt.getAnnIds(catIds=cid))
        r = rec[:, k, 0, 2]
        r = r[r > -1]
        per_class[names[cid]] = {
            "ap": ap(k), "ap50": ap(k, iou_idx=0),
            "ap_small": ap(k, area=1), "ap_medium": ap(k, area=2),
            "ap_large": ap(k, area=3),
            "ar": round(float(r.mean()), 4) if r.size else None,
            "gt_boxes": n_gt,
        }
    return {
        "overall": {"map": ap(), "map50": ap(iou_idx=0),
                    "map_small": ap(area=1), "map_medium": ap(area=2),
                    "map_large": ap(area=3)},
        "per_class": per_class,
    }


def compute(run_dir, batch_size=16):
    from pycocotools.coco import COCO
    run_dir = Path(run_dir)
    args_yaml = yaml.safe_load((run_dir / "args.yaml").read_text())
    split_root = Path(args_yaml["root"])
    if not split_root.is_absolute():
        split_root = Path(__file__).parent / split_root
    ckpt = run_dir / "model_best.pth.tar"
    preds = run_dir / "val_predictions.json"
    if not preds.exists() and not split_root.exists():
        sys.exit(f"[metrics] split dir {split_root} no longer exists — this "
                 "run cannot be rescored (its summary.csv curves remain)")

    if not preds.exists():
        cmd = [sys.executable, str(Path(__file__).parent / "validate.py"),
               str(split_root), "--dataset", "coco", "--split", "val",
               "--model", args_yaml["model"],
               "--num-classes", str(args_yaml["num_classes"]),
               "--checkpoint", str(ckpt), "-b", str(batch_size),
               "--results", str(preds)]
        print("[metrics] scoring val split:", " ".join(cmd[1:]))
        subprocess.run(cmd, check=True)

    gt = COCO(str(split_root / "annotations" / "instances_val2017.json"))
    metrics = _breakdown(gt, preds)
    (run_dir / "coco_metrics.json").write_text(json.dumps(metrics, indent=1))

    # summary.csv -> best epoch + curve endpoints
    rows = list(csv.DictReader(open(run_dir / "summary.csv")))
    best = max(rows, key=lambda r: float(r["eval_map"]))

    # merge the split's provenance manifest if present
    manifest = {}
    split_manifest = split_root / "run.json"
    if split_manifest.exists():
        manifest = json.loads(split_manifest.read_text())
    manifest.update({
        "run_dir": str(run_dir),
        "model": args_yaml["model"],
        "epochs_run": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_eval_map": round(float(best["eval_map"]), 4),
        "car_ap": (metrics["per_class"].get("car") or {}).get("ap"),
        "car_ap_large": (metrics["per_class"].get("car") or {}).get("ap_large"),
        "person_ap": (metrics["per_class"].get("person") or {}).get("ap"),
    })
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=1))
    print(f"[metrics] {run_dir.name}: car AP {manifest['car_ap']} "
          f"(large {manifest['car_ap_large']}) · person AP {manifest['person_ap']} "
          f"· best epoch {manifest['best_epoch']}")
    return manifest


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("run_dir")
    ap_.add_argument("--batch-size", type=int, default=16)
    a = ap_.parse_args()
    compute(a.run_dir, a.batch_size)
