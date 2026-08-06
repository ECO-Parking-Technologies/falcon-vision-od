#!/usr/bin/env python3
"""In-spot car accuracy: the product metric.

Scores a training run's saved val predictions against ONLY the boxes stamped
InEcoParkingSpot (preannotation/label_inspot.py); all other GT becomes a COCO
ignore region (iscrowd=1) so background/distant cars neither reward nor
penalize. Requires the run's val_predictions.json (run_metrics.py creates it).

    python3 eval_inspot.py experiments/falcon-vision-effdet/train/<run-dir>
"""
import argparse
import json
import tempfile
from pathlib import Path

REMAP_INV = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 8}
_cache = {}


def store_index(store, garage):
    if garage not in _cache:
        idx = {}
        for jf in (store / garage).glob("*/preannotations.coco.json"):
            d = json.loads(jf.read_text())
            imgs = {im["id"]: Path(im["file_name"]).name for im in d["images"]}
            for a in d["annotations"]:
                idx[(imgs[a["image_id"]],
                     tuple(round(v, 1) for v in a["bbox"]),
                     a["category_id"])] = bool(
                    a.get("attributes", {}).get("InEcoParkingSpot"))
        _cache[garage] = idx
    return _cache[garage]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    man = json.loads((args.run_dir / "run.json").read_text())
    val = json.loads((Path(man["split_dir"]) / "annotations" /
                      "instances_val2017.json").read_text())
    imgs = {i["id"]: i["file_name"] for i in val["images"]}
    kept = ign = miss = 0
    for a in val["annotations"]:
        garage, base = imgs[a["image_id"]].split("_", 1)
        inspot = store_index(args.store, garage).get(
            (base, tuple(round(v, 1) for v in a["bbox"]),
             REMAP_INV[a["category_id"]]))
        if inspot is None:
            miss += 1
            inspot = False
        if inspot:
            kept += 1
        else:
            a["iscrowd"] = 1
            ign += 1

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(val, f)
        tmp = f.name
    gt = COCO(tmp)
    dt = gt.loadRes(str(args.run_dir / "val_predictions.json"))
    E = COCOeval(gt, dt, "bbox")
    E.evaluate()
    E.accumulate()
    p = E.eval["precision"]
    k = E.params.catIds.index(3)
    def apv(i=None):
        x = p[:, :, k, 0, 2] if i is None else p[i, :, k, 0, 2]
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")
    Path(tmp).unlink()
    print(f"{args.run_dir.name}: in-spot car AP {apv()*100:.1f}% | "
          f"AP50 {apv(0)*100:.1f}% ({kept:,} in-spot GT, {ign:,} ignored, "
          f"{miss} unmatched)")


if __name__ == "__main__":
    main()
