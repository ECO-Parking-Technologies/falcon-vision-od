#!/usr/bin/env python3
"""In-spot car accuracy: the product metric.

Scores a training run's saved val predictions against ONLY the boxes stamped
InEcoParkingSpot (preannotation/label_inspot.py); all other GT becomes a COCO
ignore region (iscrowd=1) so background/distant cars neither reward nor
penalize. Requires the run's val_predictions.json (run_metrics.py creates it).

    python3 eval_inspot.py runs/train/<run-dir>
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
    ap.add_argument("run_dir", type=Path, nargs="?",
                    help="run dir (or give --split + --predictions instead, "
                         "e.g. to score another model's checkpoint on this "
                         "run's split)")
    ap.add_argument("--split", type=Path, help="split dir override")
    ap.add_argument("--predictions", type=Path,
                    help="COCO predictions json override")
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    args = ap.parse_args()
    if args.run_dir is None and not (args.split and args.predictions):
        ap.error("need a run_dir, or both --split and --predictions")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # accept a <session>/<level> dir or its train/ subdir; find manifest,
    # predictions, and the split (session layout first, legacy fallback)
    split = args.split
    if args.run_dir is not None:
        rd = args.run_dir
        train_dir = rd / "train" if (rd / "train").exists() else rd
        man = {}
        for mf in (train_dir / "run.json", train_dir.parent / "run.json"):
            if mf.exists():
                man = json.loads(mf.read_text())
                break
        if split is None:
            for cand in (train_dir.parent / "split",
                         train_dir.parent.parent / "split",
                         Path(man.get("split_dir", "/nonexistent"))):
                if (cand / "annotations" / "instances_val2017.json").exists():
                    split = cand
                    break
    if split is None or not (split / "annotations" /
                             "instances_val2017.json").exists():
        raise SystemExit(f"no split found for {args.run_dir or args.split}")
    preds = args.predictions or (train_dir / "val_predictions.json")
    name = args.predictions.stem if args.predictions else train_dir.name
    val = json.loads((split / "annotations" /
                      "instances_val2017.json").read_text())
    imgs = {i["id"]: i["file_name"] for i in val["images"]}
    kept = ign = miss = 0
    for a in val["annotations"]:
        if "attributes" in a:
            # roi_crop splits carry attributes inline (crop-shifted boxes
            # can't be matched back to the store drafts by bbox)
            inspot = bool(a["attributes"].get("InEcoParkingSpot"))
        else:
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
    dt = gt.loadRes(str(preds))
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
    print(f"{name}: in-spot car AP {apv()*100:.1f}% | "
          f"AP50 {apv(0)*100:.1f}% ({kept:,} in-spot GT, {ign:,} ignored, "
          f"{miss} unmatched)")


if __name__ == "__main__":
    main()
