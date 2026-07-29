#!/usr/bin/env python3
"""Package the annotation queue into per-garage CVAT task bundles.

For each garage: a flat directory of the queued frames (copied — filenames are
already unique: <sensor>-<run>-<mac>-cameraN.jpg) plus one merged
preannotations COCO whose file_names match the flat layout. In CVAT, create
one task per garage by selecting the whole directory from the share, then
upload the matching JSON (COCO 1.0).

    python3 preannotation/export_cvat_tasks.py
    → data/cvat_tasks/<garage>/{*.jpg, preannotations.coco.json}
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--queue", type=Path, default=None,
                    help="default: <data-root>/annotation_queue_phase1.json")
    args = ap.parse_args()
    root = args.data_root
    queue = json.loads((args.queue or root / "annotation_queue_phase1.json").read_text())
    out_root = root / "cvat_tasks"

    total_imgs = total_anns = 0
    for garage, paths in sorted(queue.items()):
        if not paths:
            continue
        gdir = out_root / garage
        gdir.mkdir(parents=True, exist_ok=True)

        # merge the garage's per-sensor preannotations, keyed by rel path
        anns_by_rel = {}
        for pj in (root / "images" / garage).glob("*/preannotations.coco.json"):
            d = json.loads(pj.read_text())
            sensor = pj.parent.name
            by_id = {im["id"]: f"{garage}/{sensor}/{Path(im['file_name']).relative_to(Path(garage) / sensor)}"
                     for im in d["images"]}
            for a in d["annotations"]:
                anns_by_rel.setdefault(by_id[a["image_id"]], []).append(a)
            cats = d["categories"]

        images, annotations = [], []
        img_id = ann_id = 1
        for rel in sorted(paths):
            src = root / "images" / rel
            flat = Path(rel).name
            dst = gdir / flat
            if not dst.exists():
                shutil.copy2(src, dst)
            images.append({"id": img_id, "file_name": flat})
            for a in anns_by_rel.get(rel, []):
                annotations.append({**a, "id": ann_id, "image_id": img_id})
                ann_id += 1
            img_id += 1

        out = {"images": images, "annotations": annotations, "categories": cats}
        (gdir / "preannotations.coco.json").write_text(json.dumps(out, indent=1))
        total_imgs += len(images)
        total_anns += len(annotations)
        print(f"  {garage:56s} {len(images):4d} imgs {len(annotations):5d} anns")

    print(f"\n{total_imgs} images, {total_anns} annotations -> {out_root}")


if __name__ == "__main__":
    main()
