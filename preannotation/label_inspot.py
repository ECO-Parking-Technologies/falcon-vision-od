#!/usr/bin/env python3
"""Stamp InEcoParkingSpot onto SAM3 draft boxes using portal spot polygons.

Walks every per-sensor preannotations.coco.json, matches each frame to its
(garage, sensor, cameraN) polygon set from data/spot_polygons.json, and adds
per-annotation CVAT-compatible attributes:

    "attributes": {"InEcoParkingSpot": true/false}   (+ "spot": name when true)

Rule: in-spot when the box CENTER lies inside a polygon OR intersection
covers >= 30% of the box area (mirrors the fusion overlap regime). Polygons
are normalized [0-1]; boxes are absolute pixels. Idempotent — rerunning
re-stamps from scratch.

    python3 preannotation/label_inspot.py [--store data/images] [--dry-run]
"""
import argparse
import json
import re
from pathlib import Path


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def clip_poly_area_over_box(poly, bx, by, bw, bh):
    """Approximate intersection(poly, box)/box_area by grid sampling (5x5)."""
    hits = 0
    for i in range(5):
        for j in range(5):
            px = bx + (i + 0.5) / 5 * bw
            py = by + (j + 0.5) / 5 * bh
            if point_in_poly(px, py, poly):
                hits += 1
    return hits / 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    ap.add_argument("--polygons", type=Path, default=Path("data/spot_polygons.json"))
    ap.add_argument("--min-overlap", type=float, default=0.30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    polys = json.loads(args.polygons.read_text())
    n_files = n_boxes = n_inspot = n_nomatch = 0

    for jf in sorted(args.store.glob("*/*/preannotations.coco.json")):
        garage, sensor = jf.parent.parent.name, jf.parent.name
        gmap = polys.get(garage, {})
        coco = json.loads(jf.read_text())
        imgs = {im["id"]: im for im in coco["images"]}
        changed = False
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            m = re.search(r"camera(\d+)", im["file_name"])
            key = f"{sensor}|camera{m.group(1)}" if m else None
            spaces = gmap.get(key or "", [])
            n_boxes += 1
            if not spaces:
                n_nomatch += 1
                a["attributes"] = {"InEcoParkingSpot": False}
                changed = True
                continue
            W, H = im.get("width", 640), im.get("height", 480)
            x, y, w, h = a["bbox"]
            nx, ny, nw, nh = x / W, y / H, w / W, h / H
            cx, cy = nx + nw / 2, ny + nh / 2
            hit = None
            for sp in spaces:
                if point_in_poly(cx, cy, sp["points"]) or \
                   clip_poly_area_over_box(sp["points"], nx, ny, nw, nh) >= args.min_overlap:
                    hit = sp["space"]
                    break
            a["attributes"] = {"InEcoParkingSpot": bool(hit)}
            if hit:
                a["attributes"]["spot"] = hit
                n_inspot += 1
            changed = True
        if changed and not args.dry_run:
            jf.write_text(json.dumps(coco, indent=2))
        n_files += 1
        if n_files % 500 == 0:
            print(f"  …{n_files} sensor files")

    print(f"{n_files} sensor files · {n_boxes:,} boxes · "
          f"{n_inspot:,} in-spot ({100*n_inspot//max(1,n_boxes)}%) · "
          f"{n_nomatch:,} boxes on sensors without polygon data"
          f"{' [DRY RUN — nothing written]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
