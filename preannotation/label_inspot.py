#!/usr/bin/env python3
"""Stamp InEcoParkingSpot onto SAM3 draft boxes — calibration-aware.

Spot calibrations change over time, so polygons are matched per-frame:
1. EXACT: the frame's filename embeds its snapshot run id
   (<sensor>-<run8>-<mac>-cameraN) -> that run's own polygon set.
2. FALLBACK: frames without a run match use the sensor-camera's timeline
   entry nearest the frame's capture month (store path <YYYY>/<MM>/).

Adds CVAT-compatible per-annotation attributes:
    "attributes": {"InEcoParkingSpot": bool, ["spot": name]}
Rule: box CENTER inside a polygon OR intersection >= 30% of box area
(mirrors the fusion overlap regime). Idempotent — rerun re-stamps.

    python3 preannotation/label_inspot.py [--dry-run]
"""
import argparse
import json
import re
from datetime import datetime
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


def overlap_frac(poly, bx, by, bw, bh):
    """intersection(poly, box)/box_area by 5x5 grid sampling."""
    hits = 0
    for i in range(5):
        for j in range(5):
            if point_in_poly(bx + (i + 0.5) / 5 * bw,
                             by + (j + 0.5) / 5 * bh, poly):
                hits += 1
    return hits / 25.0


def frame_run8(file_name):
    """Extract the snapshot run id (8 hex) from a store filename."""
    m = re.search(r"-([0-9a-f]{8})-[0-9a-f_]{17}-camera\d", Path(file_name).name)
    return m.group(1) if m else None


def frame_month(file_name):
    m = re.search(r"/(\d{4})/(\d{2})/", file_name)
    return datetime(int(m.group(1)), int(m.group(2)), 15) if m else None


def spaces_for(garage_polys, key, run8, month):
    """Exact run match first; else nearest-in-time timeline entry."""
    if run8 and run8 in garage_polys.get("runs", {}):
        sp = garage_polys["runs"][run8].get(key)
        if sp:
            return sp, "exact"
    tl = garage_polys.get("timeline", {}).get(key, [])
    if tl and month:
        best = min(tl, key=lambda e: abs(
            (datetime.fromisoformat(e["at"][:19]) - month).total_seconds()))
        sp = garage_polys["runs"].get(best["run"], {}).get(key)
        if sp:
            return sp, "timeline"
    if tl:  # no month either: newest known calibration
        sp = garage_polys["runs"].get(tl[-1]["run"], {}).get(key)
        if sp:
            return sp, "latest"
    return None, "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    ap.add_argument("--polygons", type=Path, default=Path("data/spot_polygons.json"))
    ap.add_argument("--min-overlap", type=float, default=0.30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    polys = json.loads(args.polygons.read_text())
    n_files = n_boxes = n_inspot = 0
    match_counts = {"exact": 0, "timeline": 0, "latest": 0, "none": 0}

    for jf in sorted(args.store.glob("*/*/preannotations.coco.json")):
        garage, sensor = jf.parent.parent.name, jf.parent.name
        gp = polys.get(garage, {})
        coco = json.loads(jf.read_text())
        imgs = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            n_boxes += 1
            cam = re.search(r"camera(\d+)", im["file_name"])
            key = f"{sensor}|camera{cam.group(1)}" if cam else ""
            spaces, how = spaces_for(gp, key, frame_run8(im["file_name"]),
                                     frame_month(im["file_name"]))
            match_counts[how] += 1
            if not spaces:
                a["attributes"] = {"InEcoParkingSpot": False}
                continue
            W, H = im.get("width", 640), im.get("height", 480)
            x, y, w, h = a["bbox"]
            nx, ny, nw, nh = x / W, y / H, w / W, h / H
            cx, cy = nx + nw / 2, ny + nh / 2
            hit = None
            for sp in spaces:
                if point_in_poly(cx, cy, sp["points"]) or \
                   overlap_frac(sp["points"], nx, ny, nw, nh) >= args.min_overlap:
                    hit = sp["space"]
                    break
            a["attributes"] = {"InEcoParkingSpot": bool(hit)}
            if hit:
                a["attributes"]["spot"] = hit
                n_inspot += 1
        if not args.dry_run:
            jf.write_text(json.dumps(coco, indent=2))
        n_files += 1
        if n_files % 500 == 0:
            print(f"  …{n_files} sensor files")

    print(f"{n_files} sensor files · {n_boxes:,} boxes · "
          f"{n_inspot:,} in-spot ({100*n_inspot//max(1,n_boxes)}%)")
    print(f"calibration match: exact {match_counts['exact']:,} · "
          f"nearest-in-time {match_counts['timeline']:,} · "
          f"latest-fallback {match_counts['latest']:,} · "
          f"no-polygons {match_counts['none']:,}"
          f"{'  [DRY RUN — nothing written]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
