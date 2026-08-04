#!/usr/bin/env python3
"""Render SAM3 draft annotations onto frames for visual inspection.

Samples frames across the store (or one garage/sensor), draws the boxes from
each sensor's preannotations.coco.json, and writes JPGs plus an index.html to
browse locally. LOCAL VIEWING ONLY — these are customer CCTV frames.

    python3 preannotation/render_drafts.py                     # 3/garage, all garages
    python3 preannotation/render_drafts.py --garage yaamava-north-garage --count 24
    python3 preannotation/render_drafts.py --sensor yaamava-north-garage/l1-wl-s01
"""
import argparse
import json
import random
from pathlib import Path

import cv2

COLORS = {"person": (0, 200, 255), "bicycle": (255, 180, 0),
          "car": (80, 220, 80), "motorcycle": (255, 0, 200),
          "bus": (0, 120, 255), "truck": (60, 60, 255)}


def render(img, anns, cats):
    for a in anns:
        x, y, w, h = (int(v) for v in a["bbox"])
        name = cats[a["category_id"]]
        c = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        cv2.putText(img, name, (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    ap.add_argument("--out", type=Path, default=Path("data/draft_previews"))
    ap.add_argument("--garage", help="only this garage")
    ap.add_argument("--sensor", help="only this garage/sensor")
    ap.add_argument("--count", type=int, default=3,
                    help="frames per garage (or total for --sensor)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.sensor:
        jsons = [args.store / args.sensor / "preannotations.coco.json"]
    elif args.garage:
        jsons = sorted((args.store / args.garage).glob("*/preannotations.coco.json"))
    else:
        jsons = sorted(args.store.glob("*/*/preannotations.coco.json"))

    args.out.mkdir(parents=True, exist_ok=True)
    rows, per_garage = [], {}
    for j in jsons:
        garage = j.parent.parent.name
        per_garage.setdefault(garage, []).append(j)

    for garage, files in sorted(per_garage.items()):
        budget = args.count
        rng.shuffle(files)
        for j in files:
            if budget <= 0:
                break
            d = json.loads(j.read_text())
            cats = {c["id"]: c["name"] for c in d["categories"]}
            by_img = {}
            for a in d["annotations"]:
                by_img.setdefault(a["image_id"], []).append(a)
            imgs = list(d["images"])
            rng.shuffle(imgs)
            for im in imgs[:1]:  # one frame per sensor file, spread wide
                src = args.store / im["file_name"]
                frame = cv2.imread(str(src))
                if frame is None:
                    continue
                anns = by_img.get(im["id"], [])
                name = f"{garage}__{Path(im['file_name']).name}.jpg"
                cv2.imwrite(str(args.out / name),
                            render(frame, anns, cats),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                rows.append((garage, name, len(anns)))
                budget -= 1

    body = "\n".join(
        f'<div><h3>{g} &mdash; {n} boxes</h3>'
        f'<img src="{f}" loading="lazy" style="max-width:100%"></div>'
        for g, f, n in rows)
    (args.out / "index.html").write_text(
        "<style>body{background:#111;color:#eee;font-family:sans-serif;"
        "max-width:1100px;margin:auto}img{margin-bottom:20px}</style>"
        f"<h1>SAM3 drafts &mdash; {len(rows)} frames</h1>{body}")
    print(f"{len(rows)} frames rendered -> {args.out}/index.html")


if __name__ == "__main__":
    main()
