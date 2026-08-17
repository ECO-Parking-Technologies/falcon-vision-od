#!/usr/bin/env python3
"""Per-tier spot-occupancy accuracy across the FULL validation side of the
store — the first half of the roadmap's spot-occupancy evaluator.

For every frame of every VAL-side sensor (sensor-hash split, salt falcon-v1 —
tiers never trained on these), match the frame to its own snapshot-run spot
polygons, then per spot: truth = SAM3's stamped InEcoParkingSpot vehicles;
prediction = tier detection (conf >= 0.40, vehicle) center-in-spot or >=30%
overlap. Confusion + accuracy per tier, overall and per garage.

Train-side sensors are EXCLUDED by construction (would inflate accuracy).

    python3 eval_occupancy_tiers.py [--limit-sensors N] [--out PATH]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision  # registers nms op for the ts.pt traces
from PIL import Image

from preannotation.label_inspot import (frame_month, frame_run8, point_in_poly,
                                        spaces_for)

SESSION = "runs/20260812-121457"
TIERS = {  # tier -> (ts.pt, native size, letterbox fill per v2 recipe)
    "lite0": (f"{SESSION}/lite0/export/20260812-121457-lite0.ts.pt", 320, (128, 128, 128)),
    "lite1": (f"{SESSION}/lite1/export/20260812-121457-lite1.ts.pt", 384, (0, 0, 0)),
    "lite2": (f"{SESSION}/lite2/export/20260812-121457-lite2.ts.pt", 448, (0, 0, 0)),
    "lite3": (f"{SESSION}/lite3/export/20260812-121457-lite3.ts.pt", 512, (0, 0, 0)),
    "lite4": (f"{SESSION}/lite4/export/20260812-121457-lite4.ts.pt", 640, (0, 0, 0)),
}
SALT = "falcon-v1"
VAL_FRAC = 0.2
CONF = 0.40


def is_val_sensor(garage, sensor):
    h = int.from_bytes(
        hashlib.md5(f"{SALT}:{garage}/{sensor}".encode()).digest()[:8], "big")
    return (h % 10_000) < round(10_000 * VAL_FRAC)


def overlap_frac(poly, b):
    return sum(point_in_poly(b[0] + (i + .5) / 5 * (b[2] - b[0]),
                             b[1] + (j + .5) / 5 * (b[3] - b[1]), poly)
               for i in range(5) for j in range(5)) / 25.0


def box_in_poly(b, poly):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return point_in_poly(cx, cy, poly) or overlap_frac(poly, b) >= 0.30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=Path("data/images"))
    ap.add_argument("--polygons", type=Path, default=Path("data/spot_polygons.json"))
    ap.add_argument("--limit-sensors", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for t, (p, size, fill) in TIERS.items():
        m = torch.jit.load(p, map_location=dev).eval()
        models[t] = (m, size, fill)
    print(f"5 tiers loaded on {dev}")

    polys_db = json.loads(args.polygons.read_text())
    files = sorted(args.store.glob("*/*/preannotations.coco.json"))
    val_files = [f for f in files
                 if is_val_sensor(f.parent.parent.name, f.parent.name)]
    if args.limit_sensors:
        val_files = val_files[:args.limit_sensors]
    print(f"{len(val_files)} VAL-side sensors of {len(files)} total")

    conf = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in TIERS}
    per_garage = {}
    n_frames = n_skipped = 0
    t0 = time.time()
    import re
    for jf in val_files:
        garage, sensor = jf.parent.parent.name, jf.parent.name
        gp = polys_db.get(garage, {})
        coco = json.loads(jf.read_text())
        imgs = {im["id"]: im for im in coco["images"]}
        per_img = {}
        for a in coco["annotations"]:
            per_img.setdefault(a["image_id"], []).append(a)
        for imid, im in imgs.items():
            cam = re.search(r"camera(\d+)", im["file_name"])
            key = f"{sensor}|camera{cam.group(1)}" if cam else ""
            spaces, _ = spaces_for(gp, key, frame_run8(im["file_name"]),
                                   frame_month(im["file_name"]))
            if not spaces:
                n_skipped += 1
                continue
            path = args.store / im["file_name"]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                n_skipped += 1
                continue
            W, H = img.size
            anns = per_img.get(imid, [])
            truth_spots = {a.get("attributes", {}).get("spot")
                           for a in anns
                           if a["category_id"] != 1
                           and a.get("attributes", {}).get("InEcoParkingSpot")}
            n_frames += 1
            g = per_garage.setdefault(garage, {t: {"tp":0,"fp":0,"fn":0,"tn":0}
                                               for t in TIERS})
            for tier, (m, size, fill) in models.items():
                s = size / max(W, H)
                canvas = Image.new("RGB", (size, size), fill)
                canvas.paste(img.resize((int(W*s), int(H*s)), Image.BILINEAR))
                x = torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1)
                x = x.float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(dev)
                with torch.no_grad():
                    det = m(x)[0].cpu()
                boxes = []
                for row in det:
                    x0, y0, x1, y1, sc, cls = row.tolist()
                    if sc < CONF or int(cls) == 1:
                        continue
                    boxes.append([x0/s/W, y0/s/H, x1/s/W, y1/s/H])
                for sp in spaces:
                    truth = sp["space"] in truth_spots
                    pred = any(box_in_poly(b, sp["points"]) for b in boxes)
                    k = ("tp" if truth and pred else "fp" if pred
                         else "fn" if truth else "tn")
                    conf[tier][k] += 1
                    g[tier][k] += 1
        done = val_files.index(jf) + 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  …{done}/{len(val_files)} sensors, {n_frames} frames, "
                  f"{el/60:.0f} min", flush=True)

    print(f"\n=== SPOT-OCCUPANCY ACCURACY — {n_frames} val frames, "
          f"{n_skipped} skipped (no calibration) ===")
    print(f"{'tier':6s} {'TP':>7s} {'FP':>6s} {'FN':>6s} {'TN':>8s} "
          f"{'accuracy':>9s} {'FN-rate':>8s}")
    for t, c in conf.items():
        n = sum(c.values())
        acc = (c["tp"] + c["tn"]) / n if n else 0
        fnr = c["fn"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        print(f"{t:6s} {c['tp']:7d} {c['fp']:6d} {c['fn']:6d} {c['tn']:8d} "
              f"{acc:8.2%} {fnr:7.2%}")
    out = args.out or Path(f"data/occupancy_eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps({"overall": conf, "per_garage": per_garage,
                               "frames": n_frames, "skipped": n_skipped,
                               "conf_threshold": CONF, "session": SESSION},
                              indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
