#!/usr/bin/env python3
"""Diversity sampler: pick the annotation queue from the unified image store.

Selects up to --target frames with per-garage caps, per-sensor spread, and
time-of-day/date coverage, then drops residual near-duplicates (dHash on the
picked frames only). Output is a queue JSON the preannotation runner consumes
via `queue_file:` in its config.

    python3 preannotation/sample_queue.py --target 5000
    → data/annotation_queue_phase1.json
"""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2


def dhash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8))
    return sum(1 << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).flatten()) if v)


def hour_bucket(ts):  # night / morning / day / evening
    try:
        return int(ts[11:13]) // 6
    except (ValueError, IndexError):
        return 2


def spread_pick(items, k):
    """Evenly-strided picks from a time-sorted list."""
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <data-root>/annotation_queue_phase1.json")
    ap.add_argument("--min-per-garage", type=int, default=40)
    args = ap.parse_args()
    out = args.out or args.data_root / "annotation_queue_phase1.json"
    images_root = (args.data_root / "images").resolve()

    db = sqlite3.connect(f"file:{args.data_root/'manifest.sqlite'}?mode=ro", uri=True)
    rows = db.execute(
        "select garage, sensor, ts, path from images where source='portal-snapshot'"
        " order by garage, sensor, ts").fetchall()

    by_garage = defaultdict(lambda: defaultdict(list))
    for g, s, ts, p in rows:
        by_garage[g][s].append((ts or "", p))

    n_garages = len(by_garage)
    cap = max(args.min_per_garage, args.target // n_garages)
    print(f"{len(rows)} frames, {n_garages} garages -> cap {cap}/garage, target {args.target}")

    queue = {}
    for g, sensors in sorted(by_garage.items()):
        # per-sensor quota, round-robin over hour buckets within each sensor
        per_sensor = max(1, cap // len(sensors))
        picked = []
        for s, frames in sorted(sensors.items()):
            buckets = defaultdict(list)
            for ts, p in frames:
                buckets[hour_bucket(ts)].append((ts, p))
            quota = {b: max(1, per_sensor // len(buckets)) for b in buckets}
            got = []
            for b, items in sorted(buckets.items()):
                got.extend(spread_pick(items, quota[b]))
            picked.extend(got[:per_sensor] if len(got) > per_sensor else got)
        picked = picked[:cap]
        # residual near-dup drop within the garage (consecutive per sensor)
        picked.sort()
        kept, prev = [], {}
        for ts, p in picked:
            sensor = Path(p).resolve().relative_to(images_root).parts[1]
            h = dhash(p)
            if h is not None and sensor in prev and bin(prev[sensor] ^ h).count("1") <= 6:
                continue
            if h is not None:
                prev[sensor] = h
            kept.append(p)
        queue[g] = [str(Path(p).resolve().relative_to(images_root)) for p in kept]
        print(f"  {g:56s} {len(queue[g]):4d}")

    total = sum(len(v) for v in queue.values())
    out.write_text(json.dumps(queue, indent=1))
    print(f"\nqueue: {total} frames -> {out}")


if __name__ == "__main__":
    main()
