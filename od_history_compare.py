#!/usr/bin/env python3
"""Compare a sensor's OD detection history before/after a model swap.

Pulls the same hourly archive the od-history.shtml page reads
(/archive/objects/data/camera-<n>/<Y>/<M>/<D>/<H>, post-NMS frames) for a
window before the swap (old model) and after it (new model), and prints
side-by-side detection statistics. Point it at the SAME sensor so the scene
is held constant — differences are then the model's.

    python3 od_history_compare.py --host https://<sensor> --camera 1 \
        --cutoff 2026-08-10T09:41:00-04:00 --before-hours 48 --cf-access

Cloudflare Access service-token creds are prompted (RAM only, same policy
as every portal/CVAT tool). Plain LAN sensors: --host http://<ip> and omit
--cf-access.
"""
import argparse
import statistics
from datetime import datetime, timedelta, timezone

import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def fetch_window(host, cam, start, end, headers, timeout=15):
    """All archived frames with start <= ts < end (UTC hour steps)."""
    frames = []
    hours_with_data = 0
    t = start.replace(minute=0, second=0, microsecond=0)
    while t < end:
        url = (f"{host}/archive/objects/data/camera-{cam}/"
               f"{t.year}/{t.month}/{t.day}/{t.hour}")
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            data = r.json().get("data", []) if r.ok else []
        except Exception:
            data = []
        got = 0
        for f in data:
            ts = datetime.fromtimestamp(f["ts"] / 1000, tz=timezone.utc)
            if start <= ts < end:
                frames.append(f)
                got += 1
        hours_with_data += bool(got)
        t += timedelta(hours=1)
    return frames, hours_with_data


def window_stats(frames):
    n = len(frames)
    confs, types = [], {}
    per_frame_counts, per_frame_strong = [], []
    for f in frames:
        objs = f.get("objects") or []
        if not isinstance(objs, list):
            objs = list(objs.values())
        per_frame_counts.append(len(objs))
        per_frame_strong.append(sum(1 for o in objs
                                    if (o.get("confidence") or 0) >= 0.40))
        for o in objs:
            c = o.get("confidence") or 0
            confs.append(c)
            key = o.get("subType") or o.get("type") or "?"
            types[key] = types.get(key, 0) + 1
    return {
        "frames": n,
        "objects": len(confs),
        "objs_per_frame": statistics.mean(per_frame_counts) if n else 0,
        "strong_per_frame": statistics.mean(per_frame_strong) if n else 0,
        "conf_mean": statistics.mean(confs) if confs else 0,
        "conf_p50": statistics.median(confs) if confs else 0,
        "conf_p90": (statistics.quantiles(confs, n=10)[8]
                     if len(confs) >= 10 else (max(confs) if confs else 0)),
        "ge40_frac": (sum(1 for c in confs if c >= 0.40) / len(confs)
                      if confs else 0),
        "types": types,
    }


SAM3_VEHICLE_CATS = {2, 3, 4, 6, 8}   # bicycle,car,moto,bus,truck -> VEHICLE


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sensor_boxes(frame, min_conf):
    """Archived objects -> normalized [x1,y1,x2,y2] vehicle boxes."""
    out = []
    objs = frame.get("objects") or []
    if not isinstance(objs, list):
        objs = list(objs.values())
    for o in objs:
        tl, br = o.get("topLeft"), o.get("bottomRight")
        if not tl or not br or (o.get("confidence") or 0) < min_conf:
            continue
        if (o.get("type") or "").upper() == "PERSON":
            continue
        out.append([tl["x"], tl["y"], br["x"], br["y"]])
    return out


def grade_vs_sam3(host, cam, frames, headers, sam3, n_sample, label):
    """Sample frames, fetch their archive images, draft with SAM3, and score
    the archived sensor detections against the teacher (IoU>=0.5 greedy)."""
    import cv2
    import numpy as np

    if not frames:
        return None
    step = max(1, len(frames) // n_sample)
    sample = frames[::step][:n_sample]
    # archive image listings, cached per hour
    img_index = {}

    def images_for(ts):
        t = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        key = (t.year, t.month, t.day, t.hour)
        if key not in img_index:
            url = f"{host}/archive/image/files/{key[0]}/{key[1]}/{key[2]}/{key[3]}"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                entries = r.json().get("data", []) if r.ok else []
            except Exception:
                entries = []
            img_index[key] = [
                {"fileName": e["fileName"],
                 "ts": datetime.fromisoformat(
                     e["dateTime"].replace("Z", "+00:00")).timestamp() * 1000}
                for e in entries if e.get("cameraId") == cam]
        return img_index[key]

    stats = {t: {"tp": 0, "det": 0, "gt": 0} for t in (0.25, 0.40)}
    graded = 0
    for f in sample:
        cands = images_for(f["ts"])
        if not cands:
            continue
        best = min(cands, key=lambda e: abs(e["ts"] - f["ts"]))
        if abs(best["ts"] - f["ts"]) > 10_000:   # >10s apart: not this frame
            continue
        try:
            r = requests.get(f"{host}/archive/image/image/{best['fileName']}",
                             headers=headers, timeout=20)
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8),
                               cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            continue
        H, W = img.shape[:2]
        _, dets = sam3.infer(img, input_size=(W, H))
        gt = [[d[0] / W, d[1] / H, d[2] / W, d[3] / H]
              for d in dets if int(d[5]) in SAM3_VEHICLE_CATS]
        graded += 1
        for thr in stats:
            det = sensor_boxes(f, thr)
            used = set()
            tp = 0
            for db in det:
                m = max(((iou(db, g), j) for j, g in enumerate(gt)
                         if j not in used), default=(0, -1))
                if m[0] >= 0.5:
                    used.add(m[1])
                    tp += 1
            stats[thr]["tp"] += tp
            stats[thr]["det"] += len(det)
            stats[thr]["gt"] += len(gt)
    console.print(f"[dim]{label}: graded {graded} frames vs SAM3[/dim]")
    return stats if graded else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True,
                    help="sensor base URL (https://<cf-hostname> or http://<lan-ip>)")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--cutoff", required=True,
                    help="model-swap time, ISO with offset "
                         "(e.g. 2026-08-10T09:41:00-04:00)")
    ap.add_argument("--before-hours", type=int, default=48,
                    help="old-model window: this many hours before cutoff")
    ap.add_argument("--after-hours", type=int, default=None,
                    help="new-model window length (default: cutoff -> now)")
    ap.add_argument("--sam3", type=int, default=0, metavar="N",
                    help="grade each window vs SAM3 on N sampled archive "
                         "frames (downloads images; runs the teacher locally)")
    ap.add_argument("--cf-access", action="store_true",
                    help="Cloudflare Access service token (prompted, RAM only)")
    args = ap.parse_args()

    headers = {}
    if args.cf_access:
        headers = {
            "CF-Access-Client-Id":
                Prompt.ask("[cyan]CF-Access-Client-Id[/cyan]",
                           console=console).strip(),
            "CF-Access-Client-Secret":
                Prompt.ask("[cyan]CF-Access-Client-Secret[/cyan]",
                           password=True, console=console).strip(),
        }

    cutoff = datetime.fromisoformat(args.cutoff).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    b0, b1 = cutoff - timedelta(hours=args.before_hours), cutoff
    a0 = cutoff
    a1 = (cutoff + timedelta(hours=args.after_hours)
          if args.after_hours else now)

    host = args.host.rstrip("/")
    console.print(f"old model: [bold]{b0:%m-%d %H:%M} -> {b1:%m-%d %H:%M}[/] UTC")
    console.print(f"new model: [bold]{a0:%m-%d %H:%M} -> {a1:%m-%d %H:%M}[/] UTC")
    old_frames, old_hours = fetch_window(host, args.camera, b0, b1, headers)
    new_frames, new_hours = fetch_window(host, args.camera, a0, a1, headers)
    old, new = window_stats(old_frames), window_stats(new_frames)

    t = Table(title=f"camera {args.camera} — OD archive, old vs new model")
    t.add_column("metric")
    t.add_column("old model", justify="right")
    t.add_column("new model", justify="right")
    rows = [
        ("hours with data", old_hours, new_hours),
        ("frames archived", old["frames"], new["frames"]),
        ("objects total", old["objects"], new["objects"]),
        ("objects / frame", f"{old['objs_per_frame']:.2f}",
         f"{new['objs_per_frame']:.2f}"),
        (">=0.40 (strong) / frame", f"{old['strong_per_frame']:.2f}",
         f"{new['strong_per_frame']:.2f}"),
        ("confidence mean", f"{old['conf_mean']:.3f}", f"{new['conf_mean']:.3f}"),
        ("confidence p50", f"{old['conf_p50']:.3f}", f"{new['conf_p50']:.3f}"),
        ("confidence p90", f"{old['conf_p90']:.3f}", f"{new['conf_p90']:.3f}"),
        ("frac objects >=0.40", f"{old['ge40_frac']:.2f}", f"{new['ge40_frac']:.2f}"),
    ]
    for name, a, b in rows:
        t.add_row(name, str(a), str(b))
    console.print(t)

    tt = Table(title="type / subType mix")
    tt.add_column("type")
    tt.add_column("old", justify="right")
    tt.add_column("new", justify="right")
    for k in sorted(set(old["types"]) | set(new["types"])):
        tt.add_row(k, str(old["types"].get(k, 0)), str(new["types"].get(k, 0)))
    console.print(tt)
    if not old["frames"] or not new["frames"]:
        console.print("[yellow]one window has no frames — check camera id, "
                      "date range, or whether archiving is enabled[/yellow]")

    if args.sam3:
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).parent / "preannotation"))
        from sam3_model import Sam3DraftModel
        console.print("[dim]loading SAM 3 (cache-first)…[/dim]")
        sam3 = Sam3DraftModel()
        g_old = grade_vs_sam3(host, args.camera, old_frames, headers, sam3,
                              args.sam3, "old model")
        g_new = grade_vs_sam3(host, args.camera, new_frames, headers, sam3,
                              args.sam3, "new model")
        gt = Table(title="vs SAM 3 ground truth (vehicle boxes, IoU>=0.5)")
        gt.add_column("metric")
        gt.add_column("old model", justify="right")
        gt.add_column("new model", justify="right")

        def pr(s, thr):
            if not s:
                return "—", "—"
            d = s[thr]
            p = d["tp"] / d["det"] if d["det"] else 0
            r = d["tp"] / d["gt"] if d["gt"] else 0
            return f"{p:.2f}", f"{r:.2f}"

        for thr in (0.25, 0.40):
            po, ro = pr(g_old, thr)
            pn, rn = pr(g_new, thr)
            gt.add_row(f"precision @conf>={thr}", po, pn)
            gt.add_row(f"recall    @conf>={thr}", ro, rn)
        console.print(gt)


if __name__ == "__main__":
    main()
