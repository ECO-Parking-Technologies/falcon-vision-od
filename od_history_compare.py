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
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()


DEBUG = False
# Sensors truncate their archive after a handful of days — everything pulled
# is cached here permanently (backed up with the other valuables), so a
# later matched-window rerun never depends on data the sensor has expired.
CACHE = Path("data/sensor_archive")


def host_slug(host):
    return host.split("//")[-1].split("/")[0].split(".")[0]


def cached_json(url, cache_file, headers, complete, timeout=15):
    """GET with a permanent per-hour cache. Only COMPLETE past hours are
    written (an in-progress hour would freeze a partial listing)."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    payload = None
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if DEBUG:
            ct = r.headers.get("content-type", "?")
            console.print(f"[dim]{r.status_code} {ct} {len(r.content)}B "
                          f"{url}[/dim]")
            if "json" not in ct:
                console.print(f"[dim]  body: {r.text[:120]!r}[/dim]")
        payload = r.json() if r.ok else None
    except Exception as e:
        if DEBUG:
            console.print(f"[dim]  fetch/parse failed: {type(e).__name__}[/dim]")
    if payload is not None and complete:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
    return payload or {}


def hour_complete(t):
    return t + timedelta(hours=1) < datetime.now(timezone.utc)


def fetch_window(host, cam, start, end, headers):
    """All archived frames with start <= ts < end (UTC hour steps)."""
    frames = []
    hours_with_data = 0
    base = CACHE / host_slug(host) / f"camera-{cam}"
    t = start.replace(minute=0, second=0, microsecond=0)
    while t < end:
        url = (f"{host}/archive/objects/data/camera-{cam}/"
               f"{t.year}/{t.month}/{t.day}/{t.hour}")
        cf = base / f"objects-{t:%Y-%m-%d-%H}.json"
        data = cached_json(url, cf, headers, hour_complete(t)).get("data", [])
        got = 0
        for f in data:
            ts = datetime.fromtimestamp(f["ts"] / 1000, tz=timezone.utc)
            if start <= ts < end:
                frames.append(f)
                got += 1
        hours_with_data += bool(got)
        t += timedelta(hours=1)
        if not DEBUG and hours_with_data and hours_with_data % 12 == 0 and got:
            console.print(f"[dim]  …{hours_with_data} hours fetched, "
                          f"{len(frames)} frames so far[/dim]")
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


def list_images(host, cam, start, end, headers):
    """Archived image entries for [start, end], hour-cached like detections.
    The listing endpoint is not camera-scoped — cache all, filter here."""
    base = CACHE / host_slug(host)
    images = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        url = f"{host}/archive/image/files/{t.year}/{t.month}/{t.day}/{t.hour}"
        cf = base / f"imagefiles-{t:%Y-%m-%d-%H}.json"
        entries = cached_json(url, cf, headers, hour_complete(t)).get("data", [])
        images += [
            {"fileName": e["fileName"],
             "ts": datetime.fromisoformat(
                 e["dateTime"].replace("Z", "+00:00")).timestamp() * 1000}
            for e in entries if e.get("cameraId") == cam]
        t += timedelta(hours=1)
    images.sort(key=lambda e: e["ts"])
    return images


def fetch_image(host, file_name, headers):
    """Image bytes, cached permanently (images are immutable once archived)."""
    cf = CACHE / host_slug(host) / "images" / Path(file_name).name
    if cf.exists():
        return cf.read_bytes()
    try:
        r = requests.get(f"{host}/archive/image/image/{file_name}",
                         headers=headers, timeout=20)
        if not r.ok or not r.content:
            return None
    except Exception:
        return None
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_bytes(r.content)
    return r.content


def mirror_images(host, cam, start, end, headers):
    """Download every archived image in the window into the local cache —
    run this before the sensor's rolling truncation eats the evidence."""
    entries = list_images(host, cam, start, end, headers)
    got = new = 0
    for e in entries:
        cf = CACHE / host_slug(host) / "images" / Path(e["fileName"]).name
        existed = cf.exists()
        if fetch_image(host, e["fileName"], headers) is not None:
            got += 1
            if not existed:
                new += 1
                if new % 50 == 0:
                    console.print(f"[dim]  …mirrored {new} new images[/dim]")
    console.print(f"[green]mirror: {got}/{len(entries)} images in cache "
                  f"({new} newly downloaded)[/green]")


def grade_vs_sam3(host, cam, frames, headers, sam3, n_sample, label):
    """Sample frames, fetch their archive images, draft with SAM3, and score
    the archived sensor detections against the teacher (IoU>=0.5 greedy)."""
    import cv2
    import numpy as np

    if not frames:
        return None
    frames = sorted(frames, key=lambda f: f["ts"])
    frame_ts = [f["ts"] for f in frames]

    # IMAGES are the scarce resource (archived far less often than detection
    # frames) — so sample images evenly across the window and match each to
    # its detection frame, not the other way around.
    from bisect import bisect_left
    start = datetime.fromtimestamp(frame_ts[0] / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(frame_ts[-1] / 1000, tz=timezone.utc)
    images = list_images(host, cam, start, end, headers)
    console.print(f"[dim]{label}: {len(images)} archived images in window[/dim]")
    if not images:
        return None
    step = max(1, len(images) // n_sample)
    sample = images[::step][:n_sample]

    stats = {t: {"tp": 0, "det": 0, "gt": 0} for t in (0.25, 0.40)}
    graded = 0
    for best in sample:
        # nearest detection frame to this image
        i = bisect_left(frame_ts, best["ts"])
        cands = [j for j in (i - 1, i) if 0 <= j < len(frames)]
        j = min(cands, key=lambda j: abs(frame_ts[j] - best["ts"]))
        f = frames[j]
        if abs(frame_ts[j] - best["ts"]) > 10_000:   # >10s apart: skip
            continue
        raw = fetch_image(host, best["fileName"], headers)
        img = (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
               if raw else None)
        if img is None:
            continue
        H, W = img.shape[:2]
        _, dets = sam3.infer(img, input_size=(W, H))
        if graded % 10 == 9:
            console.print(f"[dim]  …{label}: {graded + 1}/{len(sample)} "
                          "frames graded[/dim]")
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
    ap.add_argument("--new-start", default=None,
                    help="ISO start of the new-model window (default: same "
                         "as --cutoff). Set it later than the cutoff to "
                         "leave a buffer around the install for clean "
                         "old/new separation")
    ap.add_argument("--after-hours", type=int, default=None,
                    help="new-model window length (default: start -> now)")
    ap.add_argument("--sam3", type=int, default=0, metavar="N",
                    help="grade each window vs SAM3 on N sampled archive "
                         "frames (downloads images; runs the teacher locally)")
    ap.add_argument("--sam3-old", type=int, default=None, metavar="N",
                    help="override N for the old-model window (e.g. sample "
                         "a long weekend window more densely)")
    ap.add_argument("--sam3-new", type=int, default=None, metavar="N",
                    help="override N for the new-model window")
    ap.add_argument("--cf-access", action="store_true",
                    help="Cloudflare Access service token (prompted, RAM only)")
    ap.add_argument("--mirror", action="store_true",
                    help="download EVERY archived image in both windows into "
                         "data/sensor_archive/ (protects evidence from the "
                         "sensor's rolling truncation)")
    ap.add_argument("--debug", action="store_true",
                    help="print request status/content-type + enumerate the "
                         "sensor's cameras/spaces before pulling")
    args = ap.parse_args()
    global DEBUG
    DEBUG = args.debug

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
    a0 = (datetime.fromisoformat(args.new_start).astimezone(timezone.utc)
          if args.new_start else cutoff)
    a1 = (a0 + timedelta(hours=args.after_hours)
          if args.after_hours else now)

    host = args.host.rstrip("/")
    console.print(f"old model: [bold]{b0:%m-%d %H:%M} -> {b1:%m-%d %H:%M}[/] UTC")
    console.print(f"new model: [bold]{a0:%m-%d %H:%M} -> {a1:%m-%d %H:%M}[/] UTC")
    old_frames, old_hours = fetch_window(host, args.camera, b0, b1, headers)
    new_frames, new_hours = fetch_window(host, args.camera, a0, a1, headers)
    if args.mirror:
        mirror_images(host, args.camera, b0, a1, headers)
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
                              args.sam3_old or args.sam3, "old model")
        g_new = grade_vs_sam3(host, args.camera, new_frames, headers, sam3,
                              args.sam3_new or args.sam3, "new model")
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
