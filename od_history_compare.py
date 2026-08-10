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


if __name__ == "__main__":
    main()
