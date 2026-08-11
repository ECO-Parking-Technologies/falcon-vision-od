#!/usr/bin/env python3
"""Fusion channel analysis: inference (classifier) vs OD vs fused decision,
each judged against SAM 3 truth, per spot.

Reads the same archive the fusion-history.shtml page does
(/archive/fusion-status/data/<spaceId>/<Y>/<M>/<D>/<H>: per-OD-run records
with inferenceState / odState / fusionState / decisionPath), pairs each
record with the nearest SAM3-graded frame from an od_history_compare run
log, and reports per channel: accuracy, disagreements, and who was right
when the channels disagreed.

    python3 fusion_history_compare.py \
        --host https://<sensor> --camera 1 \
        --calibration /tmp/<sensor>-calibration.json \
        --od-log data/sensor_archive/<sensor>/runs/<run>.json --cf-access

Windows are taken from the od-log (same ranges as the OD pull). Results are
cached hour-by-hour like the OD archive and a JSON log is written next to
the od run logs.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.prompt import Prompt
from rich.table import Table

from od_history_compare import CACHE, cached_json, console, hour_complete, host_slug
from preannotation.label_inspot import point_in_poly


def overlap_frac(poly, b):
    hits = sum(point_in_poly(b[0] + (i + .5) / 5 * (b[2] - b[0]),
                             b[1] + (j + .5) / 5 * (b[3] - b[1]), poly)
               for i in range(5) for j in range(5))
    return hits / 25.0


def box_in_spot(b, poly):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return point_in_poly(cx, cy, poly) or overlap_frac(poly, b) >= 0.30


def state_to_bool(s):
    """DETECT -> True; NO_DETECT/BACKGROUND -> False; else None (skip)."""
    s = (s or "").upper().replace("-", "_")
    if s == "DETECT":
        return True
    if s in ("NO_DETECT", "BACKGROUND"):
        return False
    return None


def fetch_fusion(host, space_id, start, end, headers):
    base = CACHE / host_slug(host) / f"space-{space_id}"
    recs = []
    t = start.replace(minute=0, second=0, microsecond=0)
    hours = 0
    while t < end:
        url = (f"{host}/archive/fusion-status/data/{space_id}/"
               f"{t.year}/{t.month}/{t.day}/{t.hour}")
        cf = base / f"fusionstatus-{t:%Y-%m-%d-%H}.json"
        data = cached_json(url, cf, headers, hour_complete(t)).get("data", [])
        recs += data
        t += timedelta(hours=1)
        hours += 1
        if hours % 24 == 0:
            console.print(f"[dim]  …space {space_id}: {hours} hours fetched, "
                          f"{len(recs)} records[/dim]")
    for r in recs:
        ts = r.get("timestamp") or r.get("ts") or 0
        r["_ts"] = ts * 1000 if ts and ts < 1e12 else ts  # normalize to ms
    return sorted(recs, key=lambda r: r["_ts"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--calibration", type=Path, required=True,
                    help="sensor calibration json (zone/enum response)")
    ap.add_argument("--od-log", type=Path, required=True,
                    help="od_history_compare run log (SAM3 truth + windows)")
    ap.add_argument("--tolerance", type=float, default=300,
                    help="max seconds between a graded frame and a fusion "
                         "record for them to be paired")
    ap.add_argument("--cf-access", action="store_true")
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

    log = json.loads(args.od_log.read_text())
    cal = json.loads(args.calibration.read_text())
    spots = [(z["relatedId"], z.get("spotName") or z["name"],
              [(z["topLeft"]["x"], z["topLeft"]["y"]),
               (z["topRight"]["x"], z["topRight"]["y"]),
               (z["bottomRight"]["x"], z["bottomRight"]["y"]),
               (z["bottomLeft"]["x"], z["bottomLeft"]["y"])])
             for z in cal["response"]["data"]["zones"]
             if z.get("zoneType") == "s" and z.get("cameraId") == args.camera]
    if not spots:
        raise SystemExit("no spot zones for this camera in the calibration")
    host = args.host.rstrip("/")
    console.print(f"[dim]{len(spots)} spots: "
                  f"{', '.join(n for _, n, _ in spots)}[/dim]")

    windows = {"old": log["meta"]["old_window"], "new": log["meta"]["new_window"]}
    out = {"meta": {"host": host, "camera": args.camera,
                    "od_log": str(args.od_log), "windows": windows,
                    "ran": datetime.now(timezone.utc).isoformat()},
           "samples": {}}

    for side in ("old", "new"):
        w0 = datetime.fromisoformat(windows[side][0])
        w1 = datetime.fromisoformat(windows[side][1])
        fusion = {sid: fetch_fusion(host, sid, w0, w1, headers)
                  for sid, _, _ in spots}
        n_rec = sum(len(v) for v in fusion.values())
        console.print(f"[dim]{side}: {n_rec} fusion records across "
                      f"{len(spots)} spots[/dim]")

        # channel tallies + disagreement bookkeeping
        acc = {c: [0, 0] for c in ("inference", "od", "fusion")}  # [right, n]
        dis = {"n": 0, "od_right": 0, "inf_right": 0, "neither": 0,
               "fusion_sided_od": 0, "fusion_sided_inf": 0}
        paths = {}
        samples = []
        for rec in log["graded_frames"][side]:
            gt_boxes = [d[:4] for d in rec["sam3"] if d[5] in (2, 3, 4, 6, 8)]
            for sid, name, poly in spots:
                frecs = fusion[sid]
                if not frecs:
                    continue
                best = min(frecs, key=lambda r: abs(r["_ts"] - rec["ts"]))
                if abs(best["_ts"] - rec["ts"]) > args.tolerance * 1000:
                    continue
                truth = any(box_in_spot(b, poly) for b in gt_boxes)
                ch = {"inference": state_to_bool(best.get("inferenceState")),
                      "od": state_to_bool(best.get("odState")),
                      "fusion": state_to_bool(best.get("fusionState"))}
                for c, v in ch.items():
                    if v is not None:
                        acc[c][1] += 1
                        acc[c][0] += (v == truth)
                path = best.get("decisionPath") or "?"
                samples.append({"ts": rec["ts"], "spot": name, "truth": truth,
                                **ch, "path": path})
                if ch["inference"] is not None and ch["od"] is not None \
                        and ch["inference"] != ch["od"]:
                    dis["n"] += 1
                    paths[path] = paths.get(path, 0) + 1
                    if ch["od"] == truth:
                        dis["od_right"] += 1
                    elif ch["inference"] == truth:
                        dis["inf_right"] += 1
                    else:
                        dis["neither"] += 1
                    if ch["fusion"] is not None:
                        if ch["fusion"] == ch["od"]:
                            dis["fusion_sided_od"] += 1
                        elif ch["fusion"] == ch["inference"]:
                            dis["fusion_sided_inf"] += 1

        t = Table(title=f"{side} window — channel accuracy vs SAM3 "
                        f"(per spot-sample)")
        t.add_column("channel")
        t.add_column("accuracy", justify="right")
        t.add_column("n", justify="right")
        for c in ("inference", "od", "fusion"):
            right, n = acc[c]
            t.add_row(c, f"{right / n:.1%}" if n else "—", str(n))
        console.print(t)
        if dis["n"]:
            console.print(
                f"  disagreements (inference vs OD): {dis['n']} — "
                f"OD right {dis['od_right']}, inference right "
                f"{dis['inf_right']}, neither {dis['neither']}; "
                f"fusion sided with OD {dis['fusion_sided_od']}x, "
                f"inference {dis['fusion_sided_inf']}x")
            top = sorted(paths.items(), key=lambda kv: -kv[1])[:4]
            console.print("  disagreement decision paths: "
                          + ", ".join(f"{k}×{v}" for k, v in top))
        out["samples"][side] = samples

    run_dir = CACHE / host_slug(host) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = run_dir / (datetime.now().strftime("%Y%m%d-%H%M%S")
                      + f"-fusion-cam{args.camera}.json")
    logf.write_text(json.dumps(out, indent=1))
    console.print(f"[dim]fusion log: {logf}[/dim]")


if __name__ == "__main__":
    main()
