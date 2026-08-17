#!/usr/bin/env python3
"""Fleet A/B pull: whole-floor split analysis (e.g. Tarkington L2).

Phase 1 — EVERY sensor in the roster:
  - /plugin/od-model/info -> model SHA -> auto-classified against the known
    deployed builds (no hand-kept assignment sheet)
  - /plugin/calibration/zone/enum -> data/sensor_archive/<host>/calibration.json
  - writes the split assignment: data/sensor_archive/<roster>-split.json

Phase 2 — a stratified SUBSET (N per lane x model): the standard
od_history_compare treatment — archive windows, image mirror, SAM3 grading —
one run log per sensor, plus a pooled per-model summary.

    python3 fleet_pull.py --roster data/sensor_archive/tarkington-l2-roster.json \
        --domain fvg-tarkington --cutoff <install-ISO> --new-start <ISO+buffer>

CF Access service-token creds prompted ONCE (RAM only).
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.prompt import Prompt
from rich.table import Table

from od_history_compare import (CACHE, console, fetch_window, grade_vs_sam3,
                                host_slug, mirror_images, window_stats)

KNOWN_SHAS = {
    "084d3a96a843a242147e9f5bcc8321d62c3c48695a3314b66e23256e89aa7bbc":
        "lite1-a20-3x-fill0",
    "5bd82c9522ae12659cd62b77107f99cf08f1f697ff6d710be7794f58219dad08":
        "lite0-a20-3x",
}


def sensor_url(host, domain):
    return f"https://{host}.{domain}.private.ecofalcondata.com"


def get_model_sha(base, headers):
    for method in ("get", "post"):
        try:
            r = getattr(requests, method)(f"{base}/plugin/od-model/info",
                                          headers=headers, timeout=12)
            m = re.search(r"[0-9a-f]{64}", r.text)
            if m:
                return m.group(0)
        except Exception:
            pass
    return None


def get_calibration(base, headers, cameras):
    """Merge zone lists across cameras into one calibration doc (zones carry
    their own cameraId, so downstream per-camera filtering still works)."""
    zones = []
    ok = False
    for cam in cameras:
        try:
            r = requests.post(f"{base}/plugin/calibration/zone/enum",
                              data={"cameraId": cam}, headers=headers, timeout=12)
            d = r.json()
            z = d.get("response", {}).get("data", {}).get("zones")
            if z is not None:
                ok = True
                zones += z
        except Exception:
            pass
    if not ok:
        return None
    return {"response": {"retval": {"status": "OK"},
                         "data": {"zones": zones}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--domain", required=True, help="e.g. fvg-tarkington")
    ap.add_argument("--cameras", default="1,2",
                    help="camera ids to cover (default both; cameras with "
                         "no spot zones are skipped automatically)")
    ap.add_argument("--cutoff", required=True, help="install time, ISO w/ offset")
    ap.add_argument("--new-start", required=True, help="cutoff + buffer, ISO")
    ap.add_argument("--before-hours", type=int, default=48)
    ap.add_argument("--grade-per-cell", type=int, default=2,
                    help="sensors per (lane x model) for full SAM3 grading")
    ap.add_argument("--sam3", type=int, default=40, help="frames graded per window")
    args = ap.parse_args()

    cameras = [int(c) for c in args.cameras.split(",") if c.strip()]
    headers = {
        "CF-Access-Client-Id":
            Prompt.ask("[cyan]CF-Access-Client-Id[/cyan]", console=console).strip(),
        "CF-Access-Client-Secret":
            Prompt.ask("[cyan]CF-Access-Client-Secret[/cyan]", password=True,
                       console=console).strip(),
    }
    roster = json.loads(args.roster.read_text())

    # ---- phase 1: sha + calibration for every sensor ----
    console.print(f"[bold]phase 1: {len(roster)} sensors — model detect + "
                  "calibration[/bold]")
    for s in roster:
        base = sensor_url(s["host"], args.domain)
        sha = get_model_sha(base, headers)
        s["sha"] = sha
        s["model"] = KNOWN_SHAS.get(sha, "UNKNOWN" if sha else "UNREACHABLE")
        cal = get_calibration(base, headers, cameras)
        if cal is not None:
            d = CACHE / s["host"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "calibration.json").write_text(json.dumps(cal))
        s["calibration"] = cal is not None
        per_cam = {c: len([z for z in cal["response"]["data"]["zones"]
                           if z.get("zoneType") == "s" and z.get("cameraId") == c])
                   for c in cameras} if cal else {}
        s["spots_per_camera"] = per_cam
        console.print(f"  {s['host']} {s['name']:10s} -> {s['model']:20s} "
                      f"cal={'ok' if s['calibration'] else 'FAIL'} spots={per_cam}")
    split_file = args.roster.with_name(args.roster.stem.replace("-roster", "")
                                       + "-split.json")
    split_file.write_text(json.dumps(roster, indent=1))
    counts = {}
    for s in roster:
        counts[s["model"]] = counts.get(s["model"], 0) + 1
    console.print(f"[bold]assignment:[/bold] {counts} -> {split_file}")

    # ---- phase 2: stratified subset, full A/B grading ----
    cells = {}
    for s in roster:
        if s["model"] in ("UNKNOWN", "UNREACHABLE") or not s["calibration"]:
            continue
        cells.setdefault((s["lane"], s["model"]), []).append(s)
    subset = []
    for key, group in sorted(cells.items()):
        subset += group[:args.grade_per_cell]
    console.print(f"[bold]phase 2: grading {len(subset)} sensors "
                  f"({args.grade_per_cell} per lane x model)[/bold]")

    import sys
    sys.path.insert(0, str(Path(__file__).parent / "preannotation"))
    from sam3_model import Sam3DraftModel
    sam3 = Sam3DraftModel()

    cutoff = datetime.fromisoformat(args.cutoff).astimezone(timezone.utc)
    a0 = datetime.fromisoformat(args.new_start).astimezone(timezone.utc)
    from datetime import timedelta
    b0 = cutoff - timedelta(hours=args.before_hours)
    now = datetime.now(timezone.utc)

    pooled = {}   # model -> {thr: {tp,det,gt}}
    results = []
    for s in subset:
        base = sensor_url(s["host"], args.domain)
        for cam in cameras:
            if not s.get("spots_per_camera", {}).get(cam):
                continue   # no monitored spots on this camera
            console.print(f"[bold]{s['host']} {s['name']} cam{cam} "
                          f"({s['model']})[/bold]")
            old_f, _ = fetch_window(base, cam, b0, cutoff, headers)
            new_f, _ = fetch_window(base, cam, a0, now, headers)
            mirror_images(base, cam, b0, now, headers)
            g_old, rec_old = grade_vs_sam3(base, cam, old_f, headers, sam3,
                                           args.sam3, "old")
            g_new, rec_new = grade_vs_sam3(base, cam, new_f, headers, sam3,
                                           args.sam3, "new")
            run_dir = CACHE / s["host"] / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / (datetime.now().strftime("%Y%m%d-%H%M%S")
                             + f"-fleet-cam{cam}.json")
            log.write_text(json.dumps({
                "meta": {"host": base, "camera": cam, "name": s["name"],
                         "lane": s["lane"], "model": s["model"], "sha": s["sha"],
                         "old_window": [b0.isoformat(), cutoff.isoformat()],
                         "new_window": [a0.isoformat(), now.isoformat()],
                         "ran": datetime.now(timezone.utc).isoformat()},
                "summary": {"old": window_stats(old_f), "new": window_stats(new_f)},
                "sam3_stats": {"old": g_old, "new": g_new},
                "graded_frames": {"old": rec_old, "new": rec_new},
            }, indent=1))
            row = {"host": s["host"], "name": f"{s['name']} cam{cam}",
                   "lane": s["lane"], "model": s["model"]}
            for side, g in (("old", g_old), ("new", g_new)):
                if g:
                    d = g[0.40]
                    row[side] = (round(d["tp"] / d["det"], 2) if d["det"] else None,
                                 round(d["tp"] / d["gt"], 2) if d["gt"] else None)
                    if side == "new":
                        agg = pooled.setdefault(s["model"],
                                                {"tp": 0, "det": 0, "gt": 0})
                        for k in agg:
                            agg[k] += d[k]
                else:
                    row[side] = (None, None)
            results.append(row)

    t = Table(title="fleet A/B — per sensor (P, R @0.40 vs SAM3)")
    for c in ("sensor", "lane", "model", "old P/R", "new P/R"):
        t.add_column(c)
    for r in results:
        t.add_row(f"{r['host']} {r['name']}", r["lane"], r["model"],
                  str(r["old"]), str(r["new"]))
    console.print(t)
    console.print("[bold]pooled new-window by model:[/bold]")
    for m, d in pooled.items():
        p = d["tp"] / d["det"] if d["det"] else 0
        rr = d["tp"] / d["gt"] if d["gt"] else 0
        console.print(f"  {m}: P={p:.2f} R={rr:.2f} ({d['gt']} SAM3 vehicles)")


if __name__ == "__main__":
    main()
