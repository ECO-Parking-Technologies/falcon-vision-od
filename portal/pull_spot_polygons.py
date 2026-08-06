#!/usr/bin/env python3
"""Pull per-sensor parking-spot polygons from the portal — PER SNAPSHOT RUN.

Spot calibrations change over time, and `snapshotParkingSpace.bounds` is
recorded per run — so this pulls the polygon set of EVERY snapshot run (not
just the latest) and indexes it two ways:

    data/spot_polygons.json
    { "<garage>": {
        "runs":     { "<run8>": { "<sensor>|cameraN": [ {space, points} ] } },
        "timeline": { "<sensor>|cameraN": [ {at: iso, run: run8}, ... ] } } }

Store frames carry their run id in the filename (<sensor>-<run8>-<mac>-cameraN)
so the labeler matches each frame to ITS OWN run's calibration; frames without
a run id fall back to the nearest-in-time entry on the sensor's timeline.
~2.5k runs fleet-wide at the client's 0.4 s throttle ≈ 15-20 min, one-time.
Credentials: portal refresh token prompted at runtime (RAM only).

    python3 portal/pull_spot_polygons.py
"""
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt

from pull_training_images import PortalClient, slugify

console = Console()
STORE = Path(__file__).resolve().parent.parent / "data" / "images"
OUT = Path(__file__).resolve().parent.parent / "data" / "spot_polygons.json"


def norm_points(bounds):
    """bounds -> [[x,y]x4] normalized; tolerates dict/list forms, orders
    unordered points by angle around the centroid."""
    if isinstance(bounds, str):
        bounds = json.loads(bounds)
    pts = []
    for p in bounds or []:
        if isinstance(p, dict):
            pts.append((float(p.get("x", 0)), float(p.get("y", 0))))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    if len(pts) < 3:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [[round(x, 5), round(y, 5)] for x, y in pts]


def main():
    garages = sorted(p.name for p in STORE.iterdir() if p.is_dir())
    console.print(f"store garages: {len(garages)}")
    token = Prompt.ask("[cyan]Portal refresh token[/cyan]", password=True,
                       console=console)
    portal = PortalClient(token)

    sites = portal.discover_sites()
    by_slug = {}
    for s in sites:
        by_slug[slugify(f"{s['org']}-{s['display']}")] = s
        by_slug.setdefault(slugify(s["display"]), s)

    out = {}
    for garage in garages:
        site = by_slug.get(garage)
        if not site:
            console.print(f"[yellow]{garage}: no matching portal site[/yellow]")
            continue
        runs = portal.graphql(
            "query($id: Int!) { snapshots(condition: {siteId: $id}, first: 1000)"
            " { nodes { id runAt snapshotParkingSpaces { totalCount } } } }",
            {"id": site["site_id"]})["snapshots"]["nodes"]
        runs = sorted((r for r in runs if r.get("runAt")
                       and r["snapshotParkingSpaces"]["totalCount"] > 0),
                      key=lambda r: r["runAt"])
        g = {"runs": {}, "timeline": {}}
        for run in track(runs, description=f"{garage[:40]:40}", console=console,
                         transient=True):
            run8 = run["id"].replace("-", "")[:8].lower()
            try:
                nodes = portal.graphql(
                    "query($id: UUID!) { snapshot(id: $id) { snapshotParkingSpaces"
                    " { nodes { bounds originalImageUrlSigned"
                    "   parkingSpace { name }"
                    "   detectedBySensor { configurationName } } } } }",
                    {"id": run["id"]})["snapshot"]["snapshotParkingSpaces"]["nodes"]
            except Exception as e:
                console.print(f"[yellow]{garage} run {run8}: {e}[/yellow]")
                continue
            rmap = {}
            for n in nodes:
                pts = norm_points(n.get("bounds"))
                url = n.get("originalImageUrlSigned") or ""
                m = re.search(r"camera(\d+)", unquote(urlparse(url).path))
                sensor = slugify((n.get("detectedBySensor") or {})
                                 .get("configurationName") or "")
                if not pts or not m or not sensor:
                    continue
                key = f"{sensor}|camera{m.group(1)}"
                name = (n.get("parkingSpace") or {}).get("name") or "?"
                rmap.setdefault(key, {}).setdefault(name, pts)
            if not rmap:
                continue
            g["runs"][run8] = {k: [{"space": nm, "points": p}
                                   for nm, p in sorted(v.items())]
                               for k, v in rmap.items()}
            for k in rmap:
                g["timeline"].setdefault(k, []).append(
                    {"at": run["runAt"], "run": run8})
        out[garage] = g
        console.print(f"[green]{garage}[/green]: {len(g['runs'])} runs, "
                      f"{len(g['timeline'])} sensor-cameras")

    OUT.write_text(json.dumps(out, indent=1))
    n_runs = sum(len(g["runs"]) for g in out.values())
    console.print(f"\n[bold green]{n_runs} calibration snapshots -> {OUT}[/bold green]")


if __name__ == "__main__":
    main()
