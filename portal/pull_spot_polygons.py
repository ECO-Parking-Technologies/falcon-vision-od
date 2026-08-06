#!/usr/bin/env python3
"""Pull per-sensor parking-spot polygons from the portal.

For every garage in the unified store, takes the most recent snapshot runs and
collects each snapshotParkingSpace's `bounds` (normalized 4-point polygon in
the camera frame), grouped by (garage, sensor, camera). Output:

    data/spot_polygons.json
      { "<garage>": { "<sensor>|cameraN": [ {"space": name, "points": [[x,y]x4]} ] } }

Used to stamp InEcoParkingSpot onto SAM3 draft boxes (preannotation/
label_inspot.py) and for the spot-restricted eval. Credentials: portal
refresh token prompted at runtime (RAM only).

    python3 portal/pull_spot_polygons.py
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rich.console import Console
from rich.prompt import Prompt

from pull_training_images import PortalClient, slugify

console = Console()
STORE = Path(__file__).resolve().parent.parent / "data" / "images"
OUT = Path(__file__).resolve().parent.parent / "data" / "spot_polygons.json"


def norm_points(bounds):
    """bounds -> [[x,y]x4] normalized floats; tolerates dict/list forms and
    unordered points (sorted by angle around the centroid)."""
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
    import math
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
    for gi, garage in enumerate(garages, 1):
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
                      key=lambda r: r["runAt"], reverse=True)
        gmap = {}
        # a couple of recent runs: union covers sensors missing from any one run
        for run in runs[:3]:
            nodes = portal.graphql(
                "query($id: UUID!) { snapshot(id: $id) { snapshotParkingSpaces"
                " { nodes { bounds originalImageUrlSigned"
                "   parkingSpace { name }"
                "   detectedBySensor { configurationName } } } } }",
                {"id": run["id"]})["snapshot"]["snapshotParkingSpaces"]["nodes"]
            for n in nodes:
                pts = norm_points(n.get("bounds"))
                url = n.get("originalImageUrlSigned") or ""
                if not pts or not url:
                    continue
                sensor = slugify((n.get("detectedBySensor") or {})
                                 .get("configurationName") or "")
                m = re.search(r"camera(\d+)", unquote(urlparse(url).path))
                if not sensor or not m:
                    continue
                key = f"{sensor}|camera{m.group(1)}"
                name = (n.get("parkingSpace") or {}).get("name") or "?"
                spaces = gmap.setdefault(key, {})
                spaces.setdefault(name, pts)  # first (newest run) wins
        out[garage] = {k: [{"space": nm, "points": pts}
                           for nm, pts in sorted(v.items())]
                       for k, v in gmap.items()}
        n_spaces = sum(len(v) for v in out[garage].values())
        console.print(f"[green]{gi:2}/{len(garages)}[/green] {garage}: "
                      f"{len(out[garage])} sensor-cameras, {n_spaces} spaces")

    OUT.write_text(json.dumps(out, indent=1))
    tot = sum(len(s) for g in out.values() for s in g.values())
    console.print(f"\n[bold green]{tot} spot polygons -> {OUT}[/bold green]")


if __name__ == "__main__":
    main()
