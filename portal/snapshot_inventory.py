#!/usr/bin/env python3
"""Portal snapshot inventory: how much training imagery does the portal hold?

Walks every account (organization) -> site (garage) -> validations, counts the
snapshot full frames and per-space crops available, and reports a per-garage
breakdown plus an overview. Read-only: no images are downloaded, no signed
URLs are requested.

Portal API refresh token is prompted (no echo) and held ONLY in RAM.
Console shows live progress; full detail + a JSON summary land in the data
root for later review:

    python3 portal/snapshot_inventory.py            # writes data/inventory-<ts>.{log,json}
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pull_training_images import PortalClient, setup_logging, console, log  # noqa: E402

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

FRAMES_QUERY = (
    "query($id: UUID!) { validation(id: $id) { createdTimestamp"
    " validationParkingSpaces { nodes { snapshotParkingSpace {"
    " id sensorSnapshotId detectedBySensor { configurationName } } } } } }"
)


def site_row(stats, key):
    s = stats[key]
    rng = f"{s['oldest'][:10]}→{s['newest'][:10]}" if s["oldest"] else "—"
    return (s["org"], s["garage"], f"{s['completed']}/{s['validations']}",
            str(s["frames"]), str(s["crops"]), str(len(s["sensors"])), rng)


def build_table(stats, done, total):
    t = Table(title=f"Portal snapshot inventory ({done}/{total} sites scanned)",
              box=box.SIMPLE_HEAD)
    for col in ("account", "garage", "validations✓", "full frames", "space crops",
                "sensors", "date range"):
        t.add_column(col, justify="right" if col not in ("account", "garage") else "left")
    tot = dict(frames=0, crops=0, completed=0, validations=0, sensors=set())
    for key in sorted(stats):
        t.add_row(*site_row(stats, key))
        s = stats[key]
        tot["frames"] += s["frames"]; tot["crops"] += s["crops"]
        tot["completed"] += s["completed"]; tot["validations"] += s["validations"]
        tot["sensors"] |= s["sensors"]
    if stats:
        t.add_section()
        t.add_row("TOTAL", f"{len(stats)} garages",
                  f"{tot['completed']}/{tot['validations']}", str(tot["frames"]),
                  str(tot["crops"]), str(len(tot["sensors"])), "")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--garages", help="comma-separated site-name filter (default: all)")
    ap.add_argument("--max-validations", type=int, default=0,
                    help="cap validations scanned per site, 0 = all")
    args = ap.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = args.data_root / f"inventory-{stamp}.log"
    json_file = args.data_root / f"inventory-{stamp}.json"
    setup_logging(log_file)

    console.print(Panel("Read-only inventory — nothing is downloaded.\n"
                        "Token is held in memory only, never written to disk.\n"
                        f"Log: [bold]{log_file}[/bold]",
                        title="Portal snapshot inventory", border_style="cyan"))
    token = Prompt.ask("[cyan]Portal API refresh token[/cyan]", password=True,
                       console=console)
    token = "".join(token.split())  # kill line-wrap newlines from long pastes
    if not token:
        sys.exit("portal token required")
    portal = PortalClient(token)

    with console.status("Discovering organizations and sites…"):
        sites = portal.discover_sites()
    if args.garages:
        wanted = {g.strip().lower() for g in args.garages.split(",")}
        sites = [s for s in sites if s["name"].lower() in wanted
                 or s["display"].lower() in wanted]
    log.info("discovered %d sites", len(sites))
    console.print(f"[green]{len(sites)}[/green] site(s) discovered\n")

    stats = {}
    with Live(build_table(stats, 0, len(sites)), console=console,
              refresh_per_second=2) as live:
        for i, site in enumerate(sites):
            key = f"{site['org']}/{site['display']}"
            s = stats[key] = {"org": site["org"], "garage": site["display"],
                              "site_id": site["site_id"], "validations": 0,
                              "completed": 0, "frames": 0, "crops": 0,
                              "sensors": set(), "oldest": "", "newest": "",
                              "errors": 0}
            try:
                validations = portal.validations_for_site(site["site_id"])
            except Exception as e:
                log.warning("%s: validations query failed: %s", key, e)
                s["errors"] += 1
                live.update(build_table(stats, i + 1, len(sites)))
                continue
            s["validations"] = len(validations)
            completed = [v for v in validations
                         if v.get("validationState") == "completed"]
            s["completed"] = len(completed)
            if args.max_validations:
                completed = completed[: args.max_validations]
            frame_ids = set()
            for v in completed:
                try:
                    data = portal.graphql(FRAMES_QUERY, {"id": v["id"]})["validation"]
                except Exception as e:
                    log.debug("%s validation %s failed: %s", key, v["id"], e)
                    s["errors"] += 1
                    continue
                ts = data.get("createdTimestamp") or ""
                if ts:
                    s["oldest"] = min(s["oldest"] or ts, ts)
                    s["newest"] = max(s["newest"], ts)
                for node in data["validationParkingSpaces"]["nodes"]:
                    sp = node.get("snapshotParkingSpace") or {}
                    if not sp:
                        continue
                    s["crops"] += 1
                    uid = sp.get("sensorSnapshotId") or sp.get("id")
                    if uid:
                        frame_ids.add(uid)
                    sensor = (sp.get("detectedBySensor") or {}).get("configurationName")
                    if sensor:
                        s["sensors"].add(sensor.lower())
            s["frames"] = len(frame_ids)
            log.info("%s: validations=%d completed=%d frames=%d crops=%d sensors=%d",
                     key, s["validations"], s["completed"], s["frames"], s["crops"],
                     len(s["sensors"]))
            live.update(build_table(stats, i + 1, len(sites)))

    summary = {k: {**v, "sensors": sorted(v["sensors"])} for k, v in stats.items()}
    totals = {
        "garages": len(stats),
        "validations": sum(v["validations"] for v in stats.values()),
        "completed": sum(v["completed"] for v in stats.values()),
        "full_frames": sum(v["frames"] for v in stats.values()),
        "space_crops": sum(v["crops"] for v in stats.values()),
        "sensors": len(set().union(*(v["sensors"] for v in stats.values())) if stats else set()),
    }
    json_file.write_text(json.dumps({"generated": stamp, "totals": totals,
                                     "sites": summary}, indent=2))
    log.info("totals: %s", totals)
    console.print(Panel(
        f"garages: {totals['garages']} · validations: {totals['completed']}/{totals['validations']} completed\n"
        f"unique full-frame snapshots: [green]{totals['full_frames']}[/green] · "
        f"per-space crops: {totals['space_crops']} · sensors seen: {totals['sensors']}\n"
        f"summary JSON: {json_file}", title="Overview", border_style="green"))


if __name__ == "__main__":
    main()
