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

def introspect(portal):
    """Discover live schema facts for snapshots (the docs dump is stale)."""
    info = {}
    q = portal.graphql('{ __type(name: "Query") { fields { name } } }')
    fields = [f["name"] for f in q["__type"]["fields"]]
    info["query_fields_snapshotish"] = [f for f in fields if "napshot" in f]
    t = portal.graphql('{ __type(name: "Snapshot") { fields { name } } }')
    info["snapshot_fields"] = [f["name"] for f in (t["__type"] or {}).get("fields", [])]
    e = portal.graphql('{ __type(name: "SnapshotsOrderBy") { enumValues { name } } }')
    info["order_by"] = [v["name"] for v in (e["__type"] or {}).get("enumValues", [])]
    log.info("introspection: %s", info)
    return info


def pick_ts_field(info):
    for cand in ("runAt", "createdTimestamp", "timestamp"):
        if cand in info["snapshot_fields"]:
            return cand
    return None


def site_row(stats, key):
    s = stats[key]
    rng = f"{s['oldest'][:10]}→{s['newest'][:10]}" if s["oldest"] else "—"
    err = "[red]ERR[/red]" if s["errors"] else str(s["snapshots"])
    return (s["org"], s["garage"], err, f"{s['avg_images_per_run']:.0f}",
            str(s["images_est"]), rng, str(s["validations"]))


def build_table(stats, done, total):
    t = Table(title=f"Portal snapshot inventory ({done}/{total} sites scanned)",
              box=box.SIMPLE_HEAD)
    for col in ("account", "garage", "runs", "img/run", "est. images", "date range",
                "validations"):
        t.add_column(col, justify="right" if col not in ("account", "garage") else "left")
    tot = dict(snapshots=0, images_est=0, validations=0)
    for key in sorted(stats):
        t.add_row(*site_row(stats, key))
        for f in tot:
            tot[f] += stats[key][f]
    if stats:
        t.add_section()
        t.add_row("TOTAL", f"{len(stats)} garages", str(tot["snapshots"]), "",
                  str(tot["images_est"]), "", str(tot["validations"]))
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

    with console.status("Introspecting live schema for snapshots…"):
        info = introspect(portal)
    console.print(f"snapshot-ish query fields: [cyan]{info['query_fields_snapshotish']}[/cyan]")
    if "snapshots" not in info["query_fields_snapshotish"]:
        console.print("[red]No `snapshots` query on the live API[/red] — see the log for "
                      "what exists; paste this output back for the next iteration.")
        sys.exit(1)
    ts_field = pick_ts_field(info)
    asc = next((v for v in info["order_by"] if ts_field and v.startswith(
        "RUN_AT_ASC" if ts_field == "runAt" else "CREATED_TIMESTAMP_ASC")), None)
    desc = asc.replace("_ASC", "_DESC") if asc else None
    log.info("using ts_field=%s orderBy=%s/%s", ts_field, asc, desc)

    def snapshot_stats(site_id):
        """Sample the newest 50 runs for an images-per-run average; estimate the total.

        runs × avg(images/run of recent runs) ≈ images available. One combined
        request per site + one tiny query for the oldest-run date.
        """
        order = f", orderBy: {desc}" if desc else ""
        ts = ts_field or "id"
        data = portal.graphql(
            f"query($id: Int!) {{"
            f" snapshots(condition: {{siteId: $id}}{order}, first: 50) {{ totalCount"
            f"  nodes {{ {ts} snapshotParkingSpaces {{ totalCount }} }} }}"
            f" validations(condition: {{siteId: $id}}) {{ totalCount }} }}",
            {"id": site_id})
        snaps = data["snapshots"]
        runs = snaps["totalCount"]
        sampled = [n["snapshotParkingSpaces"]["totalCount"] for n in snaps["nodes"]]
        avg = (sum(sampled) / len(sampled)) if sampled else 0.0
        newest = max((n[ts] for n in snaps["nodes"] if n.get(ts)), default="")
        oldest = ""
        if runs and asc:
            r = portal.graphql(
                f"query($id: Int!) {{ snapshots(condition: {{siteId: $id}}, "
                f"orderBy: {asc}, first: 1) {{ nodes {{ {ts} }} }} }}",
                {"id": site_id})["snapshots"]["nodes"]
            oldest = r[0][ts] if r else ""
        return (runs, avg, round(runs * avg), oldest or "", newest,
                data["validations"]["totalCount"])

    stats = {}
    with Live(build_table(stats, 0, len(sites)), console=console,
              refresh_per_second=2) as live:
        for i, site in enumerate(sites):
            key = f"{site['org']}/{site['display']}"
            s = stats[key] = {"org": site["org"], "garage": site["display"],
                              "site_id": site["site_id"], "snapshots": 0,
                              "avg_images_per_run": 0.0, "images_est": 0,
                              "validations": 0, "oldest": "", "newest": "",
                              "errors": 0}
            try:
                (s["snapshots"], s["avg_images_per_run"], s["images_est"], s["oldest"],
                 s["newest"], s["validations"]) = snapshot_stats(site["site_id"])
            except Exception as e:
                log.warning("%s: snapshots query FAILED: %s", key, e)
                s["errors"] += 1
            log.info("%s: runs=%d avg_img/run=%.1f est_images=%d range=%s..%s "
                     "validations=%d errors=%d", key, s["snapshots"],
                     s["avg_images_per_run"], s["images_est"], s["oldest"],
                     s["newest"], s["validations"], s["errors"])
            live.update(build_table(stats, i + 1, len(sites)))

    totals = {
        "garages": len(stats),
        "garages_with_snapshots": sum(1 for v in stats.values() if v["snapshots"]),
        "garages_errored": sum(1 for v in stats.values() if v["errors"]),
        "snapshot_runs": sum(v["snapshots"] for v in stats.values()),
        "images_est": sum(v["images_est"] for v in stats.values()),
        "validations": sum(v["validations"] for v in stats.values()),
    }
    json_file.write_text(json.dumps({"generated": stamp, "totals": totals,
                                     "introspection": info, "sites": stats}, indent=2))
    log.info("totals: %s", totals)
    console.print(Panel(
        f"garages: {totals['garages']} ({totals['garages_with_snapshots']} with snapshots, "
        f"{totals['garages_errored']} errored)\n"
        f"snapshot runs: {totals['snapshot_runs']} · est. images: [green]{totals['images_est']}[/green] · "
        f"validations: {totals['validations']}\n"
        f"summary JSON: {json_file}", title="Overview", border_style="green"))


if __name__ == "__main__":
    main()
