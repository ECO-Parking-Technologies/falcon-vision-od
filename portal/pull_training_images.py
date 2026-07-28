#!/usr/bin/env python3
"""Unified training-image puller: sensor training-image stores + portal snapshot archive.

Distilled from falcon-vision-ml reference code (see portal/reference/). Design:
docs/planning/04-portal-integration.md.

- Credentials (portal refresh token, Cloudflare Access id/secret) are prompted
  at startup with no echo and held ONLY in RAM. Never pass them as CLI args.
- Every download is gated by a SQLite manifest keyed on stable source identity;
  files are static, so nothing is ever fetched twice. Presigned URLs are never
  persisted.
- Garages auto-discovered from the portal (organizations -> sites); gateways
  derived as https://legacy.fvg-<site-name>.private.ecofalcondata.com and
  probed — sites without a reachable gateway are skipped.

Usage:
    python3 portal/pull_training_images.py --data-root <dir> \
        --start 202607200000 --end 202607270000 [--interval-min 60] \
        [--source a|b|both] [--garages name1,name2] [--list-garages]
"""
import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()
log = logging.getLogger("pull")


def setup_logging(log_file: Path):
    """Full detail goes to the log file; the console stays reserved for rich UI."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file)],
        force=True,
    )
    for noisy in ("requests", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class PullStats:
    """Per-(garage, sensor) counters rendered as a live rich table."""

    EVENTS = ("new", "dup", "cached", "error")

    def __init__(self):
        self.rows = {}
        self.current = ""
        self.progress = {}   # garage -> (runs_done, runs_planned)
        self.data_root = None

    def bump(self, garage, sensor, event):
        row = self.rows.setdefault((garage, sensor), dict.fromkeys(self.EVENTS, 0))
        row[event] += 1

    def table(self):
        t = Table(title="Training image pull (one row per garage)", box=box.SIMPLE_HEAD)
        for col, style in (("garage", "cyan"), ("progress", "cyan"), ("new", "green"),
                           ("dup", "yellow"), ("cached", "dim"), ("errors", "red")):
            t.add_column(col, style=style, justify="left" if col == "garage" else "right")
        totals = dict.fromkeys(self.EVENTS, 0)
        for (g, s), r in sorted(self.rows.items()):
            d, n = self.progress.get(g, (0, 0))
            prog = f"{d}/{n} ({100 * d // n}%)" if n else (s if s != "•" else "-")
            t.add_row(g, prog, *(str(r[e]) for e in self.EVENTS))
            for e in self.EVENTS:
                totals[e] += r[e]
        done = sum(d for d, _ in self.progress.values())
        planned = sum(n for _, n in self.progress.values())
        if self.rows:
            t.add_section()
            overall = f"{done}/{planned} ({100 * done // planned}%)" if planned else "-"
            t.add_row("OVERALL", overall, *(str(totals[e]) for e in self.EVENTS))
        cap = self.current or ""
        if self.data_root is not None:
            cap += f"   ·   disk free {free_gb(self.data_root):.0f} GB"
        if cap:
            t.caption = cap
        return t

PORTAL_GRAPHQL = "https://api.ecoparkingtechnologies.com/graphql"
PORTAL_TOKEN_URL = "https://identity.ecoparkingtechnologies.com/token"
PORTAL_CLIENT_ID = "EcoParking.ClientApi"
GATEWAY_TEMPLATE = "https://legacy.fvg-{site}.private.ecofalcondata.com"
TIMEOUT = 30
RETRIES = 3


def http_get(session, url, **kw):
    """GET with timeout + exponential-backoff retries."""
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT, **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            log.debug("retry %s after %s", url, e)
        time.sleep(2 ** attempt)
    return r


class PortalClient:
    """OAuth2 refresh-token flow + GraphQL. Token lives in RAM only."""

    def __init__(self, refresh_token: str):
        self._refresh_token = refresh_token
        self._access_token = None
        self._expiry = 0.0
        self.session = requests.Session()

    def _token(self) -> str:
        if self._access_token and time.time() < self._expiry - 300:
            return self._access_token
        r = self.session.post(PORTAL_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": PORTAL_CLIENT_ID,
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            try:
                detail = r.json().get("error") or r.text[:300]
            except Exception:
                detail = r.text[:300]
            raise RuntimeError(
                f"token exchange failed ({r.status_code}): {detail} — "
                "check the refresh token (portal → Personal API Tokens; "
                "it may be expired/revoked or mispasted)")
        data = r.json()
        self._access_token = data["access_token"]
        self._expiry = time.time() + data.get("expires_in", 1200)
        return self._access_token

    MIN_INTERVAL = 0.4  # seconds between requests — stay under the portal rate limit
    _last_request = 0.0

    def graphql(self, query: str, variables: dict = None) -> dict:
        for attempt in range(6):
            wait = PortalClient.MIN_INTERVAL - (time.time() - PortalClient._last_request)
            if wait > 0:
                time.sleep(wait)
            PortalClient._last_request = time.time()
            r = self.session.post(
                PORTAL_GRAPHQL,
                json={"query": query, "variables": variables or {}},
                headers={"Authorization": f"Bearer {self._token()}"},
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                delay = float(r.headers.get("Retry-After") or min(30, 2 ** (attempt + 1)))
                log.info("rate limited (429), backing off %.0fs", delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            out = r.json()
            log.debug("graphql response (%s...): %.500s", query[:60].replace("\n", " "), out)
            if out.get("errors"):
                raise RuntimeError(f"GraphQL errors: {out['errors']}")
            return out["data"]
        raise RuntimeError("still rate-limited after 6 attempts")

    def discover_sites(self):
        """[{org, site_id, name, display}] for all non-deleted orgs/sites."""
        orgs = self.graphql(
            "query { organizations(condition: {isDeleted: false}, orderBy: NAME_ASC)"
            " { nodes { id name } } }"
        )["organizations"]["nodes"]
        sites = []
        for org in orgs:
            nodes = self.graphql(
                "query($id: Int!) { sites(condition: {organizationId: $id, isDeleted: false})"
                " { nodes { id name displayName } } }",
                {"id": org["id"]},
            )["sites"]["nodes"]
            for s in nodes:
                sites.append({"org": org["name"], "site_id": s["id"],
                              "name": s["name"], "display": s.get("displayName") or s["name"]})
        return sites

    def validations_for_site(self, site_id: int):
        return self.graphql(
            "query($siteId: Int!) { validations(condition: {siteId: $siteId})"
            " { nodes { id createdTimestamp validationState } } }",
            {"siteId": site_id},
        )["validations"]["nodes"]

    def validation_frames(self, validation_id: str):
        """Full-frame snapshot entries for one validation:
        [{uuid, url_signed, sensor}] (url is secret — use immediately, never store)."""
        data = self.graphql(
            "query($id: UUID!) { validation(id: $id) { createdTimestamp"
            " validationParkingSpaces { nodes { snapshotParkingSpace {"
            " id sensorSnapshotId originalImageUrlSigned"
            " detectedBySensor { configurationName } } } } } }",
            {"id": validation_id},
        )["validation"]
        frames, seen = [], set()
        for node in data["validationParkingSpaces"]["nodes"]:
            sp = node.get("snapshotParkingSpace") or {}
            uuid = sp.get("sensorSnapshotId") or sp.get("id")
            url = sp.get("originalImageUrlSigned")
            if not uuid or not url or uuid in seen:
                continue
            seen.add(uuid)
            sensor = ((sp.get("detectedBySensor") or {}).get("configurationName") or "unknown").lower()
            if not sensor.startswith("fv"):
                sensor = "fv" + sensor
            frames.append({"uuid": uuid, "url": url, "sensor": sensor,
                           "ts": data.get("createdTimestamp") or ""})
        return frames


class CloudflareAccess:
    """Session with CF Access service-token headers. Secrets in RAM only."""

    def __init__(self, client_id: str, client_secret: str):
        self.session = requests.Session()
        self.session.headers.update({
            "CF-Access-Client-Id": client_id,
            "CF-Access-Client-Secret": client_secret,
        })

    def enum_sensors(self, gateway_url: str):
        r = self.session.post(f"{gateway_url}/plugin/registered-sensor/enum",
                              json={}, timeout=TIMEOUT)
        r.raise_for_status()
        return [s["hstNm"] for s in r.json()["response"]["data"]["sensorList"]]

    def sensor_url(self, gateway_url: str, hostname: str) -> str:
        domain = gateway_url.split("://", 1)[1].split(".", 1)[1]
        return f"https://{hostname.lower()}.{domain}"


class Manifest:
    """SQLite gate: if (source, key) exists, we never re-download."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            " source TEXT, key TEXT, garage TEXT, sensor TEXT, ts TEXT,"
            " sha256 TEXT, size INTEGER, path TEXT, pulled_at TEXT,"
            " PRIMARY KEY (source, key))"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_sha ON images(sha256)")

    def has(self, source: str, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM images WHERE source=? AND key=?",
                               (source, key)).fetchone() is not None

    def path_for_sha(self, sha: str):
        row = self.db.execute("SELECT path FROM images WHERE sha256=? LIMIT 1",
                              (sha,)).fetchone()
        return row[0] if row else None

    def add(self, source, key, garage, sensor, ts, sha, size, path):
        self.db.execute(
            "INSERT OR IGNORE INTO images VALUES (?,?,?,?,?,?,?,?,?)",
            (source, key, garage, sensor, ts, sha, size, str(path),
             datetime.utcnow().isoformat(timespec="seconds")))
        self.db.commit()


def store_bytes(data: bytes, dest: Path, manifest: Manifest,
                source, key, garage, sensor, ts):
    """Write with sha dedup: identical content anywhere in the corpus is kept once."""
    sha = hashlib.sha256(data).hexdigest()
    existing = manifest.path_for_sha(sha)
    if existing:
        manifest.add(source, key, garage, sensor, ts, sha, len(data), existing)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest.add(source, key, garage, sensor, ts, sha, len(data), dest)
    return True


def backfill_legacy(legacy_root: Path, data_root: Path, manifest: "Manifest"):
    """Import existing training images into the unified store.

    Registers files under source 'sensor-store' with the same (sensor/filename)
    keys the live puller uses, so already-held images are never re-downloaded.
    Hardlinks when possible (same filesystem), copies otherwise.
    """
    import shutil
    stats = dict.fromkeys(("new", "dup", "cached", "skipped"), 0)
    for png in sorted(legacy_root.glob("*/training_images/*/*.png")):
        garage, _, sensor, name = png.parts[-4:]
        parts = name.split("-")
        if len(parts) < 4 or len(parts[2]) != 8:
            stats["skipped"] += 1
            log.debug("skip unparseable filename %s", name)
            continue
        when = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:]}T{parts[3][:2]}:{parts[3][2:4]}"
        key = f"{sensor}/{name}"
        if manifest.has("sensor-store", key):
            stats["cached"] += 1
            continue
        data = png.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        existing = manifest.path_for_sha(sha)
        if existing:
            manifest.add("sensor-store", key, garage, sensor, when, sha, len(data), existing)
            stats["dup"] += 1
            continue
        dest = data_root / "images" / garage / sensor / parts[2][:4] / parts[2][4:6] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(png, dest)
        except OSError:
            shutil.copy2(png, dest)
        manifest.add("sensor-store", key, garage, sensor, when, sha, len(data), dest)
        stats["new"] += 1
        if stats["new"] % 500 == 0:
            log.info("backfill progress: %s", stats)
            check_disk(data_root, MIN_FREE_GB_DEFAULT)
    return stats


def hour_range(start: datetime, end: datetime):
    t = start.replace(minute=0, second=0)
    while t < end:
        yield t
        t += timedelta(hours=1)


def parse_ts_arg(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def pull_source_a(cf, manifest, data_root, garage, gateway, start, end, interval_min,
                  max_per_sensor, notify):
    """Sensor training-image stores: list per hour, sample by interval, download once."""
    try:
        sensors = cf.enum_sensors(gateway)
        log.info("%s: %d sensors at %s", garage, len(sensors), gateway)
    except Exception as e:
        log.info("skip %s (gateway unreachable: %s)", garage, e)
        return 0
    pulled = 0
    for host in sensors:
        base = cf.sensor_url(gateway, host)
        sensor = host.lower()
        last_kept = None
        kept = 0
        for t in hour_range(start, end):
            if max_per_sensor and kept >= max_per_sensor:
                break
            notify(garage, sensor, None, f"{garage}/{sensor} listing {t:%Y-%m-%d %H}h")
            url = f"{base}/training-image/files/{t.year}/{t.month}/{t.day}/{t.hour}"
            try:
                r = http_get(cf.session, url)
                files = r.json().get("response", {}).get("data", {}).get("files", []) \
                    if r.ok else []
            except Exception as e:
                log.debug("list failed %s: %s", url, e)
                continue
            for f in sorted(files, key=lambda x: x.get("dateTime", "")):
                name, ts = f.get("fileName"), f.get("dateTime", "")
                if not name:
                    continue
                when = datetime.strptime(ts[:14], "%Y%m%d%H%M%S") if ts else t
                if last_kept and (when - last_kept) < timedelta(minutes=interval_min):
                    continue
                key = f"{host}/{name}"
                if manifest.has("sensor-store", key):
                    last_kept = when
                    notify(garage, sensor, "cached")
                    continue
                try:
                    img = http_get(cf.session, f"{base}/training-image/image/{name}")
                    img.raise_for_status()
                except Exception as e:
                    log.warning("download failed %s: %s", name, e)
                    notify(garage, sensor, "error")
                    continue
                dest = (data_root / "images" / garage / sensor /
                        f"{when:%Y}" / f"{when:%m}" / name)
                new = store_bytes(img.content, dest, manifest, "sensor-store", key,
                                  garage, sensor, when.isoformat())
                log.info("%s %s", "stored" if new else "dedup", key)
                notify(garage, sensor, "new" if new else "dup")
                last_kept = when
                kept += 1
                pulled += 1
                if max_per_sensor and kept >= max_per_sensor:
                    break
    return pulled


MIN_FREE_GB_DEFAULT = 50


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 1e9


_last_disk_check = 0.0


def check_disk(data_root, min_free_gb, interval_s=60):
    """Abort the pull gracefully before the disk gets tight.

    Throttled: callers can invoke freely, the actual check runs at most
    once per interval_s (pass interval_s=0 to force, e.g. at startup)."""
    global _last_disk_check
    if time.time() - _last_disk_check < interval_s:
        return
    _last_disk_check = time.time()
    g = free_gb(data_root)
    if g < min_free_gb:
        log.error("disk guard: only %.1f GB free (< %.0f GB floor) — stopping", g, min_free_gb)
        raise SystemExit(
            f"disk guard: {g:.1f} GB free on the store volume is below the "
            f"{min_free_gb:.0f} GB floor — pull stopped cleanly (resume any time "
            f"after freeing space, or lower --min-free-gb)")


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def select_runs(portal, site_id, per_garage):
    """Pick a diverse subset of snapshot runs: spread over the full date span
    AND across time-of-day buckets (night/morning/day/evening)."""
    runs = portal.graphql(
        "query($id: Int!) { snapshots(condition: {siteId: $id}, first: 1000)"
        " { nodes { id runAt snapshotParkingSpaces { totalCount } } } }",
        {"id": site_id})["snapshots"]["nodes"]
    # prune failed/empty runs (no images) and back-to-back re-runs (< 6h apart:
    # keep the run with more images, which drops aborted retries too)
    runs = sorted((r for r in runs
                   if r.get("runAt") and r["snapshotParkingSpaces"]["totalCount"] > 0),
                  key=lambda r: r["runAt"])
    spaced = []
    for r in runs:
        if spaced and (r["runAt"][:13] <= spaced[-1]["runAt"][:13] or
                       abs(datetime.fromisoformat(r["runAt"][:19]) -
                           datetime.fromisoformat(spaced[-1]["runAt"][:19]))
                       < timedelta(hours=6)):
            if (r["snapshotParkingSpaces"]["totalCount"] >
                    spaced[-1]["snapshotParkingSpaces"]["totalCount"]):
                spaced[-1] = r
            continue
        spaced.append(r)
    runs = spaced
    if len(runs) <= per_garage:
        return runs
    buckets = {}
    for r in runs:
        h = int(r["runAt"][11:13]) if len(r["runAt"]) > 12 else 12
        buckets.setdefault(h // 6, []).append(r)
    picked = []
    quota = max(1, per_garage // len(buckets))
    for b in buckets.values():  # evenly-spaced picks inside each bucket
        step = max(1, len(b) // quota)
        picked.extend(b[::step][:quota])
    for r in runs:  # top up to per_garage with widest temporal stride
        if len(picked) >= per_garage:
            break
        if r not in picked:
            picked.append(r)
    return sorted(picked[:per_garage], key=lambda r: r["runAt"])


def snapshot_frames(portal, snapshot_id):
    """Unique full frames of one snapshot run: [{uuid, url, sensor}]."""
    d = portal.graphql(
        "query($id: UUID!) { snapshot(id: $id) { snapshotParkingSpaces { nodes {"
        " sensorSnapshotId originalImageUrlSigned"
        " detectedBySensor { configurationName } } } } }",
        {"id": snapshot_id})["snapshot"]
    from urllib.parse import unquote, urlparse
    frames, seen = [], set()
    for n in d["snapshotParkingSpaces"]["nodes"]:
        url = n.get("originalImageUrlSigned")
        if not url:
            continue
        # sensorSnapshotId is often null; the blob path is a stable identity:
        # /snapshots/<run-uuid>/originals/<sensor-mac>-cameraN.jpg
        parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
        uid = n.get("sensorSnapshotId") or "/".join(parts[-3:])
        if uid in seen:
            continue
        seen.add(uid)
        sensor = slugify((n.get("detectedBySensor") or {}).get("configurationName")
                         or "unknown")
        fname = re.sub(r"[^A-Za-z0-9._-]", "_", f"{parts[-3][:8]}-{parts[-1]}")
        frames.append({"uuid": uid, "url": url, "sensor": sensor, "fname": fname})
    return frames


def pull_source_b(portal, manifest, data_root, site, runs, notify, pos=""):
    """Portal snapshot archive: pull the planned runs' full frames, download once."""
    garage = site["slug"]
    pulled = 0
    for j, run in enumerate(runs, 1):
        check_disk(data_root, pull_source_b.min_free_gb)
        ts = run.get("runAt") or ""
        notify(garage, "•", None, f"{pos}{garage} · run {j}/{len(runs)} · {ts[:16]}",
               progress=(j - 1, len(runs)))
        try:
            frames = snapshot_frames(portal, run["id"])
        except Exception as e:
            log.warning("%s snapshot %s failed: %s", garage, run["id"], e)
            notify(garage, "•", "error")
            continue
        for fr in frames:
            if manifest.has("portal-snapshot", fr["uuid"]):
                notify(garage, "•", "cached")
                continue
            try:
                # presigned URL: auth embedded, bare session; never persist it
                r = http_get(requests.Session(), fr["url"])
                r.raise_for_status()
            except Exception as e:
                log.warning("frame fetch failed %s: %s", fr["uuid"], e)
                notify(garage, "•", "error")
                continue
            dest = (data_root / "images" / garage / fr["sensor"] /
                    ts[:4] / ts[5:7] / f"{fr['sensor']}-{fr['fname']}")
            new = store_bytes(r.content, dest, manifest, "portal-snapshot", fr["uuid"],
                              garage, fr["sensor"], ts)
            notify(garage, "•", "new" if new else "dup")
            pulled += 1
    notify(garage, "•", None, progress=(len(runs), len(runs)))
    return pulled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"),
                    help="training-image store (default: repo-local 'data' symlink)")
    ap.add_argument("--backfill", type=Path, metavar="LEGACY_ROOT",
                    help="import existing <garage>/training_images/<sensor>/*.png into the "
                         "store (hardlink + manifest register, no network), then exit")
    ap.add_argument("--source", choices=["a", "b", "both"], default="both",
                    help="a = sensor training stores, b = portal snapshots")
    ap.add_argument("--start", help="YYYYMMDDHHmm UTC (source a)")
    ap.add_argument("--end", help="YYYYMMDDHHmm UTC (source a)")
    ap.add_argument("--interval-min", type=int, default=60,
                    help="min minutes between kept images per sensor (source a)")
    ap.add_argument("--max-per-sensor", type=int, default=0,
                    help="cap images per sensor per run, 0 = no cap (source a)")
    ap.add_argument("--runs-per-garage", type=int, default=12,
                    help="diverse snapshot runs to pull per garage (source b)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT,
                    help="stop pulling when the store volume's free space drops "
                         f"below this many GB (default {MIN_FREE_GB_DEFAULT})")
    ap.add_argument("--plan-only", action="store_true",
                    help="source b: build + save the diverse run selection, download nothing")
    ap.add_argument("--garages", help="comma-separated site-name filter (default: all)")
    ap.add_argument("--list-garages", action="store_true",
                    help="discover and list sites, then exit (portal token only)")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="log file (default: <data-root>/pull.log)")
    args = ap.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or args.data_root / "pull.log"
    setup_logging(log_file)

    if args.backfill:
        manifest = Manifest(args.data_root / "manifest.sqlite")
        console.print(f"Backfilling from [bold]{args.backfill}[/bold] → {args.data_root} …")
        stats = backfill_legacy(args.backfill, args.data_root, manifest)
        console.print(Panel(
            f"imported [green]{stats['new']}[/green] · content-dup [yellow]{stats['dup']}[/yellow] · "
            f"already registered [dim]{stats['cached']}[/dim] · unparseable {stats['skipped']}",
            title="Backfill done", border_style="green"))
        log.info("backfill complete: %s", stats)
        return

    # --- credentials: prompted (no echo), RAM only, never logged/persisted ---
    console.print(Panel("Credentials are held in memory only — never written to disk.\n"
                        f"Detailed log: [bold]{log_file}[/bold]",
                        title="Training image pull", border_style="cyan"))
    portal_token = Prompt.ask("[cyan]Portal API refresh token[/cyan]",
                              password=True, console=console)
    portal_token = "".join(portal_token.split())  # kill line-wrap newlines from long pastes
    if not portal_token:
        sys.exit("portal token required")
    portal = PortalClient(portal_token)

    with console.status("Discovering organizations and sites…"):
        sites = portal.discover_sites()
    if args.garages:
        wanted = {g.strip().lower() for g in args.garages.split(",")}
        sites = [s for s in sites if s["name"].lower() in wanted
                 or s["display"].lower() in wanted]
    log.info("discovered %d sites", len(sites))

    if args.list_garages:
        t = Table(title=f"{len(sites)} site(s)", box=box.SIMPLE_HEAD)
        t.add_column("organization", style="dim")
        t.add_column("site name", style="cyan")
        t.add_column("display name")
        for s in sites:
            t.add_row(s["org"], s["name"], s["display"])
        console.print(t)
        return
    console.print(f"[green]{len(sites)}[/green] site(s) discovered")

    cf = None
    if args.source in ("a", "both"):
        if not (args.start and args.end):
            sys.exit("--start/--end required for source a")
        cf = CloudflareAccess(
            Prompt.ask("[cyan]CF-Access-Client-Id[/cyan]", console=console).strip(),
            Prompt.ask("[cyan]CF-Access-Client-Secret[/cyan]",
                       password=True, console=console).strip(),
        )

    manifest = Manifest(args.data_root / "manifest.sqlite")
    pull_source_b.min_free_gb = args.min_free_gb
    check_disk(args.data_root, args.min_free_gb, interval_s=0)
    console.print(f"disk guard: {free_gb(args.data_root):.0f} GB free, "
                  f"floor {args.min_free_gb:.0f} GB")
    stats = PullStats()
    stats.data_root = args.data_root

    # phase 1 (source b): build the COMPLETE plan first so the overall total is
    # fixed before any pulling starts
    plan = {}
    if args.source in ("b", "both"):
        import json as _json
        manifest.db.execute(
            "CREATE TABLE IF NOT EXISTS garages (site_id INTEGER PRIMARY KEY,"
            " org TEXT, name TEXT, display TEXT, slug TEXT)")
        with console.status("Planning: selecting diverse runs across all garages…") as st:
            for i, s in enumerate(sites, 1):
                s["slug"] = slugify(s["display"])
                st.update(f"Planning [{i}/{len(sites)}] {s['slug']}…")
                manifest.db.execute(
                    "INSERT OR REPLACE INTO garages VALUES (?,?,?,?,?)",
                    (s["site_id"], s["org"], s["name"], s["display"], s["slug"]))
                manifest.db.commit()
                try:
                    runs = select_runs(portal, s["site_id"], args.runs_per_garage)
                except Exception as e:
                    log.warning("%s: run selection failed: %s", s["slug"], e)
                    runs = []
                plan[s["slug"]] = {"site_id": s["site_id"],
                                   "runs": [{"id": r["id"], "runAt": r.get("runAt", "")}
                                            for r in runs]}
                if runs:
                    stats.progress[s["slug"]] = (0, len(runs))
                log.info("plan %s: %d runs", s["slug"], len(runs))
        plan_file = args.data_root / "snapshot_plan.json"
        plan_file.write_text(_json.dumps(plan, indent=1))
        n_runs = sum(len(p["runs"]) for p in plan.values())
        console.print(f"plan: [green]{n_runs}[/green] runs across "
                      f"{sum(1 for p in plan.values() if p['runs'])} garages "
                      f"(saved: {plan_file})")
        if args.plan_only:
            return
    total = 0

    with Live(stats.table(), console=console, refresh_per_second=4) as live:
        def notify(garage, sensor, event, current=None, progress=None):
            if event:
                stats.bump(garage, sensor, event)
            if current is not None:
                stats.current = current
            if progress is not None:
                stats.progress[garage] = progress
            live.update(stats.table())

        if args.source in ("a", "both"):
            start, end = parse_ts_arg(args.start), parse_ts_arg(args.end)
            for s in sites:
                gateway = GATEWAY_TEMPLATE.format(site=s["name"].lower())
                total += pull_source_a(cf, manifest, args.data_root,
                                       s["name"].lower(), gateway, start, end,
                                       args.interval_min, args.max_per_sensor, notify)

        if args.source in ("b", "both"):
            # phase 2: pull against the frozen plan built before the Live table
            for i, s in enumerate(sites, 1):
                runs = plan[s["slug"]]["runs"]
                if not args.plan_only and runs:
                    total += pull_source_b(portal, manifest, args.data_root, s,
                                           runs, notify,
                                           pos=f"[garage {i}/{len(sites)}] ")


        stats.current = ""
        live.update(stats.table())

    console.print(Panel(f"[green]{total}[/green] new images pulled\n"
                        f"manifest: {args.data_root / 'manifest.sqlite'}\n"
                        f"log: {log_file}",
                        title="Done", border_style="green"))
    log.info("run complete: %d new images", total)


if __name__ == "__main__":
    main()
