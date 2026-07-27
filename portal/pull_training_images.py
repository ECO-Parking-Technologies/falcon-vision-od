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
import getpass
import hashlib
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

log = logging.getLogger("pull")

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
        r.raise_for_status()
        data = r.json()
        self._access_token = data["access_token"]
        self._expiry = time.time() + data.get("expires_in", 1200)
        return self._access_token

    def graphql(self, query: str, variables: dict = None) -> dict:
        r = self.session.post(
            PORTAL_GRAPHQL,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        out = r.json()
        if out.get("errors"):
            raise RuntimeError(f"GraphQL errors: {out['errors']}")
        return out["data"]

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


def hour_range(start: datetime, end: datetime):
    t = start.replace(minute=0, second=0)
    while t < end:
        yield t
        t += timedelta(hours=1)


def parse_ts_arg(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def pull_source_a(cf, manifest, data_root, garage, gateway, start, end, interval_min,
                  max_per_sensor):
    """Sensor training-image stores: list per hour, sample by interval, download once."""
    try:
        sensors = cf.enum_sensors(gateway)
    except Exception as e:
        log.info("skip %s (gateway unreachable: %s)", garage, e)
        return 0
    pulled = 0
    for host in sensors:
        base = cf.sensor_url(gateway, host)
        last_kept = None
        kept = 0
        for t in hour_range(start, end):
            if max_per_sensor and kept >= max_per_sensor:
                break
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
                    continue
                try:
                    img = http_get(cf.session, f"{base}/training-image/image/{name}")
                    img.raise_for_status()
                except Exception as e:
                    log.warning("download failed %s: %s", name, e)
                    continue
                dest = (data_root / "images" / garage / host.lower() /
                        f"{when:%Y}" / f"{when:%m}" / name)
                store_bytes(img.content, dest, manifest, "sensor-store", key,
                            garage, host.lower(), when.isoformat())
                last_kept = when
                kept += 1
                pulled += 1
                if max_per_sensor and kept >= max_per_sensor:
                    break
    return pulled


def pull_source_b(portal, manifest, data_root, site, max_per_site):
    """Portal snapshot archive: full frames from completed validations, download once."""
    garage = re.sub(r"[^a-z0-9_-]+", "_", site["display"].lower())
    pulled = 0
    try:
        validations = [v for v in portal.validations_for_site(site["site_id"])
                       if v.get("validationState") == "completed"]
    except Exception as e:
        log.info("skip %s validations (%s)", garage, e)
        return 0
    for v in sorted(validations, key=lambda x: x.get("createdTimestamp", ""), reverse=True):
        if max_per_site and pulled >= max_per_site:
            break
        try:
            frames = portal.validation_frames(v["id"])
        except Exception as e:
            log.debug("validation %s failed: %s", v["id"], e)
            continue
        for fr in frames:
            if manifest.has("portal-snapshot", fr["uuid"]):
                continue
            try:
                # presigned URL: auth is embedded, use a bare session; never persist it
                r = http_get(requests.Session(), fr["url"])
                r.raise_for_status()
            except Exception as e:
                log.warning("snapshot fetch failed %s: %s", fr["uuid"], e)
                continue
            when = fr["ts"][:19].replace(":", "").replace("-", "") or "unknown"
            dest = (data_root / "images" / garage / fr["sensor"] /
                    fr["ts"][:4] / fr["ts"][5:7] /
                    f"{fr['sensor']}-snapshot-{fr['uuid']}.jpg")
            store_bytes(r.content, dest, manifest, "portal-snapshot", fr["uuid"],
                        garage, fr["sensor"], fr["ts"])
            pulled += 1
            if max_per_site and pulled >= max_per_site:
                break
    return pulled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--source", choices=["a", "b", "both"], default="both",
                    help="a = sensor training stores, b = portal snapshots")
    ap.add_argument("--start", help="YYYYMMDDHHmm UTC (source a)")
    ap.add_argument("--end", help="YYYYMMDDHHmm UTC (source a)")
    ap.add_argument("--interval-min", type=int, default=60,
                    help="min minutes between kept images per sensor (source a)")
    ap.add_argument("--max-per-sensor", type=int, default=0,
                    help="cap images per sensor per run, 0 = no cap (source a)")
    ap.add_argument("--max-per-site", type=int, default=0,
                    help="cap snapshots per site per run, 0 = no cap (source b)")
    ap.add_argument("--garages", help="comma-separated site-name filter (default: all)")
    ap.add_argument("--list-garages", action="store_true",
                    help="discover and list sites, then exit (portal token only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    # --- credentials: prompted, RAM only, never logged/persisted ---
    portal_token = getpass.getpass("Portal API refresh token: ").strip()
    if not portal_token:
        sys.exit("portal token required")
    portal = PortalClient(portal_token)

    sites = portal.discover_sites()
    if args.garages:
        wanted = {g.strip().lower() for g in args.garages.split(",")}
        sites = [s for s in sites if s["name"].lower() in wanted
                 or s["display"].lower() in wanted]
    print(f"{len(sites)} site(s) discovered")
    if args.list_garages:
        for s in sites:
            print(f"  {s['org']:>20s} / {s['name']:<25s} ({s['display']})")
        return

    args.data_root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(args.data_root / "manifest.sqlite")

    total = 0
    if args.source in ("a", "both"):
        if not (args.start and args.end):
            sys.exit("--start/--end required for source a")
        cf = CloudflareAccess(
            input("CF-Access-Client-Id: ").strip(),
            getpass.getpass("CF-Access-Client-Secret: ").strip(),
        )
        start, end = parse_ts_arg(args.start), parse_ts_arg(args.end)
        for s in sites:
            gateway = GATEWAY_TEMPLATE.format(site=s["name"].lower())
            n = pull_source_a(cf, manifest, args.data_root, s["name"].lower(), gateway,
                              start, end, args.interval_min, args.max_per_sensor)
            if n:
                print(f"  {s['name']}: +{n} sensor-store images")
            total += n

    if args.source in ("b", "both"):
        for s in sites:
            n = pull_source_b(portal, manifest, args.data_root, s, args.max_per_site)
            if n:
                print(f"  {s['display']}: +{n} portal snapshots")
            total += n

    print(f"done: {total} new images (manifest: {args.data_root/'manifest.sqlite'})")


if __name__ == "__main__":
    main()
