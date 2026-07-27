# 04 — Training Image Acquisition (design settled 2026-07-27; build pending)

Goal: feed the OD model diverse vehicle imagery from **all garages**, automatically and organized — in service of the only metric that matters: **detect vehicles more accurately and faster than the off-the-shelf model**. (Spot definitions / occupancy geometry are explicitly out of scope here; a note on them at the bottom.)

## The two image sources (both proven in falcon-vision-ml code)

**A. Sensor training-image store** — each sensor keeps images at `/media/sdcard/images/training`, served over HTTPS behind **Cloudflare Access**:
- Sensor URL: `https://<sensor-host>.fvg-<garage>.private.ecofalcondata.com` (or from portal `sensorRemoteConnection {host, port, https}`)
- Enumerate sensors: `POST <gateway>/plugin/registered-sensor/enum` → `sensorList[].hstNm`
- List images per hour: `GET <sensor>/training-image/files/{yyyy}/{m}/{d}/{h}`; fetch: `GET <sensor>/training-image/image/{filename}`
- Reference impl: `falcon-vision-ml/data_pipeline/dnn/falcon_sensor_training_image_download.py`

**B. Portal snapshot archive** (the big one) — GraphQL `validations` → `validationParkingSpaces` → `snapshotParkingSpace`:
- `originalImageUrlSigned` = **full camera frame** via presigned S3 URL (plain GET); or direct from sensor: `GET <sensor_url>/plugin/snapshot/get/{sensorSnapshotId}` (base64 full frame).
- Reference impl: `falcon-vision-ml/inference/dnn/portal_validation_viewer.py` (`PortalAPIClient` ~L431-680).

**⚠️ docs/portal-api/ is stale** — live API also serves `validations`, `snapshot`, `Sensor.sensorRemoteConnection`; regenerate via GraphQL introspection during the build.

## Unified image store — merge both sources, download once, never again

Both sources yield static, immutable files → strict download-once semantics:

```
<data_root>/
  images/<garage>/<sensor>/<camera>/<YYYY>/<MM>/<sensor>-<camera>-<ts>.<ext>
  manifest.sqlite
```

- **Manifest** (SQLite) is the gate for every download: rows keyed by stable source identity — source A: `(sensor, filename)`; source B: `snapshotParkingSpace`/`sensorSnapshotId` uuid — plus garage, sensor, camera, capture timestamp, sha256, byte size, pulled_at, source. **If the key is in the manifest, no HTTP request is made.** Interrupted pulls resume for free.
- **Merge + dedup**: both sources can capture the same scene; after download, content-hash (sha256) dedup keeps one copy, manifest records both provenances pointing at it. Near-duplicate frames (static scene, nothing moved) filtered by perceptual hash before entering the annotation queue.
- One layout for both sources; the training wrapper consumes it directly (or via a thin split/symlink step, as today).

## Credentials: RAM only, never on disk

Both secrets are pasted into the terminal at start of a pull and held only in process memory:

- **Portal API refresh token** and **CF Access client id/secret** via `getpass`-style prompts (no echo). No key files (deliberately NOT the viewer's `~/.eco_portal_key.txt` pattern), no env vars written to profiles, no values in YAML/config.
- Never passed as CLI arguments (shell history), never logged, never written into the manifest or cache files.
- **Presigned URLs count as secrets** (they embed access signatures): fetch-and-discard, never persisted to manifest/logs; store the stable snapshot uuid instead.
- OAuth access tokens refreshed in memory (~20 min lifetime, refresh at expiry — don't re-exchange per request; the token endpoint is rate-limited).

## Auto-discovery (no manual garage lists)

1. GraphQL: `organizations(isDeleted:false)` → `sites(...)` → sensors with `sensorRemoteConnection`.
2. Keep sensors whose host matches `*.ecofalcondata.com` → pullable garages, discovered fresh each run. (Fallback: derive gateway `https://legacy.fvg-<site.name>.private.ecofalcondata.com` + `/plugin/registered-sensor/enum`.)

## Sampling: diversity over volume

Few images per sensor per time window, spread wide:
- Cap per sensor per pull (hourly-interval default like the existing downloader; tune later), across **all** garages rather than deep dives into one.
- Spread across time-of-day buckets and calendar days; prefer occupancy-change moments over static repeats; perceptual-hash out the near-identical frames.
- The cap + manifest make repeated runs incremental: each run tops up new time windows only.

## Build tasks

- [ ] Regenerate portal API reference via live GraphQL introspection.
- [ ] `portal/` module: RAM-only credential prompts, OAuth refresh handling, org→site→sensor discovery, gateway resolution.
- [ ] Unified puller (sources A + B) with manifest-gated download-once, sha256 + perceptual-hash dedup, politeness (timeouts, retry/backoff, modest concurrency — the reference scripts have none).
- [ ] Sampling layer (per-sensor caps, time-bucket spread) → images land in the unified layout ready for preannotation → CVAT.

*Parked (not current focus):* `snapshotParkingSpace.bounds` spot polygons and the human occupancy labels in validations remain available for the track-06 evaluator whenever spot-level evaluation becomes relevant.
