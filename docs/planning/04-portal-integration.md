# 04 — Portal Integration: Pulling Training Images (design settled 2026-07-27; build pending)

Goal: automatically pull diverse training images across **all garages**, replacing the manual external-drive workflow.

## The two image sources (both proven in falcon-vision-ml code)

**A. Sensor training-image store** — each sensor keeps images at `/media/sdcard/images/training`, served over HTTPS behind **Cloudflare Access** (service token: `CF_ACCESS_CLIENT_ID`/`SECRET`):
- Sensor URL: `https://<sensor-host>.fvg-<garage>.private.ecofalcondata.com` (or from portal `sensorRemoteConnection {host, port, https}`)
- Enumerate sensors: `POST <gateway>/plugin/registered-sensor/enum` → `sensorList[].hstNm`
- List images per hour: `GET <sensor>/training-image/files/{yyyy}/{m}/{d}/{h}`; fetch: `GET <sensor>/training-image/image/{filename}`
- Working reference implementation: `falcon-vision-ml/data_pipeline/dnn/falcon_sensor_training_image_download.py` (+ `.yml`) — time window, `interval` minutes between images, camera filter, `sensors: ['*']`, discovery caching, resume, writes the exact `{garage}/training_images/{sensor}/` layout our training consumes.

**B. Portal snapshots (the big archive)** — via GraphQL `validations` → `validationParkingSpaces` → `snapshotParkingSpace`:
- `originalImageUrlSigned` = **full camera frame** via S3-presigned URL (plain GET, no auth); `imageUrlSigned` = per-spot ROI crop.
- Alternative direct-from-sensor: `GET <sensor_url>/plugin/snapshot/get/{sensorSnapshotId}` (Cloudflare Access) → base64 full frame + per-space crops.
- Working reference: `falcon-vision-ml/inference/dnn/portal_validation_viewer.py` (`PortalAPIClient` ~L431-680, `SensorImageFetcher` ~L221-420).

**⚠️ Our extracted API docs (docs/portal-api/) are stale/incomplete**: the live API serves `validations`, `snapshot`, and `Sensor.sensorRemoteConnection`, none of which appear in the SpectaQL dump. Regenerate the reference via GraphQL introspection against the live endpoint (small task, do it during the build).

## Bonus findings (feed tracks 05/06)

- `snapshotParkingSpace.bounds` = 4-point spot polygon in **normalized 0-1 coords** → the spot-definition source for the occupancy evaluator (point order NOT guaranteed — sort explicitly; see viewer `extract_roi` L96-145).
- Validations carry **human-labeled occupancy ground truth** (`validationParkingSpaceResponses.occupancyStatus`, overrides; encoding 1=vacant 2=occupied) tied to snapshots — ready-made eval data for track 06, and a source of auto-labels for `InEcoParkingSpot`-style training signal.
- `reportedOccupancyStatus`, `rawInference/inference/filteredInference` per spot = the current production model's outputs, directly comparable.

## Auto-discovery design (no manual garage lists)

Portal is the source of truth; Cloudflare gateways do the heavy pulling:

1. GraphQL: `organizations(isDeleted:false)` → `sites(organizationId, isDeleted:false)` (skip `isDataPollingActive:false` as appropriate).
2. Per site, get sensors + `sensorRemoteConnection {host, port, https}`; keep sites whose sensor hosts match `*.ecofalcondata.com` → those are pullable garages. (Fallback: derive gateway `https://legacy.fvg-<site.name>.private.ecofalcondata.com` and enumerate via `/plugin/registered-sensor/enum`.)
3. Auto-populate the downloader config from that list — garages appear/disappear as the portal changes, no YAML edits.

## Sampling strategy (diversity over volume)

Few images per sensor per time window, spread wide:
- Cap per sensor per pull (e.g. hourly interval like the existing downloader default, or N/day), across **all** garages rather than deep dives into one.
- Spread across time-of-day buckets (morning/day/evening/night) and calendar days; prefer occupancy-change moments (`parkingSpaceDataPoints` timestamps) over static repeats.
- Dedup near-identical frames (content hash / perceptual hash) before they reach annotation.
- Manifest (JSON/SQLite) recording source, sensor, timestamp, hash → resumable, idempotent pulls.

## Build tasks

- [ ] Regenerate portal API reference via live GraphQL introspection.
- [ ] `portal/` module in this repo: OAuth2 token handling (refresh-token file like `~/.eco_portal_key.txt`), org→site→sensor discovery, gateway resolution.
- [ ] Puller combining source A (training-image store) with the auto-discovered garage list; politeness added (timeouts, retry/backoff, modest concurrency — the reference scripts have none).
- [ ] Snapshot puller for source B (validations → signed full-frame URLs) with the same manifest/dedup.
- [ ] Sampling/dedup layer per the strategy above; output into the training data layout.
- [ ] Export spot polygons per sensor from `snapshotParkingSpace.bounds` for track 06.
