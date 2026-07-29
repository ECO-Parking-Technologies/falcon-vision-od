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

## Agreed structure — v2, CLEAN SLATE (locked 2026-07-28, supersedes v1)

Decisions: previous annotations are retired (fresh CVAT project with the full attribute spec);
the store holds **only curated images, as real copies** (no hardlinks); **garage identity comes
from the portal**, and ordering follows from that.

- **Data root**: `/media/lopezemi/Expansion/falcon-vision-od-data` via the repo-local `data`
  symlink. Layout `images/<garage>/<sensor>/<YYYY>/<MM>/<file>` + `manifest.sqlite` + logs.
- **Canonical garage names = portal site names.** Discovery writes a `garages` table to the
  manifest (site_id, org, name, display); every image row references it. Legacy dir names map
  to portal names via an explicit mapping (e.g. `wpb_banyan → West Palm Beach/Banyan`,
  `yaamava-north → Yaamava'/North Garage`, `amazon → Amazon/AMAZON KCVG AirHub`,
  `arlington → Arlington Heights/Vail Avenue Garage`, `switch → City of Fishers/Switch Garage`,
  `carmel_* → Carmel/*`, `google*` → Google Alta — ambiguous ones confirmed before import).
- **Curated-only, copies-only**: images enter the store only AFTER selection (near-dup pruned,
  diversity sampled); selected files are copied (not linked) so the store is fully self-contained.
  **falcon-vision-ml's data is READ-ONLY source material — never deleted, never modified**: it
  remains the active dataset for the production image-classification model, which is still in
  heavy use. We only copy from it. The v1 hardlinked 189k mirror in OUR store gets wiped (it cost
  no disk and is superseded); that wipe touches only falcon-vision-od-data, never the originals.
- **Order of operations**: ① wipe v1 store → ② portal discovery seeds canonical garage names →
  ③ **portal snapshot pull first** (canonical names born correct) → ④ legacy import: dedup +
  diversity selection over falcon-vision-ml (dry-run scan sizing this now), then copy survivors
  under mapped names. Manifest keys unchanged (download-once still holds).
- **Sampling default: hourly** (`--interval-min 60`), breadth-first across all garages.
- **Clean-slate annotation**: new CVAT project — 6 classes; box attributes `InEcoParkingSpot`,
  `InMotion`, `Occluded`; image tags garage/sensor/time-of-day/conditions auto-filled where
  derivable (`export_cvat_labels` must emit the attribute definitions). Old arlington/wpb_banyan
  annotations are not migrated.
- **Preannotation**: Grounded SAM 2 (Apache-2.0, local 3090) as the box generator behind the
  existing COCO-1.0 converter, fine-tuned lite0 as consensus partner; humans audit in CVAT
  (confirm boxes + tick flags), export COCO → training.
- Training-wrapper adapter (consume the store layout) lands with the first annotation round.

## Snapshot inventory verdict (2026-07-28 scan)

**Portal snapshots alone are sufficient for training — sensor pulls demoted to optional.**
Scan of all 43 sites (`portal/snapshot_inventory.py`, data/inventory-20260728-125911.json):
**2,510 snapshot runs · ~2.11M per-space crop images (est.) · 40/43 garages · 2021→2026 span**,
plus 1,880 validations (human-labeled occupancy) for eval. Top sites: Yaamava ~334k, UF ~240k,
Fontainebleau ~220k, Amazon ~147k, Google ~117k. Notes:
- The 2.1M figure counts **per-space crops**; for OD training we want the **full camera frames**
  behind them (`originalImageUrlSigned`, deduped by `sensorSnapshotId`) — roughly runs × sensors
  ≈ **~100–200k unique full frames**, still ~100× today's 1,439 garage images.
- Diversity is excellent: 40 garages across climates/architectures (casinos, airports,
  universities, municipal), 5-year span; sparse sites (Midtown: 5 runs) are fixable on demand
  since snapshot runs can be triggered per garage.
- **Puller change required**: source B currently walks *validations* (only some sites run them);
  it must walk **snapshots directly** (snapshots → snapshotParkingSpaces → unique
  sensorSnapshotId → originalImageUrlSigned) with the rate-limit throttle/backoff now in
  PortalClient.

## Build tasks (v2 order)

- [ ] Wipe the v1 hardlinked store (after the dedup dry-run finishes measuring it).
- [ ] Puller: `garages` table from discovery; canonical-name layout; snapshots-walk for
      source B (snapshots → snapshotParkingSpaces → unique sensorSnapshotId →
      originalImageUrlSigned); copies not links.
- [x] **First full portal pull DONE (2026-07-29)**: 442/442 runs, 39 garages, 117,620 full frames (2.8 GB), 0 errors, org-qualified slugs, plan-first UI with per-garage %.
- [ ] Legacy import: dedup + diversity selection → copy survivors under mapped names
      (mapping confirmed for ambiguous legacy dirs first).
- [ ] Grounded SAM 2 runner behind the existing COCO converter; CVAT label config with
      attributes; auto-filled tags.

## Superseded v1 tasks

- [x] Copy reference implementations into `portal/reference/` (downloader + yml, cloudflare_auth, PortalAPIClient/SensorImageFetcher extracted from the viewer). Delete once the puller is proven.
- [x] **`portal/pull_training_images.py`** (first version, untested against live API): RAM-only credential prompts, OAuth refresh, org→site discovery, gateway derivation with graceful skip, sources A + B, SQLite manifest download-once, sha256 dedup, timeouts + retry/backoff, interval sampling + per-sensor/per-site caps, `--list-garages` mode.
- [ ] **First live run** (needs pasted tokens): `--list-garages`, then a small bounded pull; fix whatever reality disagrees with (esp. the source-A file-listing JSON shape and validation query field names vs the live schema).
- [ ] Regenerate portal API reference via live GraphQL introspection (same session as the first run).
- [ ] Perceptual-hash near-duplicate filtering before the annotation queue (sha256 exact-dedup is in).
- [ ] Wire pulled layout into preannotation/training (thin split step or config path update).

*Parked (not current focus):* `snapshotParkingSpace.bounds` spot polygons and the human occupancy labels in validations remain available for the track-06 evaluator whenever spot-level evaluation becomes relevant.

## Future: gateway-resident annotation oracle (noted 2026-07-28)

Gateway HW (Ubuntu Core 24, x86_64, snaps) can run Grounding DINO-class models for
continuous on-site auto-annotation of new snapshot runs + QA auditing of sensor decisions:
Supermicro SYS-111AD-WRN2 (Xeon/Core 12-14th gen, ~1.5-4s/frame CPU int8, PCIe room for a
T4/L4-class GPU later) and OnLogic K521 (Core Ultra 7 165H — OpenVINO can target its Arc
iGPU/NPU, ~0.5-1.5s/frame, fanless) are viable; CL250 is not. 32GB+ RAM fine (~4GB needed).
Packaging: OpenVINO int8 (GroundingDINO→ONNX→OV) in a strict snap — avoid CUDA-in-snap.
Division of labor: 3090 does historical backfill (~14k img/h); gateways handle incremental
snaps + drift detection. Not on the critical path — revisit after the first clean-slate
training round.
