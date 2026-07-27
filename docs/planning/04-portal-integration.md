# 04 — Portal Integration: Pulling Training Images (blocked on API gap)

Goal: pull training images for **all garages** automatically from the ECO Parking portal, replacing the manual external-drive workflow, with a data layout designed for scale.

## API facts (extracted → [../portal-api/](../portal-api/))

- GraphQL endpoint: `https://api.ecoparkingtechnologies.com/graphql`; API Explorer in the portal.
- Auth: OAuth2 — create a **Personal API Token** (refresh token) in the portal, exchange at `https://identity.ecoparkingtechnologies.com/token` with `grant_type=refresh_token`, `client_id=EcoParking.ClientApi` → Bearer access token (~20 min lifetime; don't re-exchange per request).
- 73 queries, no mutations exposed. Relevant entities: `sites` (garages), `siteLevels`, `sensors` (`sensorId`, `siteId`, config), `parkingSpaces`, `parkingZones`, `parkingSpaceVehicleSessions`, `vehicleRecognitions`.

## ⚠️ The gap

**No query returns sensor camera images.** The only image fields in the whole schema are LPR crops on `VehicleRecognition` (`imageUrlSigned`, `compositeImageUrlSigned`, …) and `SiteLevel.mapFilePathSigned`. Filter types reference `snapshotParkingSpacesExist` / `snapshotsExist`, so a *snapshot* table exists server-side, but **no Snapshot type or query is exposed**.

- [ ] Confirm with the portal/backend team: expose a `snapshots` query (sensor id, timestamp, signed image URL, pagination + time filters) — the `VehicleRecognition` signed-URL pattern already exists to copy.
- [ ] Until then, decide interim source (existing data_pipeline drops, direct sensor pulls, or a bulk export from the backend).

## Puller design (once images are reachable)

- [ ] Python CLI (`portal/` module): token refresh, paged GraphQL queries, resumable downloads, per-garage/sensor manifest (JSON/SQLite) recording source, timestamp, and content hash for dedup.
- [ ] Layout: `<data_root>/<garage>/<sensor>/<YYYY>/<MM>/<sensor>-<camera>-<ts>.png` + manifest — restructure of the current `training_images/` flat layout is allowed.
- [ ] **Sampling step** between pull and annotation (roadmap guidance: unique scenes > repeated frames): occupancy-change events (`parkingSpaceDataPoints` / vehicle sessions give the timestamps to target), plus coverage per time-of-day and per sensor.
- [ ] Feed into preannotation → CVAT task creation per garage+sensor (CVAT API can automate task creation; currently manual).

## Also available via API (useful beyond images)

`parkingSpaces` geometry/config could seed **spot definitions** for the occupancy evaluator ([06-evaluation.md](06-evaluation.md)), and `parkingSpaceDataPoints` provide historical occupancy ground truth from the current classifier to compare against.
