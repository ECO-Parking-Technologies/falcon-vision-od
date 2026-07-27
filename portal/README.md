# portal/ — training image acquisition

Design: [../docs/planning/04-portal-integration.md](../docs/planning/04-portal-integration.md)

## pull_training_images.py

Unified puller for both image sources, with portal-driven garage auto-discovery
and a SQLite manifest that guarantees nothing is downloaded twice.

```bash
# discover garages only (needs just the portal token)
python3 portal/pull_training_images.py --data-root /path/to/store --list-garages

# pull both sources for a week, hourly sampling, all garages
python3 portal/pull_training_images.py --data-root /path/to/store \
    --start 202607200000 --end 202607270000 --interval-min 60

# portal snapshots only, capped per site
python3 portal/pull_training_images.py --data-root /path/to/store \
    --source b --max-per-site 50
```

Credentials (portal refresh token; Cloudflare Access client id/secret for
source a) are prompted at startup with no echo and held only in RAM — never
put them in files, args, or env vars.

Output layout:

```
<data-root>/
  images/<garage>/<sensor>/<YYYY>/<MM>/<file>
  manifest.sqlite     # (source, key) rows gate downloads; sha256 dedups content
```

## reference/

Verbatim copies from `falcon-vision-ml` these were distilled from (not
imported by anything):

- `falcon_sensor_training_image_download.py` + `.yml` — sensor-store downloader
- `cloudflare_auth.py` — CF Access session handling
- `portal_api_client.py` — `PortalAPIClient` + `SensorImageFetcher` classes
  extracted from `portal_validation_viewer.py`

Delete this directory once the puller has proven itself in real use.
