#!/usr/bin/env bash
# Backs up everything irreplaceable that git DOESN'T track (~13 GB):
#   data/images (frames+drafts+attributes), runs/ (checkpoints, exports,
#   metrics), manifest.sqlite, spot_polygons.json, evidence dirs.
# Regenerable heavies (venvs, weights/, draft_previews) excluded.
#
# Manual:    ./backup_valuables.sh /mnt/nas/falcon-backup
#            (optional offsite: read -s AZURE_SAS_URL && export AZURE_SAS_URL)
# Service:   ./backup_valuables.sh --from-config
#            (config written by setup_backup_service.sh; runs unattended
#             with retries; per-machine subdir <hostname>-<machine-id8>)
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CONF_DIR="$HOME/.config/falcon-vision-backup"
TAG="$(hostname)-$(cut -c1-8 /etc/machine-id 2>/dev/null || echo unknown)"

if [ "${1:-}" = "--from-config" ]; then
  # shellcheck source=/dev/null
  source "$CONF_DIR/config"           # sets NAS_DEST (and optionally uses sas file)
  DEST="$NAS_DEST/$TAG"
  [ -f "$CONF_DIR/sas.url" ] && AZURE_SAS_URL="$(cat "$CONF_DIR/sas.url")"
else
  DEST="${1:?usage: backup_valuables.sh <dest> | --from-config}/$TAG"
fi
mkdir -p "$DEST"

retry() {  # retry <label> <cmd...> — 3 attempts, 30s/120s backoff
  local label="$1"; shift
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 3 ]; then echo "[$label] FAILED after 3 attempts"; return 1; fi
    local wait=$((n == 1 ? 30 : 120))
    echo "[$label] attempt $n failed — retrying in ${wait}s"
    sleep "$wait"
  done
}

echo "== backup -> $DEST"
retry "runs" rsync -a --info=progress2 --copy-links "$REPO/runs/" "$DEST/runs/"
SRCS=("$REPO/data/images" "$REPO/data/manifest.sqlite"
      "$REPO/data/spot_polygons.json" "$REPO/data/sam3_sandbox"
      "$REPO/data/probes" "$REPO/data/annotation_queue_phase1.json"
      "$REPO/data/sensor_archive")   # sensor OD-history mirror (sensors
                                     # truncate; this is the only copy)
EXIST=()
for s in "${SRCS[@]}"; do [ -e "$s" ] && EXIST+=("$s"); done
retry "data" rsync -a --info=progress2 --copy-links "${EXIST[@]}" "$DEST/data/"
echo "local backup complete ($(du -sh "$DEST" | cut -f1))"

if [ -n "${AZURE_SAS_URL:-}" ]; then
  if ! command -v rclone >/dev/null; then
    echo "rclone not installed (sudo apt install rclone) — cloud leg skipped"
    exit 0
  fi
  echo "== syncing to Azure Blob ($TAG)…"
  retry "azure" rclone sync "$DEST" ":azureblob,sas_url=${AZURE_SAS_URL}:/$TAG" \
    --azureblob-access-tier Cool --stats-one-line
  echo "cloud sync complete"
else
  echo "(no Azure SAS — cloud leg skipped)"
fi
