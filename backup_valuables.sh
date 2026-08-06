#!/usr/bin/env bash
# Backs up everything irreplaceable that git DOESN'T track (~13 GB):
#   - data/images        frames + SAM3 drafts + in-spot attributes
#   - runs/              checkpoints, exports (deployed shas!), metrics, splits
#   - data/manifest.sqlite, spot_polygons.json, sam3_sandbox, probes, queue
# Deliberately EXCLUDED (regenerable): venvs, weights/, draft_previews.
#
#   ./backup_valuables.sh /path/to/backup/destination
#
# Incremental (rsync); safe to run repeatedly, e.g. after each training
# session or store pull. Never uses --delete: the backup only grows.
set -euo pipefail
DEST="${1:?usage: backup_valuables.sh <destination-dir>}"
REPO="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST"

# trailing slashes + --copy-links: runs/ and weights/ are symlinks to the
# Expansion drive — copy contents, not the links
rsync -a --info=progress2 --copy-links \
  "$REPO/runs/" \
  "$DEST/runs/"

rsync -a --info=progress2 --copy-links \
  "$REPO/data/images" \
  "$REPO/data/manifest.sqlite" \
  "$REPO/data/spot_polygons.json" \
  "$REPO/data/sam3_sandbox" \
  "$REPO/data/probes" \
  "$REPO/data/annotation_queue_phase1.json" \
  "$DEST/data/"

echo "backup complete -> $DEST ($(du -sh "$DEST" | cut -f1))"
