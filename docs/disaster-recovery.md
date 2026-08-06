# Disaster Recovery — restoring onto a fresh machine

What the nightly backup protects (see `backup_valuables.sh`): everything
irreplaceable that git doesn't track — `runs/` (checkpoints, exports, metrics),
`data/images` (frames + SAM 3 drafts + in-spot attributes), `manifest.sqlite`,
`spot_polygons.json`, `sam3_sandbox`, `probes`, `annotation_queue_phase1.json`.
Everything else is regenerable from the repo (venvs, pretrained `weights/`,
`draft_previews`).

Backups are namespaced per machine: `<backup-root>/<hostname>-<machine-id8>/`.
On a fresh machine the tag will be *new*, so old backups are never overwritten —
you restore **from the dead machine's tag**, then new backups write under the
new machine's tag.

## 0. What you need

- The repo (GitHub) — clone it.
- ONE of: the NAS backup dir, or the Azure Blob container + a read SAS URL
  (prompted at runtime, never written to disk — restore is interactive).
- A data disk with ~20 GB+ free (any path works; below assumes an external
  drive mounted like the original Expansion disk).

## 1. Clone and prepare the data root

```bash
git clone <repo-url> ~/projects/falcon/falcon-vision-od
cd ~/projects/falcon/falcon-vision-od

DATA_ROOT=/media/$USER/Expansion/falcon-vision-od-data   # pick your disk
mkdir -p "$DATA_ROOT"
```

## 2. Pull the backup

From the NAS (preferred — LAN speed):

```bash
OLD_TAG=<dead-machine-hostname>-<id8>        # ls the backup root to find it
rsync -a --info=progress2 /mnt/nas/falcon-backup/$OLD_TAG/data/ "$DATA_ROOT/"
rsync -a --info=progress2 /mnt/nas/falcon-backup/$OLD_TAG/runs/ "$DATA_ROOT/runs/"
```

From Azure (if the NAS is gone too):

```bash
read -rs AZURE_SAS_URL     # container-scoped SAS, RAM only
rclone copy ":azureblob,sas_url=${AZURE_SAS_URL}:/$OLD_TAG/data" "$DATA_ROOT" --info=progress2
rclone copy ":azureblob,sas_url=${AZURE_SAS_URL}:/$OLD_TAG/runs" "$DATA_ROOT/runs" --info=progress2
unset AZURE_SAS_URL
```

## 3. Recreate the repo symlinks

The repo expects three symlinks (this exact layout — `data` is the root itself,
`runs`/`weights` are subdirs of it):

```bash
mkdir -p "$DATA_ROOT/weights"                 # regenerable, not in backup
ln -sfn "$DATA_ROOT"          data
ln -sfn "$DATA_ROOT/runs"     runs
ln -sfn "$DATA_ROOT/weights"  weights
```

## 4. Rebuild the regenerable pieces

```bash
bash setup_venv.sh              # main venv (training/eval/preannotation)
bash setup_export_venv.sh       # TFLite export toolchain
bash setup_convert_venvs.sh     # onnx2tf + TF 2.8 venvs (created on the data disk)
```

Pretrained backbones re-download automatically on first training run (timm).
SAM 3 weights are gated — re-run the HF license flow per
[preannotation/README.md](../preannotation/README.md) if you'll preannotate.

## 5. Verify before trusting it

```bash
python3 -c "import sqlite3; c=sqlite3.connect('data/manifest.sqlite'); \
  print(c.execute('select count(*) from frames').fetchone())"      # frame count sane?
xdg-open runs/report.html                                          # dashboards render?
ls runs/*/*/train/model_best.pth.tar | tail                        # checkpoints present?
falcon-vision-od-venv/bin/python eval_inspot.py runs/<latest-session>/<level>   # metric reproduces?
```

Expected reference numbers are in [training-and-experiments.md](training-and-experiments.md)
(e.g. lite1 in-spot car AP 69.1 strict / 88.7 AP50 on the 20260803 ladder session).

## 6. Re-arm backups on the new machine

```bash
./setup_backup_service.sh       # one-time; new machine tag, old backups untouched
```

Once the restore is verified and the first new-tag backup has completed, the old
tag's copy can be pruned from NAS/Azure manually (never automatically).
