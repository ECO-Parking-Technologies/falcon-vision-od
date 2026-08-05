# Preannotation — SAM 3 drafts for human audit

Runs an open-vocabulary detector over the unified image store and writes
per-sensor `preannotations.coco.json` drafts that CVAT imports as first-draft
boxes. Humans audit the drafts (see `docs/annotation-sop.md`); the audited
exports train the deployed EfficientDet.

**Default backend: SAM 3** (`facebook/sam3`). Head-to-head vs Grounding DINO
on 52 store frames it found 123 confident boxes GDINO missed (mostly distant
cars), produced zero wheel→motorcycle false positives, and its car/truck
labels match our policy better. GDINO remains available as
`model_type: grounding_dino` (legacy / second opinion for consensus checks).

## One-time: get access to the gated SAM 3 weights

Meta gates the checkpoint behind the SAM license:

1. Log in at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
   and click **Agree and access**. The request goes to **manual review** —
   approval usually lands within the hour (email + status at
   huggingface.co/settings/gated-repos). Until then downloads 403 with
   "awaiting a review from the repo authors".
2. Create a **read** token at huggingface.co/settings/tokens.
3. Just run the tool (below). On first run it **prompts for the token**
   (hidden input). Per project policy the token lives only in process RAM —
   never in files, env profiles, shell history, or logs. After the ~3.4 GB
   download is cached, no token is ever asked for again.

## Running

Always from `preannotation/` in the main venv:

```bash
cd preannotation

# Phase-1 queue only (the sampled annotation frames from queue_file):
PYTHONPATH=.. python3 run_preannotation.py --config config.yaml --skip-existing

# Whole store (pre-compute drafts for future phases; resumable, ~1.4 s/frame):
PYTHONPATH=.. python3 run_preannotation.py --config config.yaml --all-frames --skip-existing
```

Console shows a live per-garage table (sensors %, frames, boxes, skipped);
details stream to `data/preannotation.log` automatically. Ctrl-C any time —
`--skip-existing` resumes at frame granularity and also picks up frames added
by later portal pulls, merging new drafts into existing per-sensor JSONs.

Visual QA of drafts (local browsing only — customer CCTV):

```bash
python3 preannotation/render_drafts.py                      # 3 frames/garage
python3 preannotation/render_drafts.py --garage <slug> --count 24
# -> data/draft_previews/index.html
```

## After drafting: into CVAT

```bash
python3 export_cvat_tasks.py            # per-garage bundles -> data/cvat_tasks/
python3 ../cvat/create_tasks.py --host http://<host>:8085 --project "Falcon Vision v2"
```

To wipe CVAT tasks and re-import from better drafts:
`python3 ../cvat/purge_tasks.py` (deletes tasks/annotations, keeps project,
labels, and users), then re-run the two commands above.

## Config notes (`config.yaml`)

- `model_type`: `sam3` (default) | `grounding_dino` | an efficientdet variant
  (for drafting with our own fine-tuned model in later rounds).
- `sam3.score_threshold`: 0.5 default — SAM 3 scores are calibrated; lower it
  only with evidence.
- `queue_file`: the sampled-frame allowlist; `--all-frames` ignores it.
- SAM 3 backend details (per-class NMS dedup, COCO id mapping, token
  handling): `sam3_model.py`.
