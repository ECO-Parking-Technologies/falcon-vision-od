# CLAUDE.md — Falcon Vision OD: full project brief & findings

Fine-tuned EfficientDet object detectors for Eco Parking garage sensors (fork
of rwightman/efficientdet-pytorch), replacing the off-the-shelf
EfficientDet-Lite2 running in production. Core architecture: **SAM 3
distillation** — SAM 3 (open-vocab teacher) drafts boxes for every store
frame; small students train on drafts; humans grade and correct instead of
annotating volume. Deep docs: [docs/training-and-experiments.md](docs/training-and-experiments.md),
[preannotation/README.md](preannotation/README.md), [cvat/README.md](cvat/README.md),
[docs/sensor-architecture.md](docs/sensor-architecture.md),
[docs/annotation-guidelines.md](docs/annotation-guidelines.md) + [docs/annotation-sop.md](docs/annotation-sop.md).

## 1. Hard rules (each earned by an incident or standing policy)

- **`/media/lopezemi/Expansion/falcon-vision-ml` is READ-ONLY** — the active
  production classifier's dataset. Copy from it, never modify/delete.
- **Credentials are RAM-only**: portal/Cloudflare/CVAT/HF secrets are prompted
  at runtime (rich Prompt / getpass) — never in files, CLI args, env
  profiles, logs, or git. Presigned URLs are secrets: fetch-and-discard.
- **CVAT stays behind Cloudflare Zero Trust** — customer CCTV + open
  self-registration + aging CVAT. Never raw-public. Never publish store
  frames anywhere (no artifacts/uploads); renders are local-view only.
- **Never pip-install/upgrade in a venv while a long run uses it** — Python
  lazy imports crash mid-run (killed a preannotation run once).
- **Never change `split_salt`** ("falcon-v1") — it would move the frozen val
  set and invalidate every cross-run comparison since 2026-08-03.
- **Never ship a non-native-input dropin** — a 448 build of a 320 model is
  exactly 2× latency (real incident: 3 s → 6-7 s on a bench sensor). Sizes
  are in filenames now for this reason.

## 2. Environments & infra

| venv | purpose |
|---|---|
| `falcon-vision-od-venv` | training/eval/preannotation — torch 2.7, transformers **5.14** (SAM3 needs v5; GDINO still works on v5), tensorboard, cvat-sdk, pycocotools |
| `falcon-vision-od-export-venv` | export/packaging — litert-torch, ai-edge-litert/quantizer, onnx, onnxscript, flatbuffers |
| `<data>/onnx2tf-venv` | clean converter (onnx2tf + TF 2.19) — `setup_convert_venvs.sh` |
| `<data>/tf28-venv` | TF **2.8.4** — int8 static PTQ with sensor-era op versions AND the old-runtime (2.6-proxy) load/invoke compat check (+ cv2-headless) |

- `data/`, `runs/`, `weights/` are all symlinks onto the Expansion drive
  (`/media/lopezemi/Expansion/falcon-vision-od-data/…`, 1.8T) — the home
  nvme holds only code + venvs. Backups (`backup_valuables.sh <dest>`)
  should therefore target the home disk or an external/off-machine location,
  NOT Expansion. Optional unattended backup: `setup_backup_service.sh`
  (systemd user timer, 30 min after login + daily, idle-priority, per-machine
  subdir `<hostname>-<machine-id8>`, NAS rsync + optional Azure rclone —
  storing the SAS at ~/.config/falcon-vision-backup/ 0600 is an explicit
  consented exception to RAM-only). Restore runbook:
  docs/disaster-recovery.md.
- GPU: single RTX 3090 (24 GB). Before heavy work:
  `nvidia-smi; pgrep -f "run_training|run_preannotation"`.
  (Considered upgrades: 5060-class cards are DOWNGRADES — less bandwidth/VRAM;
  real options = used 4090 ~2×, 5090 ~3×/32GB, or a second used 3090 on this
  X570 board — second slot is chipset x4, fine for independent jobs; needs
  ~1200 W PSU. Cloud GPU is a data-governance question: CCTV frames.)
- Dev box: TUF X570-PLUS, Ryzen 9 3900X, 64 GB.

## 3. Data & unified store

- Store: `data/images/<garage>/<sensor>/<YYYY>/<MM>/*.jpg` — 117,485 frames,
  39 garages, ~7,738 sensors, pulled from the portal (org-qualified slugs,
  garage-local timezones, sha256 dedup, manifest-gated download-once).
- SAM3 drafts: per-sensor `preannotations.coco.json` — COMPLETE store
  coverage, 724,591 boxes (car 687k / truck 19k / person 17k / moto 850 /
  bus 195). Original COCO category ids (1,2,3,4,6,8).
- Portal API: GraphQL, OAuth2 refresh token (docs/portal-api/). ~1,880 human
  occupancy validations exist = free business-metric ground truth for the
  future spot-occupancy evaluator (NOT BUILT YET — high value).
- Redundancy insight: fixed cameras → effective dataset ≈ viewpoints ×
  conditions, NOT frame count. Diversity is the currency, not volume.

## 4. Preannotation (SAM 3)

- **Why SAM 3** (replaced Grounding DINO 2026-07-29 after a 52-frame
  head-to-head, `data/sam3_sandbox/`): +123 confident boxes GDINO missed
  (mostly distant cars ~29×29 px), ZERO wheel→motorcycle false positives,
  car/truck labels match our policy (SUV=car), ~1.4 s/frame on the 3090.
  GDINO backend kept as legacy/consensus option (`model_type: grounding_dino`).
- SAM 2 ≠ detector (promptable segmenter, no vocabulary). SAM 3 does text →
  ALL instances with box+score+mask; we call `post_process_object_detection`
  (boxes only); masks are one API swap away (`post_process_instance_segmentation`)
  if ever needed — `--with-masks` idea parked.
- **Weights are GATED** (huggingface.co/facebook/sam3, SAM license, manual
  approval ~1 h). Backend prompts for an HF read token in-process (RAM only)
  on first download; cache-first afterward. **License** (saved analysis):
  commercial use OK, no output/training restriction (unlike Llama), only
  distributing the WEIGHTS requires passing the license on — our sensors only
  ever get our own distilled models. Pin: license dated 2025-11-19.
- Runner (`preannotation/run_preannotation.py`): rich live table (per-garage
  rows), self-managed log `data/preannotation.log`, frame-level incremental
  `--skip-existing` (diffs frames, merges with id offsets — new portal pulls
  only draft the new frames), `--all-frames` ignores the queue file.
  Backend (`sam3_model.py`): all 6 class prompts batched per frame, per-class
  NMS 0.6 (kills fog dups), score floor 0.5 (calibrated), MAX_BOX_AREA guard.
- Visual QA: `preannotation/render_drafts.py [--garage|--sensor|--count]` →
  `data/draft_previews/index.html` (local only).

## 5. CVAT & the human role

- Self-hosted CVAT v2.23.1, traefik v3 + `TRAEFIK_CORE_DEFAULTRULESYNTAX: v2`,
  port 8085. Compose patch in `cvat/docker-compose.patch` (regenerate from the
  live clone's git diff). `CVAT_HOST` pinned in the clone's `.env`.
- **Routing**: traefik matches BOTH the tunnel hostname and the LAN IP
  (multi-host rule) and both origins are in `CSRF_TRUSTED_ORIGINS` (mounted
  production_settings.py — stock CVAT omits the setting entirely and every
  write 403s through https). Symptom map: 404 = host-rule mismatch; reads-ok
  writes-CSRF-fail = origin missing; TLS handshake failure at CF edge =
  hostname flip-flopped and the edge cert is re-minting (waits ~minutes).
- SDK scripts (`cvat/create_tasks.py`, `cvat/purge_tasks.py`): idempotent,
  creds prompted; `--cf-access` = Cloudflare Access service-token headers
  (Zero Trust stays up); `--host-header` = reach the LAN IP under a canonical
  host rule. Admin must be superuser to see others in Assignee dropdown.
- **Human strategy: ON HOLD (user 2026-08-06)** — no manual annotation
  running; gold auditing (~3–5k frames: eval yardstick + condition-slice
  blind-spot hunting, NO volume drawing) is a just-in-time task before any
  fleet-wide push. Exports → `data/cvat_exports/<garage>.json` **override
  drafts per-frame automatically** whenever they exist. CVAT currently holds
  39 tasks with attribute-prefilled SAM3 drafts (spot names + checkboxes);
  vehicle labels carry a `spot` TEXT attribute (spec generator in
  convert_to_cvat.py; ensure_labels also syncs missing ATTRIBUTES onto
  existing project labels).
- Annotation policy: identifiability rule (unlabeled visible vehicle =
  negative supervision — box everything identifiable, skip <12 px/smudges);
  InEcoParkingSpot/InMotion/Occluded attributes; condition tags.

## 6. Training system

- Adapter (`run_training_from_config.py`): `annotated_files: auto` discovers
  store drafts; merges to COCO train/val with symlinked images (ABSOLUTE
  symlink targets — relative base paths once produced dangling links);
  audited-export override; label remap 1-based contiguous.
- **Frozen split**: `split_by: sensor-hash` — membership =
  md5(salt:garage/sensor), whole sensors to one side (no camera leakage),
  stable as the store grows; hash-ordering makes `max_train_images` subsets
  NESTED prefixes. `val_max_images: 5000`. Shared-split placement falls back
  to level-local when an existing session split's params.json mismatches
  (never clobbers).
- **Per-tier deployment recipes (empirical, 2026-08-12)** — recipes are
  per-tier, NEVER copied across tiers untested:
  - **lite1 = a20 + 3x + fill0** (`anchor_scale: 2.0`,
    `train_steps_budget: 75000`, `--fill-color 0`) → 69.8/90.9 in-spot,
    car 56.8, person 4.3. Each lever won/tied its weekend single-arm run;
    the composed run beat both single-arm fronts. fill0 is free at 384 and
    matches the FW's black letterbox.
  - **lite0 = a20 + 3x, NO fill0** → 65.5/87.7 in-spot. Chosen by the full
    2^3 lattice (all 8 cells measured): a20 and 3x compose additively, but
    **a20×fill0 is toxic at 320** (a20+fill0 61.5/80.9, and it sank the
    triple to 62.8/82.7). Hypothesis: at 320 the pad is ~25% of the tensor
    and dense scale-2.0 anchors over hard black make junk targets a 4M
    model can't absorb. fill0's FW-match benefit is forfeited at this tier
    — field A/Bs show gray-trained models perform fine on the black-pad FW.
  - Upper tiers (lite2+, d-series): no lever validation yet — derive
    anchor_scale from the per-input-size anchor-fit analysis and validate
    per tier before folding into any retrain.
  (e.g. bifpn_sum) plumb through train.py → args.yaml → run_metrics →
  validate.py so post-run scoring always rebuilds with the training config.
  Auto-export REFUSES override-trained checkpoints until the packagers take
  the flags (silent wrong-anchor decode otherwise).
- **`roi_crop: true` + `roi_tensor: N`** trains on FW-geometry spot-region
  crops (roi_crop.py mirrors AdjustRoi incl. pads 0.1 + expandWidth/Bottom;
  calibration-matched per frame). SHELVED — see findings; kept as a flag.
- `epochs: auto` = constant gradient-step budget (`train_steps_budget: 25000`)
  → sweep points cost equal wall-clock. **Gotcha**: timm adds 10 LR-cooldown
  epochs on top (slightly favors big datasets; zero out via
  `--cooldown-epochs 0` in extra_args for strict fairness).
- COCO mixing: `include_coco` + `coco_root` (READ-ONLY legacy archive — no
  20 GB re-download) + `coco_max_frac` cap + train-only (val stays pure
  garage). 70,082 COCO images contain our classes (262k persons).
- Multi-level sessions are the DEFAULT: the config's `levels:` list (names
  or variant dicts with overrides — `{name: lite0-25k, model: lite0,
  max_train_images: 25000}`) trains everything into ONE session dir
  (`<output>/<datetime>/<level>/{train,export}` + session-shared `split/`
  reused across levels via a params.json guard; data-modified entries get
  local splits). run_sweep.py is retired — its jobs are config entries now.
- **Output root is `runs/`** (renamed from experiments/falcon-vision-effdet
  2026-08-06): `runs/<session-dt>/<level>/{train,export}` + per-session
  report.html; `runs/latest-<level>` symlinks to the newest export; global
  `runs/report.html` + `runs/ladder.html`. History migrated: the full ladder
  lives merged in `runs/20260803-182412/` (lite0…d2, one shared-format split
  per level). Artifact filenames prefix `<session>-<level>.…`. All tools
  (run_metrics, eval_inspot, build_report --session, find_ckpt globs)
  resolve both this layout and migrated dirs.
- **Augmentation today**: ONLY hflip + RandomResizePad scale-jitter 0.1–2×
  (aspect-preserving ≈ letterbox, conveniently matching FW preprocessing) +
  random interpolation. `--color-jitter` exists but is COMMENTED OUT in
  train.py. **Round-1 plan**: Tier-1 photometric (gamma/brightness/noise for
  night, sun-flare glare, sodium↔LED white balance) via albumentations, flag
  per aug, one-variable experiments on frozen val; Tier-2 lens-occluder
  overlays (spider webs/smudges — the CVAT spider-web frame is the reference)
  + random-erasing; Tier-3 alternative: mine REAL bad frames by brightness
  stats and oversample. Never change augs mid-sweep.
- Tracking: TensorBoard per run (`train/<ts>/tb/`), `run_metrics.py` →
  per-class + size-banded `coco_metrics.json` + `run.json` provenance;
  dashboard `runs/report.html` auto-refreshes;
  ladder page `runs/ladder.html`.

## 7. Metrics discipline

- Garage data ~94% car → **read car AP and person AP, never the 6-class
  mean** (mean 0.093 hid car 0.44 once; motorcycle/bus have ~no support).
- **34% of val boxes are <12 px** short-side at 320 input — sub-resolvable;
  size-banded AP (APsmall/med/large) separates "can't see" from "misses".
  AP-large ≈ the close/in-spot band that decides occupancy.
- Val labels are SAM3's → all sweep/ladder numbers = distillation fidelity;
  a student rarely exceeds its teacher; systematic teacher errors are
  invisible to self-eval (why gold audits + condition slices exist).
- Person gate: prod baseline 3.8% person AP — fleet-wide promotion bar.
- **THE product metric: in-spot car AP** (`eval_inspot.py <run-dir>`) —
  portal spot polygons (pulled PER SNAPSHOT RUN, calibration-drift-aware;
  frames matched to their own run's calibration via run8 in filenames) →
  `label_inspot.py` stamps InEcoParkingSpot/spot attributes on all 724k
  draft boxes (31% in-spot) → eval scores ONLY in-spot GT (rest = iscrowd
  ignore regions). Round-2 gate: changes commit only on a measured win here.
- Sensor-geometry insight: down-lane sensors' in-spot cars ≈ large band,
  across-lane ≈ medium band (the weaker one; capacity helps it most).
- Eval tools: `validate.py` (PyTorch), `eval_tflite_coco.py <root> <out>
  <model> [ours]` (any TFLite; `ours` = our 0-based class indices vs COCO-90
  baseline), `run_metrics.py`, `.ts.pt` via `torch.jit.load` (full NMS,
  outputs [100,6] xyxy/score/1-based-class).

## 8. Findings — data & architecture (frozen 5k val, car AP as %)

- **Prod baseline (off-shelf lite2@448 int8, tfhub)**: car 26.7 / large 51.5 /
  person 3.8 / recall 32.1 (recall-large 57.0).
- **Data curve (lite0, nested subsets, equal steps)**: 6k→50k FLAT (~46.3);
  full-store 47.6. **lite0 saturates at ~6k images.** Round-0 model beat the
  prod baseline by +58% car AP and +51% recall from 4.1k images/45 min.
- **Arm B (+COCO 25% cap): REJECTED for small models** — car −1.2, person
  +0.4 only. But pure-garage models hallucinate on out-of-domain scenes
  (desk test: 91% "cars" on furniture) — catastrophic forgetting of world
  knowledge; COCO replay remains the fix if that ever matters in prod.
- **V2 LADDER (2026-08-14, session 20260812-121457) — per-tier recipes,
  NEW BASELINES (v1 ladder retired)**. In-spot strict/AP50 · all-boxes car ·
  person: lite0 65.4/87.7·50.5·2.1 — lite1 69.7/90.6·56.5·4.0 — lite2
  72.7/92.4·60.3·7.3 — lite3 74.0/92.9·62.1·8.2 — **lite4 77.6/94.8·67.6·
  10.3** — d1 75.3/93.3·64.0·7.2 — d2 NaN'd (bs4+amp fp16, retraining
  amp-off). Recipes REPRODUCE (lite0/lite1 within 0.3 of discovery runs).
  **The v1 "~62.5 ceiling" was largely an ANCHOR artifact, not capacity:
  lite4 with anchors 2.5+fill0 beats d1 — FVS2 pick reopened (lite4 vs
  d-series-with-tuned-anchors TBD).** Person gate (3.8) clears from lite1 up.
- **v1 capacity ladder (RETIRED 2026-08-14; kept for history)**:
  lite0 47.6 · lite1 53.9 · lite2 59.3 · lite3 60.0 · lite4 60.8 · d1 62.7 ·
  d2 62.5. Person %: 2.1 / 3.2 / 5.4 / 5.3 / 7.3 / 6.1 / 2.2 (d2's person
  collapse = schedule artifact: bs4 → 2 real epochs). **Car accuracy
  saturates ~62.5% — d1 confirmed FVS2/NPU pick** (d2 buys nothing).
  - Knee at lite2 (+12 over lite0, then <1/tier) — Greg's off-the-shelf-era
    intuition reproduced under domain training.
  - **d1 beats lite4 with half the params** — d-series architecture (deeper
    BiFPN, SiLU) > scaling lite; provisional FVS2/NPU pick.
  - Person tracks capacity/resolution, NOT data — gate clears from lite2 up.
- Quantization accuracy cost (lite0): int8 −1.7 pts vs f32 with 256-frame
  calibration (was −3.6 with 64 frames — calibration diversity matters).
  TFQ (TF-converter PTQ) beat AEQ static by +1.3 car AP on lite2.
- **In-spot car AP baselines (strict/AP50 %, 8,441 in-spot val boxes)**:
  lite0 63.8/84.6 · lite1 69.1/88.7 · lite2 72.5/91.1 · d1 74.9/92.2 —
  in-spot accuracy is NOT saturated up the ladder (unlike all-boxes).
- **ROI-crop training REJECTED (2026-08-06, user: ditch it)**: lite1 trained
  on FW-geometry crops scored 73.3/92.3 in-spot vs ladder lite1's 74.1/93.1
  on the SAME cropped val — full-frame training generalizes to crops better
  than crop-specialized training (scale jitter covers the geometry; crops
  lose context/diversity). The FW's inference-time crop itself is RIGHT
  (free zoom: same lite1 is 69.1 on full frames vs 74.1 on crops).
  eval_inspot takes --split/--predictions to score any ckpt on any split.
- **Anchor-fit analysis (shape-IoU upper bound, 400 sensors)**: at default
  anchor_scale 3.0, 36% of full-frame car boxes (63% person!) can't reach
  0.5 IoU with ANY anchor; scale 1.5–2.0 dominates every slice — likely the
  structural cause of low person AP; zero latency cost. Weekend arms test it.
- **Letterbox fill mismatch (Greg's catch)**: FW pads BLACK; effdet defaults
  to ImageNet-mean GRAY (input_config.py 'mean' fallback — lite configs
  don't set fill_color). All runs to date trained gray; fill0 arm measures
  the fix (--fill-color 0). Eval pad is top-left vs FW centered (minor).
- **Why the ~62.5% all-boxes ceiling isn't what it looks like**: AP-large is
  87-91% (≈90% teacher fidelity on the operational band); the "missing"
  accuracy is (a) sub-resolvable tiny boxes SAM3 labels at 1008px that
  students can't see at 320-640, (b) strict-IoU box-jitter vs the teacher's
  own boxes. Also d1≈d2 because store frames are 640×480 — inputs beyond
  ~640 are upscaling air (changes with FVS2 4K), and d2's person collapse
  (2.2%) = bs4/2-epoch schedule artifact.
- Surface-lot probe (roof-camera image through everything,
  `data/probes/2026-08-05-surface-lot/`): garage students partially
  transfer (side-profile viewpoint gap; mid-tiers hallucinate a pavement
  mega-box; d1 conservative OOD — safest failure profile); prod baseline
  clean-but-partial; **SAM3 9/9 perfect incl. a distant pedestrian** — the
  pole/lot domain fine-tune path is pre-validated.

## 9. Export & conversion — the deep lessons

- **Legacy path (litert-torch) pollutes graphs**: lite2 = 888 ops vs Google's
  357 for the same architecture — 173 TRANSPOSE + 92 RESHAPE (layout churn)
  which then shatter int8 into float islands (147 QUANTIZE / 42 DEQUANT
  boundaries). Desktop hides this (bandwidth); the A53 amplifies it ~2×.
- **Clean path (DEFAULT since commit 86cdf5e)**: NCHW wrapper →
  `torch.onnx.export` (opset 17) → **onnx2tf `-kat images`** → quantize →
  surgery. Results: 369 ops / 1 transpose; frozen-val AP IDENTICAL to 3
  decimals; TF2.8 proxy loads+invokes; on-device 6–19% faster.
  - onnx2tf gotchas: NHWC-input ONNX gets layout-MANGLED without `-kat` and
    CRASHES (pickle bug) with it → **export NCHW + `-kat` is the working
    combo**; surgery bridges uint8-NHWC→(dequant|quant)→TRANSPOSE→NCHW
    (one input-sized transpose, ~free). `onnxsim` must be on PATH.
- **Quantizers**: f32 = onnx2tf output as-is. dyn = AEQ dynamic_wi8_afp32.
  int8 = **TF-2.8 TFLiteConverter static PTQ on the saved_model** (fused
  chain, DQ 24→2, era-appropriate op versions, +1.3 AP vs AEQ static).
  Calibration: 256 diverse garage frames, raw 0-255 NCHW (norm is in-graph).
- **Surgery** (`package_dropin.apply_surgery`): uint8 interface input
  (float body: scale-1.0 DEQUANTIZE v1; int8 body: exact requant zp+128
  QUANTIZE v2; NCHW body: + TRANSPOSE bridge), anchors from effdet Anchors
  (ycxcyhw normalized, scales all 1.0 — effdet regressions are unscaled),
  TFLite_Detection_PostProcess graft mirroring the baseline's flexbuffer
  options (max_det 25, iou 0.5, score −inf), int8 heads get DEQUANTIZE v2
  before the op. `repack_for_old_runtimes`: inline external buffers +
  **int64→int32 PAD paddings downcast** (2.6–2.8 misread int64 paddings →
  graph-prepare failures).
- **Validation harness** (every build): interface match vs
  `baseline/efficientdet_lite2.tflite` (repo-tracked), car-image detection
  overlap (IoU vs baseline), empty-frame silence, + TF2.8 no-delegates
  load/invoke as the 2.6 proxy. Desktop timing convention: 2 threads,
  BUILTIN_WITHOUT_DEFAULT_DELEGATES — but desktop LOST latency-prediction
  authority for boundary-op effects (AEQ-vs-offshelf tied on desktop, 1.8×
  apart on device). Only the sensor bench settles device latency.
- **Naming**: `<run>.dropin-<size>-{f32,dyn,int8}.tflite` (sensor-ready) ·
  `<run>.raw-{f32,dyn,int8}.tflite` (dev-only) · `<run>.ts.pt` · manifest
  keys `dropin_<size>_<q>` / `raw_<q>` / `torchscript`. One
  `package_dropin.py` call builds+validates all three variants;
  `--legacy-export` = old path (no int8). `<run>` =
  run_name_for_checkpoint = `<session-dt>-<level>[-roi]` (-roi auto-appended
  from run.json if the level name hides it) — exporters must NEVER name from
  art_dir.name (that's literally "export" in the session layout; bit us
  2026-08-06). generate_model_files reads train_sam3_full.yaml.
- Big-tier packaging RESOLVED 2026-08-05/06: lite4's "failure" was the
  validation fixture (at 640 it sees a real vehicle through a wall opening in
  the empty reference frame → empty-check now fails only >0.40, the sensor's
  strong-confirm threshold); d1's TFQ dies on 'illegal scale: INF' → packager
  auto-falls-back to AEQ static (manifest records which); d2's TFQ worked
  fine. All 7 tiers now have validated dropin-<size>-{f32,dyn,int8} sets.

## 10. On-device latency ledger (CM3 bench, all measured)

| build | latency |
|---|---|
| off-shelf lite2@448 int8 (prod) | ~3.0 s |
| lite0-320: dyn / f32 / int8(clean) | 3.0 / 2.6 / **1.6 s** |
| lite1-384 int8 clean | **3.1 s** |
| lite2-448 int8: legacy / clean-AEQ / clean-TFQ | 6.6 / 5.4 / 5.6 s |

- int8 ≈ 1.7× faster than f32 on the A53; **dynamic-range is SLOWER than f32**
  (hybrid-conv reference-kernel penalty on 2.6) — dyn is a compat variant,
  never the speed pick.
- 2.6-era **XNNPACK is float-only** (Greg's null result with int8 models was
  correct-by-mechanism); our legacy graph would shatter it into 75 delegate
  partitions — with the clean graph, f32+XNNPACK (FW rebuild) is an untested
  but plausible ~1.3–1.6 s lever.
- Residual lite2 gap vs off-the-shelf (5.4 vs 3.0 despite identical desktop
  timing): the weighted-fusion elementwise-chain hypothesis is **FALSIFIED**
  — lite2-sum (bifpn_sum retrain, accuracy-identical 72.5/91.1 in-spot,
  car 59.2) measured **5.8 s** int8 on the bench 2026-08-10, no better than
  weighted (5.4–5.6). Remaining suspects: unfused RELU6, explicit PADs,
  memory traffic. **lite2@448 is DEAD on CM3 — lite1 is the CM3 family;
  lite2 belongs to FVS2's NPU** (user decision 2026-08-10). Untested
  long-shot: clean-f32 + XNNPACK FW rebuild (Greg-side).
- OD Elapsed on the fusion page includes downstream per-object cost — more
  detections = more tracker/signature work; compare like-for-like scenes.

## 11. Deployment state & procedure

- **Targets (user decision 2026-08-05)**: PRIMARY **lite1 dropin-384-int8**
  (3.1 s, car 53.9% = 2× prod, large 87.7%); fallback **lite0-320-int8**
  (1.6 s, 47.6%). lite2 challenger pending plain-sum retrain. lite1's person
  3.2% is slightly under the 3.8% gate — fine for vehicle-focused sensors,
  round-1 target for fleet-wide.
- Swap procedure: upload dropin via sensor
  `/plugin/od-model/staging/upload` → `install` (atomic, sha256 sidecar, no
  FW change); verify Model SHA256 + Tensor Size on the Fusion Analysis page.
  Watch Override Rate — OD out-voting the classifier is where our value shows.
- Real-garage validation (L1-EL-S15, lite1-int8): spot car 98-99%, far row
  detected at 27-96%, CUSUM_CONFIDENT.
- **Field A/B (Switch test sensor, 2026-08-10, od_history_compare.py +
  SAM3 grading)**: lite1-3x vs ladder lite1 on the live sensor archive —
  precision 0.70→0.89 / recall 0.37→0.51 (@0.25); 0.77→0.94 / 0.26→0.41
  (@0.40); confidence p50 0.54→0.95. First field confirmation that
  frozen-val gains transfer. Caveat: new window was 4 daytime hours vs 66
  mixed; rerun matched-window before quoting externally. The tool
  (od_history_compare.py --sam3 N, CF Access RAM-only) is the pilot-phase
  measurement instrument. Desk/bench scenes are OUT-OF-DOMAIN
  for pure-garage models (hallucinated furniture-cars are expected there) —
  bench measures latency, frozen val + garages measure accuracy.

## 12. Sensor / firmware contract (full ref: docs/sensor-architecture.md)

CM3+ @600 MHz, TFLite **2.6** C++, 2 threads, no XNNPACK, **new-style
full-int8 WORKS** (bench-verified 2026-08-03 — old assumption overturned).
FW reads input dims FROM THE MODEL (native-size dropins need no FW change);
feeds per-camera ROI crops, letterboxed, BGR→RGB; uint8 raw-byte memcpy or
float input; parses 4-tensor TFLite_Detection_PostProcess by index (or
[1,N,6]); hardcoded COCO map but fusion only uses PERSON vs VEHICLE (our
index shifts harmless); maxDetections config default 10. Two models run:
per-spot classifier (MobileNetV3 — proof that MNv3 ops are field-proven on
this runtime; an MNv3+BiFPN experiment is a config string away) + our OD
model in ConfirmFusion (OD ≥0.40 strong-confirm, 40% duty cycle).
**Latency policy: accuracy first while OD ≤ ~4 s** (prod ran 3 s for years;
OD confirms parked cars — seconds don't matter, points do).

## 13. Future hardware tracks

- **FVS2 (1–2 yr, possibly 4K + NPU)**: bake into firmware spec EARLY —
  (1) native-res training-image capture endpoints from demo unit #1,
  (2) modern TFLite/NPU runtime (sheds all the 2.6 archaeology). No 4K
  pre-collection possible or needed; current corpus transfers as
  pretraining. 4K value = distant spots become in-scope (ROI/tiling,
  d-series territory); transposes/layout hurt NPU compilers even more.
- **The surface-lot domain is LIVE TODAY**: Eco "Roof Zonal" zone counters
  run on Jetson Xavier NX (ZED 2i stereo, JetPack r35, GPU 90%/mem 87%
  utilized — no room for SAM3 on-device). The 2026-08-05 probe image came
  from one. Opportunity: use Zonal units as FRAME SOURCES into the store →
  SAM3 drafts (proven 9/9 on that domain) → d-series fine-tune, long before
  FVS2. Jetson buys: Xavier is EOL/JetPack-frozen (torch ≤2.1 — SAM3 needs
  modern torch); **Orin NX 16GB is the pick** (pin-compatible with Xavier NX
  carriers, current JetPack, ~3-5×; Orin Nano Super $249 for prototyping).
  SAM3 at 0.2 Hz: comfortable on Orin, stack-archaeology on Xavier; a
  distilled d2 beats on-edge SAM3 whenever the vocabulary is fixed.
- Jetson lot product recipe: d2/d3-class + tiled 4K + temporal voting;
  TensorRT path; if YOLO considered, mind AGPL (Ultralytics) — YOLOX/RT-DETR
  are Apache. SAM3 = server-side oracle + annotation factory, not edge.

## 14. Roadmap / open threads

**Round-2 protocol (user): improve the deployment target, one lever at a
time, judged by eval_inspot.py; commit recipe changes ONLY on measured wins.
Annotation is ON HOLD (CVAT recreated with attribute-prefilled drafts; gold
auditing = just-in-time before any fleet-wide push).**

- **lite0 full 2^3 lever lattice (2026-08-12) — WINNER lite0-a20-3x
  65.5/87.7** (+1.7/+3.1 over ladder lite0's 63.8/84.6; packaged with
  --anchor-scale 2.0). All cells: a20 64.5/87.1 · 3x 64.9/85.2 · fill0
  63.9/84.6 · a20+3x 65.5/87.7 · a20+fill0 **61.5/80.9** · 3x+fill0
  64.7/85.2 · triple 62.8/82.7. The triple's failure = **a20×fill0
  interaction** (each fine alone; toxic together at 320 — dense small
  anchors over a 25%-black pad?). Per-tier recipes are now empirical:
  **lite1 = a20+3x+fill0, lite0 = a20+3x (NO fill0)**. Eval pads gray
  (fill0-trained models carry a small eval mismatch; fill0-alone ≈ wash so
  the interaction is real, not artifact).

**Rollout ladder (user plan 2026-08-10)**: Mon night 3x on 3 Switch sensors
(field A/B vs SAM3, od_history_compare) → Tue: composed lite1-a20-3x-fill0
(69.8/90.9, car 56.8, person 4.3 — best lite1) on all three → Wed: split
pilot (fv7a2f8e -> composed lite0, others stay lite1 — miniature of the
garage-wide config) → weekend: garage-wide lite0/lite1 split at Switch →
if good: full retrain with combo recipe as default (fold in: master-config
anchor_scale 2.0 + fill0 + 3x budget for deployment tiers, best-ckpt
selection by car AP, --cooldown-epochs decision — NEW baselines, old
comparisons reset) → beta in other garages. Gold audit + spot-occupancy
evaluator should land before beta.

1. **Weekend arm sweep RUNNING (session 20260806-155101, launched Thu
   ~16:00)**: lite1-a20 / lite1-a15 (anchor_scale) · lite1-ema · lite1-3x ·
   lite1-fill0 (black letterbox) · lite2-sum (bifpn_sum ≤4s challenger).
   All one-lever vs ladder lite1, gate = in-spot 69.1/88.7; watcher writes
   `<session>/weekend_verdict.txt`. Winners compose into the recipe; anchor
   winners need packager flag plumbing before export. (Supersedes
   train_lite1_r2.yaml — its EMA/3x are now separate arms.)
2. Next round-2 levers (specced): spot-weighted loss (polygons + teacher
   confidence weighting), per-geometry input size (lite1@448 build =
   zero-training probe), mask-tightened boxes. ROI-crop training: REJECTED,
   shelved (flag remains).
4. Round 1: photometric augmentation (night/glare/WB) + person-AP fix
5. Spot-occupancy evaluator vs portal validations (the business metric —
   half-built now that spot polygons + in-spot eval exist)
6. Greg-side: XNNPACK FW rebuild test with clean f32; f32-vs-int8 power check
7. `lite1-coco` person-gate arm (one uncomment in the master config)
8. Parked: masks in drafts (`--with-masks`), consensus GDINO+SAM3 filtering,
   MNv3-backbone experiment, SAM3 fine-tuning (github repo, not HF),
   gateway annotation oracle, SAM3 offline packaging (container/snap —
   feasible, license rides along on distribution)
