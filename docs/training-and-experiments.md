# Training, experiments & packaging (state as of 2026-08-04)

The pipeline is a **distillation loop**: SAM 3 (open-vocabulary teacher, gated
weights, see [preannotation/README.md](../preannotation/README.md)) drafts
boxes for every store frame; small EfficientDet students train on those
drafts; humans grade and correct rather than annotate from scratch.

```
unified store (data/images, 117k frames, 39 garages)
   └─ SAM 3 drafts (per-sensor preannotations.coco.json, 724k boxes)
        ├─ CVAT audit subset (~4k frames) ── human gold: eval yardstick +
        │                                    blind-spot hunting (exports in
        │                                    data/cvat_exports/ OVERRIDE drafts)
        └─ training (run_training_from_config.py / run_sweep.py)
             └─ auto: per-class metrics, dashboard row, TFLite variants,
                      sensor drop-in packages (dynamic/f32/int8 @ native res)
```

## Experiment machinery

- **`run_sweep.py`** — two modes:
  - size sweep (`--sizes 6000,12000,…,full [--arm-b]`): nested, log-spaced
    train subsets, one frozen val set, constant gradient-step budget
  - capacity ladder (`--models tf_efficientdet_lite1,…,tf_efficientdet_d2`):
    one full-store run per architecture
- **Frozen split** (`split_by: sensor-hash` in
  [config/train_sam3_full.yaml](../config/train_sam3_full.yaml)): val
  membership = hash(salt: garage/sensor) — never changes as the store grows;
  never change `split_salt`. Whole sensors on one side (no camera leakage);
  subsets are nested prefixes.
- **Tracking**: TensorBoard scalars per run (`train/<ts>/tb/`), per-class +
  size-banded `coco_metrics.json` (run_metrics.py), `run.json` provenance
  manifest, and a self-contained dashboard —
  `experiments/falcon-vision-effdet/report.html` — regenerated after every run.
- **Eval harnesses**: `validate.py` (PyTorch), `eval_tflite_coco.py` (any
  TFLite incl. the prod baseline; 4th arg `ours` for our class indices),
  `preannotation/render_drafts.py` (visual browse of drafts).
- **Reading metrics**: garage data is ~94% car — ALWAYS read per-class AP
  (car, person), never the 6-class mean (motorcycle/bus have ~no support and
  drag the mean to meaninglessness). Val labels are SAM3's: sweep numbers are
  distillation fidelity; honest accuracy comes from audited gold.

## Results so far (frozen 5k val, SAM3 labels)

- **Data curve (lite0)**: car AP flat ~0.463 from 6k→50k train images —
  **lite0 saturates at ~6k**; full-store 0.4756 (small cooldown-step caveat).
  Fixed-camera fleets: diversity is the currency, not volume.
- **COCO replay (arm B, 25% cap)**: car AP −1.2, person +0.004 → rejected for
  lite0. Person AP (~0.02 at every size) is resolution/capacity-bound, not
  data-bound — the capacity ladder's main question.
- **Round 0** = arm A full-store lite0 (train/20260803-182412): car AP 0.4756
  / AP-large 0.8508. vs prod baseline on the same yardstick: ~0.26 car AP.
- **Human annotation strategy** (revised): ~3–5k audited frames as eval gold +
  condition-slice blind-spot checks; no volume box-drawing. Audited CVAT
  exports automatically override drafts in every training build.

## Packaging & on-device latency (CM3, bench-measured)

Every training run auto-exports at the model's **native input size**:
`.dropin<size>.tflite` (dynamic-range), `.dropin<size>.f32.tflite`, and
`.dropin<size>.int8.tflite` (static PTQ, 256 garage-frame calibration) — all
byte-compatible with the sensor contract and validated vs the baseline.
Firmware reads input dims from the file (never ship a 448 build of a 320
model: exactly 2× latency — that mistake is why auto-export is size-aware).

| lite0@320 variant | on-device | car AP (dropin, frozen val) |
|---|---|---|
| dynamic | ~3.0 s | — |
| f32 | ~2.6 s | 0.438 |
| int8 | **~1.7 s** | 0.421 |

- The 2.6 FW runtime DOES run new-style int8 (verified on bench 2026-08-03).
- 2.6-era XNNPACK accelerates **float only**; our current export graph would
  fragment it into 75 delegate partitions (109 TRANSPOSEs from the converter)
  → **export cleanup (NHWC-native path) is the next latency lever**, then one
  bench session: cleaned-int8 vs cleaned-f32+XNNPACK decides the CM3 variant.
- Decision policy: accuracy first while OD latency ≤ ~4 s (OD is the fusion
  confirm voice at 40% duty; prod ran 3 s for years).

## Queued next

1. Capacity ladder (lite1→d2) — running; watch person AP vs tier
2. Export cleanup (kill the transposes) → Greg's XNNPACK rebuild test
3. Round 1: winner + Tier-1 photometric augmentation (night/glare/WB)
4. Spot-occupancy evaluator vs portal validations (the business metric)
5. FVS2 planning: native-res training-image capture + modern runtime in the
   firmware spec from demo unit #1 (see project memory / roadmap)
